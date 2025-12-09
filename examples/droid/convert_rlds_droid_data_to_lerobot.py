"""
Script for converting DROID dataset (TFRecord/RLDS) to LeRobot format.
Refactored to match HDF5 reference style, WITH episode_id frame features.
"""

import json
import hashlib
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tyro
from PIL import Image
from tqdm import tqdm

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

EPISODE_ID_LENGTH = 64

'''
def resize_image(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(image)
    return np.array(image.resize(size, resample=Image.BICUBIC))
'''

@tf.function
def resize_image_tf(tensor: tf.Tensor, target_w: int, target_h: int) -> np.ndarray:
    resized = tf.image.resize(tensor, [target_h, target_w], method='bicubic')
    return tf.cast(tf.clip_by_value(resized, 0, 255), tf.uint8)

def generate_hash_id(unique_string: str) -> str:
    return hashlib.md5(unique_string.encode("utf-8")).hexdigest()

def main(
    input_dir: Path,
    mapping_file: Path,
    repo_id: str = "droid_dataset_lerobot",
    push_to_hub: bool = False,
    output_root: Optional[Path] = None,
):
    root = output_root if output_root else HF_LEROBOT_HOME
    output_path = root / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    folder_name_to_id = {}
    if mapping_file.exists():
        print(f"Loading mapping from {mapping_file}")
        with open(mapping_file, "r") as f:
            raw_data = json.load(f)
            for eid, path in raw_data.items():
                folder = path.strip().split("/")[-1]
                if folder:
                    folder_name_to_id[folder] = eid
    else:
        print(f"Warning: Mapping file not found at {mapping_file}")

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        robot_type="panda",
        fps=15,
        features={
            "exterior_image_1_left": {"dtype": "video", "shape": (180, 320, 3), "names": ["height", "width", "channel"]},
            "exterior_image_2_left": {"dtype": "video", "shape": (180, 320, 3), "names": ["height", "width", "channel"]},
            "wrist_image_left":      {"dtype": "video", "shape": (180, 320, 3), "names": ["height", "width", "channel"]},
            "joint_position":        {"dtype": "float32", "shape": (7,), "names": ["joint_position"]},
            "gripper_position":      {"dtype": "float32", "shape": (1,), "names": ["gripper_position"]},
            "actions":               {"dtype": "float32", "shape": (8,), "names": ["actions"]},
            "episode_id":            {"dtype": "uint8", "shape": (EPISODE_ID_LENGTH,), "names": ["id_bytes"]},
        },
        image_writer_threads=30,
        image_writer_processes=0,
    )

    print(f"Loading TFRecords from: {input_dir}")
    builder = tfds.builder_from_directory(str(input_dir))
    ds = builder.as_dataset(split="train")
    total_episodes = builder.info.splits["train"].num_examples
    
    stats = {"matched": 0, "fallback": 0, "processed": 0}

    for episode in tqdm(ds, total=total_episodes, desc="Converting"):
        stats["processed"] += 1
        
        # --- ID Logic ---
        meta = episode.get("episode_metadata", {})
        file_path_raw = meta.get("file_path", b"unknown")
        file_path = (file_path_raw.numpy().decode("utf-8") if hasattr(file_path_raw, "numpy") else str(file_path_raw))
        target_folder = Path(file_path).parts[-2] if len(Path(file_path).parts) > 1 else Path(file_path).name
  
        if target_folder in folder_name_to_id:
            curr_id = folder_name_to_id[target_folder]
            stats["matched"] += 1
        elif (fuzzy := target_folder.replace("__", "_")) in folder_name_to_id:
            curr_id = folder_name_to_id[fuzzy]
            stats["matched"] += 1
        else:
            curr_id = generate_hash_id(target_folder)
            stats["fallback"] += 1

        # --- [Modification]: Convert String ID to Uint8 Array ---
        # Encode string to bytes
        id_bytes = curr_id.encode("utf-8")[:EPISODE_ID_LENGTH]
        # Create a zero-filled numpy array
        id_array = np.zeros(EPISODE_ID_LENGTH, dtype=np.uint8)
        # Fill it with the byte data
        id_array[:len(id_bytes)] = np.frombuffer(id_bytes, dtype=np.uint8)

        # --- Language Instruction ---
        language_instruction = "do something"
        for step in episode["steps"]:
            if "language_instruction" in step:
                lang_tensor = step["language_instruction"]
                if lang_tensor.dtype == tf.string:
                    language_instruction = lang_tensor.numpy().decode("utf-8")
                    break

        # --- Frame Loop ---
        for step in episode["steps"]:
            obs = step["observation"]

            joint_action = step["action"].numpy()
            gripper_action = obs["gripper_position"].numpy()
            full_action = np.concatenate([joint_action, np.atleast_1d(gripper_action)])

            dataset.add_frame({
                "exterior_image_1_left": resize_image_tf(obs["exterior_image_1_left"], 320, 180).numpy(),
                "exterior_image_2_left": resize_image_tf(obs["exterior_image_2_left"], 320, 180).numpy(),
                "wrist_image_left":      resize_image_tf(obs["wrist_image_left"], 320, 180).numpy(),
                "joint_position":        obs["joint_position"].numpy().astype(np.float32),
                "gripper_position":      obs["gripper_position"].numpy().astype(np.float32),
                "actions":               full_action.astype(np.float32),
                "task":                  language_instruction,
                "episode_id":            id_array,
            })

        dataset.save_episode()

    print(f"Summary: {stats}")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["droid", "lerobot"], 
            private=False, 
            push_videos=True, 
            license="apache-2.0"
        )

if __name__ == "__main__":
    tyro.cli(main)