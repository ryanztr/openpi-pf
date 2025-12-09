"""
Script for converting DROID dataset (in TFRecord/RLDS format) to LeRobot format.
Version 6: With final statistics summary.
"""

from pathlib import Path
import shutil
import json
import hashlib
import numpy as np
from PIL import Image
from tqdm import tqdm
import tyro

import tensorflow_datasets as tfds
import tensorflow as tf

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

REPO_NAME = "droid_dataset_from_tfrecord_lerobot_6"
EPISODE_ID_LENGTH = 64


def resize_image(image, size):
    image = Image.fromarray(image)
    return np.array(image.resize(size, resample=Image.BICUBIC))


def generate_hash_id(unique_string):
    hash_object = hashlib.md5(unique_string.encode("utf-8"))
    return hash_object.hexdigest()


def main(data_dir: str, *, push_to_hub: bool = False):
    base_dir = Path("/app")
    droid_json_dir = base_dir / "dataset" / "droid_json"
    ep_path_json = droid_json_dir / "episode_id_to_path.json"

    folder_name_to_id = {}

    print(f"Loading episode ID mapping from {ep_path_json}")
    if ep_path_json.exists():
        with open(ep_path_json) as f:
            raw_data = json.load(f)
            for eid, path in raw_data.items():
                folder_name = path.strip().split("/")[-1]
                if folder_name:
                    folder_name_to_id[folder_name] = eid
        print(f"Loaded {len(folder_name_to_id)} mappings.")
    else:
        print(f"[ERROR] Mapping file not found at {ep_path_json}")

    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)
        print(f"Cleaned up existing dataset at {output_path}")

    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=15,
        features={
            "exterior_image_1_left": {
                "dtype": "image",
                "shape": (180, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "exterior_image_2_left": {
                "dtype": "image",
                "shape": (180, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image_left": {
                "dtype": "image",
                "shape": (180, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "joint_position": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["joint_position"],
            },
            "gripper_position": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["gripper_position"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["actions"],
            },
            "episode_id": {
                "dtype": "uint8",
                "shape": (EPISODE_ID_LENGTH,),
                "names": ["episode_id_bytes"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    print(f"Loading TFRecord dataset from: {data_dir}")
    builder = tfds.builder_from_directory(data_dir)
    ds = builder.as_dataset(split="train")
    total_episodes = builder.info.splits["train"].num_examples
    print(f"Found {total_episodes} episodes.")

    count_matched = 0
    count_fallback = 0
    processed_count = 0

    for episode in tqdm(ds, total=total_episodes, desc="Converting episodes"):
        processed_count += 1

        language_instruction = "do something"
        file_path = "unknown"
        episode_metadata = episode.get("episode_metadata", {})

        if "file_path" in episode_metadata:
            raw_val = episode_metadata["file_path"]
            if hasattr(raw_val, "numpy"):
                raw_val = raw_val.numpy()
            file_path = raw_val.decode("utf-8") if isinstance(raw_val, bytes) else str(raw_val)

        current_episode_id = None

        parts = file_path.split("/")
        target_folder = parts[-2] if len(parts) >= 2 else parts[-1]

        if target_folder in folder_name_to_id:
            current_episode_id = folder_name_to_id[target_folder]
        else:
            fuzzy_folder = target_folder.replace("__", "_")
            if fuzzy_folder in folder_name_to_id:
                current_episode_id = folder_name_to_id[fuzzy_folder]

        if current_episode_id is not None:
            count_matched += 1
        else:
            count_fallback += 1
            current_episode_id = generate_hash_id(target_folder)

        ep_id_bytes = current_episode_id.encode("utf-8")
        ep_id_array = np.zeros(EPISODE_ID_LENGTH, dtype=np.uint8)
        ep_id_array[: min(len(ep_id_bytes), EPISODE_ID_LENGTH)] = np.frombuffer(ep_id_bytes[:EPISODE_ID_LENGTH], dtype=np.uint8)

        steps = episode["steps"]

        for step in steps:
            if "language_instruction" in step:
                lang_tensor = step["language_instruction"]
                if lang_tensor.dtype == tf.string:
                    language_instruction = lang_tensor.numpy().decode("utf-8")

            observation = step["observation"]

            ext_img_1 = observation["exterior_image_1_left"].numpy()
            ext_img_2 = observation["exterior_image_2_left"].numpy()
            wrist_img = observation["wrist_image_left"].numpy()

            joint_pos = observation["joint_position"].numpy()
            gripper_pos = observation["gripper_position"].numpy()
            joint_action = step["action"].numpy()

            gripper_action = gripper_pos
            if gripper_action.ndim == 0:
                gripper_action = gripper_action[None]

            full_action = np.concatenate([joint_action, gripper_action], axis=0)

            dataset.add_frame(
                {
                    "exterior_image_1_left": resize_image(ext_img_1, (320, 180)),
                    "exterior_image_2_left": resize_image(ext_img_2, (320, 180)),
                    "wrist_image_left": resize_image(wrist_img, (320, 180)),
                    "joint_position": joint_pos.astype(np.float32),
                    "gripper_position": gripper_pos.astype(np.float32),
                    "actions": full_action.astype(np.float32),
                    "task": language_instruction,
                    "episode_id": ep_id_array,
                }
            )

        dataset.save_episode()

    print(f"Total Episodes Processed: {processed_count}")
    print(f"Successful ID Matches   : {count_matched}")
    print(f"Missing IDs (Hash Used) : {count_fallback}")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["droid", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
