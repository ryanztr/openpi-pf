import numpy as np
from scipy.spatial.transform import Rotation
import torch


def extract_single_camera(cam_entry: dict):
    intr = cam_entry.get("intrinsics")
    extr = cam_entry.get("extrinsics")
    if intr is None or extr is None:
        return None
    return intr, extr


def get_all_camera_args(episode_id: str, json_data: dict) -> list[tuple[dict, list]]:
    """
    返回该 episode 内所有可用相机的 (intrinsics, extrinsics) 元组列表。
    """
    if episode_id not in json_data:
        return []

    episode_data = json_data[episode_id]
    camera_entries = []

    for cam_key in ("ext1_cam", "ext2_cam"):
        cam_entry = episode_data.get(cam_key, {})
        cam = extract_single_camera(cam_entry)
        if cam is not None:
            camera_entries.append(cam)

    return camera_entries


def get_batch_camera_args(episode_ids: list[str], json_data: dict, device: torch.device, batch_size: int):
    """
    Args:
        batch_size: int: The number of episodes to process in each batch for checking
    Returns:
        intrinsics, extrinsics, width, height, valid_mask, num_cameras
    """

    if len(episode_ids) != batch_size:
        raise ValueError(f"Number of episode_ids must be equal to batch_size, but got {len(episode_ids)} and {batch_size}")

    default_extrinsics = np.eye(4, dtype=np.float32)

    camera_sets = [get_all_camera_args(eid, json_data) for eid in episode_ids]
    num_cameras = [len(cams) for cams in camera_sets]
    batch_num_cameras = np.array(num_cameras, dtype=np.int32)

    max_cam = max(max(num_cameras, default=0), 1)

    identity_intrinsics = np.eye(3, dtype=np.float32)
    batch_intrinsics = np.repeat(identity_intrinsics[None, None, :, :], batch_size, axis=0)
    batch_intrinsics = np.repeat(batch_intrinsics, max_cam, axis=1)
    batch_extrinsics = np.repeat(default_extrinsics[None, None, :, :], batch_size, axis=0)
    batch_extrinsics = np.repeat(batch_extrinsics, max_cam, axis=1)
    batch_width = np.zeros((batch_size, max_cam), dtype=np.int32)
    batch_height = np.zeros((batch_size, max_cam), dtype=np.int32)
    batch_valid_mask = np.zeros((batch_size, max_cam), dtype=np.float32)

    for batch_idx, cams in enumerate(camera_sets):
        for cam_idx in range(max_cam):
            if cam_idx >= len(cams):
                continue

            intrinsics, extrinsics = cams[cam_idx]
            try:
                fx, cx, fy, cy = intrinsics["cameraMatrix"]
                if abs(fx) < 1e-6 or abs(fy) < 1e-6:
                    raise ValueError("Invalid intrinsics: fx/fy near zero")
                intr_np = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

                w = intrinsics.get("width", 1280)
                h = intrinsics.get("height", 720)

                ext_raw = np.array(extrinsics, dtype=np.float32)
                pos = ext_raw[0:3]
                euler = ext_raw[3:6]
                rot_mat = Rotation.from_euler("xyz", euler).as_matrix()
                ext_np = np.eye(4, dtype=np.float32)
                ext_np[:3, :3] = rot_mat
                ext_np[:3, 3] = pos

                batch_intrinsics[batch_idx, cam_idx] = intr_np
                batch_extrinsics[batch_idx, cam_idx] = ext_np
                batch_width[batch_idx, cam_idx] = w
                batch_height[batch_idx, cam_idx] = h
                batch_valid_mask[batch_idx, cam_idx] = 1.0
            except Exception as e:
                print(f"[Error] Failed to process episode {episode_ids[batch_idx]} cam_idx={cam_idx}: {e}")

    intrinsics_tensor = torch.tensor(batch_intrinsics, dtype=torch.float32, device=device)
    extrinsics_tensor = torch.tensor(batch_extrinsics, dtype=torch.float32, device=device)
    width_tensor = torch.tensor(batch_width, dtype=torch.int32, device=device)
    height_tensor = torch.tensor(batch_height, dtype=torch.int32, device=device)
    valid_mask_tensor = torch.tensor(batch_valid_mask, dtype=torch.float32, device=device)
    num_cameras_tensor = torch.tensor(batch_num_cameras, dtype=torch.int32, device=device)

    return intrinsics_tensor, extrinsics_tensor, width_tensor, height_tensor, valid_mask_tensor, num_cameras_tensor
