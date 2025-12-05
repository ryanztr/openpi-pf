import logging
import os
import pathlib

import numpy as np
import torch

from depth_anything_3.api import DepthAnything3

CAMERA_IMAGE_KEYS = ("base_1_rgb", "base_2_rgb")


def tensor_to_numpy_uint8(image):
    """Convert various tensor/array formats to uint8 HWC numpy arrays."""
    if image is None:
        return None

    arr = image.detach().cpu()
    if arr.ndim == 4:
        raise ValueError("Expected a single image tensor, but received a batched tensor.")
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] != 3:
        arr = arr.permute(1, 2, 0)
    arr = arr.numpy()

    if arr.ndim != 3 or arr.shape[-1] not in (1, 3):
        return None

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    if np.issubdtype(arr.dtype, np.floating):
        arr_min = float(arr.min())
        arr_max = float(arr.max())
        if arr_min >= -1.0 and arr_max <= 1.0:
            arr = (arr + 1.0) / 2.0
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0).astype(np.uint8)
    else:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return arr


def grab_feature(image, model, batch_size=None, num_cameras=None):
    """
    Run DepthAnything3 feature extraction on a list of images (or a single image).

    Args:
        image: Single image or list of images in numpy / torch format.
        model: Initialized DepthAnything3 model.
        batch_size: Optional batch size for logging purposes.
        num_cameras: Optional number of cameras per batch for logging purposes.

    Returns:
        Feature tensor from the requested layer, or None if unavailable.
    """
    if model is None:
        return None

    if not isinstance(image, list):
        image = [image]

    processed_images = []
    for img in image:
        np_img = tensor_to_numpy_uint8(img)
        if np_img is not None:
            processed_images.append(np_img)

    if not processed_images:
        return None

    # Log shape information if batch_size and num_cameras are provided
    if batch_size is not None and num_cameras is not None:
        expected_structure = f"[{batch_size}, {num_cameras}, 3, H, W]"
        logging.info(f"DA3 processing: {len(processed_images)} images, expected structure: {expected_structure}")

    last_layer = model.model.da3.backbone.pretrained.n_blocks - 1

    prediction = model.inference(
        image=processed_images,
        export_dir=None,
        export_format="feat_vis",
        export_feat_layers=[last_layer],
    )

    feat_key = f"feat_layer_{last_layer}"
    return prediction.aux.get(feat_key)


def resolve_da3_model_dir() -> pathlib.Path | None:
    """Locate the DepthAnything3 weights directory."""
    env_dir = os.environ.get("DA3_MODEL_DIR")
    if env_dir:
        env_path = pathlib.Path(env_dir).expanduser()
        if env_path.exists():
            return env_path

    repo_root = pathlib.Path(__file__).resolve().parents[3]
    snapshot_roots = [
        repo_root / "models" / "models--depth-anything--DA3NESTED-GIANT-LARGE" / "snapshots",
        pathlib.Path("/app/models/models--depth-anything--DA3NESTED-GIANT-LARGE/snapshots"),
    ]

    for root in snapshot_roots:
        if not root.exists():
            continue
        if root.is_file():
            return root
        snapshot_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
        if snapshot_dirs:
            return snapshot_dirs[-1]

    return None


def load_da3_model(device: torch.device):
    """Load DepthAnything3 weights if available."""
    model_dir = resolve_da3_model_dir()
    if model_dir is None:
        logging.warning("DA3 model directory not found. Skip DA3 feature extraction.")
        return None

    logging.info(f"Loading DepthAnything3 model from {model_dir}")
    da3_model = DepthAnything3.from_pretrained(str(model_dir))
    da3_model = da3_model.to(device)
    da3_model.eval()
    for param in da3_model.parameters():
        param.requires_grad_(False)  # noqa: FBT003
    logging.info("DepthAnything3 model loaded successfully.")
    return da3_model


def build_da3_features_for_batch(
    observation,
    da3_model,
    valid_mask_batch: torch.Tensor,
    max_cam: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Generate DA3 features for each observation in the batch."""
    if da3_model is None or max_cam <= 0:
        return None

    base_observation = getattr(observation, "_observation", None)
    if base_observation is None or not hasattr(base_observation, "images"):
        logging.warning("Observation is missing image data; cannot compute DA3 features.")
        return None

    images_dict = base_observation.images
    batch_size = valid_mask_batch.shape[0]
    features_cache = [[None for _ in range(max_cam)] for _ in range(batch_size)]
    feature_shape = None
    usable_cam = min(len(CAMERA_IMAGE_KEYS), max_cam)

    batched_requests: list[tuple[int, int, torch.Tensor]] = []

    for batch_idx in range(batch_size):
        for cam_idx in range(usable_cam):
            camera_key = CAMERA_IMAGE_KEYS[cam_idx]
            camera_tensor = images_dict.get(camera_key)
            if camera_tensor is None:
                continue
            try:
                batched_requests.append((batch_idx, cam_idx, camera_tensor[batch_idx]))
            except Exception as exc:
                logging.warning(f"Failed to extract image {camera_key} for sample {batch_idx}: {exc}")

    if not batched_requests:
        return None

    with torch.no_grad():
        features = grab_feature(
            [req[2] for req in batched_requests], 
            da3_model, 
            batch_size=batch_size, 
            num_cameras=max_cam
        )

    if features is None:
        return None

    features_tensor = features.detach() if torch.is_tensor(features) else torch.as_tensor(features)
    features_tensor = features_tensor.to(device=device, dtype=torch.float32)

    for req_idx, (batch_idx, cam_idx, _) in enumerate(batched_requests):
        cam_feat = features_tensor[req_idx]
        features_cache[batch_idx][cam_idx] = cam_feat
        if feature_shape is None:
            feature_shape = cam_feat.shape

    if feature_shape is None:
        return None

    stacked = torch.zeros(
        (batch_size, max_cam, *feature_shape),
        dtype=torch.float32,
        device=device,
    )

    for batch_idx in range(batch_size):
        for cam_idx in range(max_cam):
            feat = features_cache[batch_idx][cam_idx]
            if feat is None:
                continue
            stacked[batch_idx, cam_idx] = feat

    expand_dims = (1,) * (stacked.dim() - 2)
    mask = valid_mask_batch.view(batch_size, max_cam, *expand_dims)
    return stacked * mask
