import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Suppress pynvml deprecation warning from torch
warnings.filterwarnings("ignore", category=FutureWarning, message=".*pynvml.*")

# Try to import DinoV2, add src to path if needed
try:
    from openpi.models.dinov2.dinov2 import DinoV2
except ModuleNotFoundError:
    # Add src directory to path if running directly
    current_file = Path(__file__).resolve()
    src_path = current_file.parents[2]  # Go up from utils/grab_dino_feature.py to src/
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from openpi.models.dinov2.dinov2 import DinoV2

CAMERA_IMAGE_KEYS = ("base_1_rgb", "base_2_rgb")


def preprocess_image_for_dinov2(image, patch_size=14):
    """
    Preprocess image for DinoV2 model.

    Args:
        image: Image tensor or numpy array in various formats (HWC, CHW, etc.)
        patch_size: Patch size used by the model (default: 14 for DinoV2)

    Returns:
        Preprocessed image tensor in CHW format, normalized to [0, 1], resized to be multiple of patch_size
    """
    if image is None:
        return None

    # Convert to tensor if needed
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)

    # Handle different tensor formats
    if image.ndim == 4:
        raise ValueError("Expected a single image tensor, but received a batched tensor.")

    if image.ndim == 3:
        # Check if it's HWC or CHW format
        if image.shape[0] == 3 or image.shape[0] == 1:
            # CHW format
            if image.shape[-1] == 3 or image.shape[-1] == 1:
                # Actually HWC, need to permute
                image = image.permute(2, 0, 1)
        elif image.shape[-1] == 3 or image.shape[-1] == 1:
            # HWC format, convert to CHW
            image = image.permute(2, 0, 1)

    # Ensure 3 channels
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)

    # Normalize to [0, 1]
    if image.dtype == torch.uint8:
        image = image.float() / 255.0
    elif image.dtype in (torch.float32, torch.float64):
        if image.max() > 1.0:
            image = image / 255.0
        image = torch.clamp(image, 0.0, 1.0)

    # Resize image to be multiple of patch_size
    # DinoV2 expects image dimensions to be multiples of patch_size
    _, height, width = image.shape
    new_height = ((height + patch_size - 1) // patch_size) * patch_size
    new_width = ((width + patch_size - 1) // patch_size) * patch_size

    if new_height != height or new_width != width:
        # Use interpolation to resize
        image = image.unsqueeze(0)  # Add batch dimension: (1, C, H, W)
        image = F.interpolate(
            image,
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
        )
        image = image.squeeze(0)  # Remove batch dimension: (C, H, W)

    return image


def grab_dinov2_feature(image, model, out_layers=None, batch_size=None, num_cameras=None):
    """
    Run DinoV2 feature extraction on a list of images (or a single image).

    Args:
        image: Single image or list of images in numpy / torch format.
        model: Initialized DinoV2 model.
        out_layers: List of layer indices to extract features from. If None, uses all layers.
        batch_size: Optional batch size for logging purposes.
        num_cameras: Optional number of cameras per batch for logging purposes.

    Returns:
        Feature tensor from the requested layers, or None if unavailable.
    """
    if model is None:
        return None

    if not isinstance(image, list):
        image = [image]

    processed_images = []
    for img in image:
        preprocessed = preprocess_image_for_dinov2(img)
        if preprocessed is not None:
            processed_images.append(preprocessed)

    if not processed_images:
        return None

    # Log shape information if batch_size and num_cameras are provided
    if batch_size is not None and num_cameras is not None:
        expected_structure = f"[{batch_size}, {num_cameras}, 3, H, W]"
        logging.info(f"DinoV2 processing: {len(processed_images)} images, expected structure: {expected_structure}")

    # Stack images into batch format: (B, S, C, H, W)
    # B = batch_size or 1, S = num_cameras or len(processed_images)
    images_tensor = torch.stack(processed_images)  # (N, C, H, W)
    if batch_size is not None and num_cameras is not None:
        # Reshape to (batch_size, num_cameras, C, H, W)
        if images_tensor.shape[0] != batch_size * num_cameras:
            logging.warning(f"Image count mismatch: expected {batch_size * num_cameras}, got {images_tensor.shape[0]}")
        images_tensor = images_tensor.view(batch_size, num_cameras, *images_tensor.shape[1:])
    else:
        # Single batch, multiple cameras/views: (1, num_images, C, H, W)
        images_tensor = images_tensor.unsqueeze(0)  # (1, N, C, H, W)

    # Move to model device
    device = next(model.parameters()).device
    images_tensor = images_tensor.to(device)

    # Determine output layers
    if out_layers is None:
        # Use last layer by default
        if hasattr(model.pretrained, "blocks"):
            out_layers = [len(model.pretrained.blocks) - 1]
        else:
            out_layers = [-1]
    else:
        # Convert -1 to actual layer index
        if hasattr(model.pretrained, "blocks"):
            total_layers = len(model.pretrained.blocks)
            out_layers = [layer if layer >= 0 else total_layers + layer for layer in out_layers]

    with torch.no_grad():
        outputs, _ = model(images_tensor, n=out_layers)

    # Extract features from outputs
    # outputs is a tuple of (feature, camera_token) pairs
    if isinstance(outputs, tuple) and len(outputs) > 0:
        # Get the first output (feature tensor)
        features = outputs[0][0]  # First layer, first element (feature)
        # Remove batch and sequence dimensions if needed: (B, S, N, D) -> (B*S, N, D)
        batch_size_dim, seq_dim, num_patches, feat_dim = features.shape
        return features.view(batch_size_dim * seq_dim, num_patches, feat_dim)

    return None


def load_dinov2_model(
    name: str = "vitb",
    out_layers: list[int] | None = None,
    device: torch.device | None = None,
    pretrained_path: str | None = None,
    **kwargs,
):
    """
    Load DinoV2 model.

    Args:
        name: Model name, one of {"vits", "vitb", "vitl", "vitg"}
        out_layers: List of layer indices to extract features from. If None, uses [-1] (last layer).
        device: Device to load model on. If None, uses CPU.
        pretrained_path: Path to pretrained weights file. If None, model is initialized without pretrained weights.
        **kwargs: Additional arguments passed to DinoV2 constructor.

    Returns:
        Initialized DinoV2 model, or None if loading fails.
    """
    if device is None:
        device = torch.device("cpu")

    if out_layers is None:
        out_layers = [-1]

    try:
        logging.info(f"Loading DinoV2 model: {name}")
        model = DinoV2(name=name, out_layers=out_layers, **kwargs)

        # Load pretrained weights if provided
        if pretrained_path is not None:
            if os.path.exists(pretrained_path):
                logging.info(f"Loading pretrained weights from: {pretrained_path}")
                checkpoint = torch.load(pretrained_path, map_location=device)
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if "model" in checkpoint:
                        state_dict = checkpoint["model"]
                    elif "state_dict" in checkpoint:
                        state_dict = checkpoint["state_dict"]
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint

                # Load weights, handling potential key mismatches
                model.load_state_dict(state_dict, strict=False)
                logging.info("Pretrained weights loaded successfully.")
            else:
                logging.warning(f"Pretrained weights file not found: {pretrained_path}")

        model = model.to(device)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)  # noqa: FBT003
        logging.info("DinoV2 model loaded successfully.")
        return model
    except Exception as exc:
        logging.warning(f"Failed to load DinoV2 model: {exc}")
        import traceback

        logging.warning(traceback.format_exc())
        return None


def build_dinov2_features_for_batch(
    observation,
    dinov2_model,
    valid_mask_batch: torch.Tensor,
    max_cam: int,
    device: torch.device,
    out_layers: list[int] | None = None,
) -> torch.Tensor | None:
    """Generate DinoV2 features for each observation in the batch."""
    if dinov2_model is None or max_cam <= 0:
        return None

    base_observation = getattr(observation, "_observation", None)
    if base_observation is None or not hasattr(base_observation, "images"):
        logging.warning("Observation is missing image data; cannot compute DinoV2 features.")
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
        features = grab_dinov2_feature(
            [req[2] for req in batched_requests],
            dinov2_model,
            out_layers=out_layers,
            batch_size=batch_size,
            num_cameras=max_cam,
        )

    if features is None:
        return None

    features_tensor = features.detach() if torch.is_tensor(features) else torch.as_tensor(features)
    features_tensor = features_tensor.to(device=device, dtype=torch.float32)

    # features_tensor shape: (batch_size * max_cam, N, D)
    # Need to reshape to (batch_size, max_cam, N, D)
    num_features = features_tensor.shape[0]
    if num_features == batch_size * max_cam:
        features_tensor = features_tensor.view(batch_size, max_cam, *features_tensor.shape[1:])
    else:
        # Handle case where we have fewer features than expected
        logging.warning(f"Feature shape mismatch: expected {batch_size * max_cam}, got {num_features}")
        return None

    for batch_idx, cam_idx, _ in batched_requests:
        cam_feat = features_tensor[batch_idx, cam_idx]
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


def test_dinov2_feature_extraction(
    image_path: str = "/home/zhongzd/trzhang/Depth-Anything-3/assets/examples/SOH/000.png",
):
    """
    Test function to extract DinoV2 features from an image and print the feature shape.

    Args:
        image_path: Path to the input image file.
    """
    try:
        from PIL import Image
    except ImportError:
        logging.error("PIL (Pillow) is required for image loading. Install with: pip install Pillow")
        return

    # Load image
    if not os.path.exists(image_path):
        logging.error(f"Image file not found: {image_path}")
        return

    logging.info(f"Loading image from: {image_path}")
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Convert PIL image to numpy array (HWC format)
    image_array = np.array(image)

    # Load DinoV2 model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load model with pretrained weights
    pretrained_path = "/home/zhongzd/trzhang/models/dinov2_vitg14_pretrain.pth"
    model = load_dinov2_model(
        name="vitg",  # Use vitg since the weights are for vitg
        out_layers=[-1],
        device=device,
        pretrained_path=pretrained_path,
    )
    if model is None:
        logging.error("Failed to load DinoV2 model")
        return

    # Extract features
    logging.info("Extracting DinoV2 features...")
    features = grab_dinov2_feature(image_array, model, out_layers=[-1])

    if features is None:
        logging.error("Failed to extract features")
        return

    # Print feature shape
    separator = "=" * 60
    print(f"\n{separator}")
    print("DinoV2 Feature Extraction Test")
    print(separator)
    print(f"Input image path: {image_path}")
    print(f"Input image shape: {image_array.shape}")
    print("Model: DinoV2-ViT-Base")
    print("Output layers: [-1] (last layer)")
    print(f"\nFeature shape: {features.shape}")
    print(f"Feature dtype: {features.dtype}")
    print(f"Feature device: {features.device}")
    print(f"{separator}\n")

    return features


if __name__ == "__main__":
    # Run test
    test_dinov2_feature_extraction()
