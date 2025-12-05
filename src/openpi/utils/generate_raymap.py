import torch


def generate_batch_raymap(intrinsics, extrinsics, height, width, device="cuda"):
    """
    Generate batched raymaps directly on GPU WITHOUT normalization.

    Args:
        intrinsics: (B, 3, 3) Tensor, pinhole camera matrices
        extrinsics: (B, 4, 4) Tensor, camera-to-world transformation
        height: int, target image height
        width: int, target image width
        device: torch.device
    Returns:
        raymap: (B, H, W, 3) Tensor, unnormalized ray vectors in world coordinates
    """
    B = intrinsics.shape[0]  # noqa: N806

    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )

    pixels = torch.stack([x.flatten(), y.flatten(), torch.ones_like(x.flatten())], dim=0)
    pixels_batch = pixels.unsqueeze(0).expand(B, -1, -1)

    K_inv = torch.linalg.inv(intrinsics)  # noqa: N806
    rays_cam = torch.bmm(K_inv, pixels_batch)
    R_c2w = extrinsics[:, :3, :3]  # noqa: N806

    rays_world = torch.bmm(R_c2w, rays_cam)

    return rays_world.permute(0, 2, 1).reshape(B, height, width, 3)


def build_raymaps_for_batch(
    intrinsics_batch: torch.Tensor,
    extrinsics_batch: torch.Tensor,
    width_batch: torch.Tensor,
    height_batch: torch.Tensor,
    valid_mask_batch: torch.Tensor,
    target_h: int,
    target_w: int,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """Generate raymaps for a batch with potentially different numbers of cameras."""
    batch_size, max_cam = intrinsics_batch.shape[:2]

    flat_intrinsics = intrinsics_batch.view(batch_size * max_cam, 3, 3)
    flat_extrinsics = extrinsics_batch.view(batch_size * max_cam, 4, 4)
    flat_width = width_batch.view(-1).float()
    flat_height = height_batch.view(-1).float()

    safe_width = torch.where(flat_width > 0, flat_width, torch.ones_like(flat_width))
    safe_height = torch.where(flat_height > 0, flat_height, torch.ones_like(flat_height))

    scale_x = target_w / safe_width.view(-1, 1, 1)
    scale_y = target_h / safe_height.view(-1, 1, 1)

    scaled_intrinsics = flat_intrinsics.clone()
    scaled_intrinsics[:, 0, 0] *= scale_x[:, 0, 0]
    scaled_intrinsics[:, 0, 2] *= scale_x[:, 0, 0]
    scaled_intrinsics[:, 1, 1] *= scale_y[:, 0, 0]
    scaled_intrinsics[:, 1, 2] *= scale_y[:, 0, 0]

    raymaps = generate_batch_raymap(scaled_intrinsics, flat_extrinsics, target_h, target_w, device)
    mask_expanded = valid_mask_batch.view(-1, 1, 1, 1)
    raymaps = (raymaps * mask_expanded).view(batch_size, max_cam, target_h, target_w, 3)

    return raymaps, max_cam
