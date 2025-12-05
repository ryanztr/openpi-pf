"""
PyTorch training entrypoint for PI0/PI05 with multi-GPU and multi-node (DDP) support.
This script mirrors the behavior of the JAX trainer (`scripts/train.py`) but runs
entirely in PyTorch using the `PI0Pytorch` model and your existing config/data
pipeline from `src/openpi/training/config.py` and `src/openpi/training/data_loader.py`.

Usage
Single GPU:
  python scripts/train_pytorch_with_da3.py <config_name> --exp_name <run_name> --save_interval <interval>
  Example:
  python scripts/train_pytorch_with_da3.py debug --exp_name pytorch_ddp_test
  python scripts/train_pytorch_with_da3.py debug --exp_name pytorch_ddp_test --resume  # Resume from latest checkpoint
Multi-GPU (single node):
  torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch_with_da3.py <config_name> --exp_name <run_name>
  Example:
  torchrun --standalone --nnodes=1 --nproc_per_node=1 scripts/train_pytorch_with_da3.py pi05_droid_finetune_pytorch --exp_name debug
  torchrun --standalone --nnodes=1 --nproc_per_node=8 scripts/train_pytorch_with_da3.py pi05_droid_finetune_pytorch --exp_name pytorch_ddp_test
  torchrun --standalone --nnodes=1 --nproc_per_node=8 scripts/train_pytorch_with_da3.py pi05_droid_finetune_pytorch --exp_name pytorch_ddp_test --resume
Multi-Node Training:
	torchrun \
    --nnodes=<num_nodes> --nproc_per_node=<gpus_per_node> --node_rank=<rank_of_node> \
    --master_addr=<master_ip> --master_port=<port> \
    scripts/train_pytorch_with_da3.py <config_name> --exp_name=<run_name> --save_interval <interval>

"""

import dataclasses
import gc
import json
import logging
import os
import pathlib
import platform
import shutil
import time

import jax
import numpy as np
from PIL import Image
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.parallel
import tqdm
import wandb

import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data
from openpi.utils.generate_raymap import build_raymaps_for_batch
from openpi.utils.get_camera_args import get_batch_camera_args
from openpi.utils.grab_dino_feature import build_da3_features_for_batch
from openpi.utils.grab_dino_feature import load_da3_model


def init_logging():
    level_mapping = {
        "DEBUG": "D",
        "INFO": "I",
        "WARNING": "W",
        "ERROR": "E",
        "CRITICAL": "C",
    }

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(
                record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    """Initialize wandb logging."""
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory {ckpt_dir} does not exist.")

    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


def setup_ddp():
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1

    device = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    if use_ddp and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(
            backend=backend,
            init_method="env://",
            device_id=device if torch.cuda.is_available() else None,
        )

        # Set up debugging environment variables for DDP issues
        if os.environ.get("TORCH_DISTRIBUTED_DEBUG") is None:
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

    return use_ddp, local_rank, device


def cleanup_ddp():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def set_seed(seed: int, local_rank: int):
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)


class ObservationWrapper:
    def __init__(self, observation):
        # Keep the original observation object
        self._observation = observation
        # Store extra fields
        self._extra_fields = {}

    def __getattr__(self, name):
        if hasattr(self._observation, name):
            return getattr(self._observation, name)
        if name in self._extra_fields:
            return self._extra_fields[name]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setitem__(self, key, value):
        self._extra_fields[key] = value

    def __getitem__(self, key):
        return self._extra_fields[key]

    def __contains__(self, key):
        return key in self._extra_fields

    def get(self, key, default=None):
        if key in self._extra_fields:
            return self._extra_fields[key]
        if hasattr(self._observation, key):
            return getattr(self._observation, key)
        return default


def sanitize_name(name: str) -> str:
    """Make camera names filesystem friendly."""
    return name.replace("/", "_")


def to_uint8_image(array: np.ndarray) -> np.ndarray | None:
    """Convert a single image array to uint8 HWC for saving."""
    if array is None:
        return None

    arr = np.asarray(array)
    if arr.ndim == 4:
        return None  # Should pass individual samples only.

    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)

    if arr.ndim != 3:
        return None

    if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.moveaxis(arr, 0, -1)

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    if arr.shape[-1] != 3:
        return None

    if np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float32)
        arr_min = float(arr.min())
        arr_max = float(arr.max())
        arr = (arr - arr_min) / (arr_max -
                                 arr_min) if arr_max > arr_min else np.zeros_like(arr, dtype=np.float32)
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255).astype(np.uint8)
    else:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return arr


def save_batch_visuals(batch_idx: int, observation: ObservationWrapper, output_root: pathlib.Path | str):
    """Persist observation images and raymaps from a batch as JPEGs."""
    output_root = pathlib.Path(output_root)
    batch_dir = output_root / f"batch_{batch_idx:04d}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    base_observation = getattr(observation, "_observation", observation)
    images = getattr(base_observation, "images", {}) or {}

    for name, tensor in images.items():
        if tensor is None:
            continue
        array = tensor.detach().cpu().numpy() if torch.is_tensor(
            tensor) else np.asarray(tensor)
        if array.ndim == 3:
            array = array[None, ...]
        if array.ndim != 4:
            continue

        for sample_idx, sample in enumerate(array):
            img = to_uint8_image(sample)
            if img is None:
                continue
            filename = batch_dir / \
                f"sample_{sample_idx:04d}_{sanitize_name(name)}.jpg"
            Image.fromarray(img).save(filename, format="JPEG")

    raymap_tensor = observation.get("raymap", None)
    if raymap_tensor is None:
        return

    raymap_array = raymap_tensor.detach().cpu().numpy() if torch.is_tensor(
        raymap_tensor) else np.asarray(raymap_tensor)
    if raymap_array.ndim != 5:
        return

    for sample_idx in range(raymap_array.shape[0]):
        for cam_idx in range(raymap_array.shape[1]):
            sample = raymap_array[sample_idx, cam_idx]
            img = to_uint8_image(sample)
            if img is None:
                continue
            filename = batch_dir / \
                f"sample_{sample_idx:04d}_raymap_cam_{cam_idx:02d}.jpg"
            Image.fromarray(img).save(filename, format="JPEG")


def build_datasets(config: _config.TrainConfig):
    # Use the unified data loader with PyTorch framework
    data_loader = _data.create_data_loader(
        config, framework="pytorch", shuffle=True)
    return data_loader, data_loader.data_config()


def save_checkpoint(model, optimizer, global_step, config, is_main, data_config):
    """Save a checkpoint with model state, optimizer state, and metadata."""
    if not is_main:
        return

    # Only save if it's time to save or if it's the final step
    if (global_step % config.save_interval == 0 and global_step > 0) or global_step == config.num_train_steps - 1:
        # Create temporary directory for atomic checkpoint saving
        final_ckpt_dir = config.checkpoint_dir / f"{global_step}"
        tmp_ckpt_dir = config.checkpoint_dir / f"tmp_{global_step}"

        # Remove any existing temp directory and create new one
        if tmp_ckpt_dir.exists():
            shutil.rmtree(tmp_ckpt_dir)
        tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save model state using safetensors (handle shared tensors)
        model_to_save = model.module if isinstance(
            model, torch.nn.parallel.DistributedDataParallel) else model
        safetensors.torch.save_model(
            model_to_save, tmp_ckpt_dir / "model.safetensors")

        # Save optimizer state using PyTorch format
        torch.save(optimizer.state_dict(), tmp_ckpt_dir / "optimizer.pt")

        # Save training metadata (avoid saving full config to prevent JAX/Flax compatibility issues)
        metadata = {
            "global_step": global_step,
            "config": dataclasses.asdict(config),
            "timestamp": time.time(),
        }
        torch.save(metadata, tmp_ckpt_dir / "metadata.pt")

        # save norm stats
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(tmp_ckpt_dir / "assets" /
                            data_config.asset_id, norm_stats)

        # Atomically move temp directory to final location
        if final_ckpt_dir.exists():
            shutil.rmtree(final_ckpt_dir)
        tmp_ckpt_dir.rename(final_ckpt_dir)

        logging.info(
            f"Saved checkpoint at step {global_step} -> {final_ckpt_dir}")

        # Log checkpoint to wandb
        if config.wandb_enabled:
            wandb.log({"checkpoint_step": global_step}, step=global_step)


def load_checkpoint(model, optimizer, checkpoint_dir, device):
    """Load the latest checkpoint and return the global step."""
    checkpoint_steps = [int(d.name) for d in checkpoint_dir.iterdir(
    ) if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")]

    if not checkpoint_steps:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    latest_step = max(checkpoint_steps)
    ckpt_dir = checkpoint_dir / f"{latest_step}"

    # Clear memory before loading checkpoints
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "before_loading_checkpoint")

    try:
        # Load model state with error handling
        logging.info("Loading model state...")
        safetensors_path = ckpt_dir / "model.safetensors"

        if safetensors_path.exists():
            model_to_load = model.module if isinstance(
                model, torch.nn.parallel.DistributedDataParallel) else model
            safetensors.torch.load_model(
                model_to_load, safetensors_path, device=str(device))
            logging.info("Loaded model state from safetensors format")
        else:
            raise FileNotFoundError(f"No model checkpoint found at {ckpt_dir}")

        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_model")

        # Load optimizer state with error handling
        logging.info("Loading optimizer state...")
        optimizer_path = ckpt_dir / "optimizer.pt"

        if optimizer_path.exists():
            optimizer_state_dict = torch.load(
                optimizer_path, map_location=device, weights_only=False)
            logging.info("Loaded optimizer state from pt format")
        else:
            raise FileNotFoundError(
                f"No optimizer checkpoint found at {ckpt_dir}")

        optimizer.load_state_dict(optimizer_state_dict)
        del optimizer_state_dict
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_optimizer")

        # Load metadata
        logging.info("Loading metadata...")
        metadata = torch.load(ckpt_dir / "metadata.pt",
                              map_location=device, weights_only=False)
        global_step = metadata.get("global_step", latest_step)
        del metadata
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_metadata")

        logging.info(
            f"Successfully loaded all checkpoint components from step {latest_step}")
        return global_step

    except RuntimeError as e:
        if "out of memory" in str(e):
            # Clear memory and provide detailed error message
            torch.cuda.empty_cache()
            gc.collect()
            logging.error(
                f"Out of memory error while loading checkpoint: {e!s}")
            log_memory_usage(device, latest_step, "after_oom_error")
            raise RuntimeError(
                "Out of memory while loading checkpoint. Try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True") from e
        raise


def get_latest_checkpoint_step(checkpoint_dir):
    """Get the latest checkpoint step number from a checkpoint directory."""
    checkpoint_steps = [int(d.name) for d in checkpoint_dir.iterdir(
    ) if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")]
    return max(checkpoint_steps) if checkpoint_steps else None


def log_memory_usage(device, step, phase="unknown"):
    """Log detailed memory usage information."""
    if not torch.cuda.is_available():
        return

    memory_allocated = torch.cuda.memory_allocated(device) / 1e9
    memory_reserved = torch.cuda.memory_reserved(device) / 1e9
    memory_free = torch.cuda.memory_reserved(
        device) - torch.cuda.memory_allocated(device)
    memory_free = memory_free / 1e9

    # Get more detailed memory info
    memory_stats = torch.cuda.memory_stats(device)
    max_memory_allocated = memory_stats.get(
        "allocated_bytes.all.peak", 0) / 1e9
    max_memory_reserved = memory_stats.get("reserved_bytes.all.peak", 0) / 1e9

    # Get DDP info if available
    ddp_info = ""
    if dist.is_initialized():
        ddp_info = f" | DDP: rank={dist.get_rank()}, world_size={dist.get_world_size()}"

    logging.info(
        f"Step {step} ({phase}): GPU memory - allocated: {memory_allocated:.2f}GB, reserved: {memory_reserved:.2f}GB, free: {memory_free:.2f}GB, peak_allocated: {max_memory_allocated:.2f}GB, peak_reserved: {max_memory_reserved:.2f}GB{ddp_info}"
    )


def decode_episode_id_bytes(tensor_or_array):
    if hasattr(tensor_or_array, "cpu"):
        tensor_or_array = tensor_or_array.cpu().numpy()
    elif hasattr(tensor_or_array, "numpy"):
        tensor_or_array = tensor_or_array.numpy()

    tensor_or_array = np.asarray(tensor_or_array, dtype=np.uint8)
    samples = tensor_or_array if tensor_or_array.ndim > 1 else tensor_or_array[None, :]

    decoded = []
    for sample in samples:
        valid_bytes = bytes(int(b) for b in sample.tolist() if b != 0)
        decoded.append(valid_bytes.decode("utf-8"))

    return decoded if len(decoded) > 1 else decoded[0]


def train_loop(config: _config.TrainConfig):
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    set_seed(config.seed, local_rank)

    base_dir = os.environ.get("OPENPI_BASE_DIR", None)
    base_dir = pathlib.Path(
        __file__).parent.parent.parent if base_dir is None else pathlib.Path(base_dir)
    droid_json_dir = base_dir / "dataset" / "droid_json"

    # Load camera arguments
    if is_main:
        logging.info(f"Loading camera args from {droid_json_dir}...")
    with open(droid_json_dir / "external_camera_args.json") as f:
        camera_args = {k: v for k, v in json.load(f).items() if v is not None}

    # Load Depth Anything 3 Model for each rank
    if is_main:
        logging.info("Loading DA3 model...")
    da3_model = load_da3_model(device)

    # Initialize checkpoint directory and wandb
    resuming = False
    if config.resume:
        # Find checkpoint directory based on experiment name
        exp_checkpoint_dir = config.checkpoint_dir
        if exp_checkpoint_dir.exists():
            # Use validation to find the latest working checkpoint
            latest_step = get_latest_checkpoint_step(exp_checkpoint_dir)
            if latest_step is not None:
                resuming = True
                logging.info(
                    f"Resuming from experiment checkpoint directory: {exp_checkpoint_dir} at step {latest_step}")
            else:
                raise FileNotFoundError(
                    f"No valid checkpoints found in {exp_checkpoint_dir} for resume")
        else:
            raise FileNotFoundError(
                f"Experiment checkpoint directory {exp_checkpoint_dir} does not exist for resume")
    elif config.overwrite and config.checkpoint_dir.exists():
        shutil.rmtree(config.checkpoint_dir)
        logging.info(
            f"Overwriting checkpoint directory: {config.checkpoint_dir}")

    # Create checkpoint directory with experiment name
    if not resuming:
        # For new runs, create experiment-specific checkpoint directory
        exp_checkpoint_dir = config.checkpoint_dir
        exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logging.info(
            f"Created experiment checkpoint directory: {exp_checkpoint_dir}")
    else:
        # For resume, checkpoint_dir is already set to the experiment directory
        logging.info(
            f"Using existing experiment checkpoint directory: {config.checkpoint_dir}")

    # Initialize wandb (only on main process)
    if is_main:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # Build data loader using the unified data loader
    # Calculate effective batch size per GPU for DDP
    # For N GPUs, each GPU should get batch_size/N samples, so total across all GPUs is batch_size
    world_size = torch.distributed.get_world_size() if use_ddp else 1
    effective_batch_size = config.batch_size // world_size
    logging.info(
        f"Using batch size per GPU: {effective_batch_size} (total batch size across {world_size} GPUs: {config.batch_size})")

    # Pass the original batch size to data loader - it will handle DDP splitting internally
    loader, data_config = build_datasets(config)
    logging.info("Create loader successfully!")

    # Log sample images to wandb on first batch
    if is_main and config.wandb_enabled and not resuming:
        # Create a separate data loader for sample batch to avoid consuming the main loader
        sample_data_loader = _data.create_data_loader(
            config, framework="pytorch", shuffle=False)
        sample_batch = next(iter(sample_data_loader))
        # Convert observation and actions to torch tensors
        observation, actions = sample_batch
        sample_batch = observation.to_dict()
        sample_batch["actions"] = actions

        # Create sample images for wandb
        images_to_log = []
        # Get batch size from the first image tensor
        batch_size = next(iter(sample_batch["image"].values())).shape[0]
        for i in range(min(5, batch_size)):
            # Concatenate all camera views horizontally for this batch item
            # Convert from NCHW to NHWC format for wandb
            img_concatenated = torch.cat(
                [img[i].permute(1, 2, 0)
                 for img in sample_batch["image"].values()],
                axis=1,
            )
            img_concatenated = img_concatenated.cpu().numpy()
            images_to_log.append(wandb.Image(img_concatenated))

        wandb.log({"camera_views": images_to_log}, step=0)

        # Clear sample batch from memory aggressively
        del sample_batch, observation, actions, images_to_log, img_concatenated
        del sample_data_loader  # Also delete the sample data loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Cleared sample batch and data loader from memory")

    # Build model
    if not isinstance(config.model, openpi.models.pi0_config.Pi0Config):
        # Convert dataclass to Pi0Config if needed
        model_cfg = openpi.models.pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(
                config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(
                config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
        )
    else:
        model_cfg = config.model
        # Update dtype to match pytorch_training_precision
        object.__setattr__(model_cfg, "dtype",
                           config.pytorch_training_precision)

    model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(device)

    # Optimization setup
    if hasattr(model, "gradient_checkpointing_enable"):
        enable_gradient_checkpointing = True
        model.gradient_checkpointing_enable()
        logging.info("Enabled gradient checkpointing for memory optimization")
    else:
        enable_gradient_checkpointing = False
        logging.info("Gradient checkpointing is not supported for this model")

    # Log initial memory usage after model creation
    if is_main and torch.cuda.is_available():
        log_memory_usage(device, 0, "after_model_creation")

    # Enable memory optimizations for large-scale training
    if world_size >= 8:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Set memory allocation configuration
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
        logging.info("Enabled memory optimizations for 8+ GPU training")

    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=False,  # Disable for memory efficiency
            gradient_as_bucket_view=True,  # Enable for memory efficiency
            static_graph=world_size >= 8,  # Enable for 8+ GPUs
        )

    # Load weights from weight_loader if specified (for fine-tuning)
    if config.pytorch_weight_path is not None:
        logging.info(f"Loading weights from: {config.pytorch_weight_path}")

        model_path = os.path.join(
            config.pytorch_weight_path, "model.safetensors")
        safetensors.torch.load_model(
            (model.module if isinstance(
                model, torch.nn.parallel.DistributedDataParallel) else model),
            model_path,
        )
        logging.info(
            f"Loaded PyTorch weights from {config.pytorch_weight_path}")

    # Optimizer + learning rate schedule from config
    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

    # Create optimizer with config parameters
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )

    # Load checkpoint if resuming
    global_step = 0
    if resuming:
        global_step = load_checkpoint(
            model, optim, config.checkpoint_dir, device)
        logging.info(f"Resumed training from step {global_step}")

    def lr_schedule(step: int):
        if step < warmup_steps:
            # Match JAX behavior: start from peak_lr / (warmup_steps + 1)
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        # cosine decay
        progress = min(1.0, (step - warmup_steps) /
                       max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    model.train()
    start_time = time.time()
    infos = []  # Collect stats over log interval
    if is_main:
        logging.info(
            f"Running on: {platform.node()} | world_size={torch.distributed.get_world_size() if use_ddp else 1}")
        logging.info(
            f"Training config: batch_size={config.batch_size}, effective_batch_size={effective_batch_size}, num_train_steps={config.num_train_steps}")
        logging.info(
            f"Memory optimizations: gradient_checkpointing={enable_gradient_checkpointing}")
        logging.info(
            f"LR schedule: warmup={warmup_steps}, peak_lr={peak_lr:.2e}, decay_steps={decay_steps}, end_lr={end_lr:.2e}")
        logging.info(
            f"Optimizer: {type(config.optimizer).__name__}, weight_decay={config.optimizer.weight_decay}, clip_norm={config.optimizer.clip_gradient_norm}")
        logging.info("EMA is not supported for PyTorch training")
        logging.info(f"Training precision: {model_cfg.dtype}")

    # Training loop - iterate until we reach num_train_steps
    pbar = (
        tqdm.tqdm(
            total=config.num_train_steps,
            initial=global_step,
            desc="Training",
            disable=not is_main,
        )
        if is_main
        else None
    )

    while global_step < config.num_train_steps:
        # Set epoch for distributed training
        if use_ddp and hasattr(loader, "set_epoch"):
            loader.set_epoch(global_step // len(loader))

        for batch in loader:
            # Check if we've reached the target number of steps
            if global_step >= config.num_train_steps:
                break

            observation, actions, episode_id_raw = batch
            current_batch_size = actions.shape[0]

            observation = jax.tree.map(lambda x: x.to(device), observation)
            actions = actions.to(device=device, dtype=torch.float32)

            episode_ids_list = []
            if episode_id_raw is not None:
                decoded = decode_episode_id_bytes(episode_id_raw)
                episode_ids_list = decoded if isinstance(
                    decoded, list) else [decoded]

            if len(episode_ids_list) != current_batch_size:
                episode_ids_list = episode_ids_list + \
                    [None] * (current_batch_size - len(episode_ids_list))

            # Wrap Observation
            observation = ObservationWrapper(observation)
            observation["num_cameras_with_args"] = None
            observation["raymap"] = None
            observation["da3_feature"] = None

            # Get Camera Arguments
            intrinsics_batch, extrinsics_batch, width_batch, height_batch, valid_mask_batch, num_cameras_batch = get_batch_camera_args(
                episode_ids_list,
                camera_args,
                device,
                current_batch_size,
            )
            observation["num_cameras_with_args"] = num_cameras_batch

            # Generate Raymaps
            raymaps_batch, max_cam = build_raymaps_for_batch(
                intrinsics_batch,
                extrinsics_batch,
                width_batch,
                height_batch,
                valid_mask_batch,
                width=224,
                height=224,
                device=device,
            )
            observation["raymap"] = raymaps_batch
            observation["camera_valid_mask"] = valid_mask_batch

            # Grab DA3 features
            da3_features = build_da3_features_for_batch(
                observation,
                da3_model,
                valid_mask_batch,
                max_cam,
                device,
            )
            observation["da3_feature"] = da3_features

            # Update LR
            for pg in optim.param_groups:
                pg["lr"] = lr_schedule(global_step)

            # Forward pass (model will handle different cases based on num_cameras_with_args)
            losses = model(observation, actions)

            # Ensure losses is a tensor and handle different return types
            if isinstance(losses, list | tuple):
                losses = torch.stack(losses)
            elif not isinstance(losses, torch.Tensor):
                losses = torch.tensor(
                    losses, device=device, dtype=torch.float32)

            loss = losses.mean()

            # Backward pass
            loss.backward()

            # Log memory usage after backward pass
            if global_step < 5 and is_main and torch.cuda.is_available():
                log_memory_usage(device, global_step, "after_backward")

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config.optimizer.clip_gradient_norm)

            # Optimizer step
            optim.step()
            optim.zero_grad(set_to_none=True)

            # Clear gradients more aggressively
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad = None

            # Collect stats
            if is_main:
                infos.append(
                    {
                        "loss": loss.item(),
                        "learning_rate": optim.param_groups[0]["lr"],
                        "grad_norm": (float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm),
                    }
                )

            if is_main and (global_step % config.log_interval == 0):
                elapsed = time.time() - start_time

                # Average stats over log interval
                avg_loss = sum(info["loss"] for info in infos) / len(infos)
                avg_lr = sum(info["learning_rate"]
                             for info in infos) / len(infos)

                avg_grad_norm = None
                if any("grad_norm" in info for info in infos):
                    vals = [info["grad_norm"]
                            for info in infos if "grad_norm" in info and info["grad_norm"] is not None]
                    if len(vals) > 0:
                        avg_grad_norm = sum(vals) / len(vals)
                logging.info(
                    f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} grad_norm={avg_grad_norm:.2f} time={elapsed:.1f}s"
                    if avg_grad_norm is not None
                    else f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} time={elapsed:.1f}s"
                )

                # Log to wandb
                if config.wandb_enabled and len(infos) > 0:
                    log_payload = {
                        "loss": avg_loss,
                        "learning_rate": avg_lr,
                        "step": global_step,
                        "time_per_step": elapsed / config.log_interval,
                    }
                    if avg_grad_norm is not None:
                        log_payload["grad_norm"] = avg_grad_norm
                    wandb.log(log_payload, step=global_step)

                start_time = time.time()
                infos = []  # Reset stats collection

            global_step += 1
            # Save checkpoint using the new mechanism
            save_checkpoint(model, optim, global_step,
                            config, is_main, data_config)

            # Update progress bar
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "lr": f"{optim.param_groups[0]['lr']:.2e}",
                        "step": global_step,
                    }
                )

    # Close progress bar
    if pbar is not None:
        pbar.close()

    # Finish wandb run
    if is_main and config.wandb_enabled:
        wandb.finish()

    cleanup_ddp()


# Check data loader
def inspect_data_loop(config: _config.TrainConfig):
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    set_seed(config.seed, local_rank)

    if is_main:
        logging.info("Enter data inspection mode")
        logging.info(
            "No model loaded, only check data loader, step id decode, camera args and raymap generation")

    world_size = torch.distributed.get_world_size() if use_ddp else 1
    effective_batch_size = config.batch_size // world_size

    loader, _data_config = build_datasets(config)

    if is_main:
        logging.info(
            f"Loader created successfully! Batch Size: {config.batch_size} (Per GPU: {effective_batch_size})")

    base_dir = os.environ.get("OPENPI_BASE_DIR", None)
    base_dir = pathlib.Path(
        __file__).parent.parent.parent if base_dir is None else pathlib.Path(base_dir)
    visuals_dir = base_dir / "inspection_visuals"
    if is_main:
        visuals_dir.mkdir(parents=True, exist_ok=True)

    droid_json_dir = base_dir / "dataset" / "droid_json"

    with open(droid_json_dir / "external_camera_args.json") as f:
        camera_args = {k: v for k, v in json.load(f).items() if v is not None}

    logging.info("Camera Args loaded successfully")
    logging.info("Start traversing DataLoader...")

    da3_model = load_da3_model(device)

    if use_ddp and hasattr(loader, "set_epoch"):
        loader.set_epoch(0)

    max_batches = 3

    for batch_idx, (observation, actions, episode_id) in enumerate(loader, start=1):
        if batch_idx > max_batches:
            break

        current_batch_size = actions.shape[0]

        # Move data to GPU
        observation = jax.tree.map(lambda x: x.to(device), observation)  # noqa: PLW2901
        actions = actions.to(device=device, dtype=torch.float32)  # noqa: PLW2901

        # Decode Episode ID
        if episode_id is not None:
            decoded = decode_episode_id_bytes(episode_id)
            episode_ids_list = decoded if isinstance(
                decoded, list) else [decoded]

        if len(episode_ids_list) != current_batch_size:
            logging.warning(
                f"Episode ID count mismatch (Ids={len(episode_ids_list)}, Batch={current_batch_size}). Padding with None.")
            episode_ids_list = episode_ids_list + \
                [None] * (current_batch_size - len(episode_ids_list))

        # Calculate Camera number
        intrinsics_batch, extrinsics_batch, width_batch, height_batch, valid_mask_batch, num_cameras_batch = get_batch_camera_args(
            episode_ids_list, camera_args, device, current_batch_size)

        # Wrap Observation
        observation = ObservationWrapper(observation)  # noqa: PLW2901
        observation["num_cameras_with_args"] = num_cameras_batch
        observation["raymap"] = None
        observation["da3_feature"] = None

        TARGET_H = 224  # noqa: N806
        TARGET_W = 224  # noqa: N806

        raymaps_batch, max_cam = build_raymaps_for_batch(
            intrinsics_batch,
            extrinsics_batch,
            width_batch,
            height_batch,
            valid_mask_batch,
            TARGET_H,
            TARGET_W,
            device,
        )

        logging.info(
            f"Raymap Batch Generated: {raymaps_batch.shape}, Valid Cameras: {valid_mask_batch.sum().item()}, Max Cam: {max_cam}")

        observation["raymap"] = raymaps_batch
        observation["camera_valid_mask"] = valid_mask_batch

        da3_features = build_da3_features_for_batch(
            observation,
            da3_model,
            valid_mask_batch,
            max_cam,
            device,
        )
        observation["da3_feature"] = da3_features

        if is_main:
            save_batch_visuals(batch_idx, observation, visuals_dir)
        log_batch_contents(batch_idx, observation, actions)

    cleanup_ddp()


def log_batch_contents(batch_idx: int, observation: ObservationWrapper, actions: torch.Tensor):
    """Log the structure of observation and actions for inspection."""
    base_observation = getattr(observation, "_observation", observation)
    base_field_names = list(vars(base_observation).keys()) if hasattr(
        base_observation, "__dict__") else []
    extra_fields = list(getattr(observation, "_extra_fields", {}).keys()) if hasattr(
        observation, "_extra_fields") else []

    logging.info(
        "[Inspect][Batch %d] Observation base fields: %s | Extra fields: %s",
        batch_idx,
        base_field_names,
        extra_fields,
    )

    image_info = {}
    for key, value in base_observation.images.items():
        shape = tuple(value.shape) if hasattr(value, "shape") else "unknown"
        dtype = getattr(value, "dtype", "unknown")
        image_info[key] = {"shape": shape, "dtype": str(dtype)}
    logging.info("[Inspect][Batch %d] Images: %s", batch_idx, image_info)

    state_shape = tuple(base_observation.state.shape) if hasattr(
        base_observation.state, "shape") else "unknown"
    logging.info("[Inspect][Batch %d] State shape: %s", batch_idx, state_shape)

    logging.info(
        "[Inspect][Batch %d] Actions shape: %s, dtype: %s",
        batch_idx,
        tuple(actions.shape),
        actions.dtype,
    )

    if "raymap" in observation and observation["raymap"] is not None:
        raymap_tensor = observation["raymap"]
        logging.info(
            "[Inspect][Batch %d] Raymap shape: %s, dtype: %s",
            batch_idx,
            tuple(raymap_tensor.shape),
            raymap_tensor.dtype,
        )
    else:
        logging.info("[Inspect][Batch %d] Raymap missing or None", batch_idx)

    if "da3_feature" in observation and observation["da3_feature"] is not None:
        da3_tensor = observation["da3_feature"]
        logging.info(
            "[Inspect][Batch %d] DA3 feature shape: %s, dtype: %s",
            batch_idx,
            tuple(da3_tensor.shape),
            da3_tensor.dtype,
        )
    else:
        logging.info(
            "[Inspect][Batch %d] DA3 feature missing or None", batch_idx)


def main():
    init_logging()
    config = _config.cli()
    # train_loop(config)
    inspect_data_loop(config)


if __name__ == "__main__":
    main()
