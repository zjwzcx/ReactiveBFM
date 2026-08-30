"""Training datasets, dataloaders, collation, and dataset-specific features."""

from reactivebfm.data.datasets.registry import (
    BONES_SEED_PATH_ENV,
    DATASET_ROOT_ENV,
    DEFAULT_BONES_SEED_PATH,
    DEFAULT_BONES_SEED_V2_PATH,
    DEFAULT_BONES_SEED_V3_PATH,
    DEFAULT_DATASET_ROOT,
    DatasetRegistry,
    dataset_path,
    get_dataset_root,
)

_MOTION_TEXT_EXPORTS = {
    "MotionDataset",
    "MotionTextDataset",
    "denormalize_motion",
    "normalize_motion",
}

_COLLATE_EXPORTS = {
    "collate",
    "collate_smooth",
    "collate_tensors",
    "cross_prefix_collate_smooth",
    "ensure_float32",
    "lengths_to_mask",
    "self_rollout_collate",
    "prefix_collate",
    "prefix_collate_smooth",
}

_LOADER_EXPORTS = {"get_dataset_loader", "get_dataset_raw"}


def __getattr__(name):
    if name in _MOTION_TEXT_EXPORTS:
        from reactivebfm.data.datasets import motion_text

        return getattr(motion_text, name)
    if name in _COLLATE_EXPORTS:
        from reactivebfm.data.datasets import batch

        return getattr(batch, name)
    if name in _LOADER_EXPORTS:
        from reactivebfm.data.datasets import dataset

        return getattr(dataset, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BONES_SEED_PATH_ENV",
    "DATASET_ROOT_ENV",
    "DEFAULT_BONES_SEED_PATH",
    "DEFAULT_BONES_SEED_V2_PATH",
    "DEFAULT_BONES_SEED_V3_PATH",
    "DEFAULT_DATASET_ROOT",
    "DatasetRegistry",
    "MotionDataset",
    "MotionTextDataset",
    "collate",
    "collate_smooth",
    "collate_tensors",
    "cross_prefix_collate_smooth",
    "dataset_path",
    "denormalize_motion",
    "ensure_float32",
    "get_dataset_loader",
    "get_dataset_raw",
    "get_dataset_root",
    "lengths_to_mask",
    "normalize_motion",
    "self_rollout_collate",
    "prefix_collate",
    "prefix_collate_smooth",
]
