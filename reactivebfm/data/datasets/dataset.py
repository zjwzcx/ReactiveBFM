"""ReactiveBFM dataset and dataloader construction."""

import math
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

from reactivebfm.data.datasets.batch import collate as all_collate
from reactivebfm.data.datasets.batch import (
    collate_smooth,
    cross_prefix_collate_smooth,
    prefix_collate,
    prefix_collate_smooth,
    self_rollout_collate,
)
from reactivebfm.data.datasets.motion_text import MotionTextDataset
from reactivebfm.data.datasets import DatasetRegistry


class DistributedEvalSampler(Sampler):
    """Shard evaluation without padding or duplicating held-out samples."""

    def __init__(self, dataset, rank, world_size):
        self.dataset = dataset
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self):
        return max(0, math.ceil((len(self.dataset) - self.rank) / self.world_size))


class CompositeMotionTextDataset(ConcatDataset):
    """Runtime concatenation that retains each subset's normalization/cache."""

    def __init__(self, datasets, dataset_names):
        super().__init__(datasets)
        self.dataset_names = list(dataset_names)
        self.cache = {
            name: dataset.cache for name, dataset in zip(self.dataset_names, datasets)
        }
        self.component_lengths = {
            name: len(dataset) for name, dataset in zip(self.dataset_names, datasets)
        }


def _component_data_path(dataset_name, data_dir, is_composite):
    if not data_dir:
        return DatasetRegistry.get_data_path(dataset_name)
    if not is_composite:
        return data_dir
    candidate = Path(data_dir).expanduser() / dataset_name
    if not candidate.is_dir():
        raise FileNotFoundError(
            f"Composite --data_dir must be a common parent containing "
            f"{dataset_name!r}; missing {candidate}. Leave --data_dir empty to "
            "use each registered dataset path."
        )
    return str(candidate)


def _component_cache(cache, dataset_name, is_composite):
    if cache is None:
        return None
    if is_composite:
        if not isinstance(cache, dict):
            raise TypeError("Composite dataset cache must map dataset names to caches.")
        return cache.get(dataset_name)
    return cache


def _get_collate_fn(
    name,
    pred_len=0,
    use_smooth=False,
    collate_mode="prefix",
    cross_prob=0.0,
):
    """Return the planner-training collator for a dataset."""
    if collate_mode == "self_rollout":
        return self_rollout_collate
    if collate_mode == "cross":
        return lambda x: cross_prefix_collate_smooth(
            x, pred_len=pred_len, cross_prob=cross_prob
        )
    if DatasetRegistry.is_text_conditioned(name):
        assert pred_len > 0
        if use_smooth:
            return lambda x: prefix_collate_smooth(x, pred_len=pred_len)
        return lambda x: prefix_collate(x, pred_len=pred_len)
    if use_smooth:
        return collate_smooth
    return all_collate


def get_dataset_loader(
    name,
    batch_size,
    data_dir=None,
    split="train",
    hml_type=None,
    unit_length=4,
    abs_path="diffusion_planner",
    fixed_len=0,
    pred_len=0,
    device=None,
    drop_last=True,
    return_keys=False,
    relative_root_xy=False,
    num_workers=0,
    use_smooth=True,
    collate_mode="prefix",
    cross_prob=0.0,
    pin_memory=None,
    persistent_workers=None,
    prefetch_factor=2,
    distributed=False,
    rank=0,
    world_size=1,
    seed=0,
    shuffle=True,
    deterministic=False,
    cache=None,
    minimum_length=0,
):
    if not DatasetRegistry.is_supported_dataset(name):
        raise ValueError(f"Unsupported dataset name for ReactiveBFM loader: [{name}]")

    dataset_names = DatasetRegistry.parse_dataset_names(name)
    is_composite = len(dataset_names) > 1
    component_datasets = []
    for dataset_name in dataset_names:
        component_datasets.append(
            MotionTextDataset(
                dataset_name=dataset_name,
                unit_length=unit_length,
                data_path=_component_data_path(dataset_name, data_dir, is_composite),
                abs_path=abs_path,
                hml_type=hml_type,
                split=split,
                fixed_len=fixed_len,
                device=device,
                return_keys=return_keys,
                relative_root_xy=relative_root_xy,
                deterministic=deterministic,
                cache=_component_cache(cache, dataset_name, is_composite),
                minimum_length=minimum_length,
                key_namespace=dataset_name if is_composite else None,
            )
        )
    if is_composite:
        dataset = CompositeMotionTextDataset(component_datasets, dataset_names)
        print(
            f"[CompositeDataset] split={split}, subsets={dataset.component_lengths}, "
            f"total={len(dataset)}"
        )
    else:
        dataset = component_datasets[0]

    collate = _get_collate_fn(
        name,
        pred_len,
        use_smooth=use_smooth,
        collate_mode=collate_mode,
        cross_prob=cross_prob,
    )

    if pin_memory is None:
        pin_memory = torch.cuda.is_available() and device is not None and str(device).startswith("cuda")
    if persistent_workers is None:
        persistent_workers = num_workers > 0

    sampler = None
    if distributed:
        if shuffle:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=seed,
                drop_last=drop_last,
            )
        else:
            sampler = DistributedEvalSampler(dataset, rank, world_size)

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle and sampler is None,
        "sampler": sampler,
        "num_workers": num_workers,
        "drop_last": drop_last,
        "collate_fn": collate,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers and num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(dataset, **loader_kwargs)


def get_dataset_raw(
    name,
    data_dir=None,
    split="train",
    hml_type=None,
    unit_length=4,
    abs_path="diffusion_planner",
    device=None,
    relative_root_xy=False,
):
    """Load the raw dataset without collation for fine-tuning."""
    if not DatasetRegistry.is_supported_dataset(name):
        raise ValueError(f"Unsupported dataset name for ReactiveBFM loader: [{name}]")

    dataset_names = DatasetRegistry.parse_dataset_names(name)
    is_composite = len(dataset_names) > 1
    datasets = [
        MotionTextDataset(
            dataset_name=dataset_name,
            unit_length=unit_length,
            data_path=_component_data_path(dataset_name, data_dir, is_composite),
            abs_path=abs_path,
            hml_type=hml_type,
            split=split,
            fixed_len=0,
            device=device,
            return_keys=False,
            relative_root_xy=relative_root_xy,
            key_namespace=dataset_name if is_composite else None,
        )
        for dataset_name in dataset_names
    ]
    return CompositeMotionTextDataset(datasets, dataset_names) if is_composite else datasets[0]



__all__ = [
    "CompositeMotionTextDataset",
    "DistributedEvalSampler",
    "get_dataset_loader",
    "get_dataset_raw",
]
