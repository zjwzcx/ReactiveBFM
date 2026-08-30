#!/usr/bin/env python3
"""Build train/val/test metadata for a versioned ReactiveBFM dataset.

The motion pickle is linked rather than rewritten. Normalization statistics are
recomputed from train motions only, so validation and test frames never affect
training normalization. A trailing ``_M`` can be stripped for grouped splitting
to keep BONES-SEED originals and author-provided mirrors together.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np


CORE_FILES = {
    "full_train.pkl",
    "Mean.npy",
    "Std.npy",
    "train.txt",
    "val.txt",
    "test.txt",
    "split_summary.json",
    "dataset_summary.json",
}


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.01)
    parser.add_argument("--test-ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-std", type=float, default=1e-6)
    parser.add_argument(
        "--mirror-suffix",
        default="",
        help="Strip this suffix to form split groups (use _M for BONES-SEED).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _group_key(name: str, mirror_suffix: str) -> str:
    if mirror_suffix and name.endswith(mirror_suffix):
        return name[: -len(mirror_suffix)]
    return name


def _link_or_copy(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        try:
            destination.symlink_to(source.resolve())
            return "symlink"
        except OSError:
            shutil.copy2(source, destination)
            return "copy"


def _write_lines(path: Path, names: list[str]):
    path.write_text("".join(f"{name}\n" for name in names), encoding="utf-8")


def build(args):
    source = args.source_dir.expanduser().resolve()
    output = args.out_dir.expanduser().resolve()
    if source == output:
        raise ValueError("source-dir and out-dir must differ")
    source_pickle = source / "full_train.pkl"
    if not source_pickle.is_file():
        raise FileNotFoundError(source_pickle)
    if not 0.0 <= args.val_ratio < 1.0 or not 0.0 <= args.test_ratio < 1.0:
        raise ValueError("val-ratio and test-ratio must be in [0, 1)")
    if args.val_ratio + args.test_ratio >= 1.0:
        raise ValueError("val-ratio + test-ratio must be less than 1")

    output.mkdir(parents=True, exist_ok=True)
    existing = [output / name for name in CORE_FILES if (output / name).exists() or (output / name).is_symlink()]
    if existing and not args.overwrite:
        raise FileExistsError(f"output already contains generated files: {existing}")
    for path in existing:
        path.unlink()

    print(f"Loading {source_pickle} ...")
    with source_pickle.open("rb") as handle:
        cache = pickle.load(handle)
    names = list(cache["name_list"])
    data_dict = cache["data_dict"]
    if len(names) != len(set(names)):
        raise ValueError("full_train.pkl contains duplicate motion names")

    groups = defaultdict(list)
    for name in names:
        groups[_group_key(name, args.mirror_suffix)].append(name)
    group_names = np.asarray(sorted(groups), dtype=object)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(group_names)
    n_val_groups = int(round(len(group_names) * args.val_ratio))
    n_test_groups = int(round(len(group_names) * args.test_ratio))
    if args.val_ratio and n_val_groups == 0:
        n_val_groups = 1
    if args.test_ratio and n_test_groups == 0:
        n_test_groups = 1
    if n_val_groups + n_test_groups >= len(group_names):
        raise ValueError("not enough groups for non-empty train/val/test splits")

    val_groups = set(group_names[:n_val_groups].tolist())
    test_groups = set(group_names[n_val_groups:n_val_groups + n_test_groups].tolist())
    split_by_name = {}
    split_names = {"train": [], "val": [], "test": []}
    for name in names:
        group = _group_key(name, args.mirror_suffix)
        split = "val" if group in val_groups else "test" if group in test_groups else "train"
        split_by_name[name] = split
        split_names[split].append(name)

    print(f"Computing train-only Mean/Std from {len(split_names['train'])} motions ...")
    total = None
    total_sq = None
    train_frames = 0
    motion_dim = None
    for index, name in enumerate(split_names["train"], 1):
        motion = np.asarray(data_dict[name]["motion"], dtype=np.float64)
        if motion.ndim != 2:
            raise ValueError(f"{name}: expected [frames, channels], got {motion.shape}")
        if motion_dim is None:
            motion_dim = motion.shape[1]
            total = np.zeros(motion_dim, dtype=np.float64)
            total_sq = np.zeros(motion_dim, dtype=np.float64)
        elif motion.shape[1] != motion_dim:
            raise ValueError(f"{name}: motion dim {motion.shape[1]} != {motion_dim}")
        total += motion.sum(axis=0)
        total_sq += np.square(motion).sum(axis=0)
        train_frames += len(motion)
        if index % 100000 == 0:
            print(f"  processed {index}/{len(split_names['train'])} train motions")
    mean = total / train_frames
    variance = np.maximum(total_sq / train_frames - np.square(mean), 0.0)
    std = np.maximum(np.sqrt(variance), args.min_std)

    pickle_link_type = _link_or_copy(source_pickle, output / "full_train.pkl")
    np.save(output / "Mean.npy", mean.astype(np.float32))
    np.save(output / "Std.npy", std.astype(np.float32))
    for split, split_list in split_names.items():
        _write_lines(output / f"{split}.txt", split_list)

    audit_files = {}
    for path in sorted(source.iterdir()):
        if not path.is_file() or path.name in CORE_FILES:
            continue
        audit_files[path.name] = _link_or_copy(path, output / path.name)

    split_group_sets = {
        split: {_group_key(name, args.mirror_suffix) for name in split_list}
        for split, split_list in split_names.items()
    }
    leakage = {
        "train_val": len(split_group_sets["train"] & split_group_sets["val"]),
        "train_test": len(split_group_sets["train"] & split_group_sets["test"]),
        "val_test": len(split_group_sets["val"] & split_group_sets["test"]),
    }
    summary = {
        "dataset": args.dataset_name,
        "source_dataset": str(source),
        "seed": args.seed,
        "requested_ratios": {
            "train": 1.0 - args.val_ratio - args.test_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "motions": {split: len(values) for split, values in split_names.items()},
        "motion_ratios": {
            split: len(values) / len(names) for split, values in split_names.items()
        },
        "groups": {split: len(values) for split, values in split_group_sets.items()},
        "total_motions": len(names),
        "total_groups": len(groups),
        "mirror_suffix": args.mirror_suffix or None,
        "mirror_groups_with_multiple_items": sum(len(values) > 1 for values in groups.values()),
        "group_leakage": leakage,
        "train_frames_for_statistics": train_frames,
        "motion_dim": motion_dim,
        "normalization_scope": "train_only",
        "full_train_link": pickle_link_type,
        "audit_file_links": audit_files,
    }
    if any(leakage.values()):
        raise RuntimeError(f"group leakage detected: {leakage}")
    with (output / "split_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


def main():
    build(_parse_args())


if __name__ == "__main__":
    main()
