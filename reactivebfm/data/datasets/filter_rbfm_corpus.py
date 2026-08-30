#!/usr/bin/env python3
"""Derive a ReactiveBFM corpus from an accepted source-file manifest.

The source corpus remains unchanged. Motions selected through ``texts.csv`` are
written to a new ``full_train.pkl`` with fresh normalization statistics and
train/test splits.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
from pathlib import Path

import numpy as np


MOTION_DIM = 36
CANONICAL_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument(
        "--accepted-files",
        type=Path,
        help="Optional source-file manifest. Omit to derive from every motion in source-dir.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--test-ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-std", type=float, default=1e-6)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all accepted samples")
    parser.add_argument(
        "--max-abs-root-xy",
        type=float,
        default=0.0,
        help="Reject motions whose absolute root X or Y exceeds this many meters; 0 disables.",
    )
    parser.add_argument(
        "--max-root-xy-displacement",
        type=float,
        default=0.0,
        help="Reject motions whose root XY end-to-start displacement exceeds this many meters; 0 disables.",
    )
    parser.add_argument(
        "--joint-limits-json",
        type=Path,
        help="Canonical G1 joint limits; omitted to disable joint-limit filtering.",
    )
    parser.add_argument(
        "--joint-limit-margin-abs",
        type=float,
        default=0.0,
        help="Absolute joint-limit allowance in radians.",
    )
    parser.add_argument(
        "--joint-limit-margin-ratio",
        type=float,
        default=0.0,
        help="Joint-limit allowance as a fraction of each joint range.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def filtered_source_name(source_file: str) -> str:
    """Map legacy ``0/file.npz`` paths to filtered ``00_file.npz`` names."""
    path = Path(source_file)
    if len(path.parts) < 2:
        raise ValueError(f"source_file has no shard component: {source_file!r}")
    try:
        shard = int(path.parts[-2])
    except ValueError as exc:
        raise ValueError(f"source_file has a non-numeric shard: {source_file!r}") from exc
    return f"{shard:02d}_{path.name}"


def read_accepted_names(path: Path) -> set[str]:
    names = {Path(line.strip()).name for line in path.open(encoding="utf-8") if line.strip()}
    if not names:
        raise ValueError(f"accepted manifest is empty: {path}")
    return names


def select_rows(texts_csv: Path, accepted_names: set[str]) -> tuple[list[dict[str, str]], set[str]]:
    with texts_csv.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        required = {"motion_name", "total_frames", "caption", "source_file"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{texts_csv} is missing columns: {sorted(required)}")
        rows = [row for row in reader if filtered_source_name(row["source_file"]) in accepted_names]
    matched_names = {filtered_source_name(row["source_file"]) for row in rows}
    missing = accepted_names - matched_names
    if missing:
        sample = sorted(missing)[:5]
        raise ValueError(f"{len(missing)} accepted files are absent from source corpus; sample={sample}")
    keys = {row["motion_name"] for row in rows}
    if len(keys) != len(rows):
        raise ValueError("selected texts.csv rows contain duplicate motion_name values")
    return rows, keys


def read_all_rows(texts_csv: Path) -> tuple[list[dict[str, str]], set[str]]:
    with texts_csv.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        required = {"motion_name", "total_frames", "caption", "source_file"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{texts_csv} is missing columns: {sorted(required)}")
        rows = list(reader)
    keys = {row["motion_name"] for row in rows}
    if len(keys) != len(rows):
        raise ValueError("texts.csv rows contain duplicate motion_name values")
    return rows, keys


def filter_root_xy_tails(
    data_dict: dict,
    keys: list[str],
    max_abs_root_xy: float,
    max_root_xy_displacement: float,
) -> tuple[list[str], dict[str, int]]:
    kept = []
    counts = {
        "max_abs_root_xy": 0,
        "root_xy_displacement": 0,
        "rejected_unique": 0,
    }
    for index, key in enumerate(keys, 1):
        motion = np.asarray(data_dict[key]["motion"], dtype=np.float32)
        if motion.ndim != 2 or motion.shape[1] != MOTION_DIM:
            raise ValueError(f"{key}: expected motion shape (T,{MOTION_DIM}), got {motion.shape}")
        reject = False
        if max_abs_root_xy > 0 and float(np.max(np.abs(motion[:, :2]))) > max_abs_root_xy:
            counts["max_abs_root_xy"] += 1
            reject = True
        if (
            max_root_xy_displacement > 0
            and float(np.linalg.norm(motion[-1, :2] - motion[0, :2]))
            > max_root_xy_displacement
        ):
            counts["root_xy_displacement"] += 1
            reject = True
        if reject:
            counts["rejected_unique"] += 1
        else:
            kept.append(key)
        if index % 10000 == 0:
            print(
                f"[root-tail] {index}/{len(keys)} motions, "
                f"rejected={counts['rejected_unique']}",
                flush=True,
            )
    return kept, counts


def read_joint_limits(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw = raw.get("joint_limits", raw)
    missing = [name for name in CANONICAL_JOINT_NAMES if name not in raw]
    extra = sorted(set(raw) - set(CANONICAL_JOINT_NAMES))
    if missing or extra:
        raise ValueError(
            f"joint-limit names differ from canonical G1 order; missing={missing}, extra={extra}"
        )
    limits = np.asarray([raw[name] for name in CANONICAL_JOINT_NAMES], dtype=np.float64)
    if limits.shape != (29, 2) or not np.all(np.isfinite(limits)):
        raise ValueError(f"expected 29 finite [low, high] joint limits, got {limits.shape}")
    if np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError("every joint limit must satisfy low < high")
    return limits[:, 0], limits[:, 1]


def filter_joint_limit_tails(
    data_dict: dict,
    keys: list[str],
    low: np.ndarray,
    high: np.ndarray,
    margin_abs: float,
    margin_ratio: float,
) -> tuple[list[str], dict]:
    if margin_abs < 0 or margin_ratio < 0:
        raise ValueError("joint-limit margins must be non-negative")
    margin = np.maximum(margin_abs, margin_ratio * (high - low))
    lower_bound = low - margin
    upper_bound = high + margin
    kept = []
    per_joint = {name: 0 for name in CANONICAL_JOINT_NAMES}
    max_excess_rad = 0.0
    rejected_unique = 0
    for index, key in enumerate(keys, 1):
        motion = np.asarray(data_dict[key]["motion"], dtype=np.float32)
        if motion.ndim != 2 or motion.shape[1] != MOTION_DIM:
            raise ValueError(f"{key}: expected motion shape (T,{MOTION_DIM}), got {motion.shape}")
        joints = motion[:, 7:].astype(np.float64, copy=False)
        excess = np.maximum(lower_bound - joints, joints - upper_bound)
        violated = np.max(excess, axis=0) > 0.0
        if np.any(violated):
            rejected_unique += 1
            max_excess_rad = max(max_excess_rad, float(np.max(excess[:, violated])))
            for joint_index in np.flatnonzero(violated):
                per_joint[CANONICAL_JOINT_NAMES[int(joint_index)]] += 1
        else:
            kept.append(key)
        if index % 10000 == 0:
            print(
                f"[joint-limit] {index}/{len(keys)} motions, rejected={rejected_unique}",
                flush=True,
            )
    return kept, {
        "rejected_unique": rejected_unique,
        "max_excess_beyond_margin_rad": max_excess_rad,
        "by_joint": {name: count for name, count in per_joint.items() if count},
    }


def compute_stats(data_dict: dict, keys: list[str], min_std: float) -> tuple[np.ndarray, np.ndarray]:
    total_sum = np.zeros(MOTION_DIM, dtype=np.float64)
    total_sum_sq = np.zeros(MOTION_DIM, dtype=np.float64)
    n_frames = 0
    for index, key in enumerate(keys, 1):
        motion = np.asarray(data_dict[key]["motion"], dtype=np.float32)
        if motion.ndim != 2 or motion.shape[1] != MOTION_DIM:
            raise ValueError(f"{key}: expected motion shape (T,{MOTION_DIM}), got {motion.shape}")
        motion64 = motion.astype(np.float64, copy=False)
        total_sum += motion64.sum(axis=0)
        total_sum_sq += np.square(motion64).sum(axis=0)
        n_frames += motion64.shape[0]
        if index % 10000 == 0:
            print(f"[stats] {index}/{len(keys)} motions", flush=True)
    if n_frames == 0:
        raise ValueError("selected corpus has no frames")
    mean64 = total_sum / n_frames
    variance = np.maximum(total_sum_sq / n_frames - np.square(mean64), 0.0)
    return mean64.astype(np.float32), np.maximum(np.sqrt(variance), min_std).astype(np.float32)


def split_keys(keys: list[str], test_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    if not 0.0 <= test_ratio <= 1.0:
        raise ValueError(f"test-ratio must be in [0,1], got {test_ratio}")
    rng = np.random.default_rng(seed)
    indices = np.arange(len(keys))
    rng.shuffle(indices)
    n_test = int(round(len(keys) * test_ratio))
    n_test = min(max(n_test, 1 if len(keys) > 1 and test_ratio > 0 else 0), max(len(keys) - 1, 0))
    test_indices = set(indices[:n_test].tolist())
    train = [key for index, key in enumerate(keys) if index not in test_indices]
    test = [key for index, key in enumerate(keys) if index in test_indices]
    return train, test


def write_lines(path: Path, values: list[str]) -> None:
    path.write_text("\n".join(values) + ("\n" if values else ""), encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_pkl = args.source_dir / "full_train.pkl"
    texts_csv = args.source_dir / "texts.csv"
    output_pkl = args.out_dir / "full_train.pkl"
    if output_pkl.exists() and not args.overwrite:
        raise FileExistsError(f"{output_pkl} exists; pass --overwrite to rebuild")

    if args.accepted_files is not None:
        accepted_names = read_accepted_names(args.accepted_files)
        rows, selected_keys = select_rows(texts_csv, accepted_names)
    else:
        rows, selected_keys = read_all_rows(texts_csv)

    print(f"[load] {source_pkl}", flush=True)
    with source_pkl.open("rb") as source:
        corpus = pickle.load(source)
    source_names = corpus["name_list"]
    source_lengths = corpus["length_list"]
    source_data = corpus["data_dict"]
    if len(source_names) != len(source_lengths) or len(source_names) != len(source_data):
        raise ValueError("source corpus name/length/data counts differ")

    selected_names = [key for key in source_names if key in selected_keys]
    if args.max_samples > 0:
        selected_names = selected_names[: args.max_samples]
        selected_keys = set(selected_names)
        rows = [row for row in rows if row["motion_name"] in selected_keys]
    if len(selected_names) != len(selected_keys):
        missing = sorted(selected_keys - set(selected_names))[:5]
        raise ValueError(f"selected keys are absent from full_train.pkl: {missing}")

    selected_names, root_tail_counts = filter_root_xy_tails(
        source_data,
        selected_names,
        args.max_abs_root_xy,
        args.max_root_xy_displacement,
    )
    joint_limit_counts = None
    if args.joint_limits_json is not None:
        low, high = read_joint_limits(args.joint_limits_json)
        selected_names, joint_limit_counts = filter_joint_limit_tails(
            source_data,
            selected_names,
            low,
            high,
            args.joint_limit_margin_abs,
            args.joint_limit_margin_ratio,
        )
    selected_keys = set(selected_names)
    rows = [row for row in rows if row["motion_name"] in selected_keys]
    if not selected_names:
        raise ValueError("root XY tail filters rejected every selected motion")

    length_by_name = dict(zip(source_names, source_lengths))
    selected_lengths = [int(length_by_name[key]) for key in selected_names]
    mean, std = compute_stats(source_data, selected_names, args.min_std)
    train_keys, test_keys = split_keys(selected_names, args.test_ratio, args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    temp_pkl = output_pkl.with_suffix(".pkl.tmp")
    print(f"[write] {output_pkl}", flush=True)
    with temp_pkl.open("wb") as output:
        pickle.dump(
            {
                "name_list": selected_names,
                "length_list": selected_lengths,
                "data_dict": {key: source_data[key] for key in selected_names},
            },
            output,
            protocol=4,
        )
    os.replace(temp_pkl, output_pkl)

    np.save(args.out_dir / "Mean.npy", mean)
    np.save(args.out_dir / "Std.npy", std)
    write_lines(args.out_dir / "train.txt", train_keys)
    write_lines(args.out_dir / "test.txt", test_keys)

    row_by_key = {row["motion_name"]: row for row in rows}
    with (args.out_dir / "texts.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=["motion_name", "total_frames", "caption", "source_file"])
        writer.writeheader()
        writer.writerows(row_by_key[key] for key in selected_names)

    summary = {
        "dataset": args.dataset_name,
        "source_dataset": str(args.source_dir.resolve()),
        "accepted_manifest": (
            str(args.accepted_files.resolve()) if args.accepted_files is not None else None
        ),
        "motions": len(selected_names),
        "train": len(train_keys),
        "test": len(test_keys),
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "motion_dim": MOTION_DIM,
        "length_min": min(selected_lengths),
        "length_max": max(selected_lengths),
        "root_tail_thresholds_m": {
            "max_abs_root_xy": args.max_abs_root_xy,
            "max_root_xy_displacement": args.max_root_xy_displacement,
        },
        "root_tail_rejections": root_tail_counts,
        "joint_limit_filter": (
            {
                "limits_file": str(args.joint_limits_json.resolve()),
                "margin_abs_rad": args.joint_limit_margin_abs,
                "margin_range_ratio": args.joint_limit_margin_ratio,
                "rejections_after_root_filter": joint_limit_counts,
            }
            if args.joint_limits_json is not None
            else None
        ),
    }
    (args.out_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"[done] dataset={args.dataset_name} motions={len(selected_names)} "
        f"train={len(train_keys)} test={len(test_keys)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
