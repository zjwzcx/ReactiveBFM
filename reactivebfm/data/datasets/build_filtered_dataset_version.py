#!/usr/bin/env python3
"""Build a filtered ReactiveBFM dataset from an accepted motion-name manifest.

The source cache is never modified.  Source train/val/test membership is
preserved so held-out comparisons remain stable across dataset versions.

When building a filtered dataset version, preserve the corresponding source
CSV files alongside the planner cache. The builder reads the exact
`motion_name` to `source_file` mapping from the source cache's `texts.csv` and
can extract only accepted files directly from `g1.tar.gz`:

```bash
python -m reactivebfm.data.datasets.build_filtered_dataset_version \
    --source-dir reactivebfm/data/datasets/assets/bones_seed_v2_g1_36dim \
    --accepted-motion-names /path/to/accepted_motion_names.txt \
    --out-dir /path/to/bones_seed_v3_g1_36dim \
    --dataset-name bones_seed_v3_g1_36dim \
    --mirror-suffix _M \
    --raw-source-root /mnt/oss/egoscale/humanoidvla_data/bones-studio_seed \
    --raw-out-dir /mnt/oss/egoscale/humanoidvla_data/bones-studio_seed_v3 \
    --overwrite
```

The raw output contains the accepted `g1/csv` files, filtered metadata, the
accepted manifest, the motion-to-source mapping, and a checksummed summary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import shutil
import tarfile
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--accepted-motion-names", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--mirror-suffix", default="")
    parser.add_argument("--filter-report", type=Path)
    parser.add_argument(
        "--raw-source-root",
        type=Path,
        help=(
            "BONES-SEED root containing extracted g1/csv files or g1.tar.gz. "
            "When set, preserve the accepted source CSV files as a raw dataset version."
        ),
    )
    parser.add_argument(
        "--raw-out-dir",
        type=Path,
        help="Destination for accepted raw CSV files (required with --raw-source-root).",
    )
    parser.add_argument("--min-std", type=float, default=1e-6)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read_lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_lines(path: Path, values: list[str]) -> None:
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _group_key(name: str, mirror_suffix: str) -> str:
    return name[: -len(mirror_suffix)] if mirror_suffix and name.endswith(mirror_suffix) else name


def _validate_group_selection(
    source_names: list[str], accepted: set[str], mirror_suffix: str
) -> None:
    if not mirror_suffix:
        return
    groups: defaultdict[str, list[str]] = defaultdict(list)
    for name in source_names:
        groups[_group_key(name, mirror_suffix)].append(name)
    partial = [members for members in groups.values() if accepted.intersection(members) and not accepted.issuperset(members)]
    if partial:
        raise ValueError(
            f"accepted manifest partially selects {len(partial)} mirror groups; "
            f"sample={partial[:3]}"
        )


def _source_splits(source: Path, source_names: list[str]) -> dict[str, list[str]]:
    splits = {name: _read_lines(source / f"{name}.txt") for name in ("train", "val", "test")}
    flat = [name for values in splits.values() for name in values]
    if len(flat) != len(set(flat)):
        raise ValueError("source split files overlap or contain duplicate names")
    if set(flat) != set(source_names):
        missing = set(source_names) - set(flat)
        extra = set(flat) - set(source_names)
        raise ValueError(
            f"source splits do not partition the cache: missing={list(sorted(missing))[:5]} "
            f"extra={list(sorted(extra))[:5]}"
        )
    return splits


def _compute_stats(
    data_dict: dict[str, dict[str, Any]], names: list[str], min_std: float
) -> tuple[np.ndarray, np.ndarray, int, int]:
    total = None
    total_sq = None
    frames = 0
    motion_dim = 0
    for index, name in enumerate(names, 1):
        motion = np.asarray(data_dict[name]["motion"], dtype=np.float64)
        if motion.ndim != 2:
            raise ValueError(f"{name}: expected [frames, channels], got {motion.shape}")
        if total is None:
            motion_dim = int(motion.shape[1])
            total = np.zeros(motion_dim, dtype=np.float64)
            total_sq = np.zeros(motion_dim, dtype=np.float64)
        elif motion.shape[1] != motion_dim:
            raise ValueError(f"{name}: motion dim {motion.shape[1]} != {motion_dim}")
        total += motion.sum(axis=0)
        total_sq += np.square(motion).sum(axis=0)
        frames += len(motion)
        if index % 20_000 == 0:
            print(f"[stats] {index}/{len(names)}", flush=True)
    if total is None or total_sq is None or frames == 0:
        raise ValueError("filtered train split contains no frames")
    mean = total / frames
    variance = np.maximum(total_sq / frames - np.square(mean), 0.0)
    std = np.maximum(np.sqrt(variance), min_std)
    return mean.astype(np.float32), std.astype(np.float32), frames, motion_dim


def _write_filtered_texts(source: Path, output: Path, accepted: set[str]) -> int:
    source_path = source / "texts.csv"
    if not source_path.is_file():
        return 0
    count = 0
    seen_names: set[str] = set()
    with source_path.open(newline="", encoding="utf-8") as source_handle, output.open(
        "w", newline="", encoding="utf-8"
    ) as output_handle:
        reader = csv.DictReader(source_handle)
        if not reader.fieldnames or "motion_name" not in reader.fieldnames:
            raise ValueError(f"{source_path} has no motion_name column")
        writer = csv.DictWriter(output_handle, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if row["motion_name"] in accepted:
                writer.writerow(row)
                count += 1
                seen_names.add(row["motion_name"])
    missing = accepted - seen_names
    if missing:
        raise ValueError(f"texts.csv is missing {len(missing)} accepted motions; sample={sorted(missing)[:5]}")
    return count


def _accepted_source_files(source: Path, accepted: set[str]) -> dict[str, list[str]]:
    """Map each accepted motion to its source files recorded by the converter."""
    source_path = source / "texts.csv"
    if not source_path.is_file():
        raise FileNotFoundError(f"raw preservation requires {source_path}")
    result: defaultdict[str, set[str]] = defaultdict(set)
    with source_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"motion_name", "source_file"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{source_path} is missing columns: {sorted(missing)}")
        for row in reader:
            name = row["motion_name"]
            relative = row["source_file"].strip()
            if name in accepted and relative:
                result[name].add(relative)
    missing_names = accepted - set(result)
    if missing_names:
        raise ValueError(
            f"texts.csv has no source_file for {len(missing_names)} accepted motions; "
            f"sample={sorted(missing_names)[:5]}"
        )
    return {name: sorted(paths) for name, paths in sorted(result.items())}


def _safe_relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe source_file path: {value}")
    return path


def _write_filtered_metadata(source: Path, output: Path, wanted: set[str]) -> int:
    with source.open(newline="", encoding="utf-8") as source_handle, output.open(
        "w", newline="", encoding="utf-8"
    ) as output_handle:
        reader = csv.DictReader(source_handle)
        if not reader.fieldnames or "move_g1_path" not in reader.fieldnames:
            raise ValueError(f"{source} has no move_g1_path column")
        writer = csv.DictWriter(output_handle, fieldnames=reader.fieldnames)
        writer.writeheader()
        count = 0
        for row in reader:
            if row["move_g1_path"] in wanted:
                writer.writerow(row)
                count += 1
    return count


def _preserve_raw_sources(
    *,
    source: Path,
    accepted: set[str],
    raw_source_root: Path,
    raw_output: Path,
    dataset_name: str,
    manifest: Path,
    overwrite: bool,
) -> dict[str, Any]:
    if raw_output.exists() and not overwrite:
        raise FileExistsError(f"{raw_output} already exists; pass --overwrite to replace it")
    source_files = _accepted_source_files(source, accepted)
    relative_paths = sorted(
        {_safe_relative_path(value) for values in source_files.values() for value in values},
        key=str,
    )
    extracted_available = all((raw_source_root / path).is_file() for path in relative_paths)
    archive = raw_source_root / "g1.tar.gz"
    if not extracted_available and not archive.is_file():
        missing = [str(path) for path in relative_paths if not (raw_source_root / path).is_file()]
        raise FileNotFoundError(
            f"raw source has no g1.tar.gz and is missing {len(missing)} files; sample={missing[:5]}"
        )

    raw_output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{raw_output.name}.", dir=raw_output.parent))
    try:
        wanted = {path.as_posix() for path in relative_paths}
        copied: set[str] = set()
        if extracted_available:
            for path in relative_paths:
                destination = staging / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(raw_source_root / path, destination)
                copied.add(path.as_posix())
            storage = "extracted_files"
        else:
            with tarfile.open(archive, "r:gz") as handle:
                for member in handle:
                    if not member.isfile() or member.name not in wanted:
                        continue
                    source_handle = handle.extractfile(member)
                    if source_handle is None:
                        continue
                    destination = staging / _safe_relative_path(member.name)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with source_handle, destination.open("wb") as output_handle:
                        shutil.copyfileobj(source_handle, output_handle)
                    copied.add(member.name)
            storage = "g1.tar.gz"
        missing_files = wanted - copied
        if missing_files:
            raise FileNotFoundError(
                f"raw source is missing {len(missing_files)} accepted files; "
                f"sample={sorted(missing_files)[:5]}"
            )

        metadata_rows = None
        metadata_source = raw_source_root / "metadata" / "seed_metadata_v004.csv"
        if metadata_source.is_file():
            metadata_output = staging / "metadata" / metadata_source.name
            metadata_output.parent.mkdir(parents=True, exist_ok=True)
            metadata_rows = _write_filtered_metadata(metadata_source, metadata_output, wanted)
            if metadata_rows != len(wanted):
                raise ValueError(
                    f"metadata contains {metadata_rows} of {len(wanted)} accepted source files"
                )
        shutil.copy2(manifest, staging / "accepted_motion_names.txt")
        (staging / "motion_source_files.json").write_text(
            json.dumps(source_files, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        total_bytes = sum((staging / path).stat().st_size for path in relative_paths)
        summary = {
            "dataset": dataset_name,
            "source_root": str(raw_source_root),
            "source_storage": storage,
            "accepted_motions": len(accepted),
            "raw_files": len(relative_paths),
            "raw_bytes": total_bytes,
            "metadata_rows": metadata_rows,
            "accepted_motion_names_sha256": _sha256(manifest),
        }
        (staging / "raw_dataset_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if raw_output.exists():
            shutil.rmtree(raw_output)
        os.replace(staging, raw_output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def _group_sets(
    splits: dict[str, list[str]], mirror_suffix: str
) -> dict[str, set[str]]:
    return {
        split: {_group_key(name, mirror_suffix) for name in names}
        for split, names in splits.items()
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source_dir.expanduser().resolve()
    output = args.out_dir.expanduser().resolve()
    manifest = args.accepted_motion_names.expanduser().resolve()
    report = args.filter_report.expanduser().resolve() if args.filter_report else None
    raw_source_arg = getattr(args, "raw_source_root", None)
    raw_output_arg = getattr(args, "raw_out_dir", None)
    if bool(raw_source_arg) != bool(raw_output_arg):
        raise ValueError("--raw-source-root and --raw-out-dir must be provided together")
    raw_source_root = raw_source_arg.expanduser().resolve() if raw_source_arg else None
    raw_output = raw_output_arg.expanduser().resolve() if raw_output_arg else None
    if source == output:
        raise ValueError("source-dir and out-dir must differ")
    if raw_output is not None and raw_output == output:
        raise ValueError("out-dir and raw-out-dir must differ")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} already exists; pass --overwrite to replace it")
    if raw_output is not None and raw_output.exists() and not args.overwrite:
        raise FileExistsError(f"{raw_output} already exists; pass --overwrite to replace it")
    if report is not None and not report.is_file():
        raise FileNotFoundError(report)

    accepted_lines = _read_lines(manifest)
    accepted = set(accepted_lines)
    if not accepted or len(accepted) != len(accepted_lines):
        raise ValueError("accepted manifest is empty or contains duplicate names")
    if raw_source_root is not None and not raw_source_root.is_dir():
        raise FileNotFoundError(raw_source_root)

    print(f"[load] {source / 'full_train.pkl'}", flush=True)
    with (source / "full_train.pkl").open("rb") as handle:
        corpus = pickle.load(handle)
    source_names = list(corpus["name_list"])
    source_lengths = list(corpus["length_list"])
    source_data = corpus["data_dict"]
    if len(source_names) != len(source_lengths) or len(source_names) != len(source_data):
        raise ValueError("source cache name/length/data counts differ")
    unknown = accepted - set(source_names)
    if unknown:
        raise ValueError(f"manifest contains {len(unknown)} unknown names; sample={sorted(unknown)[:5]}")
    _validate_group_selection(source_names, accepted, args.mirror_suffix)

    selected_names = [name for name in source_names if name in accepted]
    length_by_name = dict(zip(source_names, source_lengths))
    selected_lengths = [int(length_by_name[name]) for name in selected_names]
    source_splits = _source_splits(source, source_names)
    splits = {
        split: [name for name in names if name in accepted]
        for split, names in source_splits.items()
    }
    mean, std, train_frames, motion_dim = _compute_stats(
        source_data, splits["train"], args.min_std
    )

    group_sets = _group_sets(splits, args.mirror_suffix)
    leakage = {
        "train_val": len(group_sets["train"] & group_sets["val"]),
        "train_test": len(group_sets["train"] & group_sets["test"]),
        "val_test": len(group_sets["val"] & group_sets["test"]),
    }
    if any(leakage.values()):
        raise RuntimeError(f"filtered split group leakage detected: {leakage}")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        print(f"[write] filtered cache with {len(selected_names)} motions", flush=True)
        with (staging / "full_train.pkl").open("wb") as handle:
            pickle.dump(
                {
                    "name_list": selected_names,
                    "length_list": selected_lengths,
                    "data_dict": {name: source_data[name] for name in selected_names},
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        np.save(staging / "Mean.npy", mean)
        np.save(staging / "Std.npy", std)
        for split, names in splits.items():
            _write_lines(staging / f"{split}.txt", names)
        caption_rows = _write_filtered_texts(source, staging / "texts.csv", accepted)
        if (source / "joint_names.npy").is_file():
            shutil.copy2(source / "joint_names.npy", staging / "joint_names.npy")

        conversion = {}
        if (source / "conversion.json").is_file():
            conversion = json.loads((source / "conversion.json").read_text(encoding="utf-8"))
        conversion.update(
            {
                "num_motions": len(selected_names),
                "source_num_motions": len(source_names),
                "filtered_from": str(source),
                "accepted_motion_manifest": str(manifest),
            }
        )
        (staging / "conversion.json").write_text(
            json.dumps(conversion, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        groups = defaultdict(list)
        for name in selected_names:
            groups[_group_key(name, args.mirror_suffix)].append(name)
        summary = {
            "dataset": args.dataset_name,
            "source_dataset": str(source),
            "source_motions": len(source_names),
            "total_motions": len(selected_names),
            "removed_motions": len(source_names) - len(selected_names),
            "motions": {split: len(names) for split, names in splits.items()},
            "motion_ratios": {
                split: len(names) / len(selected_names) for split, names in splits.items()
            },
            "groups": {split: len(values) for split, values in group_sets.items()},
            "total_groups": len(groups),
            "mirror_suffix": args.mirror_suffix or None,
            "mirror_groups_with_multiple_items": sum(len(values) > 1 for values in groups.values()),
            "group_leakage": leakage,
            "split_policy": "preserve_source_split_membership",
            "normalization_scope": "filtered_train_only",
            "train_frames_for_statistics": train_frames,
            "motion_dim": motion_dim,
            "caption_rows": caption_rows,
            "filter_provenance": {
                "accepted_motion_names": str(manifest),
                "accepted_motion_names_sha256": _sha256(manifest),
                "filter_report": str(report) if report else None,
                "filter_report_sha256": _sha256(report) if report else None,
            },
        }
        (staging / "split_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        shutil.copy2(manifest, staging / "accepted_motion_names.txt")
        if report is not None:
            audit_dir = staging / "filter_audit"
            audit_dir.mkdir()
            shutil.copy2(report, audit_dir / "filter_report.jsonl")
            for name in (
                "filter_summary.json",
                "analysis_summary.json",
                "rejected_groups.jsonl",
            ):
                source_path = report.parent / name
                if source_path.is_file():
                    shutil.copy2(source_path, audit_dir / name)
        if output.exists():
            shutil.rmtree(output)
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    if raw_source_root is not None and raw_output is not None:
        print(f"[write] preserving accepted raw sources in {raw_output}", flush=True)
        summary["raw_dataset"] = _preserve_raw_sources(
            source=source,
            accepted=accepted,
            raw_source_root=raw_source_root,
            raw_output=raw_output,
            dataset_name=args.dataset_name,
            manifest=manifest,
            overwrite=args.overwrite,
        )

    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def main() -> None:
    build(_parse_args())


if __name__ == "__main__":
    main()
