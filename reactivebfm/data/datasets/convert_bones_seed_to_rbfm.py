#!/usr/bin/env python3
"""Convert BONES-SEED Unitree-G1 CSV files to a ReactiveBFM dataset.

BONES-SEED distributes the G1 retarget as CSV files, rather than the NPZ
``qpos`` files used by the other ReactiveBFM corpora.  A row contains::

    Frame, root_translateX/Y/Z, root_rotateX/Y/Z,
    <the 29 G1 *_joint_dof columns>

The source is 120 Hz.  This converter writes the standard ReactiveBFM
representation at ``--target-fps`` (60 Hz by default)::

    qpos[:, 0:3]  root position in metres, world XYZ (Z-up)
    qpos[:, 3:7]  root quaternion in XYZW order
    qpos[:, 7:36] 29 G1 joint positions in radians

The column order is checked against the repository's canonical G1 order.  A
dataset directory containing ``full_train.pkl``, ``Mean.npy``, ``Std.npy`` and
the train/test split files is produced directly, so no intermediate NPZ pass
is needed.  Both an extracted ``g1/csv`` directory and ``g1.tar.gz`` are
supported.  The latter is useful when the 22 GB archive has not been unpacked.
Only ``content_natural_desc_1`` through ``content_natural_desc_4`` are used as
training captions; technical, short, and temporal labels are intentionally not
mixed into full-motion conditioning.

Example (small smoke test)::

  python -m reactivebfm.data.datasets.convert_bones_seed_to_rbfm \
      --bones-root /mnt/oss/egoscale/humanoidvla_data/bones-studio_seed \
      --out-dir /mnt/oss/egoscale/humanoidvla_data/bones_seed_g1_36dim \
      --max-samples 100 --overwrite

For the complete corpus, omit ``--max-samples``.  The full pickle is large
(several GB); make sure the destination has enough space.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import pickle
import re
import tarfile
from pathlib import Path
from typing import Iterator, TextIO

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


DOF_NAMES = [
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    "waist_yaw",
    "waist_roll",
    "waist_pitch",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_roll",
    "left_wrist_pitch",
    "left_wrist_yaw",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
]
DOF_COLUMNS = [f"{name}_joint_dof" for name in DOF_NAMES]
EXPECTED_COLUMNS = [
    "Frame",
    "root_translateX",
    "root_translateY",
    "root_translateZ",
    "root_rotateX",
    "root_rotateY",
    "root_rotateZ",
    *DOF_COLUMNS,
]
JOINT_NAMES = np.asarray(
    ["root", *[f"{name}_joint" for name in DOF_NAMES]], dtype="<U64"
)
NATURAL_CAPTION_COLUMNS = (
    "content_natural_desc_1",
    "content_natural_desc_2",
    "content_natural_desc_3",
    "content_natural_desc_4",
)
WORD_PATTERN = re.compile(r"[A-Za-z]+")


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bones-root", type=Path, required=True)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Metadata CSV (default: <bones-root>/metadata/seed_metadata_v004.csv).",
    )
    parser.add_argument(
        "--g1-archive",
        type=Path,
        default=None,
        help="G1 tar.gz (default: <bones-root>/g1.tar.gz when extracted CSV is absent).",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=_positive_int, default=None)
    parser.add_argument("--target-fps", type=float, default=60.0)
    parser.add_argument("--source-fps", type=float, default=120.0)
    parser.add_argument(
        "--root-translation-scale",
        type=float,
        default=0.01,
        help="Scale CSV root translations to metres (BONES-SEED G1 is cm; default 0.01).",
    )
    parser.add_argument(
        "--root-euler-order",
        default="xyz",
        choices=("xyz", "zyx", "XYZ", "ZYX"),
        help="Euler order used by root_rotateX/Y/Z (default: xyz).",
    )
    parser.add_argument("--test-ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-std", type=float, default=1e-6)
    parser.add_argument(
        "--allow-missing-text",
        action="store_true",
        help="Keep motions without a caption (normally they are skipped).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _simple_tokens(caption: str) -> list[str]:
    return [f"{word}/OTHER" for word in WORD_PATTERN.findall(caption.lower())]


def _captions(row: dict[str, str]) -> list[str]:
    result: list[str] = []
    for column in NATURAL_CAPTION_COLUMNS:
        value = (row.get(column) or "").strip()
        if value and value not in result:
            result.append(value)
    return result


def _read_metadata(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"filename", "move_g1_path", *NATURAL_CAPTION_COLUMNS}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"metadata is missing columns: {sorted(missing)}")
        return list(reader)


def _read_csv_motion(handle: TextIO, *, source_name: str, translation_scale: float, euler_order: str, source_fps: float, target_fps: float) -> np.ndarray:
    reader = csv.reader(handle)
    try:
        header = next(reader)
    except StopIteration as exc:
        raise ValueError(f"{source_name}: empty CSV") from exc
    positions = {name: idx for idx, name in enumerate(header)}
    missing = [name for name in EXPECTED_COLUMNS if name not in positions]
    if missing:
        raise ValueError(f"{source_name}: missing columns {missing}")

    values: list[list[float]] = []
    use_columns = [positions[name] for name in EXPECTED_COLUMNS]
    for line_number, row in enumerate(reader, 2):
        if not row or not any(cell.strip() for cell in row):
            continue
        if len(row) < len(header):
            raise ValueError(f"{source_name}:{line_number}: short row ({len(row)} < {len(header)})")
        try:
            values.append([float(row[idx]) for idx in use_columns])
        except ValueError as exc:
            raise ValueError(f"{source_name}:{line_number}: non-numeric value") from exc
    if not values:
        raise ValueError(f"{source_name}: no motion rows")

    raw = np.asarray(values, dtype=np.float64)
    root_pos = raw[:, 1:4] * float(translation_scale)
    root_rot = Rotation.from_euler(euler_order, raw[:, 4:7], degrees=True)
    dof = np.deg2rad(raw[:, 7:]).astype(np.float64)
    root_quat = root_rot.as_quat()
    # q and -q encode the same orientation, but sign jumps are undesirable
    # for interpolation and for a model that directly consumes quaternion data.
    if len(root_quat) > 1:
        signs = np.where(np.sum(root_quat[:-1] * root_quat[1:], axis=1) < 0.0, -1.0, 1.0)
        root_quat[1:] *= np.cumprod(signs)[:, None]
    qpos = np.concatenate([root_pos, root_quat, dof], axis=1)
    qpos = _resample(qpos, source_fps=source_fps, target_fps=target_fps)
    return qpos.astype(np.float32)


def _resample(qpos: np.ndarray, *, source_fps: float, target_fps: float) -> np.ndarray:
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("source and target FPS must be positive")
    if qpos.shape[0] < 2 or abs(source_fps - target_fps) < 1e-8:
        return qpos
    duration = (qpos.shape[0] - 1) / source_fps
    n_target = max(2, int(round(duration * target_fps)) + 1)
    source_times = np.arange(qpos.shape[0], dtype=np.float64) / source_fps
    target_times = np.linspace(0.0, duration, n_target)
    root_pos = np.column_stack([
        np.interp(target_times, source_times, qpos[:, axis]) for axis in range(3)
    ])
    rotations = Rotation.from_quat(qpos[:, 3:7])
    root_quat = Slerp(source_times, rotations)(target_times).as_quat()
    dof = np.column_stack([
        np.interp(target_times, source_times, qpos[:, axis])
        for axis in range(7, qpos.shape[1])
    ])
    return np.concatenate([root_pos, root_quat, dof], axis=1)


def _metadata_index(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        path = (row.get("move_g1_path") or "").strip()
        name = (row.get("filename") or "").strip()
        if not path or not name:
            continue
        if path in result:
            raise ValueError(f"duplicate move_g1_path in metadata: {path}")
        result[path] = row
    return result


def _iter_extracted(root: Path, paths: set[str]) -> Iterator[tuple[str, TextIO]]:
    for relative in sorted(paths):
        path = root / relative
        if not path.is_file():
            continue
        yield relative, path.open("r", newline="", encoding="utf-8")


def _iter_archive(archive: Path, paths: set[str]) -> Iterator[tuple[str, TextIO]]:
    with tarfile.open(archive, mode="r|gz") as handle:
        for member in handle:
            if not member.isfile() or member.name not in paths:
                continue
            extracted = handle.extractfile(member)
            if extracted is None:
                continue
            # TarFile's Python 3.12 streaming ExFileObject does not implement
            # ``seekable()``, which TextIOWrapper queries during construction.
            # A single motion CSV is small enough to decode eagerly here.
            yield member.name, io.StringIO(extracted.read().decode("utf-8"), newline="")


def _write_outputs(out_dir: Path, records: list[tuple[str, np.ndarray, list[str], str]], *, test_ratio: float, seed: int, min_std: float) -> None:
    if not records:
        raise RuntimeError("no valid BONES-SEED motions were converted")
    out_dir.mkdir(parents=True, exist_ok=True)
    total = np.zeros(36, dtype=np.float64)
    total_sq = np.zeros(36, dtype=np.float64)
    n_frames = 0
    data_dict: dict[str, dict] = {}
    names: list[str] = []
    lengths: list[int] = []
    text_rows: list[tuple[str, int, str, str]] = []
    for key, motion, captions, source_file in records:
        total += motion.astype(np.float64).sum(axis=0)
        total_sq += np.square(motion.astype(np.float64)).sum(axis=0)
        n_frames += len(motion)
        names.append(key)
        lengths.append(len(motion))
        data_dict[key] = {
            "motion": motion,
            "length": len(motion),
            "text": [{"caption": caption, "tokens": _simple_tokens(caption)} for caption in captions],
        }
        for caption in captions:
            text_rows.append((key, len(motion), caption, source_file))
    mean = total / n_frames
    variance = np.maximum(total_sq / n_frames - np.square(mean), 0.0)
    std = np.maximum(np.sqrt(variance), min_std).astype(np.float32)
    with (out_dir / "full_train.pkl").open("wb") as handle:
        pickle.dump({"name_list": names, "length_list": lengths, "data_dict": data_dict}, handle, protocol=4)
    np.save(out_dir / "Mean.npy", mean.astype(np.float32))
    np.save(out_dir / "Std.npy", std)

    if not 0.0 <= test_ratio <= 1.0:
        raise ValueError("test-ratio must be in [0, 1]")
    rng = np.random.default_rng(seed)
    shuffled = np.arange(len(names))
    rng.shuffle(shuffled)
    n_test = int(round(len(names) * test_ratio))
    if test_ratio > 0 and len(names) > 1:
        n_test = max(1, min(n_test, len(names) - 1))
    test_indices = set(shuffled[:n_test].tolist())
    (out_dir / "train.txt").write_text("\n".join(key for i, key in enumerate(names) if i not in test_indices) + "\n", encoding="utf-8")
    (out_dir / "test.txt").write_text("\n".join(key for i, key in enumerate(names) if i in test_indices) + ("\n" if test_indices else ""), encoding="utf-8")
    with (out_dir / "texts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["motion_name", "total_frames", "caption", "source_file"])
        writer.writerows(text_rows)
    np.save(out_dir / "joint_names.npy", JOINT_NAMES)


def main() -> None:
    args = _parse_args()
    root = args.bones_root
    if not root.is_dir():
        raise FileNotFoundError(root)
    metadata = args.metadata or root / "metadata" / "seed_metadata_v004.csv"
    if not metadata.is_file():
        raise FileNotFoundError(metadata)
    out_dir = args.out_dir
    if (out_dir / "full_train.pkl").exists() and not args.overwrite:
        raise FileExistsError(f"{out_dir / 'full_train.pkl'} exists; pass --overwrite")
    if args.target_fps <= 0 or args.source_fps <= 0:
        raise ValueError("FPS must be positive")

    rows = _read_metadata(metadata)
    index = _metadata_index(rows)
    archive = args.g1_archive or root / "g1.tar.gz"
    extracted_root = root / "g1" / "csv"
    use_extracted = extracted_root.is_dir()
    if not use_extracted and not archive.is_file():
        raise FileNotFoundError(f"neither extracted G1 CSV directory nor archive exists under {root}")

    records: list[tuple[str, np.ndarray, list[str], str]] = []
    used_keys: set[str] = set()
    missing_text = 0
    skipped = 0
    source_iter = _iter_extracted(root, set(index)) if use_extracted else _iter_archive(archive, set(index))
    for relative, handle in source_iter:
        try:
            row = index[relative]
            captions = _captions(row)
            if not captions and not args.allow_missing_text:
                missing_text += 1
                continue
            motion = _read_csv_motion(
                handle,
                source_name=relative,
                translation_scale=args.root_translation_scale,
                euler_order=args.root_euler_order,
                source_fps=args.source_fps,
                target_fps=args.target_fps,
            )
            key = row["filename"]
            if key in used_keys:
                key = f"{key}__dup{len(records):06d}"
            used_keys.add(key)
            records.append((key, motion, captions, relative))
            if len(records) == 1 or len(records) % 1000 == 0:
                print(f"[convert] {len(records)} motions, latest={relative}, frames={len(motion)}", flush=True)
            if args.max_samples is not None and len(records) >= args.max_samples:
                break
        except Exception as exc:
            skipped += 1
            print(f"[skip] {relative}: {exc}", flush=True)
        finally:
            handle.close()
    if not records:
        raise RuntimeError("no valid motions found; check archive extraction and metadata paths")
    _write_outputs(out_dir, records, test_ratio=args.test_ratio, seed=args.seed, min_std=args.min_std)
    with (out_dir / "conversion.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "source": str(root),
                "metadata": str(metadata),
                "source_fps": args.source_fps,
                "target_fps": args.target_fps,
                "root_translation_scale": args.root_translation_scale,
                "root_euler_order": args.root_euler_order,
                "root_quaternion_order": "xyzw",
                "caption_fields": list(NATURAL_CAPTION_COLUMNS),
                "joint_names": JOINT_NAMES.tolist(),
                "num_motions": len(records),
                "skipped": skipped,
                "missing_text": missing_text,
            },
            handle,
            indent=2,
        )
    print(f"[done] motions={len(records)} skipped={skipped} missing_text={missing_text}")
    print(f"[done] output={out_dir}")
    print(f"[done] mean/std={out_dir / 'Mean.npy'} {out_dir / 'Std.npy'}")


if __name__ == "__main__":
    main()
