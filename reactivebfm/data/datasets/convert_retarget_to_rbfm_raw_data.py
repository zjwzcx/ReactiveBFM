#!/usr/bin/env python3
"""Direct converter: Hymotion retargeted -> ReactiveBFM-like 36-dim at 60 FPS.

This script does one thing: convert each source `.npz` under:
  /mnt/home/chenxiao/hymotion/retargeted
to an output `.npz` under:
  /mnt/home/chenxiao/hymotion/retargeted_rbfm
with the SAME shard/folder structure and file order.

Converted representation:
  qpos: (T, 36), float32
    [0:3]   root_pos (xyz)
    [3:7]   root_quat (xyzw, normalized)
    [7:36]  dof_pos (29)
  frequency: target fps (default 60)

Usage:
# 全量处理
python /mnt/home/chenxiao/ReactiveBFM-Internal/reactivebfm/data/datasets/convert_retarget_to_rbfm_raw_data.py all --overwrite

# 处理前 100 条（按文件夹/文件顺序）
python /mnt/home/chenxiao/ReactiveBFM-Internal/reactivebfm/data/datasets/convert_retarget_to_rbfm_raw_data.py 100
python /mnt/home/chenxiao/ReactiveBFM-Internal/reactivebfm/data/datasets/convert_retarget_to_rbfm_raw_data.py 100 --overwrite
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


EXPECTED_DOF_ORDER = [
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
CANONICAL_JOINT_NAMES = np.asarray(
    ["root"] + [f"{name}_joint" for name in EXPECTED_DOF_ORDER], dtype="<U64"
)
META_STEM_PATTERN = re.compile(r"^(?P<base>.+?)_000_smplx(?:_(?P<hz>\d+)Hz_29dof)?$")


def _parse_count(value: str) -> Optional[int]:
    if value.lower() == "all":
        return None
    try:
        count = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("count must be 'all' or a positive integer") from exc
    if count <= 0:
        raise argparse.ArgumentTypeError("count must be > 0 when using a number")
    return count


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interpolate Hymotion retargeted motions to 60 FPS in 36-dim qpos format."
    )
    parser.add_argument(
        "count",
        type=_parse_count,
        help="How many files to process in folder/file order: 'all' or a positive integer.",
    )
    parser.add_argument(
        "--src-root",
        type=Path,
        default=Path("/mnt/home/chenxiao/hymotion/retargeted"),
        help="Source retargeted root. Default: /mnt/home/chenxiao/hymotion/retargeted",
    )
    parser.add_argument(
        "--dst-root",
        type=Path,
        default=Path("/mnt/home/chenxiao/hymotion/retargeted_rbfm"),
        help="Output root. Default: /mnt/home/chenxiao/hymotion/retargeted_rbfm",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=60.0,
        help="Target FPS. Default: 60",
    )
    parser.add_argument(
        "--default-src-fps",
        type=float,
        default=30.0,
        help="Fallback source FPS when no frequency field exists. Default: 30",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/mnt/home/chenxiao/hymotion/raw"),
        help="Raw root for loading text from *_meta.json if retargeted npz has no text.",
    )
    parser.add_argument(
        "--src-quat-order",
        type=str,
        default="wxyz",
        choices=("wxyz", "xyzw", "auto"),
        help="Quaternion order in source qpos[:,3:7]. Output is always xyzw.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args()


def _slerp(q0: np.ndarray, q1: np.ndarray, t: np.ndarray) -> np.ndarray:
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    sign = np.where(dot < 0.0, -1.0, 1.0)
    q1 = q1 * sign
    dot = np.abs(dot)

    out = np.empty_like(q0)
    linear_mask = dot.squeeze(-1) > 0.9995
    if linear_mask.any():
        out[linear_mask] = q0[linear_mask] + t[linear_mask, None] * (q1[linear_mask] - q0[linear_mask])

    slerp_mask = ~linear_mask
    if slerp_mask.any():
        clipped_dot = np.clip(dot[slerp_mask].squeeze(-1), -1.0, 1.0)
        theta = np.arccos(clipped_dot)
        sin_theta = np.sin(theta)
        w0 = np.sin((1.0 - t[slerp_mask]) * theta) / sin_theta
        w1 = np.sin(t[slerp_mask] * theta) / sin_theta
        out[slerp_mask] = w0[:, None] * q0[slerp_mask] + w1[:, None] * q1[slerp_mask]

    out /= np.maximum(np.linalg.norm(out, axis=-1, keepdims=True), 1e-12)
    return out


def _normalize_quat_inplace(qpos36: np.ndarray) -> None:
    quat = qpos36[:, 3:7]
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    qpos36[:, 3:7] = quat / np.maximum(norm, 1e-12)


def _infer_quat_order(qpos36: np.ndarray) -> str:
    """Infer quaternion order in qpos[:,3:7] with a simple heuristic."""
    q = np.asarray(qpos36[:, 3:7], dtype=np.float64)
    score_wxyz = float(np.mean(np.abs(q[:, 0])))
    score_xyzw = float(np.mean(np.abs(q[:, 3])))
    if score_wxyz > score_xyzw + 0.05:
        return "wxyz"
    if score_xyzw > score_wxyz + 0.05:
        return "xyzw"
    # tie-break by first frame closeness to identity
    q0 = q[0]
    c_wxyz = abs(abs(q0[0]) - 1.0) + float(np.linalg.norm(q0[1:4]))
    q0_as_wxyz = np.roll(q0, 1)  # if source is xyzw
    c_xyzw = abs(abs(q0_as_wxyz[0]) - 1.0) + float(np.linalg.norm(q0_as_wxyz[1:4]))
    return "wxyz" if c_wxyz <= c_xyzw else "xyzw"


def _to_xyzw_quat_inplace(qpos36: np.ndarray, src_quat_order: str) -> None:
    if src_quat_order == "xyzw":
        _normalize_quat_inplace(qpos36)
        return
    if src_quat_order != "wxyz":
        raise ValueError(f"Unsupported src_quat_order: {src_quat_order}")
    q = qpos36[:, 3:7].copy()
    qpos36[:, 3:7] = np.concatenate([q[:, 1:4], q[:, 0:1]], axis=1)
    _normalize_quat_inplace(qpos36)


def _resample_qpos36(qpos36: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    qpos36 = np.asarray(qpos36, dtype=np.float32)
    if qpos36.ndim != 2 or qpos36.shape[1] != 36:
        raise ValueError(f"Expected qpos shape (T, 36), got {qpos36.shape}")

    if qpos36.shape[0] < 2 or abs(src_fps - dst_fps) < 1e-6:
        _normalize_quat_inplace(qpos36)
        return qpos36

    root_pos = qpos36[:, 0:3]
    root_quat = qpos36[:, 3:7]
    dof_pos = qpos36[:, 7:36]

    total_src = qpos36.shape[0]
    duration = (total_src - 1) / src_fps
    total_dst = max(2, int(round(duration * dst_fps)) + 1)

    xs = np.linspace(0.0, total_src - 1.0, total_dst)
    i0 = np.clip(np.floor(xs).astype(np.int64), 0, total_src - 2)
    i1 = i0 + 1
    alpha = (xs - i0).astype(np.float64)

    out_root_pos = (1.0 - alpha[:, None]) * root_pos[i0] + alpha[:, None] * root_pos[i1]
    out_dof_pos = (1.0 - alpha[:, None]) * dof_pos[i0] + alpha[:, None] * dof_pos[i1]
    out_root_quat = _slerp(root_quat[i0], root_quat[i1], alpha)

    out = np.concatenate([out_root_pos, out_root_quat, out_dof_pos], axis=1).astype(np.float32)
    _normalize_quat_inplace(out)
    return out


def _npz_text(payload: dict) -> str:
    for key in ("text", "caption", "prompt"):
        if key in payload:
            value = payload[key]
            if isinstance(value, np.ndarray):
                if value.size == 0:
                    continue
                if value.ndim == 0:
                    value = value.item()
                else:
                    value = value.reshape(-1)[0]
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="ignore")
            text = str(value).strip()
            if text:
                return text
    return ""


def _base_from_stem(stem: str) -> str:
    m = META_STEM_PATTERN.match(stem)
    return m.group("base") if m else stem


def _meta_text(raw_root: Path, shard: str, base: str) -> str:
    meta_path = raw_root / shard / f"{base}_meta.json"
    if not meta_path.is_file():
        return ""
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return ""

    rewrites = meta.get("text_rewrite")
    if isinstance(rewrites, list) and rewrites:
        first = rewrites[0]
        if isinstance(first, str) and first.strip():
            return first.strip()
    text = meta.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    return ""


def _norm_joint_name(name: str) -> str:
    name = name.strip().lower()
    if name.endswith("_joint"):
        name = name[:-6]
    return name


def _align_dof_order_if_needed(qpos36: np.ndarray, npz_data: dict) -> np.ndarray:
    """Align dof_pos to ReactiveBFM canonical 29-DoF order using joint_names."""
    if "joint_names" not in npz_data:
        return qpos36

    joint_names = [str(x) for x in np.asarray(npz_data["joint_names"]).tolist()]
    if len(joint_names) == 30 and _norm_joint_name(joint_names[0]) == "root":
        src_dof_names = joint_names[1:]
    elif len(joint_names) == 29:
        src_dof_names = joint_names
    else:
        raise ValueError(f"joint_names length {len(joint_names)} is not 29/30")

    src_norm = [_norm_joint_name(n) for n in src_dof_names]
    tgt_norm = [_norm_joint_name(n) for n in EXPECTED_DOF_ORDER]
    if src_norm == tgt_norm:
        return qpos36

    src_index = {}
    for idx, name in enumerate(src_norm):
        if name in src_index:
            raise ValueError(f"duplicate joint name in source: {name}")
        src_index[name] = idx

    missing = [name for name in tgt_norm if name not in src_index]
    if missing:
        raise ValueError(f"joint_names missing expected joints: {missing}")

    perm = [src_index[name] for name in tgt_norm]
    qpos_out = qpos36.copy()
    qpos_out[:, 7:36] = qpos36[:, 7:36][:, perm]
    return qpos_out


def _iter_source_files(src_root: Path) -> Iterable[Path]:
    shard_dirs = sorted([p for p in src_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    for shard_dir in shard_dirs:
        for npz_path in sorted(shard_dir.glob("*.npz")):
            yield npz_path


def _copy_optional_field(out_dict: dict, data: dict, key: str) -> None:
    if key in data:
        out_dict[key] = data[key]


def main() -> None:
    args = _parse_args()
    src_root: Path = args.src_root
    dst_root: Path = args.dst_root
    target_fps: float = float(args.target_fps)
    default_src_fps: float = float(args.default_src_fps)
    limit: Optional[int] = args.count
    raw_root: Path = args.raw_root

    if not src_root.is_dir():
        raise FileNotFoundError(f"Source root not found: {src_root}")
    if raw_root and not raw_root.is_dir():
        print(f"[warn] raw root does not exist, text fallback disabled: {raw_root}")

    processed = 0
    skipped = 0
    text_from_npz = 0
    text_from_raw = 0
    text_missing = 0
    dst_root.mkdir(parents=True, exist_ok=True)

    for src_path in _iter_source_files(src_root):
        if limit is not None and processed >= limit:
            break

        rel = src_path.relative_to(src_root)
        dst_path = dst_root / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        if dst_path.exists() and not args.overwrite:
            print(f"[skip-exists] {rel}")
            skipped += 1
            continue

        try:
            with np.load(src_path, allow_pickle=True) as data:
                if "qpos" not in data:
                    raise KeyError("missing qpos")
                qpos = np.asarray(data["qpos"], dtype=np.float32)
                if qpos.ndim != 2 or qpos.shape[1] != 36:
                    raise ValueError(f"qpos shape {qpos.shape} is not (T, 36)")
                qpos = _align_dof_order_if_needed(qpos, data)
                src_quat_order = _infer_quat_order(qpos) if args.src_quat_order == "auto" else args.src_quat_order
                _to_xyzw_quat_inplace(qpos, src_quat_order)

                if "frequency" in data:
                    src_fps = float(data["frequency"])
                else:
                    src_fps = default_src_fps

                qpos_out = _resample_qpos36(qpos, src_fps=src_fps, dst_fps=target_fps)

                text = _npz_text(data)
                if text:
                    text_from_npz += 1
                else:
                    base = _base_from_stem(src_path.stem)
                    shard = src_path.parent.name
                    text = _meta_text(raw_root, shard, base) if raw_root and raw_root.is_dir() else ""
                    if text:
                        text_from_raw += 1
                    else:
                        text_missing += 1

                out_data = {
                    "qpos": qpos_out.astype(np.float32),
                    "frequency": np.array(target_fps, dtype=np.float64),
                    "split_points": np.array([0, qpos_out.shape[0]], dtype=np.int64),
                    "joint_names": CANONICAL_JOINT_NAMES,
                    "njnt": np.array(30, dtype=np.int64),
                    "text": np.array(text),
                    "quat_order": np.array("xyzw"),
                }
                _copy_optional_field(out_data, data, "jnt_type")

            np.savez_compressed(dst_path, **out_data)
            processed += 1
            print(f"[ok {processed}] {rel} -> {dst_path.relative_to(dst_root)}  T={qpos_out.shape[0]}")
        except Exception as exc:
            skipped += 1
            print(f"[skip-error] {rel} ({exc})")

    print(
        f"[done] processed={processed} skipped={skipped} "
        f"count={'all' if limit is None else limit} target_fps={target_fps:g}"
    )
    print(
        f"[done] text source: npz={text_from_npz} raw_meta={text_from_raw} missing={text_missing}"
    )
    print(f"[done] src={src_root}")
    print(f"[done] dst={dst_root}")


if __name__ == "__main__":
    main()
