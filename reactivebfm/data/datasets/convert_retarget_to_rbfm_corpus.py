#!/usr/bin/env python3
"""Build a ReactiveBFM training dataset from converted RBFM NPZ files.

This script intentionally does NOT retarget, resample, reorder joints, or change
quaternion conventions. It consumes the physically meaningful files produced by
`convert_retarget_to_rbfm_raw_data.py` or any converter that writes the same
NPZ schema.

Expected per-file representation:
  qpos: (T, 36), float32
    [0:3]   root_pos, world-frame xyz, meters, Z-up
    [3:7]   root_quat, xyzw
    [7:36]  dof_pos, G1 29-DoF joint order
  frequency: 60.0
  text: non-empty caption
  quat_order: "xyzw"

Output dataset directory:
  <out-dir>/
    full_train.pkl
    Mean.npy / Std.npy
    train.txt / test.txt
    texts.csv

Example:
  python -m reactivebfm.data.datasets.convert_retarget_to_rbfm_corpus \
    --src-root /data/retargeted_rbfm \
    --out-dir reactivebfm/data/datasets/assets/my_g1_36dim \
    --dataset-name my_g1_36dim \
    --max-samples all \
    --overwrite
"""

from __future__ import annotations

import argparse
import csv
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


WORD_PATTERN = re.compile(r"[A-Za-z]+")
STEM_PATTERN = re.compile(r"^(?P<base>.+?)_000_smplx(?:_\d+Hz_29dof)?$")
MOTION_DIM = 36
EXPECTED_JOINT_NAMES = np.asarray(
    [
        "root",
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
    ],
    dtype="<U64",
)


@dataclass
class LoadedMotion:
    key: str
    rel_path: str
    motion: np.ndarray
    caption: str


@dataclass
class CorpusArtifacts:
    name_list: list[str]
    length_list: list[int]
    data_dict: dict[str, dict]
    mean: np.ndarray
    std: np.ndarray


def _parse_max_samples(value: str) -> int | None:
    if value.lower() == "all":
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--max-samples must be 'all' or a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--max-samples must be > 0")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a ReactiveBFM full_train.pkl dataset from converted RBFM NPZ files."
    )
    parser.add_argument(
        "--src-root",
        type=Path,
        required=True,
        help="Converted RBFM NPZ root containing shard directories or .npz files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="ReactiveBFM dataset output directory.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="Canonical dataset name used in logs and summaries.",
    )
    parser.add_argument(
        "--max-samples",
        type=_parse_max_samples,
        default=None,
        help="How many files to pack, in shard/file order. Use 'all' for full corpus. Default: all",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.01,
        help="Held-out split ratio in [0,1]. Default: 0.01",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/test split.")
    parser.add_argument("--min-std", type=float, default=1e-6, help="Std clamp for normalization.")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="spacy",
        choices=("spacy", "simple"),
        help="Caption tokenization for ReactiveBFM text conditioning. Default: spacy",
    )
    parser.add_argument(
        "--spacy-batch-size",
        type=int,
        default=256,
        help="Batch size for spaCy nlp.pipe when --tokenizer spacy. Default: 256",
    )
    parser.add_argument(
        "--expected-fps",
        type=float,
        default=60.0,
        help="Required source frequency in retargeted_rbfm files. Default: 60",
    )
    parser.add_argument(
        "--allow-missing-text",
        action="store_true",
        help="Keep samples with empty text by writing empty caption/tokens instead of skipping.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing dataset files in --out-dir.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    return parser.parse_args()


def _progress(iterable, *, total: int | None, desc: str, enabled: bool):
    if not enabled:
        return iterable
    try:
        from tqdm import tqdm
    except ImportError:
        print(f"[progress] tqdm is not installed; continuing without progress bar for {desc}.")
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit="sample")


def collect_npz_files(src_root: Path, max_samples: int | None) -> list[Path]:
    paths: list[Path] = []
    for npz_path in sorted(src_root.glob("*.npz")):
        paths.append(npz_path)
        if max_samples is not None and len(paths) >= max_samples:
            return paths
    shard_dirs = sorted((p for p in src_root.iterdir() if p.is_dir()), key=lambda p: p.name)
    for shard_dir in shard_dirs:
        for npz_path in sorted(shard_dir.glob("*.npz")):
            paths.append(npz_path)
            if max_samples is not None and len(paths) >= max_samples:
                return paths
    return paths


def _base_from_stem(stem: str) -> str:
    match = STEM_PATTERN.match(stem)
    return match.group("base") if match else stem


def _unique_key(base: str, counter: dict[str, int]) -> str:
    if base not in counter:
        counter[base] = 0
        return base
    counter[base] += 1
    return f"{base}__dup{counter[base]:03d}"


def _scalar_to_str(value) -> str:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        if value.ndim == 0:
            value = value.item()
        else:
            value = value.reshape(-1)[0]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip()
    return str(value).strip()


def _simple_tokens(caption: str) -> list[str]:
    return [f"{word}/OTHER" for word in WORD_PATTERN.findall(caption.lower())]


def _spacy_doc_tokens(doc) -> list[str]:
    words: list[str] = []
    pos: list[str] = []
    for token in doc:
        word = token.text
        if not word.isalpha():
            continue
        if (token.pos_ == "NOUN" or token.pos_ == "VERB") and word != "left":
            words.append(token.lemma_)
        else:
            words.append(word)
        pos.append(token.pos_)
    return [f"{word}/{tag}" for word, tag in zip(words, pos)]


def _load_tokenizer(name: str):
    if name == "simple":
        return None
    try:
        import spacy
    except ImportError as exc:
        raise ImportError(
            "spaCy is required for --tokenizer spacy. Install it or rerun with --tokenizer simple."
        ) from exc
    try:
        return spacy.load("en_core_web_sm")
    except OSError as exc:
        raise OSError(
            "spaCy model en_core_web_sm is missing. Run `python -m spacy download en_core_web_sm` "
            "or rerun with --tokenizer simple."
        ) from exc


def tokenize_captions(
    captions: list[str],
    *,
    tokenizer_name: str,
    nlp,
    batch_size: int,
    show_progress: bool,
) -> list[list[str]]:
    if tokenizer_name == "simple":
        return [_simple_tokens(caption) for caption in captions]

    prepared = [caption.replace("-", "") for caption in captions]
    token_lists: list[list[str]] = []
    caption_iter = _progress(
        prepared,
        total=len(prepared),
        desc="Tokenizing captions",
        enabled=show_progress,
    )
    for doc in nlp.pipe(caption_iter, batch_size=batch_size):
        token_lists.append(_spacy_doc_tokens(doc))
    return token_lists


def _check_joint_names(payload, rel_path: str) -> None:
    if "joint_names" not in payload:
        raise ValueError(f"{rel_path}: missing joint_names")
    names = np.asarray(payload["joint_names"], dtype="<U64")
    if names.shape != EXPECTED_JOINT_NAMES.shape or not np.array_equal(names, EXPECTED_JOINT_NAMES):
        raise ValueError(f"{rel_path}: joint_names are not canonical G1 29-DoF order")


def _check_quat_xyzw(qpos: np.ndarray, payload, rel_path: str) -> None:
    if "quat_order" in payload:
        order = _scalar_to_str(payload["quat_order"]).lower()
        if order != "xyzw":
            raise ValueError(f"{rel_path}: quat_order={order!r}, expected 'xyzw'")

    quat = qpos[:, 3:7]
    norm = np.linalg.norm(quat, axis=-1)
    if np.abs(norm - 1.0).max() > 1e-3:
        raise ValueError(
            f"{rel_path}: root quaternion is not unit after raw-data conversion "
            f"(min={norm.min():.6f}, max={norm.max():.6f})"
        )


def load_motion_sample(
    npz_path: Path,
    src_root: Path,
    *,
    expected_fps: float,
    allow_missing_text: bool,
    key_counter: dict[str, int],
) -> LoadedMotion:
    rel_path = str(npz_path.relative_to(src_root))
    with np.load(npz_path, allow_pickle=True) as payload:
        if "qpos" not in payload:
            raise ValueError(f"{rel_path}: missing qpos")
        qpos = np.asarray(payload["qpos"], dtype=np.float32)
        if qpos.ndim != 2 or qpos.shape[1] != MOTION_DIM:
            raise ValueError(f"{rel_path}: expected qpos shape (T,36), got {qpos.shape}")

        fps = float(payload["frequency"]) if "frequency" in payload else 0.0
        if abs(fps - expected_fps) > 1e-6:
            raise ValueError(f"{rel_path}: frequency={fps}, expected {expected_fps}")

        _check_joint_names(payload, rel_path)
        _check_quat_xyzw(qpos, payload, rel_path)

        caption = _scalar_to_str(payload["text"]) if "text" in payload else ""
        if not caption and not allow_missing_text:
            raise ValueError(f"{rel_path}: missing/empty text")

    base = _base_from_stem(npz_path.stem)
    key = _unique_key(base, key_counter)
    return LoadedMotion(key=key, rel_path=rel_path, motion=qpos, caption=caption)


def load_motion_corpus(
    npz_paths: list[Path],
    src_root: Path,
    *,
    expected_fps: float,
    allow_missing_text: bool,
    show_progress: bool,
) -> tuple[list[LoadedMotion], int]:
    loaded: list[LoadedMotion] = []
    key_counter: dict[str, int] = {}
    skipped = 0

    for npz_path in _progress(npz_paths, total=len(npz_paths), desc="Loading RBFM NPZ", enabled=show_progress):
        try:
            loaded.append(
                load_motion_sample(
                    npz_path,
                    src_root,
                    expected_fps=expected_fps,
                    allow_missing_text=allow_missing_text,
                    key_counter=key_counter,
                )
            )
        except Exception as exc:
            skipped += 1
            print(f"[skip] {npz_path.relative_to(src_root)}: {exc}")
            continue
        if not show_progress and (len(loaded) == 1 or len(loaded) % 1000 == 0):
            print(f"[load] {len(loaded)} samples, latest={loaded[-1].rel_path}")

    return loaded, skipped


def build_corpus_artifacts(
    motions: list[LoadedMotion],
    tokens_list: list[list[str]],
    *,
    min_std: float,
) -> CorpusArtifacts:
    if len(motions) != len(tokens_list):
        raise ValueError("motion/token count mismatch")

    name_list: list[str] = []
    length_list: list[int] = []
    data_dict: dict[str, dict] = {}
    total_sum = np.zeros(MOTION_DIM, dtype=np.float64)
    total_sum_sq = np.zeros(MOTION_DIM, dtype=np.float64)
    n_frames = 0

    for motion, tokens in zip(motions, tokens_list):
        motion64 = motion.motion.astype(np.float64, copy=False)
        n_frames += motion64.shape[0]
        total_sum += motion64.sum(axis=0)
        total_sum_sq += np.square(motion64).sum(axis=0)

        name_list.append(motion.key)
        length_list.append(int(motion64.shape[0]))
        data_dict[motion.key] = {
            "motion": motion.motion,
            "length": int(motion64.shape[0]),
            "text": [{"caption": motion.caption, "tokens": tokens}],
        }

    if n_frames == 0:
        raise RuntimeError("No frames collected for normalization stats.")

    mean = (total_sum / n_frames).astype(np.float32)
    variance = np.maximum(total_sum_sq / n_frames - np.square(total_sum / n_frames), 0.0)
    std = np.maximum(np.sqrt(variance), min_std).astype(np.float32)
    return CorpusArtifacts(
        name_list=name_list,
        length_list=length_list,
        data_dict=data_dict,
        mean=mean,
        std=std,
    )


def write_split_files(out_dir: Path, keys: list[str], test_ratio: float, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not 0.0 <= test_ratio <= 1.0:
        raise ValueError(f"test_ratio should be in [0,1], got {test_ratio}")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(keys))
    rng.shuffle(indices)

    n_test = int(round(len(keys) * test_ratio))
    n_test = min(max(n_test, 1 if len(keys) > 1 and test_ratio > 0.0 else 0), max(len(keys) - 1, 0))
    test_indices = set(indices[:n_test].tolist())

    train_keys = [k for i, k in enumerate(keys) if i not in test_indices]
    test_keys = [k for i, k in enumerate(keys) if i in test_indices]

    (out_dir / "train.txt").write_text("\n".join(train_keys) + ("\n" if train_keys else ""), encoding="utf-8")
    (out_dir / "test.txt").write_text("\n".join(test_keys) + ("\n" if test_keys else ""), encoding="utf-8")


def write_dataset_outputs(
    out_dir: Path,
    motions: list[LoadedMotion],
    corpus: CorpusArtifacts,
    *,
    test_ratio: float,
    seed: int,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    full_train_path = out_dir / "full_train.pkl"
    print(f"[write] {full_train_path}")
    with full_train_path.open("wb") as f:
        pickle.dump(
            {
                "name_list": corpus.name_list,
                "length_list": corpus.length_list,
                "data_dict": corpus.data_dict,
            },
            f,
            protocol=4,
        )

    print(f"[write] normalization stats -> {out_dir}")
    np.save(out_dir / "Mean.npy", corpus.mean)
    np.save(out_dir / "Std.npy", corpus.std)

    write_split_files(out_dir, corpus.name_list, test_ratio, seed)

    with (out_dir / "texts.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["motion_name", "total_frames", "caption", "source_file"])
        for motion in motions:
            writer.writerow([motion.key, motion.motion.shape[0], motion.caption, motion.rel_path])

    return full_train_path


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    if not args.src_root.is_dir():
        raise FileNotFoundError(f"source root not found: {args.src_root}")
    if out_dir.exists() and not out_dir.is_dir():
        raise NotADirectoryError(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    full_train_path = out_dir / "full_train.pkl"
    if full_train_path.exists() and not args.overwrite:
        raise FileExistsError(f"{full_train_path} exists; pass --overwrite to rebuild")

    show_progress = not args.no_progress
    npz_paths = collect_npz_files(args.src_root, args.max_samples)
    if not npz_paths:
        raise RuntimeError(f"No NPZ files found under {args.src_root}")

    motions, skipped = load_motion_corpus(
        npz_paths,
        args.src_root,
        expected_fps=args.expected_fps,
        allow_missing_text=args.allow_missing_text,
        show_progress=show_progress,
    )
    if not motions:
        raise RuntimeError("No valid retargeted_rbfm samples were loaded.")

    nlp = _load_tokenizer(args.tokenizer)
    tokens_list = tokenize_captions(
        [motion.caption for motion in motions],
        tokenizer_name=args.tokenizer,
        nlp=nlp,
        batch_size=args.spacy_batch_size,
        show_progress=show_progress,
    )
    corpus = build_corpus_artifacts(motions, tokens_list, min_std=args.min_std)
    full_train_path = write_dataset_outputs(
        out_dir,
        motions,
        corpus,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    length_list = corpus.length_list
    print(f"[done] loaded={len(motions)} skipped={skipped}")
    print(f"[done] dataset={args.dataset_name}")
    print(f"[done] output={out_dir}")
    print(f"[done] full_train={full_train_path}")
    print(
        f"[done] length min/mean/max = "
        f"{min(length_list)}/{sum(length_list) / len(length_list):.1f}/{max(length_list)}"
    )


if __name__ == "__main__":
    main()
