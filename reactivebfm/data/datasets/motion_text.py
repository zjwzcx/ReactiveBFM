"""ReactiveBFM motion-text datasets."""

from argparse import Namespace
import codecs as cs
import os
from os.path import join as pjoin
import pickle
import random

import numpy as np
from torch.utils import data

from reactivebfm.data.datasets.registry import DatasetRegistry

def normalize_motion(motion, mean, std, relative_root_xy=False):
    normalized = (motion - mean) / std
    if relative_root_xy:
        normalized[..., :2] -= normalized[..., :1, :2]
    return normalized


def denormalize_motion(motion, mean, std, relative_root_xy=False):
    denormalized = motion * std + mean
    if relative_root_xy:
        denormalized[..., :2] = motion[..., :2] * std[:2]
    return denormalized


def _build_dataset_options(
    dataset_name,
    data_root,
    unit_length=4,
):
    """Build the planner dataset options without a legacy config file."""
    if not DatasetRegistry.is_supported_dataset(dataset_name):
        raise KeyError(f"Unsupported dataset: {dataset_name}")

    opt = Namespace()
    opt.unit_length = int(unit_length)

    opt.data_root = data_root
    opt.max_motion_length = 196
    return opt


def _motion_base_key(key: str) -> str:
    """Strip eval suffix `{name}__ep{idx}` to the base motion id."""
    if "__ep" in key:
        return key.rsplit("__ep", 1)[0]
    return key


def _filter_name_list(name_list, length_list, data_dict, split_file=None, exclude_keys=None):
    """Filter pkl keys by split file and/or excluded motion ids."""
    exclude = set(exclude_keys or [])
    allowed = None
    if split_file is not None and os.path.isfile(split_file):
        with cs.open(split_file, "r") as f:
            allowed = {line.strip() for line in f.readlines() if line.strip()}
        if not allowed:
            allowed = None

    new_names, new_lengths = [], []
    for name, length in zip(name_list, length_list):
        base = _motion_base_key(name)
        if exclude and (name in exclude or base in exclude):
            continue
        if allowed is not None and name not in allowed and base not in allowed:
            continue
        if name not in data_dict:
            continue
        new_names.append(name)
        new_lengths.append(length)

    return new_names, new_lengths


class PlannerMotionDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file, exclude_keys=None, cache=None):
        self.opt = opt
        self.min_motion_length = 20
        if getattr(self.opt, "minimum_length", 0) > 0:
            self.min_motion_length = self.opt.minimum_length
        elif self.opt.fixed_len > 0:
            self.min_motion_length = self.opt.fixed_len
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length

        data_path = os.path.join(opt.data_root, "full_train.pkl")
        if cache is not None:
            print(f"[Data] Reusing motions loaded from: {data_path}...")
        elif os.path.exists(data_path):
            print(f"[Data] Loading motions from: {data_path}...")
            with open(data_path, "rb") as fp:
                cache = pickle.load(fp)
        else:
            raise FileNotFoundError(f"Cannot find motion data in {data_path}")

        name_list = cache["name_list"]
        length_list = cache["length_list"]
        data_dict = cache["data_dict"]

        self.cache = cache
        self.mean = mean
        self.std = std
        self.data_dict = data_dict
        name_list, length_list = _filter_name_list(
            name_list, length_list, data_dict, split_file, exclude_keys
        )

        kept_names, kept_lengths = [], []
        n_drop = 0
        for name, length in zip(name_list, length_list):
            text_entries = (
                data_dict[name].get("text", [])
                if isinstance(data_dict[name], dict)
                else []
            )
            if not text_entries:
                n_drop += 1
                continue
            kept_names.append(name)
            kept_lengths.append(length)
        if n_drop:
            print(
                f"[PlannerMotionDataset] dropped {n_drop} motions with "
                "empty/missing text"
            )
        name_list, length_list = kept_names, kept_lengths

        if len(name_list) == 0:
            raise RuntimeError(
                f"No motions left after split/exclude (split_file={split_file}, "
                f"exclude={len(exclude_keys or [])})"
            )
        print(
            f"[PlannerMotionDataset] Using {len(name_list)} motions "
            f"(split={os.path.basename(split_file) if split_file else 'all'}, "
            f"excluded={len(exclude_keys or [])})"
        )
        self.length_arr = np.array(length_list)
        self.name_list = name_list
        self.reset_min_motion_length(self.min_motion_length)

        self.prompt_to_key = {}
        for key in self.name_list:
            item = self.data_dict[key]
            for text_item in item["text"]:
                if isinstance(text_item, dict) and "caption" in text_item:
                    caption = text_item["caption"]
                    self.prompt_to_key.setdefault(caption, []).append(key)

    def reset_min_motion_length(self, length):
        assert length <= self.max_motion_length
        if not np.all(self.length_arr[:-1] <= self.length_arr[1:]):
            sorted_indices = np.argsort(self.length_arr)
            self.length_arr = self.length_arr[sorted_indices]
            self.name_list = [self.name_list[i] for i in sorted_indices]
        self.pointer = np.searchsorted(self.length_arr, length)
        remaining_samples = len(self.name_list) - self.pointer
        print(
            f"Pointer Pointing at {self.pointer} (filtered {self.pointer} samples "
            f"with length < {length}, remaining {remaining_samples} samples out of "
            f"{len(self.name_list)} in split)"
        )
        self.min_motion_length = length

    def inv_transform(self, data):
        return denormalize_motion(
            data,
            self.mean,
            self.std,
            relative_root_xy=self.opt.relative_root_xy,
        )

    def __len__(self):
        return len(self.name_list) - self.pointer

    def remove_motion(self, key):
        idx = self.name_list.index(key)
        assert self.name_list[idx] == key
        if "text" in self.data_dict[key] and self.data_dict[key]["text"]:
            for text_item in self.data_dict[key]["text"]:
                caption = text_item.get("caption") if isinstance(text_item, dict) else None
                if caption in self.prompt_to_key:
                    if len(self.prompt_to_key[caption]) == 1:
                        del self.prompt_to_key[caption]
                    else:
                        key_idx = self.prompt_to_key[caption].index(key)
                        self.prompt_to_key[caption] = (
                            self.prompt_to_key[caption][:key_idx]
                            + self.prompt_to_key[caption][key_idx + 1:]
                        )
        del self.data_dict[key]
        self.name_list = self.name_list[:idx] + self.name_list[idx + 1:]
        self.length_arr = np.concatenate([self.length_arr[:idx], self.length_arr[idx + 1:]])

    def remove_by_text(self, text):
        n_removed = 0
        for item in text:
            if item in self.prompt_to_key:
                key = self.prompt_to_key[item][0]
                self.remove_motion(key)
                n_removed += 1
        return n_removed

    def __getitem__(self, item):
        idx = self.pointer + item
        sample_index = int(idx)
        key = self.name_list[idx]
        data_item = self.data_dict[key]
        motion, m_length = data_item["motion"], data_item["length"]

        if self.opt.deterministic:
            text_data = data_item["text"][0]
        else:
            text_data = random.choice(data_item["text"])
        caption = text_data["caption"]
        tokens = list(text_data["tokens"])

        if self.opt.fixed_len > 0:
            m_length = self.opt.fixed_len
        else:
            if self.opt.unit_length < 10:
                coin2 = np.random.choice(["single", "single", "double"])
            else:
                coin2 = "single"

            if coin2 == "double":
                m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
            elif coin2 == "single":
                m_length = (m_length // self.opt.unit_length) * self.opt.unit_length

        if m_length > self.max_motion_length:
            m_length = (self.max_motion_length // self.opt.unit_length) * self.opt.unit_length
            if m_length == 0:
                m_length = self.max_motion_length

        if m_length > len(motion):
            m_length = (len(motion) // self.opt.unit_length) * self.opt.unit_length
            if m_length == 0:
                m_length = len(motion)

        if len(motion) < m_length:
            m_length = len(motion)

        if self.opt.deterministic:
            idx = max(0, (len(motion) - m_length) // 2)
        else:
            idx = random.randint(0, max(0, len(motion) - m_length))
        if self.opt.disable_offset_aug and not self.opt.deterministic:
            idx = random.randint(0, min(self.opt.unit_length, max(0, len(motion) - m_length)))

        motion = motion[idx:idx + m_length]
        motion = normalize_motion(
            motion,
            self.mean,
            self.std,
            relative_root_xy=self.opt.relative_root_xy,
        )

        if m_length < self.max_motion_length:
            motion = np.concatenate(
                [motion, np.zeros((self.max_motion_length - m_length, motion.shape[1]))],
                axis=0,
            )

        tokens_str = "_".join(tokens) if tokens else ""
        motion = np.asarray(motion, dtype=np.float32)
        sample = {
            "motion": motion,
            "length": m_length,
            "text": caption,
            "tokens": tokens_str,
        }
        if self.opt.return_keys:
            returned_key = (
                f"{self.opt.key_namespace}:{key}"
                if self.opt.key_namespace
                else key
            )
            sample.update({"key": returned_key, "sample_index": sample_index})
        return sample


class MotionDataset(data.Dataset):
    """Sliding-window motion snippets for pkl data."""

    def __init__(self, opt, mean, std, split_file, exclude_keys=None):
        self.opt = opt
        data_path = os.path.join(opt.data_root, "full_train.pkl")
        if not os.path.exists(data_path):
            raise FileNotFoundError(data_path)
        print(f"[MotionDataset] Loading {data_path} ...")
        with open(data_path, "rb") as fp:
            cache = pickle.load(fp)
        name_list = cache["name_list"]
        length_list = cache["length_list"]
        data_dict = cache["data_dict"]
        name_list, _ = _filter_name_list(
            name_list, length_list, data_dict, split_file, exclude_keys
        )
        self.mean = mean
        self.std = std
        self.data = []
        self.lengths = []
        for name in name_list:
            motion = np.asarray(data_dict[name]["motion"], dtype=np.float32)
            if motion.shape[0] < opt.window_size:
                continue
            self.lengths.append(motion.shape[0] - opt.window_size)
            self.data.append(motion)
        self.cumsum = np.cumsum([0] + self.lengths)
        print(
            f"[MotionDataset] {len(self.data)} motions, {self.cumsum[-1]} snippets "
            f"(window={opt.window_size})"
        )
        if self.cumsum[-1] == 0:
            raise RuntimeError("MotionDataset: no snippets (increase data or lower window_size)")

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, item):
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0
        motion = self.data[motion_id][idx:idx + self.opt.window_size]
        motion = (motion - self.mean) / self.std
        return motion


class MotionTextDataset(data.Dataset):
    def __init__(
        self,
        dataset_name,
        unit_length,
        data_path,
        device,
        split="train",
        fixed_len=0,
        return_keys=False,
        relative_root_xy=False,
        abs_path="diffusion_planner",
        hml_type=None,
        deterministic=False,
        cache=None,
        minimum_length=0,
        key_namespace=None,
    ):
        data_root = pjoin(abs_path, data_path)

        opt = _build_dataset_options(
            dataset_name,
            data_root,
            unit_length=unit_length,
        )
        opt.fixed_len = fixed_len
        if opt.fixed_len > 0:
            opt.max_motion_length = opt.fixed_len
        opt.disable_offset_aug = False
        opt.return_keys = return_keys
        opt.relative_root_xy = bool(relative_root_xy)
        opt.deterministic = bool(deterministic)
        opt.minimum_length = int(minimum_length)
        opt.key_namespace = key_namespace
        self.opt = opt
        print(f"Loading planner dataset {dataset_name} ...")

        name = "" if hml_type is None else f"_{hml_type}"
        mean_path = pjoin(opt.data_root, f"Mean{name}.npy")
        std_path = pjoin(opt.data_root, f"Std{name}.npy")
        missing_stats = [path for path in (mean_path, std_path) if not os.path.isfile(path)]
        if missing_stats:
            raise FileNotFoundError(
                f"Dataset {dataset_name!r} resolved to {opt.data_root!r}, but required "
                f"normalization files are missing: {missing_stats}. Set --data_dir, "
                "REACTIVEBFM_DATASET_ROOT, or the dataset-specific path override."
            )
        self.mean = np.load(mean_path)
        self.std = np.load(std_path)

        self.split_file = pjoin(opt.data_root, f"{split}.txt")
        self.samples = PlannerMotionDataset(
            self.opt, self.mean, self.std, self.split_file, cache=cache
        )
        self.cache = self.samples.cache
        self.num_actions = 1

        assert len(self.samples) >= 1, "Loaded an empty G1 dataset."

    def remove_by_text(self, text):
        return self.samples.remove_by_text(text)

    def __getitem__(self, item):
        return self.samples.__getitem__(item)

    def __len__(self):
        return self.samples.__len__()


__all__ = [
    "MotionDataset",
    "MotionTextDataset",
    "denormalize_motion",
    "normalize_motion",
]
