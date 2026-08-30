"""Active planner dataset names and their filesystem locations."""

import os
from pathlib import Path
from typing import List, Optional


DEFAULT_DATASET_ROOT = Path(__file__).resolve().parent / "assets"
DATASET_ROOT_ENV = "REACTIVEBFM_DATASET_ROOT"
BONES_SEED_PATH_ENV = "REACTIVEBFM_BONES_SEED_PATH"
DEFAULT_BONES_SEED_PATH = Path(
    "/mnt/oss/egoscale/humanoidvla_data/bones_seed_g1_36dim"
)
DEFAULT_BONES_SEED_V2_PATH = Path(
    "/mnt/oss/egoscale/humanoidvla_data/bones_seed_v2_g1_36dim"
)
DEFAULT_BONES_SEED_V3_PATH = Path(
    "/mnt/oss/egoscale/humanoidvla_data/bones_seed_v3_g1_36dim"
)


def get_dataset_root() -> str:
    return os.environ.get(DATASET_ROOT_ENV, str(DEFAULT_DATASET_ROOT))


def dataset_path(*parts: str) -> str:
    return str(Path(get_dataset_root(), *parts))


class DatasetRegistry:
    """Canonical names for datasets accepted by planner training."""

    AMASS = "amass_g1_36dim"
    MOTIONMILLION_100STYLE = "100style_g1_36dim"
    MOTIONMILLION_100STYLE_EVAL = "100style_g1_eval_36dim"
    MOTIONMILLION_100STYLE_EXCL_EVAL = "100style_g1_36dim_excl_eval"
    MOTIONMILLION_KUNGFU = "kungfu_g1_36dim"
    MOTIONMILLION_KUNGFU_1P5 = "kungfu_g1_1p5_36dim"
    MOTIONMILLION_KUNGFU_2X = "kungfu_g1_2x_36dim"
    MIXED_AMASS_100STYLE_KUNGFU = "amass_100style_kungfu_g1_36dim"
    HYMOTION = "hymotion_g1_36dim"
    HYMOTION_1K = "hymotion_1k_g1_36dim"
    HYMOTION_V1 = "hymotion_v1_g1_36dim"
    HYMOTION_V2 = "hymotion_v2_g1_36dim"
    HYMOTION_V3 = "hymotion_v3_g1_36dim"
    BONES_SEED = "bones_seed_g1_36dim"
    BONES_SEED_V2 = "bones_seed_v2_g1_36dim"
    BONES_SEED_V3 = "bones_seed_v3_g1_36dim"
    COMPOSITE_SEPARATOR = ","

    @classmethod
    def get_text_motion_datasets(cls) -> List[str]:
        return [
            cls.AMASS,
            cls.MOTIONMILLION_100STYLE,
            cls.MOTIONMILLION_100STYLE_EVAL,
            cls.MOTIONMILLION_100STYLE_EXCL_EVAL,
            cls.MOTIONMILLION_KUNGFU,
            cls.MOTIONMILLION_KUNGFU_1P5,
            cls.MOTIONMILLION_KUNGFU_2X,
            cls.MIXED_AMASS_100STYLE_KUNGFU,
            cls.HYMOTION,
            cls.HYMOTION_1K,
            cls.HYMOTION_V1,
            cls.HYMOTION_V2,
            cls.HYMOTION_V3,
            cls.BONES_SEED,
            cls.BONES_SEED_V2,
            cls.BONES_SEED_V3,
        ]

    @classmethod
    def get_all_datasets(cls) -> List[str]:
        return cls.get_text_motion_datasets()

    @classmethod
    def parse_dataset_names(cls, name: str) -> List[str]:
        """Parse a comma-separated runtime dataset composition."""
        if not isinstance(name, str):
            raise TypeError(f"Dataset specification must be a string, got {type(name)!r}.")
        names = [part.strip() for part in name.split(cls.COMPOSITE_SEPARATOR)]
        if not names or any(not part for part in names):
            raise ValueError(f"Invalid dataset specification: {name!r}")
        if len(names) != len(set(names)):
            raise ValueError(
                f"Dataset specification contains duplicates: {name!r}. "
                "Explicit sampling weights are not supported yet."
            )
        return names

    @classmethod
    def supports_default_runtime_eval(cls, name: str) -> bool:
        eval_datasets = {cls.HYMOTION_V3, cls.BONES_SEED_V2, cls.BONES_SEED_V3}
        try:
            names = cls.parse_dataset_names(name)
        except (TypeError, ValueError):
            return False
        return bool(names) and all(item in eval_datasets for item in names)

    @classmethod
    def is_supported_dataset(cls, name: str) -> bool:
        try:
            names = cls.parse_dataset_names(name)
        except (TypeError, ValueError):
            return False
        supported = set(cls.get_all_datasets())
        return all(item in supported for item in names)

    @classmethod
    def is_text_conditioned(cls, name: str) -> bool:
        """Whether samples contain captions for planner conditioning."""
        if not cls.is_supported_dataset(name):
            return False
        text_datasets = set(cls.get_text_motion_datasets())
        return all(item in text_datasets for item in cls.parse_dataset_names(name))

    @classmethod
    def get_data_path(cls, name: str) -> Optional[str]:
        if not cls.is_supported_dataset(name):
            return None
        names = cls.parse_dataset_names(name)
        if len(names) != 1:
            return None
        name = names[0]
        bones_datasets = (cls.BONES_SEED, cls.BONES_SEED_V2, cls.BONES_SEED_V3)
        if name in bones_datasets:
            explicit_path = os.environ.get(BONES_SEED_PATH_ENV)
            if explicit_path:
                explicit_path = Path(explicit_path).expanduser()
                if name != cls.BONES_SEED and explicit_path.name in bones_datasets:
                    explicit_path = explicit_path.with_name(name)
                return str(explicit_path)
            # Preserve the common dataset-root override when the caller set it.
            # On the internal training cluster, make the converted corpus work
            # out of the box without requiring a shell-local export.
            default_paths = {
                cls.BONES_SEED: DEFAULT_BONES_SEED_PATH,
                cls.BONES_SEED_V2: DEFAULT_BONES_SEED_V2_PATH,
                cls.BONES_SEED_V3: DEFAULT_BONES_SEED_V3_PATH,
            }
            default_path = default_paths[name]
            if DATASET_ROOT_ENV not in os.environ and default_path.is_dir():
                return str(default_path)
        return dataset_path(name)
