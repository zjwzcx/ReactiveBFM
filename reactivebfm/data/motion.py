"""Canonical motion metadata and kinematic topology for data processing.

Motion transforms and feature/recovery operations live in
``reactivebfm.utils.motion``; this module owns the data representation
constants used by those utilities and the planner data pipeline.
"""

import importlib
import sys
import types

import numpy as np


HML_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]
NUM_HML_JOINTS = len(HML_JOINT_NAMES)
HML_EE_JOINT_NAMES = ["left_foot", "right_foot", "left_wrist", "right_wrist", "head"]
HML_LOWER_BODY_JOINTS = [
    HML_JOINT_NAMES.index(name)
    for name in [
        "pelvis",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "left_foot",
        "right_foot",
    ]
]
HML_ROOT_BINARY = np.array([True] + [False] * (NUM_HML_JOINTS - 1))
HML_ROOT_MASK = np.concatenate((
    [True] * (1 + 2 + 1),
    HML_ROOT_BINARY[1:].repeat(3),
    HML_ROOT_BINARY[1:].repeat(6),
    HML_ROOT_BINARY.repeat(3),
    [False] * 4,
))
HML_ROOT_HORIZONTAL_MASK = np.concatenate((
    [True] * (1 + 2) + [False],
    np.zeros_like(HML_ROOT_BINARY[1:].repeat(3)),
    np.zeros_like(HML_ROOT_BINARY[1:].repeat(6)),
    np.zeros_like(HML_ROOT_BINARY.repeat(3)),
    [False] * 4,
))
HML_LOWER_BODY_JOINTS_BINARY = np.array([
    idx in HML_LOWER_BODY_JOINTS for idx in range(NUM_HML_JOINTS)
])
HML_LOWER_BODY_MASK = np.concatenate((
    [True] * (1 + 2 + 1),
    HML_LOWER_BODY_JOINTS_BINARY[1:].repeat(3),
    HML_LOWER_BODY_JOINTS_BINARY[1:].repeat(6),
    HML_LOWER_BODY_JOINTS_BINARY.repeat(3),
    [True] * 4,
))
HML_UPPER_BODY_MASK = ~HML_LOWER_BODY_MASK

t2m_raw_offsets = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [0, 0, 1],
        [0, -1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, -1, 0],
    ]
)
t2m_kinematic_chain = [
    [0, 2, 5, 8, 11],
    [0, 1, 4, 7, 10],
    [0, 3, 6, 9, 12, 15],
    [9, 14, 17, 19, 21],
    [9, 13, 16, 18, 20],
]


class _LazyModuleAlias(types.ModuleType):
    """Load a compatibility target only when one of its attributes is used."""

    def __init__(self, name, target):
        super().__init__(name)
        self._target = target
        self.__file__ = None
        self.__package__ = name.rpartition(".")[0]

    def __getattr__(self, name):
        return getattr(importlib.import_module(self._target), name)


_current_module = sys.modules[__name__]
sys.modules[f"{__name__}.constants"] = _current_module
sys.modules[f"{__name__}.param_util"] = _current_module
for _legacy_name, _target in {
    "processing": "reactivebfm.utils.motion.recovery",
    "quaternion": "reactivebfm.utils.motion.quaternion",
    "skeleton": "reactivebfm.utils.motion.skeleton",
    "torch_processing": "reactivebfm.utils.motion.features",
}.items():
    sys.modules[f"{__name__}.{_legacy_name}"] = _LazyModuleAlias(
        f"{__name__}.{_legacy_name}", _target
    )

_FUNCTION_EXPORTS = {
    "extract_features_t2m": "reactivebfm.utils.motion.features",
    "recover_from_ric": "reactivebfm.utils.motion.recovery",
    "recover_root_rot_pos": "reactivebfm.utils.motion.recovery",
}


def __getattr__(name):
    target = _FUNCTION_EXPORTS.get(name)
    if target is not None:
        return getattr(importlib.import_module(target), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "HML_EE_JOINT_NAMES",
    "HML_JOINT_NAMES",
    "HML_LOWER_BODY_JOINTS",
    "HML_LOWER_BODY_MASK",
    "HML_ROOT_HORIZONTAL_MASK",
    "HML_ROOT_MASK",
    "HML_UPPER_BODY_MASK",
    "NUM_HML_JOINTS",
    "extract_features_t2m",
    "recover_from_ric",
    "recover_root_rot_pos",
    "t2m_kinematic_chain",
    "t2m_raw_offsets",
]
