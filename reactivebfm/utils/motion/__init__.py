"""Motion recovery, representation, kinematics, and rotation utilities."""

import importlib


_EXPORTS = {
    "extract_features_t2m": "reactivebfm.utils.motion.features",
    "recover_from_ric": "reactivebfm.utils.motion.recovery",
    "recover_root_rot_pos": "reactivebfm.utils.motion.recovery",
}


def __getattr__(name):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(target), name)

__all__ = ["extract_features_t2m", "recover_from_ric", "recover_root_rot_pos"]
