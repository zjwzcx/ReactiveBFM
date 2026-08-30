import os
from datetime import datetime


def resolve_save_dir(save_dir: str, suffix_fmt: str = "%m%d%H%M") -> str:
    """
    If save_dir already exists, append a timestamp suffix like '_01080803'.
    Keeps appending an increment ('_2', '_3', ...) if collisions still happen.
    """
    if save_dir is None:
        return save_dir

    save_dir = str(save_dir)
    if not os.path.exists(save_dir):
        return save_dir

    suffix = datetime.now().strftime(suffix_fmt)
    candidate = f"{save_dir}_{suffix}"
    if not os.path.exists(candidate):
        return candidate

    # Extremely unlikely (same minute rerun), but make it robust.
    i = 2
    while True:
        candidate_i = f"{candidate}_{i}"
        if not os.path.exists(candidate_i):
            return candidate_i
        i += 1
