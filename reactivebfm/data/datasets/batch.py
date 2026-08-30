"""Batch collation for ReactiveBFM motion datasets."""

import random as _rng

import torch


def lengths_to_mask(lengths, max_len):
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len)
    return mask < lengths.unsqueeze(1)


def ensure_float32(motion, cond):
    motion = motion.float()
    if isinstance(cond, dict) and "y" in cond and isinstance(cond["y"], dict):
        cond["y"] = {
            k: (v.float() if (torch.is_tensor(v) and v.is_floating_point()) else v)
            for k, v in cond["y"].items()
        }
    return motion, cond


def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, tensor in enumerate(batch):
        sub_tensor = canvas[i]
        for dim in range(dims):
            sub_tensor = sub_tensor.narrow(dim, 0, tensor.size(dim))
        sub_tensor.add_(tensor)
    return canvas


def _base_collate(batch, with_smooth=False):
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b["inp"] for b in notnone_batches]
    if "lengths" in notnone_batches[0]:
        lenbatch = [b["lengths"] for b in notnone_batches]
    else:
        lenbatch = [len(b["inp"][0][0]) for b in notnone_batches]

    databatch_tensor = collate_tensors(databatch)
    lenbatch_tensor = torch.as_tensor(lenbatch)
    maskbatch_tensor = lengths_to_mask(lenbatch_tensor, databatch_tensor.shape[-1])
    maskbatch_tensor = maskbatch_tensor.unsqueeze(1).unsqueeze(1)

    motion = databatch_tensor
    cond = {"y": {"mask": maskbatch_tensor, "lengths": lenbatch_tensor}}

    if with_smooth and motion.shape[-1] > 1:
        velocity_gt = motion[:, :, :, 1:] - motion[:, :, :, :-1]
        velocity_mask = maskbatch_tensor[:, :, :, 1:] & maskbatch_tensor[:, :, :, :-1]
        cond["y"].update({"velocity_gt": velocity_gt, "velocity_mask": velocity_mask})

        if motion.shape[-1] > 2:
            acceleration_gt = velocity_gt[:, :, :, 1:] - velocity_gt[:, :, :, :-1]
            acceleration_mask = velocity_mask[:, :, :, 1:] & velocity_mask[:, :, :, :-1]
            cond["y"].update(
                {"acceleration_gt": acceleration_gt, "acceleration_mask": acceleration_mask}
            )

    if "text" in notnone_batches[0] and notnone_batches[0]["text"] is not None:
        cond["y"].update({"text": [b["text"] for b in notnone_batches]})

    if "tokens" in notnone_batches[0] and notnone_batches[0]["tokens"] is not None:
        cond["y"].update({"tokens": [b["tokens"] for b in notnone_batches]})

    if "action" in notnone_batches[0]:
        cond["y"].update({"action": torch.as_tensor([b["action"] for b in notnone_batches]).unsqueeze(1)})

    if "action_text" in notnone_batches[0]:
        cond["y"].update({"action_text": [b["action_text"] for b in notnone_batches]})

    if "prefix" in notnone_batches[0]:
        cond["y"].update({"prefix": collate_tensors([b["prefix"] for b in notnone_batches])})

    if notnone_batches[0].get("key") is not None:
        cond["y"].update({"db_key": [b["key"] for b in notnone_batches]})

    if notnone_batches[0].get("sample_index") is not None:
        cond["y"].update(
            {"sample_index": [int(b["sample_index"]) for b in notnone_batches]}
        )

    return ensure_float32(motion, cond)


def collate(batch):
    return _base_collate(batch, with_smooth=False)


def collate_smooth(batch):
    return _base_collate(batch, with_smooth=True)


def _prefix_adapt_batch(batch, pred_len, smooth=False):
    adapted_batch = [
        {
            "inp": torch.tensor(b["motion"].T).float().unsqueeze(1)[..., -pred_len:],
            "prefix": torch.tensor(b["motion"].T).float().unsqueeze(1)[..., :-pred_len],
            "text": b["text"],
            "tokens": b["tokens"],
            "lengths": pred_len,
            "key": b.get("key"),
            "sample_index": b.get("sample_index"),
        }
        for b in batch
    ]
    return collate_smooth(adapted_batch) if smooth else collate(adapted_batch)


def prefix_collate(batch, pred_len):
    return _prefix_adapt_batch(batch, pred_len, smooth=False)


def prefix_collate_smooth(batch, pred_len):
    return _prefix_adapt_batch(batch, pred_len, smooth=True)


def cross_prefix_collate_smooth(batch, pred_len, cross_prob=0.0):
    n = len(batch)
    clips = []
    for b in batch:
        clips.append(
            {
                "motion_full": torch.tensor(b["motion"].T).float().unsqueeze(1),
                "text": b["text"],
                "tokens": b["tokens"],
                "key": b.get("key"),
                "sample_index": b.get("sample_index"),
            }
        )

    context_len = clips[0]["motion_full"].shape[-1] - pred_len
    items = []
    for i in range(n):
        is_cross = n > 1 and _rng.random() < cross_prob
        if is_cross:
            j = _rng.randint(0, n - 1)
            while j == i:
                j = _rng.randint(0, n - 1)
            prefix = clips[j]["motion_full"][..., -context_len:]
            inp = clips[i]["motion_full"][..., :pred_len]
        else:
            prefix = clips[i]["motion_full"][..., :context_len]
            inp = clips[i]["motion_full"][..., context_len:]

        items.append(
            {
                "prefix": prefix,
                "inp": inp,
                "text": clips[i]["text"],
                "tokens": clips[i]["tokens"],
                "lengths": pred_len,
                "key": clips[i]["key"],
                "sample_index": clips[i]["sample_index"],
            }
        )

    return collate_smooth(items)


def self_rollout_collate(batch):
    adapted_batch = [
        {
            "inp": torch.tensor(b["motion"].T).float().unsqueeze(1),
            "text": b["text"],
            "tokens": b["tokens"],
            "lengths": b["length"],
            "key": b.get("key"),
            "sample_index": b.get("sample_index"),
        }
        for b in batch
    ]
    notnone_batches = [b for b in adapted_batch if b is not None]
    databatch = [b["inp"] for b in notnone_batches]

    motion_full = collate_tensors(databatch)
    lenbatch_tensor = torch.as_tensor([b["lengths"] for b in notnone_batches])
    cond = {"y": {"lengths": lenbatch_tensor}}

    if notnone_batches[0]["text"] is not None:
        cond["y"]["text"] = [b["text"] for b in notnone_batches]
    if notnone_batches[0]["tokens"] is not None:
        cond["y"]["tokens"] = [b["tokens"] for b in notnone_batches]
    if notnone_batches[0].get("key") is not None:
        cond["y"]["db_key"] = [b["key"] for b in notnone_batches]
    if notnone_batches[0].get("sample_index") is not None:
        cond["y"]["sample_index"] = [
            int(b["sample_index"]) for b in notnone_batches
        ]

    return ensure_float32(motion_full, cond)


__all__ = [
    "collate",
    "collate_smooth",
    "collate_tensors",
    "cross_prefix_collate_smooth",
    "ensure_float32",
    "lengths_to_mask",
    "self_rollout_collate",
    "prefix_collate",
    "prefix_collate_smooth",
]
