"""
Helpers for distributed training.
"""

import os
import socket
from datetime import timedelta

import torch as th
import torch.distributed as dist

used_device = 0


def setup_dist(device=0):
    """Initialize torchrun-style distributed training and bind this process."""
    global used_device
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", device)))

    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if th.cuda.is_available() else "gloo"
        timeout_seconds = int(os.environ.get("TORCH_DISTRIBUTED_TIMEOUT", "10800"))
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(seconds=timeout_seconds),
        )

    used_device = local_rank if world_size > 1 else int(device)
    if th.cuda.is_available() and used_device >= 0:
        th.cuda.set_device(used_device)
    return dev()


def cleanup_dist():
    """Destroy the active process group without imposing a final barrier."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_distributed():
    return dist.is_initialized() and dist.get_world_size() > 1


def get_rank():
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def get_local_rank():
    if is_distributed():
        return int(os.environ.get("LOCAL_RANK", get_rank()))
    return used_device


def is_main_process():
    return get_rank() == 0


def barrier():
    if dist.is_initialized():
        dist.barrier()


def broadcast_object(value, src=0):
    if not dist.is_initialized():
        return value
    values = [value]
    dist.broadcast_object_list(values, src=src)
    return values[0]


def reduce_mean(value):
    """Return the mean scalar/tensor across ranks on every process."""
    if not th.is_tensor(value):
        value = th.tensor(float(value), device=dev())
    else:
        value = value.detach().clone()
        if not value.is_floating_point():
            value = value.float()
    if dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value.div_(get_world_size())
    return value


def reduce_sum(value):
    """Return the summed scalar/tensor across ranks on every process."""
    if not th.is_tensor(value):
        value = th.tensor(float(value), device=dev())
    else:
        value = value.detach().clone()
    if dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def all_true(value):
    """Return True only when every rank reports a truthy value."""
    flag = th.tensor(int(bool(value)), device=dev(), dtype=th.int32)
    if dist.is_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def reduce_weighted_means(values):
    """Combine named local means and counts across all ranks.

    Training metrics are scalar values with deterministic names on every
    rank.  Use tensor collectives for the actual reduction instead of
    ``all_gather_object`` (which serializes Python objects and uses an
    additional NCCL broadcast under the hood).  A small object gather is
    retained only as a preflight key check so a rank-local metric mismatch
    fails explicitly rather than deadlocking the next forward pass.
    """
    local_values = {
        name: (float(value), int(count))
        for name, (value, count) in values.items()
        if int(count) > 0
    }
    if not dist.is_initialized():
        return {name: value for name, (value, _) in local_values.items()}

    local_names = tuple(sorted(local_values))
    gathered_names = [None] * get_world_size()
    dist.all_gather_object(gathered_names, local_names)
    if any(names != local_names for names in gathered_names):
        raise RuntimeError(
            "Distributed metric keys differ across ranks: "
            + repr(gathered_names)
        )

    if not local_names:
        return {}
    stats = th.tensor(
        [[local_values[name][0] * local_values[name][1], local_values[name][1]]
         for name in local_names],
        device=dev(),
        dtype=th.float64,
    )
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return {
        name: float(stats[index, 0].item() / max(stats[index, 1].item(), 1.0))
        for index, name in enumerate(local_names)
    }


def dev():
    """
    Get the device to use for torch.distributed.
    """
    global used_device
    if th.cuda.is_available() and used_device>=0:
        return th.device(f"cuda:{used_device}")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    return th.load(path, **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    if not dist.is_initialized():
        return
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
