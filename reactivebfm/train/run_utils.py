import json
import os

from reactivebfm.utils.runtime import distributed as dist_util
from reactivebfm.utils.runtime.seed import fixseed
from reactivebfm.utils.runtime.paths import resolve_save_dir

from .logging_platforms import (
    ClearmlPlatform,
    NoPlatform,
    TensorboardPlatform,
    WandBPlatform,
)


TRAIN_PLATFORMS = {
    "ClearmlPlatform": ClearmlPlatform,
    "NoPlatform": NoPlatform,
    "TensorboardPlatform": TensorboardPlatform,
    "WandBPlatform": WandBPlatform,
}


def prepare_training_run(args):
    dist_util.setup_dist(args.device)
    rank = dist_util.get_rank()
    world_size = dist_util.get_world_size()
    fixseed(args.seed + rank)

    if args.save_dir is None:
        raise FileNotFoundError("save_dir was not specified.")

    if args.batch_size_local <= 0:
        raise ValueError(
            f"batch_size_local must be positive, got {args.batch_size_local}."
        )
    eval_batch_size_local = getattr(args, "eval_batch_size_local", 0)
    if eval_batch_size_local < 0:
        raise ValueError(
            "eval_batch_size_local must be non-negative; "
            f"got {eval_batch_size_local}."
        )

    save_dir = args.save_dir
    if dist_util.is_main_process() and not getattr(args, "resume_checkpoint", ""):
        save_dir = resolve_save_dir(save_dir)
    args.save_dir = dist_util.broadcast_object(save_dir)

    args.distributed = dist_util.is_distributed()
    args.rank = rank
    args.local_rank = dist_util.get_local_rank()
    args.world_size = world_size
    args.batch_size_global = args.batch_size_local * world_size

    if dist_util.is_main_process():
        os.makedirs(args.save_dir, exist_ok=True)
        args_path = os.path.join(args.save_dir, "args.json")
        with open(args_path, "w") as fw:
            json.dump(vars(args), fw, indent=4, sort_keys=True)
        print("args:", args)
        print(
            f"[Distributed] world_size={world_size}, "
            f"batch_size_local={args.batch_size_local}, "
            f"batch_size_global={args.batch_size_global}"
        )

    dist_util.barrier()
    return args


def prepare_scheduled_forcing_run(args):
    n_primitives = getattr(args, "n_primitives", 1)
    if n_primitives < 2:
        raise ValueError(
            f"Scheduled forcing requires n_primitives >= 2, got {n_primitives}. "
            "Use train_planner_teacher_forcing.py for pure teacher forcing."
        )

    rollout_mode = getattr(args, "self_rollout_mode", "random_replace")
    if rollout_mode not in {"random_replace", "continuous"}:
        raise ValueError(f"Unsupported self_rollout_mode={rollout_mode!r}.")

    if getattr(args, "training_rtc", True):
        max_delay = int(getattr(args, "rtc_max_delay", 6))
        prefix_noise_std = float(getattr(args, "rtc_prefix_noise_std", 0.0))
        if max_delay < 0 or max_delay >= int(args.pred_len):
            raise ValueError(
                "training-time RTC requires 0 <= rtc_max_delay < pred_len; "
                f"got rtc_max_delay={max_delay}, pred_len={args.pred_len}."
            )
        if prefix_noise_std < 0.0:
            raise ValueError(
                f"rtc_prefix_noise_std must be non-negative, got {prefix_noise_std}."
            )

    full_len = args.context_len + n_primitives * args.pred_len
    args = prepare_training_run(args)
    if dist_util.is_main_process():
        print("[Scheduled Forcing Config]")
        print(f"  n_primitives:     {n_primitives}")
        print(f"  context_len:      {args.context_len}")
        print(f"  pred_len:         {args.pred_len}")
        print(f"  max_replace_prob: {args.max_replace_prob}")
        print(f"  warmup_steps:     {args.num_warmup_steps}")
        print(f"  ramp_steps:       {max(args.num_steps - args.num_warmup_steps, 1)}")
        print(f"  rollout_mode:     {rollout_mode}")
        print(f"  full_len/sample:  {full_len} frames")
    return args, full_len


def create_train_platform(args):
    if not dist_util.is_main_process():
        return NoPlatform(args.save_dir)
    try:
        platform_cls = TRAIN_PLATFORMS[args.train_platform_type]
    except KeyError as exc:
        available = ", ".join(sorted(TRAIN_PLATFORMS))
        raise ValueError(
            f"Unsupported train_platform_type={args.train_platform_type}. "
            f"Choose one of: {available}"
        ) from exc
    return platform_cls(args.save_dir)
