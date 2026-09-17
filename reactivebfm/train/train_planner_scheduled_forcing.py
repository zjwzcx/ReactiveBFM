"""Public DiT scheduled self-rollout training entrypoint."""

from reactivebfm.data.datasets import get_dataset_loader
from reactivebfm.utils.runtime import distributed as dist_util
from reactivebfm.utils.runtime.arguments import train_args_scheduled_forcing
from reactivebfm.utils.training.models import create_model_and_diffusion_smooth

from .run_utils import create_train_platform, prepare_scheduled_forcing_run
from .scheduled_forcing_loop import ScheduledForcingLoop


def main():
    args = train_args_scheduled_forcing()
    args, full_len = prepare_scheduled_forcing_run(args)

    if dist_util.is_main_process():
        print("creating distributed data loader...")
    data = get_dataset_loader(
        name=args.dataset,
        batch_size=args.batch_size_local,
        data_dir=args.data_dir or None,
        hml_type=args.hml_type,
        unit_length=args.unit_length,
        fixed_len=full_len,
        pred_len=args.pred_len,
        abs_path="",
        device=dist_util.dev(),
        num_workers=getattr(args, "num_workers", 0),
        return_keys=getattr(args, "log_batch_samples", False),
        relative_root_xy=getattr(args, "relative_root_xy", False),
        collate_mode="self_rollout",
        distributed=args.distributed,
        rank=args.rank,
        world_size=args.world_size,
        seed=args.seed,
    )

    if dist_util.is_main_process():
        print("creating DiT model and flow-matching objective...")
    model, diffusion = create_model_and_diffusion_smooth(args, data)
    model.to(dist_util.dev())
    if dist_util.is_main_process():
        print(
            "Total params: %.2fM"
            % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0)
        )
        print("Training DiT with scheduled self-rollout...")

    train_platform = create_train_platform(args)
    try:
        train_platform.report_args(args, name="Args")
        ScheduledForcingLoop(args, train_platform, model, diffusion, data).run_loop()
    finally:
        train_platform.close()
        dist_util.cleanup_dist()


if __name__ == "__main__":
    main()
