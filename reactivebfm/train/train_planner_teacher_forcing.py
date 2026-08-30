"""Classic ReactiveBFM planner teacher-forcing training recipe."""

from reactivebfm.data.datasets import get_dataset_loader
from reactivebfm.utils.runtime import distributed as dist_util
from reactivebfm.utils.training.models import create_model_and_diffusion_smooth
from reactivebfm.utils.runtime.arguments import train_args

from .run_utils import (
    create_train_platform,
    prepare_training_run,
)
from .teacher_forcing_loop import TeacherForcingLoop


def main():
    args = train_args()
    args = prepare_training_run(args)

    if dist_util.is_main_process():
        print("creating data loader...")
    data = get_dataset_loader(
        name=args.dataset,
        batch_size=args.batch_size_local,
        data_dir=args.data_dir or None,
        hml_type=args.hml_type,
        unit_length=args.unit_length,
        fixed_len=args.pred_len + args.context_len,
        pred_len=args.pred_len,
        abs_path="",
        device=dist_util.dev(),
        num_workers=getattr(args, "num_workers", 0),
        return_keys=getattr(args, "log_batch_samples", False),
        relative_root_xy=getattr(args, "relative_root_xy", False),
        distributed=args.distributed,
        rank=args.rank,
        world_size=args.world_size,
        seed=args.seed,
    )

    if dist_util.is_main_process():
        print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion_smooth(args, data)
    model.to(dist_util.dev())

    if dist_util.is_main_process():
        print(
            "Total params: %.2fM"
            % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0)
        )
        print("Training with teacher forcing...")
    train_platform = create_train_platform(args)
    train_platform.report_args(args, name="Args")
    TeacherForcingLoop(args, train_platform, model, diffusion, data).run_loop()
    train_platform.close()


if __name__ == "__main__":
    main()
