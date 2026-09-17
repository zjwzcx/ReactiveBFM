"""Factories for motion-planner architectures and training objectives."""

from reactivebfm.utils.runtime.defaults import (
    DATA_REP as STANDARD_DATA_REP,
    FRAME_DIM as STANDARD_FRAME_DIM,
    NFEATS as STANDARD_NFEATS,
)

from .architectures import DiTMotionPlanner
from .objectives.diffusion import gaussian_diffusion as gd
from .objectives.diffusion.respace import (
    SpacedDiffusionSmoothStandard,
    space_timesteps,
)
from .objectives.flow import FlowMatchingSmoothStandard


def create_model_and_diffusion_smooth(args, data=None):
    """Build a planner and the configured smooth generative objective."""
    model = create_motion_planner_model(args, data)
    model_type = getattr(args, "model_type", "diffusion")
    if model_type == "diffusion":
        process = create_gaussian_diffusion_smooth(args)
    elif model_type == "flow":
        process = create_flow_matching_smooth(args)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")
    return model, process


def create_motion_planner_model(args, data=None):
    """Build the requested motion-planner architecture."""
    model_args = get_model_args(args, data)
    architecture = model_args["arch"]
    if architecture == "dit":
        return DiTMotionPlanner(**model_args)
    raise ValueError(
        f"Unknown planner_arch: {architecture!r}; the public release only supports 'dit'."
    )


def get_model_args(args, data=None):
    """Translate command-line configuration into planner constructor arguments."""
    del data  # Reserved for dataset-dependent architecture settings.

    planner_arch = getattr(args, "planner_arch", "dit")
    is_dit = planner_arch == "dit"
    dit_ff_size = getattr(args, "dit_ff_size", None)
    if is_dit:
        ff_size = dit_ff_size or 4 * args.latent_dim
        dropout = getattr(args, "dit_dropout", 0.0)
    else:
        ff_size = 1024
        dropout = 0.1
    frame_dim = int(getattr(args, "frame_dim", STANDARD_FRAME_DIM))
    nfeats = int(getattr(args, "nfeats", STANDARD_NFEATS))
    data_rep = getattr(args, "data_rep", STANDARD_DATA_REP)
    if frame_dim != STANDARD_FRAME_DIM or nfeats != STANDARD_NFEATS:
        raise ValueError(
            f"Standard path expects frame_dim={STANDARD_FRAME_DIM} "
            f"and nfeats={STANDARD_NFEATS}, got frame_dim={frame_dim}, "
            f"nfeats={nfeats}."
        )
    if nfeats != STANDARD_NFEATS:
        raise ValueError(
            f"ReactiveBFM planner expects nfeats={STANDARD_NFEATS}, got {nfeats}."
        )
    if data_rep != STANDARD_DATA_REP:
        raise ValueError(
            f"Standard path expects data_rep={STANDARD_DATA_REP!r}, got {data_rep!r}."
        )

    return {
        "modeltype": "",
        "njoints": frame_dim,
        "nfeats": nfeats,
        "num_actions": 1,
        "translation": True,
        "pose_rep": "rot6d",
        "glob": True,
        "glob_rot": True,
        "latent_dim": args.latent_dim,
        "ff_size": ff_size,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "dropout": dropout,
        "activation": "gelu",
        "dit_role_embedding": getattr(args, "dit_role_embedding", False),
        "dit_rtc_time_cache": getattr(args, "dit_rtc_time_cache", False),
        "dit_rope": getattr(args, "dit_rope", False),
        "dit_text_kv_cache": getattr(args, "dit_text_kv_cache", False),
        "data_rep": data_rep,
        "cond_mode": "text",
        "cond_mask_prob": args.cond_mask_prob,
        "arch": planner_arch,
        "dataset": args.dataset,
        "text_encoder": getattr(args, "text_encoder", "bert"),
        "pos_embed_max_len": args.pos_embed_max_len,
        "mask_frames": args.mask_frames,
        "pred_len": args.pred_len,
        "context_len": args.context_len,
    }


def create_gaussian_diffusion_smooth(args):
    """Build the Gaussian diffusion objective with motion-smoothness losses."""
    _validate_standard_losses(
        args, objective_name="diffusion", base_loss_name="rot_mse"
    )

    steps = args.denoise_steps
    timestep_respacing = [steps]
    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_betas=1.0)

    return SpacedDiffusionSmoothStandard(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=gd.ModelMeanType.START_X,
        model_var_type=(
            gd.ModelVarType.FIXED_SMALL
            if args.sigma_small
            else gd.ModelVarType.FIXED_LARGE
        ),
        loss_type=gd.LossType.MSE,
        rescale_timesteps=False,
        lambda_velocity=args.lambda_velocity,
        lambda_acceleration=args.lambda_acceleration,
        lambda_velocity_prefix=args.lambda_velocity_prefix,
    )


def create_flow_matching_smooth(args):
    """Build the conditional flow-matching objective."""
    _validate_standard_losses(
        args, objective_name="flow matching", base_loss_name="flow_mse"
    )

    return FlowMatchingSmoothStandard(
        num_timesteps=args.solver_steps,
        lambda_velocity=args.lambda_velocity,
        lambda_acceleration=args.lambda_acceleration,
        lambda_velocity_prefix=args.lambda_velocity_prefix,
        sampler=getattr(args, "flow_sampler", "euler"),
        train_time_sampler=getattr(args, "flow_train_time_sampler", "beta"),
        time_min=getattr(args, "flow_time_min", 0.001),
        time_beta_alpha=getattr(args, "flow_time_beta_alpha", 1.5),
        time_beta_beta=getattr(args, "flow_time_beta_beta", 1.0),
    )


def _validate_standard_losses(args, objective_name, base_loss_name):
    unsupported_losses = {
        "lambda_rcxyz": getattr(args, "lambda_rcxyz", 0.0),
        "lambda_vel": getattr(args, "lambda_vel", 0.0),
        "lambda_fc": getattr(args, "lambda_fc", 0.0),
    }
    enabled = [name for name, value in unsupported_losses.items() if value != 0.0]
    if enabled:
        raise ValueError(
            f"Standard {objective_name} supports only {base_loss_name}, velocity, "
            "acceleration and velocity-prefix losses. Disable: "
            f"{', '.join(enabled)}."
        )


__all__ = [
    "create_flow_matching_smooth",
    "create_gaussian_diffusion_smooth",
    "create_model_and_diffusion_smooth",
    "create_motion_planner_model",
    "get_model_args",
]
