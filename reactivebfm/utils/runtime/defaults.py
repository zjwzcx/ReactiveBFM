"""Default planner settings and fixed motion representation constants."""

from __future__ import annotations

from copy import deepcopy


FRAME_DIM = 36
NFEATS = 1
DATA_REP = "hml_vec"


SCHEDULED_FORCING_RECIPE = {
    "save_dir": "save/reactivebfm_tf400k_sr3_1m",
    "dataset": "reactivebfm_dataset",
    "model_type": "flow",
    "solver_steps": 10,
    "planner_arch": "dit",
    "latent_dim": 512,
    "num_layers": 16,
    "num_heads": 8,
    "dit_ff_size": 2048,
    "dit_dropout": 0.0,
    "pos_embed_max_len": 256,
    "flow_sampler": "euler",
    "flow_train_time_sampler": "beta",
    "flow_time_beta_alpha": 1.5,
    "flow_time_beta_beta": 1.0,
    "flow_time_min": 0.001,
    "context_len": 20,
    "pred_len": 40,
    "n_primitives": 3,
    "max_replace_prob": 0.8,
    "num_warmup_steps": 400000,
    "num_steps": 1000000,
    "cond_mask_prob": 0.1,
    "mask_frames": True,
    "overwrite": True,
    "use_ema": True,
    "autoregressive": True,
    "train_platform_type": "WandBPlatform",
    "lambda_velocity": 0.0,
    "lambda_acceleration": 0.0,
    "lambda_velocity_prefix": 0.0,
    "frame_dim": FRAME_DIM,
    "nfeats": NFEATS,
    "data_rep": DATA_REP,
}


def scheduled_forcing_defaults() -> dict:
    """Return a mutable copy of the scheduled-forcing defaults."""
    return deepcopy(SCHEDULED_FORCING_RECIPE)
