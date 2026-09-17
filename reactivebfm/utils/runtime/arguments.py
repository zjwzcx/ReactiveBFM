from argparse import ArgumentParser
import argparse

from reactivebfm.data.datasets import DatasetRegistry
from reactivebfm.utils.runtime.defaults import scheduled_forcing_defaults

def apply_rules(args):
    # For prefix completion
    if args.pred_len == 0:
        args.pred_len = args.context_len

    return args

def add_base_options(parser):
    group = parser.add_argument_group('base')
    group.add_argument("--cuda", default=True, type=bool, help="Use cuda device, otherwise use CPU.")
    group.add_argument("--device", default=0, type=int, help="Device id to use.")
    group.add_argument("--seed", default=0, type=int, help="For fixing random seed.")
    group.add_argument(
        "--batch_size_local",
        default=32,
        type=int,
        help="Training batch size per rank/GPU. Global batch is this value times WORLD_SIZE.",
    )
    group.add_argument("--train_platform_type", default='NoPlatform', choices=['NoPlatform', 'ClearmlPlatform', 'TensorboardPlatform', 'WandBPlatform'], type=str,
                       help="Choose platform to log results. NoPlatform means no logging.")
    group.add_argument("--external_mode", default=False, type=bool, help="For backward cometability, do not change or delete.")


def add_diffusion_options(parser):
    group = parser.add_argument_group('diffusion')
    group.add_argument("--model_type", default='diffusion', choices=['diffusion', 'flow'], type=str,
                       help="Generative process to train and sample: diffusion or flow matching.")
    group.add_argument("--noise_schedule", default='cosine', choices=['linear', 'cosine'], type=str,
                       help="Noise schedule type")
    group.add_argument("--denoise_steps", default=10, type=int,
                       help="Number of diffusion denoising steps.")
    group.add_argument("--solver_steps", default=10, type=int,
                       help="Number of ODE solver steps for flow-matching sampling.")
    group.add_argument("--sigma_small", default=True, type=bool, help="Use smaller sigma values.")
    group.add_argument("--flow_sampler", default='euler', choices=['euler', 'heun'], type=str,
                       help="ODE solver used by flow-matching sampling.")
    group.add_argument("--flow_train_time_sampler", default='beta',
                       choices=['beta', 'uniform', 'uniform_discrete'], type=str,
                       help="Flow-matching training time sampler. "
                            "OpenPI/pi0 uses beta; uniform_discrete preserves the old diffusion-style behavior.")
    group.add_argument("--flow_time_min", default=0.001, type=float,
                       help="Minimum flow time/noise coefficient during training; avoids exact clean/noise endpoints.")
    group.add_argument("--flow_time_beta_alpha", default=1.5, type=float,
                       help="Alpha parameter for OpenPI-style Beta flow-time sampling.")
    group.add_argument("--flow_time_beta_beta", default=1.0, type=float,
                       help="Beta parameter for OpenPI-style Beta flow-time sampling.")


def add_model_options(parser):
    group = parser.add_argument_group('model')
    group.add_argument("--num_layers", default=16, type=int,
                       help="Number of layers.")
    group.add_argument("--num_heads", default=8, type=int,
                       help="Number of attention heads.")
    group.add_argument("--latent_dim", default=512, type=int,
                       help="Transformer width.")
    group.add_argument(
        "--text_encoder",
        default="bert",
        type=str,
        help=(
            "Frozen text encoder: bert, t5-base (T5 v1.1 base), t5-xl "
            "(T5 v1.1 XL), another supported preset, a T5/BERT HuggingFace "
            "ID, or a local model directory."
        ),
    )
    group.add_argument("--planner_arch", default="dit", choices=["dit"], type=str,
                       help="Motion planner architecture (the release provides text-conditioned DiT only).")
    group.add_argument("--dit_ff_size", default=2048, type=int,
                       help="DiT MLP width.")
    group.add_argument("--dit_dropout", default=0.0, type=float,
                       help="Dropout in DiT attention and MLP branches.")
    group.add_argument(
        "--dit_role_embedding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add prefix/action role embeddings to DiT tokens.",
    )
    group.add_argument(
        "--dit_rtc_time_cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache the two RTC AdaLN time conditions.",
    )
    group.add_argument(
        "--dit_rope",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use rotary position embeddings in DiT self-attention.",
    )
    group.add_argument(
        "--dit_text_kv_cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache projected text keys and values during inference rollout.",
    )
    group.add_argument("--cond_mask_prob", default=0.1, type=float,
                       help="The probability of masking the condition during training."
                            " For classifier-free guidance learning.")
    group.add_argument("--mask_frames", action=argparse.BooleanOptionalAction, default=True,
                       help="If true, will fix Rotem's bug and mask invalid frames.")
    group.add_argument("--lambda_rcxyz", default=0.0, type=float, help="Joint positions loss.")
    group.add_argument("--lambda_vel", default=0.0, type=float, help="Joint velocity loss.")
    group.add_argument("--lambda_fc", default=0.0, type=float, help="Foot contact loss.")
    group.add_argument("--lambda_velocity", default=0.0, type=float, help="Temporal velocity loss (first-order difference).")
    group.add_argument("--lambda_acceleration", default=0.0, type=float, help="Temporal acceleration loss (second-order difference).")
    group.add_argument("--lambda_velocity_prefix", default=0.0, type=float, help="Temporal velocity prefix loss (first-order difference).")
    group.add_argument(
        "--training_rtc",
        "--rtc",
        dest="training_rtc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train with a uniformly sampled clean committed action prefix.",
    )
    group.add_argument(
        "--rtc_max_delay",
        "--max_delay",
        dest="rtc_max_delay",
        default=6,
        type=int,
        help="Maximum RTC delay in action frames; must be smaller than pred_len.",
    )
    group.add_argument(
        "--rtc_delay_distribution",
        choices=("uniform",),
        default="uniform",
        help="RTC delay sampler.",
    )
    group.add_argument(
        "--rtc_prefix_noise_std",
        default=0.0,
        type=float,
        help="Shared Gaussian offset for the committed RTC prefix.",
    )
    group.add_argument("--pos_embed_max_len", default=256, type=int,
                       help="Pose embedding max length.")
    group.add_argument("--use_ema", action=argparse.BooleanOptionalAction, default=True,
                       help="If True, will use EMA model averaging.")



    # Prefix completion model
    group.add_argument("--context_len", default=25, type=int, help="If larger than 0, will do prefix completion.")
    group.add_argument("--pred_len", default=50, type=int, help="If context_len larger than 0, will do prefix completion. If pred_len will not be specified - will use the same length as context_len")
    


def add_data_options(parser):
    group = parser.add_argument_group('dataset')
    group.add_argument(
        "--dataset",
        default=DatasetRegistry.AMASS,
        type=str,
        help=(
            "Registered dataset name, or comma-separated names for runtime "
            "composition (for example dataset_a,dataset_b)."
        ),
    )
    group.add_argument("--data_dir", default="", type=str,
                       help="If empty, will use defaults according to the specified dataset.")
    group.add_argument("--hml_type", default=None, type=str, choices=[None, 'global_root'],
                       help="Optional representation variant. Use None for the standard recipe.")
    group.add_argument("--unit_length", default=4, type=int,
                       help="Motion length quantization unit used by T2M-style cropping.")
    group.add_argument("--num_workers", default=0, type=int,
                       help="DataLoader num_workers. 0 = no subprocess (no orphan workers on exit/Ctrl+C). >0 may leave workers if interrupted.")
    group.add_argument(
        "--log_batch_samples",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Append every sampled motion index/key to batch_samples_rank<RANK>.jsonl.",
    )
    group.add_argument("--use_bf16", action='store_true',
                       help="Use CUDA bfloat16 autocast for model forward/loss. "
                            "This follows OpenPI-style fast training on bf16-capable GPUs.")
    group.add_argument(
        "--relative_root_xy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Represent root XY relative to the first frame of each cropped motion clip.",
    )


def add_training_options(parser):
    group = parser.add_argument_group('training')
    group.add_argument("--save_dir", default="save/debug", type=str,
                       help="Path to save checkpoints and results.")
    group.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True,
                       help="If True, will enable to use an already existing save_dir.")
    group.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")
    group.add_argument("--weight_decay", default=0.0, type=float, help="Optimizer weight decay.")
    group.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Global gradient norm clipping threshold. Set <=0 to disable clipping.",
    )
    group.add_argument("--lr_anneal_steps", default=0, type=int, help="Number of learning rate anneal steps.")
    group.add_argument("--log_interval", default=1_000, type=int,
                       help="Log losses each N steps")
    group.add_argument("--save_interval", default=100000, type=int,
                       help="Save checkpoints each N steps")
    group.add_argument("--num_steps", default=1000000, type=int,
                       help="Training will stop after the specified number of steps.")
    group.add_argument("--num_frames", default=60, type=int,
                       help="Limit for the maximal number of frames.")
    group.add_argument("--resume_checkpoint", default="", type=str,
                       help="If not empty, will start from the specified checkpoint (path to model###.pt file).")
    group.add_argument("--finetune_from", default="", type=str,
                       help="Path to pretrained checkpoint for fine-tuning. "
                            "Loads model weights only — step counter resets to 0 and optimizer "
                            "starts fresh. Use this for pretrain→finetune pipelines.")
    group.add_argument("--gen_guidance_param", default=7.5, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")
    group.add_argument("--avg_model_beta", default=0.9999, type=float, help="Average model beta.")
    group.add_argument("--adam_beta2", default=0.999, type=float, help="Adam beta2.")
    group.add_argument(
        "--ddp_find_unused_parameters",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable DDP unused-parameter detection. The planner has conditionally used "
             "trainable branches, so this should normally remain enabled.",
    )
    group.add_argument("--autoregressive", action=argparse.BooleanOptionalAction, default=True,
                       help="If true, and we use a prefix model will generate motions in an autoregressive loop.")
    group.add_argument("--autoregressive_include_prefix", action='store_true', help="If true, include the init prefix in the output, otherwise, will drop it.")
    group.add_argument("--autoregressive_init", default='data', type=str, choices=['data', 'isaac'], 
                        help="Sets the source of the init frames, either from the dataset or isaac init poses.")
    group.add_argument("--prefix_noise_strategy", action='store_true', help="If true, will add noise to the prefix to improve robustness.")
    group.add_argument("--base_noise_scale", default=0.03, type=float, help="Base noise scale for prefix noise strategy.")

    # Self-Rollout (Scheduled Sampling) for streaming robustness
    group.add_argument("--n_primitives", default=1, type=int,
                        help="Number of consecutive motion primitives per training sample for self-rollout. "
                             "1 = standard teacher forcing. >1 = self-rollout with N consecutive chunks.")
    group.add_argument("--max_replace_prob", default=0.8, type=float,
                        help="Maximum probability that a sample enters contiguous self-rollout; once active, all remaining primitives use model predictions.")
    group.add_argument("--num_warmup_steps", default=0, type=int,
                        help="Number of initial steps to keep pure teacher forcing before self-rollout replacement starts.")
    group.add_argument(
        "--self_rollout_mode",
        default="random_replace",
        choices=("random_replace", "continuous"),
        help=(
            "Self-rollout prefix policy. 'random_replace' preserves the legacy "
            "per-primitive Bernoulli replacement schedule. 'continuous' draws "
            "h uniformly from {0,...,n_primitives-1}, continuously rolls in to "
            "depth h, and trains only the query primitive at depth h."
        ),
    )

    group.add_argument("--cross_prob", default=0.0, type=float,
                        help="Probability of switching text and target motion while retaining rollout history.")


def train_args():
    parser = ArgumentParser(allow_abbrev=False)
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_training_options(parser)
    return apply_rules(parser.parse_args())


def train_args_scheduled_forcing():
    parser = ArgumentParser(allow_abbrev=False)
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_training_options(parser)
    parser.set_defaults(**scheduled_forcing_defaults())
    return apply_rules(parser.parse_args())
