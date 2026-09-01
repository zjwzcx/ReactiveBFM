from argparse import ArgumentParser
import argparse
import os
import json
from reactivebfm.data.datasets import DatasetRegistry
from reactivebfm.utils.runtime.defaults import scheduled_forcing_defaults

def parse_and_load_from_model(parser):
    # args according to the loaded model
    # do not try to specify them from cmd line since they will be overwritten
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    args = parser.parse_args()
    args_to_overwrite = []
    for group_name in ['dataset', 'model', 'diffusion']:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)

    # load args from model
    if args.model_path != '':  # if not using external results file
        args = load_args_from_model(args, args_to_overwrite)

    if args.cond_mask_prob == 0:
        args.guidance_param = 1
    
    return apply_rules(args)

def load_args_from_model(args, args_to_overwrite):
    model_path = get_model_path_from_args()
    args_path = os.path.join(os.path.dirname(model_path), 'args.json')
    # hf_handler.get_dependencies()
    assert os.path.exists(args_path), 'Arguments json file was not found!'
    with open(args_path, 'r') as fr:
        model_args = json.load(fr)

    for a in args_to_overwrite:
        if a in model_args.keys():
            setattr(args, a, model_args[a])
        elif a == "denoise_steps":
            # Backward compatibility for older diffusion checkpoints/configs.
            if "denoise_steps" in model_args:
                setattr(args, a, model_args["denoise_steps"])
            elif "diffusion_steps" in model_args:
                setattr(args, a, model_args["diffusion_steps"])
            else:
                print('Warning: was not able to load [{}], using default value [{}] instead.'.format(a, args.__dict__[a]))
        elif a == "solver_steps":
            # Backward compatibility for older flow-matching checkpoints/configs.
            if "solver_steps" in model_args:
                setattr(args, a, model_args["solver_steps"])
            elif "flow_steps" in model_args and model_args["flow_steps"] is not None:
                setattr(args, a, model_args["flow_steps"])
            else:
                print('Warning: was not able to load [{}], using default value [{}] instead.'.format(a, args.__dict__[a]))
        elif a == "num_warmup_steps" and "self_rollout_warmup_steps" in model_args:
            # Backward compatibility for checkpoints saved before the CLI rename.
            setattr(args, a, model_args["self_rollout_warmup_steps"])

        else:
            print('Warning: was not able to load [{}], using default value [{}] instead.'.format(a, args.__dict__[a]))
    return args

def apply_rules(args):
    # For prefix completion
    if args.pred_len == 0:
        args.pred_len = args.context_len

    return args


def get_args_per_group_name(parser, args, group_name):
    for group in parser._action_groups:
        if group.title == group_name:
            group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
            return list(argparse.Namespace(**group_dict).__dict__.keys())
    return ValueError('group_name was not found.')

def get_model_path_from_args():
    try:
        dummy_parser = ArgumentParser()
        dummy_parser.add_argument('--model_path')
        dummy_args, _ = dummy_parser.parse_known_args()
        return dummy_args.model_path
    except:
        raise ValueError('model_path argument must be specified.')


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
    group.add_argument("--pos_embed_max_len", default=256, type=int,
                       help="Pose embedding max length.")
    group.add_argument("--use_ema", action=argparse.BooleanOptionalAction, default=True,
                       help="If True, will use EMA model averaging.")



    # Prefix completion model
    group.add_argument("--context_len", default=20, type=int, help="If larger than 0, will do prefix completion.")
    group.add_argument("--pred_len", default=40, type=int, help="If context_len larger than 0, will do prefix completion. If pred_len will not be specified - will use the same length as context_len")
    


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
    group.add_argument("--gen_during_training", action='store_true',
                       help="If True, will generate motions during training, on each save interval.")
    group.add_argument("--gen_num_samples", default=3, type=int,
                       help="Number of samples to sample while generating")
    group.add_argument("--gen_num_repetitions", default=2, type=int,
                       help="Number of repetitions, per sample (text prompt/action)")
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

    # Cross-condition training for streaming text transitions
    group.add_argument("--cross_prob", default=0.0, type=float,
                        help="Probability of swapping prefix with a different sample's prefix in cross training. "
                             "Creates (prefix_A, text_B, target_B) triplets that teach text-following "
                             "even when prefix comes from a different action.")


def add_sampling_options(parser):
    group = parser.add_argument_group('sampling')
    # group.add_argument("--model_path", required=True, type=str,
    group.add_argument("--model_path", type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--output_dir", default='', type=str,
                       help="Path to results dir (auto created by the script). "
                            "If empty, will create dir in parallel to checkpoint.")
    group.add_argument("--num_samples", default=1, type=int,
                       help="Maximal number of prompts to sample, "
                            "if loading dataset from file, this field will be ignored.")
    group.add_argument("--num_repetitions", default=1, type=int,
                       help="Number of repetitions, per sample (text prompt/action)")
    group.add_argument("--guidance_param", default=7.5, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")
    group.add_argument("--recon_param", default=1e3, type=float,
                       help="Reconstruction parameter.")
    group.add_argument("--recon_step_start", default=-1, type=int, help="Highest step index to perform recon_guidance. Default -1 means from first step.")
    group.add_argument("--recon_step_stop", default=2, type=int, help="Lowest step index to perform recon_guidance. 0 means until the last step.")
    group.add_argument("--recon_frame_start", default=0, type=int, help="First frame index to perform recon_guidance.")
    group.add_argument("--recon_frame_stop", default=-1, type=int, help="Last frame index to perform recon_guidance. Default -1 means from last step.")
    group.add_argument("--cfg_type", default='none', choices=['none', 'text'], type=str, help="For classifier-guidance conditioning.")
    group.add_argument("--autoregressive", action='store_true', help="If true, and we use a prefix model will generate motions in an autoregressive loop.")
    group.add_argument("--autoregressive_include_prefix", action='store_true', help="If true, include the init prefix in the output, otherwise, will drop it.")
    group.add_argument("--autoregressive_init", default='data', type=str, choices=['data', 'isaac'], 
                        help="Sets the source of the init frames, either from the dataset or isaac init poses.")


def add_dip_ft_options(parser):
    """ DiP fine-tuning sampling and generation options """
    group = parser.add_argument_group('fine-tuning')
    group.add_argument("--model_path", type=str,
                       help="Path to DiP model####.pt file to be sampled.")
    group.add_argument("--output_dir", default='', type=str,
                       help="Path to results dir (auto created by the script). "
                            "If empty, will create dir in parallel to checkpoint.")
    group.add_argument("--num_samples", default=1, type=int,
                       help="Maximal number of prompts to sample, "
                            "if loading dataset from file, this field will be ignored.")
    group.add_argument("--num_repetitions", default=1, type=int,
                       help="Number of repetitions, per sample (text prompt/action)")
    group.add_argument("--guidance_param", default=7.5, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")
    group.add_argument("--recon_param", default=1e3, type=float,
                       help="Reconstruction parameter.")
    group.add_argument("--recon_step_start", default=-1, type=int, help="Highest step index to perform recon_guidance. Default -1 means from first step.")
    group.add_argument("--recon_step_stop", default=2, type=int, help="Lowest step index to perform recon_guidance. 0 means until the last step.")
    group.add_argument("--recon_frame_start", default=0, type=int, help="First frame index to perform recon_guidance.")
    group.add_argument("--recon_frame_stop", default=-1, type=int, help="Last frame index to perform recon_guidance. Default -1 means from last step.")
    group.add_argument("--cfg_type", default='none', choices=['none', 'text'], type=str, help="For classifier-guidance conditioning.")
    group.add_argument("--motion_length", default=6.0, type=float,
                       help="The length of the sampled motion [in seconds].")
    group.add_argument("--input_text", default='', type=str,
                       help="Path to a text file lists text prompts to be synthesized. If empty, will take text prompts from dataset.")
    group.add_argument("--action_file", default='', type=str,
                       help="Path to a text file that lists names of actions to be synthesized. Names must be a subset of dataset/uestc/info/action_classes.txt if sampling from uestc, "
                            "or a subset of [warm_up,walk,run,jump,drink,lift_dumbbell,sit,eat,turn steering wheel,phone,boxing,throw] if sampling from humanact12. "
                            "If no file is specified, will take action names from dataset.")
    group.add_argument("--text_prompt", default='', type=str,
                       help="A text prompt to be generated. If empty, will take text prompts from dataset.")
    group.add_argument("--action_name", default='', type=str,
                       help="An action name to be generated. If empty, will take text prompts from dataset.")

def add_generate_options(parser):
    group = parser.add_argument_group('generate')
    group.add_argument("--motion_length", default=6.0, type=float,
                       help="The length of the sampled motion [in seconds].")
    group.add_argument("--input_text", default='', type=str,
                       help="Path to a text file lists text prompts to be synthesized. If empty, will take text prompts from dataset.")
    group.add_argument("--action_file", default='', type=str,
                       help="Path to a text file that lists names of actions to be synthesized. Names must be a subset of dataset/uestc/info/action_classes.txt if sampling from uestc, "
                            "or a subset of [warm_up,walk,run,jump,drink,lift_dumbbell,sit,eat,turn steering wheel,phone,boxing,throw] if sampling from humanact12. "
                            "If no file is specified, will take action names from dataset.")
    group.add_argument("--text_prompt", default='', type=str,
                       help="A text prompt to be generated. If empty, will take text prompts from dataset.")
    group.add_argument("--action_name", default='', type=str,
                       help="An action name to be generated. If empty, will take text prompts from dataset.")
def add_edit_options(parser):
    group = parser.add_argument_group('edit')
    group.add_argument("--edit_mode", default='in_between', choices=['in_between', 'upper_body'], type=str,
                       help="Defines which parts of the input motion will be edited.\n"
                            "(1) in_between - suffix and prefix motion taken from input motion, "
                            "middle motion is generated.\n"
                            "(2) upper_body - lower body joints taken from input motion, "
                            "upper body is generated.")
    group.add_argument("--text_condition", default='', type=str,
                       help="Editing will be conditioned on this text prompt. "
                            "If empty, will perform unconditioned editing.")
    # group.add_argument("--prefix_end", default=0.25, type=float,
    #                    help="For in_between editing - Defines the end of input prefix (ratio from all frames).")
    # group.add_argument("--suffix_start", default=0.75, type=float,
    #                    help="For in_between editing - Defines the start of input suffix (ratio from all frames).")
    group.add_argument("--prefix_end", default=10, type=int,
                       help="For in_between editing - Defines the end of input prefix (ratio from all frames).")
    group.add_argument("--suffix_start", default=196, type=int,
                       help="For in_between editing - Defines the start of input suffix (ratio from all frames).")
    group.add_argument("--apply_cond", action='store_true',
                help="If True, will use EMA model averaging.")

def add_evaluation_options(parser):
    group = parser.add_argument_group('eval')
    group.add_argument("--model_path", default='', type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--external_results_file", default='',type=str, 
                       help="Path to an npy file containing the external results of another model.")
    group.add_argument("--do_unique", action='store_true', help="If true, select only one motion for each db key.")
    group.add_argument("--eval_name", default='', type=str, help="Optional for wandb. if empty will use the model name instead.")
    group.add_argument("--eval_mode", default='wo_mm', choices=['wo_mm', 'mm_short', 'debug', 'full'], type=str,
                       help="wo_mm (t2m only) - 20 repetitions without multi-modality metric; "
                            "mm_short (t2m only) - 5 repetitions with multi-modality metric; "
                            "debug - short run, less accurate results."
                            "full (a2m only) - 20 repetitions.")
    group.add_argument("--autoregressive", action='store_true', help="If true, and we use a prefix model will generate motions in an autoregressive loop.")
    group.add_argument("--autoregressive_include_prefix", action='store_true', help="If true, include the init prefix in the output, otherwise, will drop it.")
    group.add_argument("--autoregressive_init", default='data', type=str, choices=['data', 'isaac'], 
                        help="Sets the source of the init frames, either from the dataset or isaac init poses.")
    group.add_argument("--guidance_param", default=7.5, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")


def get_cond_mode(args):
    return 'text'


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




def train_dip_ft_args(config_dict=None, env_device=None):
    """
    Arguments parser for DiP fine-tuning from a pretrained checkpoint.
    Similar to train_args(), but supports loading model parameters from a checkpoint.
    The model_path argument is required, and model/diffusion/dataset parameters will be
    loaded from the checkpoint's args.json file.
    
    Args:
        config_dict: Optional dictionary with configuration values to override defaults.
                    Keys should match argument names (e.g., 'batch_size_local', 'context_len', etc.)
        env_device: Optional torch.device or int for device setting. If provided, will convert to int.
    
    Returns:
        args: Parsed arguments object with values loaded from checkpoint and config_dict
    """
    parser = ArgumentParser(allow_abbrev=False)
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_training_options(parser)
    # NOTE: Add options needed for fine-tuning
    # add_sampling_options(parser)
    # add_generate_options(parser)
    add_dip_ft_options(parser)
    
    
    # Parse arguments
    args = parser.parse_args()
    
    # Load model parameters from checkpoint's args.json
    args_to_overwrite = []
    for group_name in ['dataset', 'model', 'diffusion']:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)

    # Apply rules (e.g., set pred_len = context_len if pred_len == 0)
    args = apply_rules(args)
    
    # Override with config_dict values if provided
    if config_dict is not None:
        for key, value in config_dict.items():
            if hasattr(args, key):
                setattr(args, key, value)
            else:
                print(f"Warning: config_dict contains unknown key '{key}', ignoring.")
    
    # Handle device conversion if env_device is provided
    if env_device is not None:
        import torch
        if isinstance(env_device, torch.device):
            if env_device.type == 'cuda':
                args.device = env_device.index if env_device.index is not None else 0
            else:
                args.device = -1  # CPU
        elif isinstance(env_device, int):
            args.device = env_device
        else:
            args.device = 0
    
    # Set resume_checkpoint to model_path for loading model weights
    # Only override if explicitly specified in config_dict
    if config_dict is not None and 'resume_checkpoint' in config_dict and config_dict['resume_checkpoint']:
        args.resume_checkpoint = config_dict['resume_checkpoint']
    else:
        args.resume_checkpoint = args.model_path
    
    return args


def generate_args():
    parser = ArgumentParser(allow_abbrev=False)
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_generate_options(parser)
    args = parse_and_load_from_model(parser)
    cond_mode = get_cond_mode(args)

    if (args.input_text or args.text_prompt) and cond_mode != 'text':
        raise Exception('Arguments input_text and text_prompt should not be used for an action condition. Please use action_file or action_name.')
    elif (args.action_file or args.action_name) and cond_mode != 'action':
        raise Exception('Arguments action_file and action_name should not be used for a text condition. Please use input_text or text_prompt.')

    return args


def edit_args():
    parser = ArgumentParser(allow_abbrev=False)
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_edit_options(parser)
    return parse_and_load_from_model(parser)


def evaluation_parser():
    parser = ArgumentParser(allow_abbrev=False)
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_evaluation_options(parser)
    return parse_and_load_from_model(parser)
