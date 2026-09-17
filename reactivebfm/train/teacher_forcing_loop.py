import contextlib
import copy
import functools
import os
import warnings
import math
import re
from os.path import join as pjoin
from typing import Optional

import blobfile as bf
import torch
from torch.optim import AdamW

from reactivebfm.model.motion_planner.objectives.diffusion import logger
from reactivebfm.utils.runtime import distributed as dist_util
from reactivebfm.model.motion_planner.objectives.diffusion.fp16_util import (
    MixedPrecisionTrainer,
)
from reactivebfm.model.motion_planner.objectives.diffusion.resample import (
    LossAwareSampler,
    create_named_schedule_sampler,
)
from tqdm import tqdm
from reactivebfm.utils.training.models import load_model_wo_clip
from reactivebfm.data.motion import HML_ROOT_HORIZONTAL_MASK

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


def wrap_ddp_model(model, device, args):
    if not dist_util.is_distributed():
        return model
    return torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[device.index] if device.type == 'cuda' else None,
        output_device=device.index if device.type == 'cuda' else None,
        # The planner has no rank-local mutable buffers.  Broadcasting the
        # frozen text/positional buffers before every forward adds a NCCL
        # collective to the hot path and can mask the real failing rank when
        # another rank exits early.
        broadcast_buffers=False,
        find_unused_parameters=bool(
            getattr(args, 'ddp_find_unused_parameters', False)
        ),
        gradient_as_bucket_view=True,
    )


class TeacherForcingLoop:
    def __init__(self, args, train_platform, model, diffusion, data):
        self.args = args
        self.train_platform = train_platform
        self.model = model
        self.model_avg = None
        if self.args.use_ema:
            self.model_avg = copy.deepcopy(self.model)
        self.diffusion = diffusion
        self.cond_mode = model.cond_mode
        self.data = data
        self.batch_size_local = args.batch_size_local
        self.batch_size_global = args.batch_size_global
        self.microbatch = self.batch_size_local
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.resume_checkpoint = args.resume_checkpoint
        self.use_fp16 = False
        bf16_requested = bool(getattr(args, 'use_bf16', False))
        bf16_supported = (
            torch.cuda.is_available()
            and hasattr(torch.cuda, "is_bf16_supported")
            and torch.cuda.is_bf16_supported()
        )
        self.use_bf16 = bf16_requested and bf16_supported
        if bf16_requested and not self.use_bf16:
            logger.log("[bf16] Requested but not supported on this GPU/runtime; falling back to fp32.")
        self.fp16_scale_growth = 1e-3
        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps
        self.training_rtc = bool(getattr(args, 'training_rtc', True))
        self.rtc_max_delay = int(getattr(args, 'rtc_max_delay', 6))
        self.rtc_prefix_noise_std = float(
            getattr(args, 'rtc_prefix_noise_std', 0.0)
        )
        if self.training_rtc and not 0 <= self.rtc_max_delay < int(args.pred_len):
            raise ValueError(
                "training-time RTC requires 0 <= rtc_max_delay < pred_len; "
                f"got rtc_max_delay={self.rtc_max_delay}, pred_len={args.pred_len}."
            )
        if self.rtc_prefix_noise_std < 0.0:
            raise ValueError(
                "rtc_prefix_noise_std must be non-negative, "
                f"got {self.rtc_prefix_noise_std}."
            )
        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size_global
        self.num_steps = args.num_steps
        # Use dataset length instead of DataLoader length to handle cases where
        # drop_last=True causes DataLoader length to be 0 when dataset is smaller than batch_size
        dataset_len = len(self.data.dataset) if hasattr(self.data, 'dataset') else len(self.data)
        if dataset_len == 0:
            raise ValueError(f"Dataset is empty! Cannot train with 0 samples.")
        
        # Calculate effective batches per epoch (accounting for drop_last)
        # If DataLoader length is 0, it means drop_last=True and dataset is smaller than batch_size
        if len(self.data) > 0:
            batches_per_epoch = len(self.data)
        else:
            # When drop_last=True and dataset < batch_size, DataLoader returns 0 batches
            # This means the training loop won't execute any iterations
            # Warn the user and calculate based on dataset size
            warnings.warn(
                f"DataLoader has 0 batches because dataset size ({dataset_len}) < batch_size_local ({self.batch_size_local}) "
                f"and drop_last=True. Training loop will not execute. Consider setting drop_last=False "
                f"or reducing batch_size_local for small datasets.",
                UserWarning
            )
            # Use 1 batch per epoch as minimum to avoid division by zero
            batches_per_epoch = 1
        
        self.num_epochs = self.num_steps // batches_per_epoch + 1 if batches_per_epoch > 0 else 1

        self.sync_cuda = torch.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
            max_grad_norm=args.max_grad_norm,
        )

        self.save_dir = args.save_dir
        self.overwrite = args.overwrite

        if self.args.use_ema:
            self.opt = AdamW(
                # with amp, we don't need to use the mp_trainer's master_params
                (self.model.parameters()
                 if self.use_fp16 else self.mp_trainer.master_params),
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=(0.9, self.args.adam_beta2),
            )
        else:
            self.opt = AdamW(
                self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
            )

        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.

        self.device = torch.device("cpu")
        if torch.cuda.is_available() and dist_util.dev() != 'cpu':
            self.device = torch.device(dist_util.dev())

        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)

        self.use_ddp = dist_util.is_distributed()
        self.ddp_model = wrap_ddp_model(self.model, self.device, self.args)

    def _forward_context(self):
        if self.use_bf16 and self.device.type == 'cuda':
            return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _load_and_sync_parameters(self):
        if self.resume_checkpoint:
            resume_checkpoint = self.find_resume_checkpoint() or self.resume_checkpoint
            self.step += 1
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            state_dict = dist_util.load_state_dict(
                resume_checkpoint, map_location=dist_util.dev())

            if 'model_avg' in state_dict:
                print('loading both model and model_avg')
                state_dict, state_dict_avg = state_dict['model'], state_dict[
                    'model_avg']
                load_model_wo_clip(self.model, state_dict)
                load_model_wo_clip(self.model_avg, state_dict_avg)
            else:
                load_model_wo_clip(self.model, state_dict)
                if self.args.use_ema:
                    print('loading model_avg from model')
                    self.model_avg.load_state_dict(self.model.state_dict(), strict=True)
        elif getattr(self.args, 'finetune_from', ''):
            ckpt_path = self.args.finetune_from
            logger.log(f"[Fine-tune] Loading pretrained weights: {ckpt_path}")
            logger.log(f"[Fine-tune] Step counter = 0, optimizer = fresh")
            state_dict = dist_util.load_state_dict(
                ckpt_path, map_location=dist_util.dev())
            if 'model_avg' in state_dict:
                load_model_wo_clip(self.model, state_dict['model'])
                if self.args.use_ema:
                    load_model_wo_clip(self.model_avg, state_dict['model_avg'])
            else:
                load_model_wo_clip(self.model, state_dict)
                if self.args.use_ema:
                    self.model_avg.load_state_dict(
                        self.model.state_dict(), strict=True)

        dist_util.sync_params(self.model.parameters())
        dist_util.sync_params(self.model.buffers())
        if self.model_avg is not None:
            dist_util.sync_params(self.model_avg.parameters())
            dist_util.sync_params(self.model_avg.buffers())

    def _load_optimizer_state(self):
        main_checkpoint = self.find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:09}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )

            if self.use_fp16:
                if 'scaler' not in state_dict:
                    print("scaler state not found ... not loading it.")
                else:
                    # load grad scaler state
                    self.scaler.load_state_dict(state_dict['scaler'])
                    # for the rest
                    state_dict = state_dict['opt']

            tgt_wd = self.opt.param_groups[0]['weight_decay']
            print('target weight decay:', tgt_wd)
            self.opt.load_state_dict(state_dict)
            print('loaded weight decay (will be replaced):',
                  self.opt.param_groups[0]['weight_decay'])
            # preserve the weight decay parameter
            for group in self.opt.param_groups:
                group['weight_decay'] = tgt_wd
            self.opt.param_groups[0]['capturable'] = True

    def cond_modifiers(self, cond, motion):
        # All modifiers must be in-place
        self.spatial_cond_modifier(cond, motion)
        self.prefix_noise_modifier(cond, motion)

    def spatial_cond_modifier(self, cond, motion):
        spatial_condition = getattr(self.args, 'spatial_condition', None)
        if spatial_condition is not None:
            if spatial_condition == 'traj':
                cond['condition_mask'] = torch.tensor(HML_ROOT_HORIZONTAL_MASK[None, :, None, None])
            else:
                raise ValueError(f'unsupported spatial_condition [{spatial_condition}]')

    def prefix_noise_modifier(self, cond, motion):
        """
        Apply noise to the motion prefix to improve robustness against exposure bias.
        This implementation uses a curriculum strategy (noise increases over time)
        while preserving the state condition for every sample.
        """
        if not self.args.prefix_noise_strategy:
            return

        # ---------------------------------------------------------
        # 1. Curriculum Schedule: Calculate training progress
        # ---------------------------------------------------------
        # Calculate the ratio of current steps to total steps, clamped between 0 and 1.
        # We use this to gradually increase the difficulty.
        current_step = self.total_step()
        total_steps = self.num_steps
        progress = max(0.0, min(current_step / total_steps, 1.0))

        # ---------------------------------------------------------
        # 2. Dynamic Probability Configuration (Exponential schedule)
        # ---------------------------------------------------------
        # The probability is very small when progress < 0.5 and rises rapidly after that.
        # For example, use an exponential curve: p = max_noise_prob * sigmoid(alpha * (progress - 0.5))
        max_noise_prob = 0.5  # In late training, 50% of samples will be noisy
        alpha = 10.0  # Controls the rapidness of growth; larger means sharper rise at 0.5
        # sigmoid curve centered at 0.5
        current_prob = max_noise_prob * (1.0 / (1.0 + math.exp(-alpha * (progress - 0.5))))
        
        # Decide which samples in the batch will receive noise/corruption
        bs = motion.shape[0]
        device = motion.device
        
        # Create a boolean mask: True means "apply noise to this sample"
        apply_mask = torch.rand(bs, device=device) < current_prob
        
        # Optimization: If no samples need noise, return early
        if not apply_mask.any():
            return

        # Reshape for broadcasting: [Batch, 1, 1, 1]
        apply_mask = apply_mask.view(-1, 1, 1, 1)

        # ---------------------------------------------------------
        # 3. Strategy A: Gaussian Noise (Jitter)
        # ---------------------------------------------------------
        # Load dataset statistics for adaptive noise scaling
        dataset_std = self.data.dataset.samples.std
        std_tensor = torch.tensor(dataset_std, device=device).view(1, -1, 1, 1)
        
        # Base scale increases with progress (e.g., from 0.1 to self.args.base_noise_scale)
        # If args.base_noise_scale is 1.0, it means we add 1.0 standard deviation of noise at the end.
        target_scale = self.args.base_noise_scale
        current_scale = target_scale * progress
        
        # Generate Gaussian noise scaled by the dataset's standard deviation
        gaussian_noise = torch.randn_like(cond['prefix']) * std_tensor * current_scale
        
        # Apply Gaussian noise to the selected samples
        noisy_prefix = cond['prefix'] + gaussian_noise

        # Keep a noisy but present prefix: planner training is always state-conditioned.
        # Combine original prefix (for clean samples) and modified prefix (for noisy samples)
        cond['prefix'] = torch.where(apply_mask, noisy_prefix, cond['prefix'])
        
    def run_loop(self):
        is_main_process = dist_util.is_main_process()
        if is_main_process:
            print('train steps:', self.num_steps)
            if self.training_rtc:
                print(
                    f'[Training RTC] enabled, uniform delay in '
                    f'[0, {self.rtc_max_delay}] action frames'
                )
        for epoch in range(self.num_epochs):
            if hasattr(self.data.sampler, 'set_epoch'):
                self.data.sampler.set_epoch(epoch)
            if epoch % 100 == 0 and is_main_process:
                print(f'Starting epoch {epoch}')
            for motion, cond in tqdm(self.data, disable=not is_main_process):
                if not (not self.lr_anneal_steps or self.total_step() < self.lr_anneal_steps):
                    break

                self.cond_modifiers(cond['y'], motion) # Modify in-place for efficiency
                non_blocking = self.device.type == 'cuda'
                motion = motion.to(self.device, non_blocking=non_blocking)
                cond['y'] = {
                    key: val.to(self.device, non_blocking=non_blocking) if torch.is_tensor(val) else val
                    for key, val in cond['y'].items()
                }

                self.run_step(motion, cond)

                # NOTE: log
                if self.total_step() % self.log_interval == 0:
                    metrics = self._collect_global_metrics()
                    report_metrics = {}
                    for k, v in metrics.items():
                        if k == 'loss' and is_main_process:
                            print('step[{}]: loss[{:0.5f}]'.format(self.total_step(), v))

                        if k in ['step', 'samples'] or '_q' in k:
                            continue
                        report_metrics[k] = v
                    if is_main_process:
                        self.train_platform.report_scalars(
                            report_metrics,
                            iteration=self.total_step(),
                            group_name='Loss',
                        )

                # NOTE: save
                if self.total_step() % self.save_interval == 0 and self.total_step() > 0:
                    dist_util.barrier()
                    if is_main_process:
                        self.save()
                    dist_util.barrier()

                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.total_step() > 0:
                        return
                
                self.step += 1

            if not (not self.lr_anneal_steps or self.total_step() < self.lr_anneal_steps):
                break
        
        # Save the last checkpoint if it wasn't already saved.
        # if (self.total_step() - 1) % self.save_interval != 0:
        if self.total_step() % self.save_interval > self.save_interval / 2:
            dist_util.barrier()
            if is_main_process:
                self.save()
            dist_util.barrier()

    def _collect_global_metrics(self):
        current_logger = logger.get_current()
        local_metrics = {
            key: (value, current_logger.name2cnt.get(key, 1))
            for key, value in current_logger.name2val.items()
        }
        current_logger.name2val.clear()
        current_logger.name2cnt.clear()
        return dist_util.reduce_weighted_means(local_metrics)

    def run_step(self, batch, cond):
        """
        Parameters:
            batch: [bs, n_joints, 1, pred_len]
        """
        did_backward = self.forward_backward(batch, cond)
        if did_backward:
            self.mp_trainer.optimize(self.opt)
            self.update_average_model()
        else:
            self.mp_trainer.zero_grad()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            # Eliminates the microbatch feature
            assert i == 0
            assert self.microbatch == self.batch_size_local
            micro_batch = batch
            micro_cond = cond
            if self.training_rtc:
                micro_cond['y']['rtc_delay'] = torch.randint(
                    0,
                    self.rtc_max_delay + 1,
                    (micro_batch.shape[0],),
                    device=micro_batch.device,
                    dtype=torch.long,
                )
                micro_cond['y']['rtc_prefix_noise_std'] = self.rtc_prefix_noise_std
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro_batch.shape[0], dist_util.dev())    # diffusion steps

            # NOTE: run model.forward(), actually training_losses() in gaussian_diffusion.py
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro_batch,  # [bs, n_joints, 1, pred_len]
                t,  # [bs](int) sampled timesteps
                model_kwargs=micro_cond,
                dataset=self.data.dataset
            )

            if last_batch or not self.use_ddp:
                with self._forward_context():
                    losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    with self._forward_context():
                        losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            finite_on_all_ranks = dist_util.all_true(torch.isfinite(loss).item())
            if not finite_on_all_ranks:
                if dist_util.is_main_process():
                    logger.log(
                        f"[warn] Non-finite loss at step={self.total_step()}; "
                        "skipping optimizer step on every rank for this batch."
                    )
                return False
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)
        return True

    def update_average_model(self):
        # update the average model using exponential moving average
        if self.args.use_ema:
            # master params are FP32
            params = self.model.parameters(
            ) if self.use_fp16 else self.mp_trainer.master_params
            for param, avg_param in zip(params, self.model_avg.parameters()):
                if not avg_param.requires_grad:
                    continue
                # avg = avg + (param - avg) * (1 - alpha)
                # avg = avg + param * (1 - alpha) - (avg - alpha * avg)
                # avg = alpha * avg + param * (1 - alpha)
                avg_param.data.mul_(self.args.avg_model_beta).add_(
                    param.data, alpha=1 - self.args.avg_model_beta)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = self.total_step() / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.total_step())
        logger.logkv("samples", (self.total_step() + 1) * self.global_batch)

    def ckpt_file_name(self):
        return f"model{(self.total_step()):09d}.pt"

    def find_resume_checkpoint(self) -> Optional[str]:
        matches = {file: re.match(r'model(\d+).pt$', file) for file in os.listdir(self.args.save_dir)}
        models = {int(match.group(1)): file for file, match in matches.items() if match}

        return pjoin(self.args.save_dir, models[max(models)]) if models else None
    
    def total_step(self):
        return self.step + self.resume_step
    
    def save(self):
        def save_checkpoint():
            def del_clip(state_dict):
                # Do not save CLIP weights
                clip_weights = [
                    e for e in state_dict.keys() if e.startswith('clip_model.')
                ]
                for e in clip_weights:
                    del state_dict[e]

            if self.use_fp16:
                state_dict = self.model.state_dict()
            else:
                state_dict = self.mp_trainer.master_params_to_state_dict(
                    self.mp_trainer.master_params)
            del_clip(state_dict)

            if self.args.use_ema:
                # save both the model and the average model
                state_dict_avg = self.model_avg.state_dict()
                del_clip(state_dict_avg)
                state_dict = {'model': state_dict, 'model_avg': state_dict_avg}

            logger.log(f"saving model...")
            filename = self.ckpt_file_name()
            with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                torch.save(state_dict, f)

        save_checkpoint()

        # with bf.BlobFile(
        #     bf.join(self.save_dir, f"opt{(self.total_step()):09d}.pt"),
        #     "wb",
        # ) as f:
        #     opt_state = self.opt.state_dict()
        #     if self.use_fp16:
        #         # with fp16 we also save the state dict
        #         opt_state = {
        #             'opt': opt_state,
        #             'scaler': self.scaler.state_dict(),
        #         }

        #     torch.save(opt_state, f)


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
