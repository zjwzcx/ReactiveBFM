"""
Self-Rollout (Scheduled Sampling) training loop for streaming-robust motion generation.

Core idea (inspired by TextOp):
  Instead of pure Teacher Forcing (always feeding GT prefix), we train on N consecutive
  motion primitives per sample. With linearly increasing probability, a sample enters
  self-rollout and then keeps consuming the model's OWN x_0 predictions for all remaining
  primitives, forcing the model to learn to recover from its own imperfect outputs.

Optional continuous mode (self_rollout_mode=continuous):
  Each sample draws a continuous roll-in depth h uniformly from
  [0, n_primitives-1].
  Depth zero is teacher forcing. For h > 0, primitives 1..h always consume
  the preceding model output, and only primitive h contributes training loss.
  This removes the replacement-probability and warmup schedules while retaining
  a clean teacher-forcing fraction through the h=0 samples.

  This directly addresses the exposure bias that causes streaming collapse when:
    - Starting from a default/T-pose
    - Switching text prompts mid-stream
    - Accumulating autoregressive errors over long horizons

Optional cross-condition mode (cross_prob > 0):
  Per-sample, with probability cross_prob, a switch point k_switch is chosen
  in [1, n_primitives-1] and a donor sample j.  At primitive k >= k_switch,
  BOTH the GT target AND text switch to sample j (re-indexed from j's start).
  The prefix remains the model's self-rollout from the original action A.
  This correctly simulates streaming text transitions: the model accumulates
  its own history from action A, then at the switch point must follow text B
  to generate action B's beginning — with matched GT supervision.

Primitive layout for context_len=C, pred_len=P, n_primitives=N:
  Full motion: [-------C+N*P frames-------]
  Prim 0:  [C prefix][P prediction]
  Prim 1:           [C prefix][P prediction]
  Prim 2:                    [C prefix][P prediction]
  ...
  Prim k:  starts at frame k*P, prefix = [k*P : k*P+C], pred = [k*P+C : (k+1)*P+C]
"""

import contextlib
import copy
import json
import os
import random as pyrandom
import warnings

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
from reactivebfm.model.motion_planner.objectives.diffusion.resample import create_named_schedule_sampler
from tqdm import tqdm
from reactivebfm.utils.training.models import load_model_wo_clip
from reactivebfm.data.motion import HML_ROOT_HORIZONTAL_MASK

INITIAL_LOG_LOSS_SCALE = 20.0


class ScheduledForcingLoop:
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
        self.batch_size_local = int(args.batch_size_local)
        self.batch_size_global = int(args.batch_size_global)
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

        self.n_primitives = args.n_primitives
        self.context_len = args.context_len
        self.pred_len = args.pred_len
        self.training_rtc = bool(getattr(args, 'training_rtc', True))
        self.rtc_max_delay = int(getattr(args, 'rtc_max_delay', 6))
        self.rtc_prefix_noise_std = float(
            getattr(args, 'rtc_prefix_noise_std', 0.0)
        )
        if self.training_rtc and not 0 <= self.rtc_max_delay < self.pred_len:
            raise ValueError(
                "training-time RTC requires 0 <= rtc_max_delay < pred_len; "
                f"got rtc_max_delay={self.rtc_max_delay}, pred_len={self.pred_len}."
            )
        if self.rtc_prefix_noise_std < 0.0:
            raise ValueError(
                "rtc_prefix_noise_std must be non-negative, "
                f"got {self.rtc_prefix_noise_std}."
            )
        self.max_replace_prob = args.max_replace_prob
        self.num_warmup_steps = max(
            0, getattr(args, 'num_warmup_steps', 0))
        self.self_rollout_mode = getattr(
            args, 'self_rollout_mode', 'random_replace'
        )
        if self.self_rollout_mode not in {'random_replace', 'continuous'}:
            raise ValueError(
                f"Unsupported self_rollout_mode={self.self_rollout_mode!r}."
            )
        self.use_continuous_rollout = self.self_rollout_mode == 'continuous'
        self.cross_prob = getattr(args, 'cross_prob', 0.0)
        if not 0.0 <= float(self.cross_prob) <= 1.0:
            raise ValueError(f"cross_prob must be in [0, 1], got {self.cross_prob}.")
        if self.cross_prob > 0.0 and self.n_primitives < 2:
            raise ValueError(
                "cross_prob > 0 requires n_primitives >= 2 so a transition point exists."
            )
        if self.use_continuous_rollout and self.cross_prob > 0.0:
            raise ValueError(
                "cross_prob is not supported in continuous self-rollout mode; "
                "run the two curricula separately."
            )
        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size_global
        self.num_steps = args.num_steps
        self.self_rollout_ramp_steps = max(
            self.num_steps - self.num_warmup_steps, 1)
        dataset_len = len(self.data.dataset) if hasattr(self.data, 'dataset') else len(self.data)
        if dataset_len == 0:
            raise ValueError("Dataset is empty! Cannot train with 0 samples.")

        if len(self.data) > 0:
            batches_per_epoch = len(self.data)
        else:
            warnings.warn(
                f"DataLoader has 0 batches because dataset size ({dataset_len}) < "
                f"batch_size_local ({self.batch_size_local}) and drop_last=True. "
                "Consider reducing batch_size_local for small datasets.",
                UserWarning
            )
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
        self.batch_trace_file = None
        if getattr(args, 'log_batch_samples', False):
            trace_path = os.path.join(
                self.save_dir, f"batch_samples_rank{args.rank}.jsonl"
            )
            self.batch_trace_file = open(trace_path, "a", encoding="utf-8", buffering=1)
            self.batch_trace_file.write(json.dumps({
                "event": "run_start",
                "rank": args.rank,
                "world_size": args.world_size,
                "resume_step": self.resume_step,
                "resume_checkpoint": self.resume_checkpoint,
            }, sort_keys=True) + "\n")
            self.batch_trace_file.flush()

        if self.args.use_ema:
            self.opt = AdamW(
                (self.model.parameters() if self.use_fp16 else self.mp_trainer.master_params),
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

        self.device = torch.device("cpu")
        if torch.cuda.is_available() and dist_util.dev() != 'cpu':
            self.device = torch.device(dist_util.dev())

        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)

        self.use_ddp = dist_util.is_distributed()
        self.ddp_model = self.model
        if self.use_ddp:
            self.ddp_model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.device.index] if self.device.type == 'cuda' else None,
                output_device=self.device.index if self.device.type == 'cuda' else None,
                find_unused_parameters=bool(
                    getattr(self.args, 'ddp_find_unused_parameters', False)
                ),
                gradient_as_bucket_view=True,
            )

    def _forward_context(self):
        if self.use_bf16 and self.device.type == 'cuda':
            return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        return contextlib.nullcontext()

    # ─── replacement probability schedule ───────────────────────────
    def _replacement_prob(self):
        """Warm up with teacher forcing, then linearly ramp self-rollout."""
        step_after_warmup = self.total_step() - self.num_warmup_steps
        if step_after_warmup <= 0:
            return 0.0

        progress = min(step_after_warmup / self.self_rollout_ramp_steps, 1.0)
        return self.max_replace_prob * progress

    def _sample_rollout_depths(self, batch_size):
        """Sample one continuous roll-in depth per sample.

        Depth zero is teacher forcing. A sample with depth ``h`` uses model
        prefixes for primitives 1..h and contributes supervision only at
        primitive ``h``. The random draw is device-local and follows the
        training seed established by ``prepare_training_run``.
        """
        return torch.randint(
            low=0,
            high=self.n_primitives,
            size=(batch_size,),
            device=self.device,
            dtype=torch.long,
        )

    @staticmethod
    def _continuous_step_masks(rollout_depths, primitive):
        """Return model-prefix and query masks for one continuous primitive."""
        return rollout_depths >= primitive, rollout_depths == primitive

    @staticmethod
    def _mask_per_sample(value, sample_mask, batch_size):
        """Multiply a per-sample tensor by a [B] mask without bad broadcasting."""
        if (
            torch.is_tensor(value)
            and value.ndim > 0
            and value.shape[0] == batch_size
        ):
            shape = (batch_size,) + (1,) * (value.ndim - 1)
            return value * sample_mask.view(shape)
        return value

    # ─── build per-primitive condition dict ──────────────────────────
    def _build_primitive_cond(self, prefix, text_list, token_list, bs,
                              pred_mask=None, pred_lengths=None,
                              text_embed=None):
        """Construct the model_kwargs dict for a single primitive."""
        if pred_mask is None:
            pred_mask = torch.ones(bs, 1, 1, self.pred_len, dtype=torch.bool, device=self.device)
        if pred_lengths is None:
            pred_lengths = torch.full((bs,), self.pred_len, dtype=torch.long, device=self.device)
        prim_cond = {
            'y': {
                'prefix': prefix,
                'mask': pred_mask,
                'lengths': pred_lengths,
            }
        }
        if text_embed is not None:
            prim_cond['y']['text_embed'] = text_embed
        elif text_list is not None:
            prim_cond['y']['text'] = text_list
        if token_list is not None:
            prim_cond['y']['tokens'] = token_list
        return prim_cond

    def _encode_text_cache(self, text_list):
        """Cache frozen text encoder outputs once per batch.

        This mirrors OpenPI's prefix-cache idea: the language context is the
        expensive, repeated prefix, while each primitive only changes the noisy
        motion suffix and motion prefix.
        """
        if text_list is None or 'text' not in self.model.cond_mode:
            return None
        model = self.ddp_model.module if hasattr(self.ddp_model, 'module') else self.ddp_model
        with torch.no_grad():
            return model.encode_text(text_list)

    def _select_text_cache(self, text_cache, source_indices):
        if text_cache is None:
            return None
        idx = torch.as_tensor(source_indices, dtype=torch.long, device=self.device)
        if isinstance(text_cache, tuple):
            enc_text, text_mask = text_cache
            return enc_text.index_select(1, idx), text_mask.index_select(0, idx)
        return text_cache.index_select(1, idx)

    def _build_primitive_mask(self, lengths, source_indices, primitive_starts):
        """Mask out target frames that come from fixed_len padding."""
        valid_lengths = []
        for src, start in zip(source_indices, primitive_starts):
            motion_len = int(lengths[src].item())
            target_start = start + self.context_len
            valid = max(0, min(self.pred_len, motion_len - target_start))
            valid_lengths.append(valid)
        pred_lengths = torch.tensor(valid_lengths, dtype=torch.long, device=self.device)
        frame_ids = torch.arange(self.pred_len, device=self.device).view(1, 1, 1, -1)
        pred_mask = frame_ids < pred_lengths.view(-1, 1, 1, 1)
        return pred_mask, pred_lengths

    def _compute_smooth_targets(self, gt_pred, mask):
        """Compute velocity/acceleration GT tensors for a single primitive (on-the-fly)."""
        extras = {}
        if getattr(self.args, 'lambda_velocity', 0.0) > 0 and self.pred_len > 1:
            vel_gt = gt_pred[:, :, :, 1:] - gt_pred[:, :, :, :-1]
            vel_mask = mask[:, :, :, 1:] & mask[:, :, :, :-1]
            extras['velocity_gt'] = vel_gt
            extras['velocity_mask'] = vel_mask
            if getattr(self.args, 'lambda_acceleration', 0.0) > 0 and self.pred_len > 2:
                acc_gt = vel_gt[:, :, :, 1:] - vel_gt[:, :, :, :-1]
                acc_mask = vel_mask[:, :, :, 1:] & vel_mask[:, :, :, :-1]
                extras['acceleration_gt'] = acc_gt
                extras['acceleration_mask'] = acc_mask
        return extras

    # ─── condition modifiers (per-primitive, no prefix_noise) ────────
    def _apply_cond_modifiers(self, cond_y):
        """Apply spatial modifiers. Prefix noise is intentionally
        skipped because self-rollout itself provides prefix augmentation."""
        spatial_condition = getattr(self.args, 'spatial_condition', None)
        if spatial_condition is not None:
            if spatial_condition == 'traj':
                cond_y['condition_mask'] = torch.tensor(
                    HML_ROOT_HORIZONTAL_MASK[None, :, None, None])

    # ─── main training loop ─────────────────────────────────────────
    def _compute_cross_schedule(self, bs):
        """Per-sample cross-switch schedule for simulating streaming text transitions.

        For each sample i, with probability cross_prob, picks a switch point
        k_switch in [1, n_primitives-1] and a donor sample j != i.
        At primitive k >= k_switch the GT source and text switch from sample i
        to sample j (re-indexed from j's beginning), so the model sees:
          prefix = self-rollout accumulated from action A (sample i)
          text   = sample j's text (action B)
          GT     = sample j's motion starting from primitive 0 (action B's start)

        Returns:
            switch_at: list[int] of length bs, primitive index to switch at
                       (n_primitives means no switch)
            switch_to: list[int] of length bs, donor sample index j
        """
        switch_at = [self.n_primitives] * bs
        switch_to = list(range(bs))
        if self.cross_prob > 0 and bs > 1 and self.n_primitives >= 2:
            for i in range(bs):
                if pyrandom.random() < self.cross_prob:
                    switch_at[i] = pyrandom.randint(1, self.n_primitives - 1)
                    j = pyrandom.randint(0, bs - 1)
                    while j == i:
                        j = pyrandom.randint(0, bs - 1)
                    switch_to[i] = j
        return switch_at, switch_to

    def run_loop(self):
        if dist_util.is_main_process():
            print(f'[Scheduled Forcing] n_primitives={self.n_primitives}, '
                  f'context_len={self.context_len}, pred_len={self.pred_len}, '
                  f'max_replace_prob={self.max_replace_prob}, cross_prob={self.cross_prob}')
            if self.use_continuous_rollout:
                print(
                    '[Scheduled Forcing] mode=continuous, '
                    'max_replace_prob and warmup_steps are ignored'
                )
            else:
                print(f'[Scheduled Forcing] warmup_steps={self.num_warmup_steps}, '
                      f'ramp_steps={self.self_rollout_ramp_steps}')
            print(f'[Scheduled Forcing] full_len per sample = '
                  f'{self.context_len + self.n_primitives * self.pred_len} frames')
            if self.training_rtc:
                print(
                    f'[Training RTC] enabled, uniform delay in '
                    f'[0, {self.rtc_max_delay}] action frames'
                )
            print(f'[Scheduled Forcing] train steps: {self.num_steps}')

        for epoch in range(self.num_epochs):
            if hasattr(self.data.sampler, 'set_epoch'):
                self.data.sampler.set_epoch(epoch)
            if epoch % 100 == 0 and dist_util.is_main_process():
                print(f'Starting epoch {epoch}')
            for motion_full, cond in tqdm(
                self.data, disable=not dist_util.is_main_process()
            ):
                if self.total_step() >= self.num_steps:
                    break
                if not (not self.lr_anneal_steps or self.total_step() < self.lr_anneal_steps):
                    break

                self._log_batch_samples(cond, epoch)

                non_blocking = self.device.type == 'cuda'
                motion_full = motion_full.to(self.device, non_blocking=non_blocking)
                cond['y'] = {
                    key: val.to(self.device, non_blocking=non_blocking) if torch.is_tensor(val) else val
                    for key, val in cond['y'].items()
                }
                bs = motion_full.shape[0]
                text_list = cond['y'].get('text', None)
                token_list = cond['y'].get('tokens', None)
                motion_lengths = cond['y']['lengths']
                text_cache = self._encode_text_cache(text_list)

                p_replace = (
                    self._replacement_prob()
                    if not self.use_continuous_rollout
                    else 0.0
                )
                rollout_depths = (
                    self._sample_rollout_depths(bs)
                    if self.use_continuous_rollout
                    else None
                )
                train_primitives = self.n_primitives

                # ── cross-switch schedule for this batch ──
                switch_at, switch_to = self._compute_cross_schedule(bs)

                # ── gradient accumulation across N primitives ──
                self.mp_trainer.zero_grad()
                accumulated_log = {}
                prev_model_output = None
                prev_prefix = None
                # A sample enters self-rollout at most once per training
                # example.  After activation, all remaining primitives use
                # the model-generated prefix, making the rollout contiguous
                # instead of independently toggling each primitive.
                rollout_active = torch.zeros(bs, dtype=torch.bool, device=self.device)
                skip_step_due_to_nonfinite = False

                for k in range(train_primitives):
                    # ── per-sample GT and text (cross-switch aware) ──
                    # Before switch_at: GT/text from original sample i, primitive k
                    # From switch_at:   GT/text from donor sample j, re-indexed
                    #                   from j's beginning (k_new = k - switch_at)
                    gt_prefix_parts = []
                    gt_pred_parts = []
                    source_indices = []
                    primitive_starts = []
                    cur_text = list(text_list) if text_list is not None else None
                    cur_tokens = list(token_list) if token_list is not None else None

                    for i in range(bs):
                        switched = (k >= switch_at[i])
                        src = switch_to[i] if switched else i
                        k_eff = (k - switch_at[i]) if switched else k
                        s = k_eff * self.pred_len
                        source_indices.append(src)
                        primitive_starts.append(s)
                        gt_prefix_parts.append(
                            motion_full[src:src+1, ..., s:s + self.context_len])
                        gt_pred_parts.append(
                            motion_full[src:src+1, ...,
                                        s + self.context_len:
                                        s + self.context_len + self.pred_len])
                        if switched:
                            if cur_text is not None:
                                cur_text[i] = text_list[src]
                            if cur_tokens is not None:
                                cur_tokens[i] = token_list[src]

                    gt_prefix = torch.cat(gt_prefix_parts, dim=0)
                    gt_pred = torch.cat(gt_pred_parts, dim=0)
                    pred_mask, pred_lengths = self._build_primitive_mask(
                        motion_lengths, source_indices, primitive_starts)
                    cur_text_embed = self._select_text_cache(text_cache, source_indices)

                    # ── decide prefix: GT or model's own prediction ──
                    if k == 0 or prev_model_output is None:
                        cur_prefix = gt_prefix
                    else:
                        if self.use_continuous_rollout:
                            # Once a sample enters rollout, it stays on its
                            # model-generated prefix until its query depth.
                            use_model, _ = self._continuous_step_masks(
                                rollout_depths, k
                            )
                            use_model = use_model.view(bs, 1, 1, 1)
                        else:
                            rollout_active |= (
                                torch.rand(bs, device=self.device) < p_replace
                            )
                            use_model = rollout_active.view(bs, 1, 1, 1)
                        switched_samples = torch.as_tensor(
                            [k >= switch_at[i] for i in range(bs)],
                            dtype=torch.bool,
                            device=self.device,
                        ).view(bs, 1, 1, 1)
                        use_model = use_model | switched_samples
                        combined = torch.cat([prev_prefix, prev_model_output], dim=-1)
                        model_prefix = combined[..., -self.context_len:]
                        cur_prefix = torch.where(use_model, model_prefix, gt_prefix)

                    # ── build condition for this primitive ──
                    prim_cond = self._build_primitive_cond(
                        cur_prefix, cur_text, cur_tokens, bs,
                        pred_mask=pred_mask, pred_lengths=pred_lengths,
                        text_embed=cur_text_embed)
                    smooth_extras = self._compute_smooth_targets(
                        gt_pred, prim_cond['y']['mask'])
                    prim_cond['y'].update(smooth_extras)
                    if self.training_rtc:
                        prim_cond['y']['rtc_delay'] = torch.randint(
                            0,
                            self.rtc_max_delay + 1,
                            (bs,),
                            device=self.device,
                            dtype=torch.long,
                        )
                        prim_cond['y']['rtc_prefix_noise_std'] = (
                            self.rtc_prefix_noise_std
                        )
                    self._apply_cond_modifiers(prim_cond['y'])

                    # ── diffusion training step ──
                    t, weights = self.schedule_sampler.sample(bs, self.device)
                    sync_context = (
                        self.ddp_model.no_sync()
                        if self.use_ddp and k < train_primitives - 1
                        else contextlib.nullcontext()
                    )
                    with sync_context:
                        with self._forward_context():
                            terms = self.diffusion.training_losses(
                                self.ddp_model, gt_pred, t,
                                model_kwargs=prim_cond,
                                dataset=self.data.dataset,
                                return_model_output=True,
                            )

                        weighted_loss = terms['loss'] * weights
                        if self.use_continuous_rollout:
                            # Every sample contributes exactly once, at its
                            # sampled query depth. Earlier forwards are
                            # roll-in-only and do not train the model.
                            _, query_mask = self._continuous_step_masks(
                                rollout_depths, k
                            )
                            query_mask = query_mask.to(weighted_loss.dtype)
                            prim_loss = (weighted_loss * query_mask).mean()
                        else:
                            query_mask = None
                            prim_loss = weighted_loss.mean() / self.n_primitives
                        finite_on_all_ranks = dist_util.all_true(
                            torch.isfinite(prim_loss).item()
                        )
                        if not finite_on_all_ranks:
                            if dist_util.is_main_process():
                                logger.log(
                                    f"[warn] Non-finite primitive loss at step={self.total_step()} primitive={k}; "
                                    "skipping optimizer step on every rank for this batch."
                                )
                            skip_step_due_to_nonfinite = True
                        else:
                            self.mp_trainer.backward(prim_loss)

                    if skip_step_due_to_nonfinite:
                        break

                    # ── save detached x_0 for the next rollout prefix ──
                    prev_model_output = terms['model_output']
                    prev_prefix = cur_prefix.detach()

                    # ── accumulate logs ──
                    for key, val in terms.items():
                        if key == 'model_output':
                            continue
                        if self.use_continuous_rollout:
                            val = self._mask_per_sample(val, query_mask, bs)
                        if key not in accumulated_log:
                            accumulated_log[key] = val.detach()
                        else:
                            accumulated_log[key] = accumulated_log[key] + val.detach()

                # ── optimize ──
                if skip_step_due_to_nonfinite:
                    self.mp_trainer.zero_grad()
                else:
                    self.mp_trainer.optimize(self.opt)
                    self.update_average_model()
                self._anneal_lr()

                # ── logging ──
                if self.total_step() % self.log_interval == 0:
                    logger.logkv("step", self.total_step())
                    logger.logkv("samples", (self.total_step() + 1) * self.global_batch)
                    if self.use_continuous_rollout:
                        logger.logkv(
                            "rollout_depth_mean",
                            dist_util.reduce_mean(rollout_depths.float().mean()).item(),
                        )
                    else:
                        logger.logkv("p_replace", p_replace)
                    local_n_cross = sum(1 for sa in switch_at if sa < self.n_primitives)
                    global_n_cross = dist_util.reduce_sum(local_n_cross).item()
                    logger.logkv("n_cross", global_n_cross)
                    for key, val in accumulated_log.items():
                        local_avg = (
                            val.mean()
                            if self.use_continuous_rollout
                            else (val / self.n_primitives).mean()
                        )
                        global_avg = dist_util.reduce_mean(local_avg).item()
                        logger.logkv_mean(key, global_avg)
                    report_metrics = {}
                    for log_key, log_val in logger.get_current().dumpkvs().items():
                        if log_key == 'loss' and dist_util.is_main_process():
                            if self.use_continuous_rollout:
                                print(
                                    f'step[{self.total_step()}]: loss[{log_val:0.5f}] '
                                    f'rollout_depth_mean[{rollout_depths.float().mean().item():.2f}]'
                                )
                            else:
                                print(
                                    f'step[{self.total_step()}]: loss[{log_val:0.5f}] '
                                    f'p_replace[{p_replace:.3f}]'
                                )
                        if log_key in ['step', 'samples'] or '_q' in log_key:
                            continue
                        report_metrics[log_key] = log_val
                    if dist_util.is_main_process():
                        self.train_platform.report_scalars(
                            report_metrics,
                            iteration=self.total_step(),
                            group_name='Loss',
                        )

                if self.total_step() % self.save_interval == 0 and self.total_step() > 0:
                    dist_util.barrier()
                    if dist_util.is_main_process():
                        self.save()
                    dist_util.barrier()
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.total_step() > 0:
                        return

                self.step += 1

            if self.total_step() >= self.num_steps:
                break
            if not (not self.lr_anneal_steps or self.total_step() < self.lr_anneal_steps):
                break

        if self.total_step() % self.save_interval > self.save_interval / 2:
            dist_util.barrier()
            if dist_util.is_main_process():
                self.save()
            dist_util.barrier()

    def _log_batch_samples(self, cond, epoch):
        if self.batch_trace_file is None:
            return
        sample_indices = cond.get('y', {}).get('sample_index')
        motion_keys = cond.get('y', {}).get('db_key')
        if sample_indices is None or motion_keys is None:
            raise RuntimeError(
                "Batch sample tracing requires dataset return_keys with indices."
            )
        if len(sample_indices) != len(motion_keys):
            raise RuntimeError("Batch trace index/key counts differ.")
        self.batch_trace_file.write(json.dumps({
            "event": "batch",
            "step": self.total_step(),
            "epoch": epoch,
            "rank": self.args.rank,
            "sample_indices": sample_indices,
            "motion_keys": motion_keys,
        }, sort_keys=True) + "\n")
        self.batch_trace_file.flush()

    # ─── boilerplate (shared with other training loops) ─────────────

    def update_average_model(self):
        if self.args.use_ema:
            params = (self.model.parameters()
                      if self.use_fp16 else self.mp_trainer.master_params)
            for param, avg_param in zip(params, self.model_avg.parameters()):
                if not avg_param.requires_grad:
                    continue
                avg_param.data.mul_(self.args.avg_model_beta).add_(
                    param.data, alpha=1 - self.args.avg_model_beta)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = self.total_step() / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def total_step(self):
        return self.step + self.resume_step

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
                state_dict, state_dict_avg = state_dict['model'], state_dict['model_avg']
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
            bf.dirname(main_checkpoint), f"opt{self.resume_step:09}.pt")
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev())
            if self.use_fp16:
                if 'scaler' not in state_dict:
                    print("scaler state not found ... not loading it.")
                else:
                    self.scaler.load_state_dict(state_dict['scaler'])
                    state_dict = state_dict['opt']
            tgt_wd = self.opt.param_groups[0]['weight_decay']
            self.opt.load_state_dict(state_dict)
            for group in self.opt.param_groups:
                group['weight_decay'] = tgt_wd
            self.opt.param_groups[0]['capturable'] = True

    def ckpt_file_name(self):
        return f"model{(self.total_step()):09d}.pt"

    def _checkpoint_state(self):
        def del_clip(state_dict):
            for key in [key for key in state_dict if key.startswith('clip_model.')]:
                del state_dict[key]

        if self.use_fp16:
            state_dict = self.model.state_dict()
        else:
            state_dict = self.mp_trainer.master_params_to_state_dict(
                self.mp_trainer.master_params
            )
        del_clip(state_dict)
        if self.args.use_ema:
            state_dict_avg = self.model_avg.state_dict()
            del_clip(state_dict_avg)
            return {'model': state_dict, 'model_avg': state_dict_avg}
        return state_dict

    def find_resume_checkpoint(self) -> Optional[str]:
        matches = {file: re.match(r'model(\d+).pt$', file)
                   for file in os.listdir(self.args.save_dir)}
        models = {int(match.group(1)): file for file, match in matches.items() if match}
        return pjoin(self.args.save_dir, models[max(models)]) if models else None

    def save(self):
        def save_checkpoint():
            logger.log("saving model...")
            filename = self.ckpt_file_name()
            with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                torch.save(self._checkpoint_state(), f)

        save_checkpoint()
        with bf.BlobFile(
            bf.join(self.save_dir, f"opt{(self.total_step()):09d}.pt"), "wb"
        ) as f:
            torch.save(self.opt.state_dict(), f)


def parse_resume_step_from_filename(filename):
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0
