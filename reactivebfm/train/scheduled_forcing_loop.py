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
import functools
import hashlib
import json
import os
import random as pyrandom
import warnings
from types import SimpleNamespace
import numpy as np

import re
from os.path import join as pjoin
from typing import Optional

import blobfile as bf
import torch
import torch.distributed as dist
from torch.optim import AdamW

from reactivebfm.model.motion_planner.objectives.diffusion import logger
from reactivebfm.utils.runtime import distributed as dist_util
from reactivebfm.model.motion_planner.objectives.diffusion.fp16_util import (
    MixedPrecisionTrainer,
)
from reactivebfm.model.motion_planner.objectives.diffusion.resample import (
    LossAwareSampler,
    UniformSampler,
    create_named_schedule_sampler,
)
from tqdm import tqdm
from reactivebfm.data.datasets import lengths_to_mask
from reactivebfm.utils.training.models import load_model_wo_clip
from reactivebfm.data.motion import HML_ROOT_HORIZONTAL_MASK
from reactivebfm.utils.training.losses import masked_motion_metrics

INITIAL_LOG_LOSS_SCALE = 20.0


class ScheduledForcingLoop:
    def __init__(self, args, train_platform, model, diffusion, data, eval_loaders=None):
        self.args = args
        self.train_platform = train_platform
        self.model = model
        self.model_avg = None
        if self.args.use_ema:
            self.model_avg = copy.deepcopy(self.model)
        self.diffusion = diffusion
        self.cond_mode = model.cond_mode
        self.data = data
        self.eval_loaders = dict(eval_loaders or {})
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
        if self.use_continuous_rollout and self.cross_prob > 0.0:
            raise ValueError(
                "cross_prob is not supported in continuous self-rollout mode; "
                "run the two curricula separately."
            )
        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size_global
        self.num_steps = args.num_steps
        self.val_eval_interval = int(getattr(args, 'val_eval_interval', 0) or 0)
        self.test_eval_interval = int(getattr(args, 'test_eval_interval', 0) or 0)
        self.eval_seed = int(getattr(args, 'eval_seed', 12345))
        self.eval_primary_metric = str(
            getattr(args, 'eval_primary_metric', 'auto') or 'auto'
        )
        if self.eval_primary_metric == 'auto':
            self.eval_primary_metric = (
                'tmr_r_at_3' if getattr(args, 'g1_tmr_checkpoint', '') else 'mse'
            )
        self.best_val_score = (
            float('-inf') if self.eval_primary_metric == 'tmr_r_at_3' else float('inf')
        )
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
        self._load_best_metrics()
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

        self.use_motion_tokenizer_latent = False

        # Held-out evaluators are intentionally outside the public baseline.
        self.g1_tmr_evaluator = None

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

    def _setup_motion_tokenizer(self):
        ckpt_path = getattr(self.args, 'motion_tokenizer_ckpt', '')
        if not ckpt_path:
            raise ValueError("motion_tokenizer_ckpt is required for tokenizer latent training.")
        ckpt = torch.load(ckpt_path, map_location='cpu')
        cfg = tokenizer_config_from_payload(ckpt['cfg'])
        tokenizer = ReactiveMotionVQVAE(cfg).to(self.device)
        tokenizer.load_state_dict(ckpt['model'], strict=True)
        tokenizer.eval()
        for param in tokenizer.parameters():
            param.requires_grad_(False)
        self.motion_tokenizer = tokenizer
        self.motion_tokenizer_frames_per_token = int(tokenizer.frames_per_token)
        expected = int(getattr(self.args, 'motion_tokenizer_frames_per_token', self.motion_tokenizer_frames_per_token))
        if expected != self.motion_tokenizer_frames_per_token:
            raise ValueError(
                f"Tokenizer frames_per_token mismatch: args={expected}, checkpoint={self.motion_tokenizer_frames_per_token}"
            )
        self._setup_motion_latent_normalization()
        logger.log(
            f"[Tokenizer Latent] Loaded frozen tokenizer from {ckpt_path}; "
            f"frames_per_token={self.motion_tokenizer_frames_per_token}, code_dim={cfg.code_dim}, "
            f"latent_norm={getattr(self.args, 'motion_tokenizer_latent_norm', 'codebook')}, "
            f"decode_quantize={self.motion_tokenizer_decode_quantize}"
        )

    def _setup_motion_latent_normalization(self):
        mode = getattr(self.args, 'motion_tokenizer_latent_norm', 'codebook')
        if mode == 'none':
            self.motion_latent_mean = None
            self.motion_latent_std = None
            return
        if mode != 'codebook':
            raise ValueError(f"Unknown motion_tokenizer_latent_norm={mode}")
        embed = self.motion_tokenizer.quantizer.embed.detach().float()
        mean = embed.mean(dim=1).reshape(1, -1, 1, 1)
        std = embed.std(dim=1).reshape(1, -1, 1, 1).clamp_min(1.0e-6)
        self.motion_latent_mean = mean.to(self.device)
        self.motion_latent_std = std.to(self.device)
        logger.log(
            f"[Tokenizer Latent] codebook normalization: mean_abs={mean.abs().mean().item():.4f}, "
            f"std_mean={std.mean().item():.4f}, std_min={std.min().item():.4f}, std_max={std.max().item():.4f}"
        )

    def _normalize_motion_latents(self, latent_motion):
        if self.motion_latent_mean is None or self.motion_latent_std is None:
            return latent_motion
        mean = self.motion_latent_mean.to(device=latent_motion.device, dtype=latent_motion.dtype)
        std = self.motion_latent_std.to(device=latent_motion.device, dtype=latent_motion.dtype)
        return (latent_motion - mean) / std

    def _denormalize_motion_latents(self, latent_motion):
        if self.motion_latent_mean is None or self.motion_latent_std is None:
            return latent_motion
        mean = self.motion_latent_mean.to(device=latent_motion.device, dtype=latent_motion.dtype)
        std = self.motion_latent_std.to(device=latent_motion.device, dtype=latent_motion.dtype)
        return latent_motion * std + mean

    @torch.no_grad()
    def _quantize_motion_latents(self, latent_motion):
        q, _indices, _commit, _perplexity = self.motion_tokenizer.quantizer(latent_motion.squeeze(2))
        return q.unsqueeze(2)

    @torch.no_grad()
    def _encode_motion_latents(self, motion_full):
        if self.motion_tokenizer is None:
            return motion_full
        if motion_full.ndim != 4 or motion_full.shape[2] != 1:
            raise ValueError(f"Expected raw motion shape [B,D,1,T], got {tuple(motion_full.shape)}")
        motion_seq = motion_full.squeeze(2).transpose(1, 2).contiguous()
        z = self.motion_tokenizer.encoder(motion_seq)
        q, _indices, _commit, _perplexity = self.motion_tokenizer.quantizer(z)
        q = q.detach().unsqueeze(2)
        return self._normalize_motion_latents(q)

    @torch.no_grad()
    def _decode_motion_latents(self, latent_motion, target_len=None):
        if self.motion_tokenizer is None:
            return latent_motion
        if latent_motion.ndim != 4 or latent_motion.shape[2] != 1:
            raise ValueError(f"Expected latent motion shape [B,C,1,T], got {tuple(latent_motion.shape)}")
        if target_len is None:
            target_len = latent_motion.shape[-1] * self.motion_tokenizer_frames_per_token
        latent_motion = self._denormalize_motion_latents(latent_motion)
        if self.motion_tokenizer_decode_quantize:
            latent_motion = self._quantize_motion_latents(latent_motion)
        decoded = self.motion_tokenizer.decoder(latent_motion.squeeze(2), int(target_len))
        return decoded.transpose(1, 2).unsqueeze(2).contiguous()

    @torch.no_grad()
    def _decoded_raw_motion_metrics(self, pred_latent, gt_raw, raw_mask, prefix_latent=None):
        prefix_raw = None
        if prefix_latent is not None:
            # Match inference/generation: decode prefix+prediction together, then crop the
            # prediction frames. The temporal decoder is convolutional, so decoding the
            # prediction chunk alone gives a pessimistic and inconsistent metric.
            full_latent = torch.cat([prefix_latent, pred_latent], dim=-1)
            full_raw = self._decode_motion_latents(
                full_latent,
                target_len=self.motion_raw_context_len + gt_raw.shape[-1],
            )
            prefix_raw = full_raw[..., :self.motion_raw_context_len]
            pred_raw = full_raw[..., self.motion_raw_context_len:self.motion_raw_context_len + gt_raw.shape[-1]]
        else:
            pred_raw = self._decode_motion_latents(pred_latent, target_len=gt_raw.shape[-1])
        return masked_motion_metrics(pred_raw, gt_raw, raw_mask, prefix=prefix_raw)

    def _latent_lengths(self, raw_lengths, max_latent_len):
        lengths = torch.div(raw_lengths, self.motion_tokenizer_frames_per_token, rounding_mode='floor')
        return lengths.clamp(min=0, max=max_latent_len).to(dtype=torch.long)

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

    def _load_best_metrics(self):
        path = os.path.join(self.args.save_dir, 'best_metrics.json')
        if not os.path.isfile(path):
            return
        try:
            with open(path, 'r', encoding='utf-8') as handle:
                payload = json.load(handle)
            saved_metric = str(payload.get('selection_metric', 'metrics_val/mse'))
            expected_metric = f'metrics_val/{self.eval_primary_metric}'
            if saved_metric != expected_metric:
                logger.log(
                    f"[eval] Ignoring best score selected by {saved_metric}; "
                    f"current selection uses {expected_metric}."
                )
                return
            self.best_val_score = float(payload[expected_metric])
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
            logger.log(f"[eval] Ignoring invalid {path}: {exc}")

    def _eval_noise(self, keys, split, primitive, shape, dtype):
        samples = []
        for key in keys:
            material = f"{self.eval_seed}:{split}:{key}:{primitive}".encode('utf-8')
            seed = int.from_bytes(hashlib.blake2b(material, digest_size=8).digest(), 'little')
            generator = torch.Generator(device='cpu')
            generator.manual_seed(seed & ((1 << 63) - 1))
            samples.append(torch.randn(shape[1:], generator=generator, dtype=dtype))
        return torch.stack(samples, dim=0).to(self.device)

    def _qpos_stats_for_keys(self, loader, keys):
        """Return per-sample qpos normalization stats, including composites."""
        dataset = loader.dataset
        if hasattr(dataset, 'datasets') and hasattr(dataset, 'dataset_names'):
            components = {
                name: component
                for name, component in zip(dataset.dataset_names, dataset.datasets)
            }
            means, stds = [], []
            for key in keys:
                if ':' not in key:
                    raise ValueError(
                        "Composite G1-TMR evaluation requires namespaced motion keys."
                    )
                dataset_name = key.split(':', 1)[0]
                component = components[dataset_name]
                means.append(torch.as_tensor(component.mean))
                stds.append(torch.as_tensor(component.std))
            mean = torch.stack(means, dim=0)
            std = torch.stack(stds, dim=0)
        else:
            mean = torch.as_tensor(dataset.mean).unsqueeze(0).expand(len(keys), -1)
            std = torch.as_tensor(dataset.std).unsqueeze(0).expand(len(keys), -1)
        return (
            mean.to(self.device, dtype=torch.float32),
            std.to(self.device, dtype=torch.float32),
        )

    def _denormalize_eval_qpos(self, normalized_motion, mean, std):
        qpos = normalized_motion.squeeze(2).transpose(1, 2).float()
        qpos = qpos * std[:, None, :] + mean[:, None, :]
        if getattr(self.args, 'relative_root_xy', False):
            normalized = normalized_motion.squeeze(2).transpose(1, 2).float()
            qpos[..., :2] = normalized[..., :2] * std[:, None, :2]
        return qpos

    @staticmethod
    def _merge_g1_tmr_records(records):
        if not records:
            return {}
        array_fields = (
            'text', 'generated', 'real', 'generated_feet', 'real_feet', 'lengths'
        )
        merged = {
            name: np.concatenate([record[name] for record in records], axis=0)
            for name in array_fields
        }
        merged['positive_ids'] = [
            positive_id for record in records for positive_id in record['positive_ids']
        ]
        return merged

    @torch.no_grad()
    def evaluate_split(self, split):
        loader = self.eval_loaders[split]
        eval_model = self.model_avg if self.args.use_ema else self.model
        was_training = eval_model.training
        eval_model.eval()
        metric_sums = {}
        metric_counts = {}
        local_samples = 0
        g1_tmr_records = []

        for batch_idx, (raw_motion_full, cond) in enumerate(loader):
            non_blocking = self.device.type == 'cuda'
            raw_motion_full = raw_motion_full.to(self.device, non_blocking=non_blocking)
            raw_lengths = cond['y']['lengths'].to(self.device, non_blocking=non_blocking)
            text_list = cond['y'].get('text')
            token_list = cond['y'].get('tokens')
            keys = cond['y'].get('db_key')
            if keys is None:
                raise RuntimeError("Held-out evaluation requires motion keys.")

            if self.use_motion_tokenizer_latent:
                motion_full = self._encode_motion_latents(raw_motion_full)
                lengths = self._latent_lengths(raw_lengths, motion_full.shape[-1])
            else:
                motion_full = raw_motion_full
                lengths = raw_lengths

            bs = motion_full.shape[0]
            text_cache = None
            if text_list is not None and 'text' in self.cond_mode:
                text_cache = eval_model.encode_text(text_list)
            prefix = motion_full[..., :self.context_len]
            initial_prefix = prefix
            generated = []

            for primitive in range(self.n_primitives):
                starts = [primitive * self.pred_len] * bs
                source_indices = list(range(bs))
                pred_mask, pred_lengths = self._build_primitive_mask(
                    lengths, source_indices, starts
                )
                prim_cond = self._build_primitive_cond(
                    prefix,
                    text_list,
                    token_list,
                    bs,
                    pred_mask=pred_mask,
                    pred_lengths=pred_lengths,
                    text_embed=text_cache,
                )
                self._apply_cond_modifiers(prim_cond['y'])
                noise = self._eval_noise(
                    keys,
                    split,
                    primitive,
                    (bs, motion_full.shape[1], motion_full.shape[2], self.pred_len),
                    motion_full.dtype,
                )
                with self._forward_context():
                    pred = self.diffusion.sample_loop(
                        eval_model,
                        noise.shape,
                        noise=noise,
                        model_kwargs=prim_cond,
                        device=self.device,
                    )
                pred = pred.to(dtype=motion_full.dtype)
                generated.append(pred)
                prefix = torch.cat([prefix, pred], dim=-1)[..., -self.context_len:]

            pred_full = torch.cat(generated, dim=-1)
            target = motion_full[
                ..., self.context_len:self.context_len + self.n_primitives * self.pred_len
            ]
            valid_lengths = (
                lengths - self.context_len
            ).clamp(min=0, max=target.shape[-1])
            mask = lengths_to_mask(valid_lengths, target.shape[-1]).unsqueeze(1).unsqueeze(1)

            if self.use_motion_tokenizer_latent:
                latent_full = torch.cat([initial_prefix, pred_full], dim=-1)
                decoded = self._decode_motion_latents(
                    latent_full,
                    target_len=self.motion_raw_context_len + self.n_primitives * self.motion_raw_pred_len,
                )
                pred_for_metrics = decoded[..., self.motion_raw_context_len:]
                target_for_metrics = raw_motion_full[
                    ...,
                    self.motion_raw_context_len:
                    self.motion_raw_context_len + self.n_primitives * self.motion_raw_pred_len,
                ]
                prefix_for_metrics = decoded[..., :self.motion_raw_context_len]
                raw_valid = (
                    raw_lengths - self.motion_raw_context_len
                ).clamp(min=0, max=target_for_metrics.shape[-1])
                mask = lengths_to_mask(raw_valid, target_for_metrics.shape[-1]).unsqueeze(1).unsqueeze(1)
            else:
                pred_for_metrics = pred_full
                target_for_metrics = target
                prefix_for_metrics = initial_prefix

            metrics = masked_motion_metrics(
                pred_for_metrics,
                target_for_metrics,
                mask,
                prefix=prefix_for_metrics,
            )
            for name, values in metrics.items():
                short_name = name.removeprefix('metrics/')
                if short_name.startswith('mse_acc'):
                    eligible = raw_valid >= 3 if self.use_motion_tokenizer_latent else valid_lengths >= 3
                elif short_name.startswith('mse_velocity'):
                    eligible = raw_valid >= 2 if self.use_motion_tokenizer_latent else valid_lengths >= 2
                else:
                    eligible = raw_valid >= 1 if self.use_motion_tokenizer_latent else valid_lengths >= 1
                metric_sums[short_name] = (
                    metric_sums.get(short_name, 0.0)
                    + values[eligible].double().sum()
                )
                metric_counts[short_name] = (
                    metric_counts.get(short_name, 0) + eligible.sum()
                )
            local_samples += bs

            if self.g1_tmr_evaluator is not None:
                if text_list is None:
                    raise RuntimeError("G1-TMR evaluation requires text captions.")
                metric_lengths = raw_valid if self.use_motion_tokenizer_latent else valid_lengths
                mean, std = self._qpos_stats_for_keys(loader, keys)
                generated_qpos = self._denormalize_eval_qpos(
                    pred_for_metrics, mean, std
                )
                real_qpos = self._denormalize_eval_qpos(
                    target_for_metrics, mean, std
                )
                g1_tmr_records.append(
                    self.g1_tmr_evaluator.encode_batch(
                        generated_qpos,
                        real_qpos,
                        metric_lengths,
                        list(text_list),
                        list(keys),
                    )
                )

            if batch_idx and batch_idx % 100 == 0 and dist_util.is_main_process():
                logger.log(f"[eval:{split}] processed at least {batch_idx * loader.batch_size} samples")

        global_samples = int(dist_util.reduce_sum(local_samples).item())
        if global_samples == 0:
            raise RuntimeError(f"The {split} split contains no evaluable samples.")
        results = {}
        for name, total in metric_sums.items():
            count = int(dist_util.reduce_sum(metric_counts[name]).item())
            if count > 0:
                results[name] = dist_util.reduce_sum(total).item() / count
        if self.g1_tmr_evaluator is not None:
            local_records = self._merge_g1_tmr_records(g1_tmr_records)
            if dist.is_initialized():
                gathered_records = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(gathered_records, local_records)
            else:
                gathered_records = [local_records]
            if dist_util.is_main_process():
                results.update(self.g1_tmr_evaluator.aggregate(gathered_records))
        results['num_samples'] = global_samples
        if was_training:
            eval_model.train()
        return results

    def _run_evaluation(self, split):
        dist_util.barrier()
        logger.log(f"[eval:{split}] starting full generative rollout at step {self.total_step()}")
        results = self.evaluate_split(split)
        if dist_util.is_main_process():
            group = f"metrics_{split}"
            reported_results = results
            if self.g1_tmr_evaluator is not None and not getattr(
                self.args, 'eval_log_diagnostics', False
            ):
                headline = {
                    'mse',
                    'tmr_r_at_3',
                    'tmr_fid',
                    'foot_skate_cm_s',
                    'grounding_score',
                    'real_foot_skate_cm_s',
                    'real_grounding_score',
                    'num_samples',
                }
                reported_results = {
                    name: value for name, value in results.items() if name in headline
                }
            self.train_platform.report_scalars(
                dict(sorted(reported_results.items())),
                iteration=self.total_step(),
                group_name=group,
            )
            logger.log(
                f"[eval:{split}] step={self.total_step()} samples={results['num_samples']} "
                f"mse={results['mse']:.6f}"
                + (
                    f" r@3={results['tmr_r_at_3']:.6f} "
                    f"fid={results['tmr_fid']:.6f} "
                    f"skate={results['foot_skate_cm_s']:.3f}cm/s"
                    if 'tmr_r_at_3' in results
                    else ''
                )
            )
            if split == 'val':
                if self.eval_primary_metric not in results:
                    raise RuntimeError(
                        f"Validation selection metric {self.eval_primary_metric!r} "
                        f"was not produced; available={sorted(results)}"
                    )
                value = float(results[self.eval_primary_metric])
                maximize = self.eval_primary_metric == 'tmr_r_at_3'
                improved = value > self.best_val_score if maximize else value < self.best_val_score
                if improved:
                    self.best_val_score = value
                    self.save_best(results)
        dist_util.barrier()

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
                raw_motion_lengths = cond['y']['lengths']
                raw_motion_full = motion_full
                if self.use_motion_tokenizer_latent:
                    motion_full = self._encode_motion_latents(motion_full)
                    motion_lengths = self._latent_lengths(raw_motion_lengths, motion_full.shape[-1])
                else:
                    motion_lengths = raw_motion_lengths
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
                    raw_gt_pred_parts = []
                    source_indices = []
                    primitive_starts = []
                    raw_primitive_starts = []
                    cur_text = list(text_list) if text_list is not None else None
                    cur_tokens = list(token_list) if token_list is not None else None

                    for i in range(bs):
                        switched = (k >= switch_at[i])
                        src = switch_to[i] if switched else i
                        k_eff = (k - switch_at[i]) if switched else k
                        s = k_eff * self.pred_len
                        raw_s = k_eff * self.motion_raw_pred_len
                        source_indices.append(src)
                        primitive_starts.append(s)
                        raw_primitive_starts.append(raw_s)
                        gt_prefix_parts.append(
                            motion_full[src:src+1, ..., s:s + self.context_len])
                        gt_pred_parts.append(
                            motion_full[src:src+1, ...,
                                        s + self.context_len:
                                        s + self.context_len + self.pred_len])
                        if self.use_motion_tokenizer_latent:
                            raw_gt_pred_parts.append(
                                raw_motion_full[src:src+1, ...,
                                                raw_s + self.motion_raw_context_len:
                                                raw_s + self.motion_raw_context_len + self.motion_raw_pred_len])
                        if switched:
                            if cur_text is not None:
                                cur_text[i] = text_list[src]
                            if cur_tokens is not None:
                                cur_tokens[i] = token_list[src]

                    gt_prefix = torch.cat(gt_prefix_parts, dim=0)
                    gt_pred = torch.cat(gt_pred_parts, dim=0)
                    pred_mask, pred_lengths = self._build_primitive_mask(
                        motion_lengths, source_indices, primitive_starts)
                    raw_gt_pred = None
                    raw_pred_mask = None
                    if self.use_motion_tokenizer_latent:
                        raw_gt_pred = torch.cat(raw_gt_pred_parts, dim=0)
                        raw_valid_lengths = []
                        for src, raw_start in zip(source_indices, raw_primitive_starts):
                            motion_len = int(raw_motion_lengths[src].item())
                            target_start = raw_start + self.motion_raw_context_len
                            valid = max(0, min(self.motion_raw_pred_len, motion_len - target_start))
                            raw_valid_lengths.append(valid)
                        raw_pred_lengths = torch.tensor(raw_valid_lengths, dtype=torch.long, device=self.device)
                        raw_frame_ids = torch.arange(self.motion_raw_pred_len, device=self.device).view(1, 1, 1, -1)
                        raw_pred_mask = raw_frame_ids < raw_pred_lengths.view(-1, 1, 1, 1)
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

                        if self.use_motion_tokenizer_latent:
                            raw_metrics = self._decoded_raw_motion_metrics(
                                terms['model_output'], raw_gt_pred, raw_pred_mask, prefix_latent=cur_prefix)
                            terms = {key: value for key, value in terms.items() if not key.startswith('metrics/')}
                            terms.update(raw_metrics)

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
                        self.model.eval()
                        self.generate_during_training()
                        self.model.train()
                    dist_util.barrier()
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.total_step() > 0:
                        return

                if (
                    self.val_eval_interval > 0
                    and self.total_step() > 0
                    and self.total_step() % self.val_eval_interval == 0
                ):
                    self._run_evaluation('val')
                if (
                    self.test_eval_interval > 0
                    and self.total_step() > 0
                    and self.total_step() % self.test_eval_interval == 0
                ):
                    self._run_evaluation('test')

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
                    self.model_avg.load_state_dict(self.model.state_dict(), strict=False)
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
                        self.model.state_dict(), strict=False)

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

    def save_best(self, metrics):
        value = metrics[self.eval_primary_metric]
        logger.log(
            f"[eval:val] new best {self.eval_primary_metric}={value:.6f}; "
            "saving best.pt"
        )
        with bf.BlobFile(bf.join(self.save_dir, 'best.pt'), 'wb') as handle:
            torch.save(self._checkpoint_state(), handle)
        payload = {
            'step': self.total_step(),
            'selection_metric': f'metrics_val/{self.eval_primary_metric}',
            **{f'metrics_val/{key}': value for key, value in metrics.items()},
        }
        with open(os.path.join(self.save_dir, 'best_metrics.json'), 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    def generate_during_training(self):
        if not self.args.gen_during_training:
            return
        raise NotImplementedError(
            "gen_during_training needs a standard sampler; the legacy sampler was removed."
        )

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
