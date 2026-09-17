"""Deployment API for closed-loop ReactiveBFM planning."""

from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from reactivebfm.data.datasets import DatasetRegistry
from reactivebfm.model.motion_planner.factory import create_model_and_diffusion_smooth
from reactivebfm.utils.training.models import load_saved_model
from reactivebfm.utils.training.sampling import TextConditionCFGSampleModel


MOTION_DIM = 36
G1_29DOF_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_args(checkpoint: Path) -> Namespace:
    path = checkpoint.parent / "args.json"
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint arguments not found: {path}")
    with path.open(encoding="utf-8") as handle:
        return Namespace(**json.load(handle))


def _dataset_root(
    dataset_name: str,
    data_dir: str | Path | None,
    stats_dataset: str | None,
) -> Path:
    if not DatasetRegistry.is_supported_dataset(dataset_name):
        raise ValueError(f"Unsupported dataset: {dataset_name!r}")
    names = DatasetRegistry.parse_dataset_names(dataset_name)
    if len(names) > 1 and not stats_dataset:
        raise ValueError("Composite checkpoints require stats_dataset")
    selected = stats_dataset or names[0]
    if selected not in names:
        raise ValueError(
            f"stats_dataset={selected!r} is not part of dataset={dataset_name!r}"
        )
    if data_dir is None:
        root = Path(DatasetRegistry.get_data_path(selected))
    else:
        root = Path(data_dir).expanduser()
        if len(names) > 1:
            root = root / selected
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    return root


def _load_stats(
    dataset_name: str,
    data_dir: str | Path | None,
    hml_type: str | None,
    stats_dataset: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    root = _dataset_root(dataset_name, data_dir, stats_dataset)
    suffix = "" if not hml_type else f"_{hml_type}"
    mean_path = root / f"Mean{suffix}.npy"
    std_path = root / f"Std{suffix}.npy"
    if not mean_path.is_file() or not std_path.is_file():
        raise FileNotFoundError(
            f"Missing normalization files: {mean_path}, {std_path}"
        )
    mean = np.asarray(np.load(mean_path), dtype=np.float32)
    std = np.asarray(np.load(std_path), dtype=np.float32)
    if mean.shape != (MOTION_DIM,) or std.shape != (MOTION_DIM,):
        raise ValueError(
            f"Expected 36-D statistics, got mean={mean.shape}, std={std.shape}"
        )
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Normalization statistics must be finite with positive std")
    return mean, std


class _TensorRTDenoiser(nn.Module):
    def __init__(
        self,
        source_model: nn.Module,
        engine: torch.jit.ScriptModule,
        metadata: dict,
    ) -> None:
        super().__init__()
        self.clip_model = source_model.clip_model
        self.engine = engine
        self.text_tokens = int(metadata["text_tokens"])
        self.text_dim = int(metadata["text_dim"])
        self.tensorrt_cfg = bool(metadata["cfg"])
        self.cond_mask_prob = float(source_model.cond_mask_prob)
        self.engine_dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }[metadata["precision"]]

    def encode_text(self, texts):
        encoded, valid_mask = self.clip_model(texts)
        return encoded.permute(1, 0, 2), ~valid_mask

    def _pad_text(self, text_embed, text_mask):
        if text_embed.shape[0] > self.text_tokens:
            raise ValueError(
                f"Text encoder produced {text_embed.shape[0]} tokens; "
                f"engine supports {self.text_tokens}"
            )
        if text_embed.shape[2] != self.text_dim:
            raise ValueError(
                f"Engine expects text_dim={self.text_dim}, got {text_embed.shape[2]}"
            )
        missing = self.text_tokens - text_embed.shape[0]
        if missing:
            text_embed = torch.cat(
                [
                    text_embed,
                    torch.zeros(
                        missing,
                        *text_embed.shape[1:],
                        device=text_embed.device,
                        dtype=text_embed.dtype,
                    ),
                ]
            )
            text_mask = torch.cat(
                [
                    text_mask,
                    torch.ones(
                        text_mask.shape[0],
                        missing,
                        device=text_mask.device,
                        dtype=torch.bool,
                    ),
                ],
                dim=1,
            )
        return text_embed, text_mask

    def forward(self, x, timesteps, y=None):
        if y is None:
            raise ValueError("TensorRT planner expects conditioning dict y")
        text_embed, text_mask = y["text_embed"]
        if y.get("text_uncond", False):
            text_embed = torch.zeros_like(text_embed)
        text_embed, text_mask = self._pad_text(text_embed, text_mask)
        flow_time = y.get("flow_time")
        if flow_time is None:
            flow_time = torch.zeros(
                x.shape[0], device=x.device, dtype=self.engine_dtype
            )
        scale = y.get("scale")
        if scale is None:
            scale = torch.ones(x.shape[0], device=x.device, dtype=self.engine_dtype)
        output = self.engine(
            x.to(dtype=self.engine_dtype),
            timesteps,
            y["prefix"].to(dtype=self.engine_dtype),
            y["mask"].to(dtype=torch.bool),
            text_embed.to(dtype=self.engine_dtype),
            text_mask.to(dtype=torch.bool),
            flow_time.to(dtype=self.engine_dtype),
            scale.to(dtype=self.engine_dtype),
        )
        return output.to(dtype=x.dtype)


def _load_tensorrt_denoiser(
    source_model: nn.Module,
    engine_path: str | Path,
    device: torch.device,
    train_args: Namespace,
    checkpoint_path: Path,
) -> _TensorRTDenoiser:
    path = Path(engine_path).expanduser().resolve()
    metadata_path = path.with_suffix(path.suffix + ".json")
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"TensorRT engine or metadata missing: {path}")
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    expected = {
        "format_version": 3,
        "planner_input_contract": "reactivebfm_dit_qpos36_v1",
        "planner_arch": "dit",
        "model_type": str(train_args.model_type),
        "context_len": int(train_args.context_len),
        "pred_len": int(train_args.pred_len),
        "dit_role_embedding": bool(train_args.dit_role_embedding),
        "dit_rope": bool(train_args.dit_rope),
    }
    mismatched = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatched:
        raise ValueError(f"TensorRT engine does not match checkpoint: {mismatched}")
    expected_hash = metadata.get("checkpoint_sha256")
    if expected_hash and _file_sha256(checkpoint_path) != expected_hash:
        raise ValueError("TensorRT engine checkpoint hash does not match model_path")
    try:
        import torch_tensorrt  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Loading the planner engine requires torch_tensorrt") from exc
    engine = torch.jit.load(str(path), map_location=device).eval()
    return _TensorRTDenoiser(source_model, engine, metadata).to(device).eval()


class ReactiveBFMPlanner:
    """Generate raw qpos36 action chunks from raw qpos36 history."""

    def __init__(
        self,
        *,
        model: nn.Module,
        process: Any,
        train_args: Namespace,
        mean: np.ndarray,
        std: np.ndarray,
        device: torch.device,
        guidance_scale: float,
    ) -> None:
        self.model = model
        self.process = process
        self.train_args = train_args
        self.mean = mean
        self.std = std
        self.device = device
        self.context_len = int(train_args.context_len)
        self.pred_len = int(train_args.pred_len)
        self.raw_context_len = self.context_len
        self.raw_pred_len = self.pred_len
        self.guidance_scale = float(guidance_scale)
        if self.guidance_scale != 1.0 and not getattr(model, "tensorrt_cfg", False):
            if model.cond_mask_prob <= 0:
                raise ValueError(
                    "guidance_scale != 1 requires training with cond_mask_prob > 0"
                )
            self.sampler_model = TextConditionCFGSampleModel(model)
        else:
            self.sampler_model = model
        self._cached_prompt: str | None = None
        self._cached_text_embed: tuple[torch.Tensor, torch.Tensor] | None = None

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        device: str | torch.device = "cuda",
        data_dir: str | Path | None = None,
        dataset: str | None = None,
        hml_type: str | None = None,
        stats_dataset: str | None = None,
        use_ema: bool | None = None,
        guidance_scale: float | None = None,
        compile_model: bool = False,
        tensorrt_engine: str | Path | None = None,
    ) -> "ReactiveBFMPlanner":
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        train_args = _load_args(checkpoint_path)
        if str(train_args.planner_arch) != "dit":
            raise ValueError("Deployment supports the current DiT action head only")
        if int(getattr(train_args, "frame_dim", MOTION_DIM)) != MOTION_DIM or int(
            getattr(train_args, "nfeats", 1)
        ) != 1:
            raise ValueError("Deployment requires the raw G1 qpos36 representation")
        dataset_name = dataset or str(train_args.dataset)
        mean, std = _load_stats(dataset_name, data_dir, hml_type, stats_dataset)
        device_obj = torch.device(device)
        model, process = create_model_and_diffusion_smooth(train_args, data=None)
        model.eval().to(device_obj)
        load_saved_model(
            model,
            str(checkpoint_path),
            use_avg=bool(
                getattr(train_args, "use_ema", False)
                if use_ema is None
                else use_ema
            ),
        )
        if tensorrt_engine is not None:
            model = _load_tensorrt_denoiser(
                model, tensorrt_engine, device_obj, train_args, checkpoint_path
            )
        if compile_model and tensorrt_engine is not None:
            raise ValueError("compile_model and tensorrt_engine are mutually exclusive")
        if compile_model:
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        default_scale = getattr(
            train_args,
            "gen_guidance_param",
            getattr(train_args, "guidance_param", 1.0),
        )
        return cls(
            model=model,
            process=process,
            train_args=train_args,
            mean=mean,
            std=std,
            device=device_obj,
            guidance_scale=float(
                default_scale if guidance_scale is None else guidance_scale
            ),
        )

    def _text_embedding(self, prompt: str | None):
        prompt = "" if prompt is None else str(prompt)
        if prompt != self._cached_prompt or self._cached_text_embed is None:
            encoded = self.model.encode_text([prompt])
            self._cached_prompt = prompt
            self._cached_text_embed = tuple(item.detach() for item in encoded)
        return self._cached_text_embed

    @torch.inference_mode()
    def generate_chunk(
        self,
        prefix_motion36: np.ndarray,
        prompt: str | None = None,
    ) -> np.ndarray:
        prefix_raw = np.asarray(prefix_motion36, dtype=np.float32)
        expected = (self.context_len, MOTION_DIM)
        if prefix_raw.shape != expected:
            raise ValueError(f"Expected prefix shape {expected}, got {prefix_raw.shape}")
        if not np.isfinite(prefix_raw).all():
            raise ValueError("Prefix contains NaN or Inf")
        prefix_norm = (prefix_raw - self.mean) / self.std
        prefix = torch.from_numpy(prefix_norm.T[None, :, None, :]).to(self.device)
        condition = {
            "prefix": prefix,
            "mask": torch.ones(
                (1, 1, 1, self.pred_len), dtype=torch.bool, device=self.device
            ),
            "lengths": torch.full(
                (1,), self.pred_len, dtype=torch.long, device=self.device
            ),
            "text_embed": self._text_embedding(prompt),
        }
        if self.guidance_scale != 1.0:
            condition["scale"] = torch.full(
                (1,), self.guidance_scale, device=self.device
            )
        sampled = self.process.p_sample_loop_gt_prefix(
            self.sampler_model,
            (1, MOTION_DIM, 1, self.pred_len),
            model_kwargs={"y": condition},
            device=self.device,
            progress=False,
        )
        output = sampled[0, :, 0, :].T.cpu().numpy()
        return (output * self.std + self.mean).astype(np.float32)


__all__ = ["G1_29DOF_JOINT_NAMES", "MOTION_DIM", "ReactiveBFMPlanner"]
