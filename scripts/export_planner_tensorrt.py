#!/usr/bin/env python3
"""Export a ReactiveBFM planner denoiser with Torch-TensorRT.

The text encoder deliberately remains in PyTorch. Its output is cached by the
deployment planner, while the repeatedly evaluated DiT/MDM denoiser runs in
TensorRT. Text CFG is fused into one batched denoiser invocation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
import sys
import time
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reactivebfm.model.motion_planner.factory import create_model_and_diffusion_smooth  # noqa: E402
from reactivebfm.utils.training.models import load_saved_model  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_checkpoint(source: Path, args_path: Path, output: Path) -> tuple[Path, Path, str]:
    """Copy a stable checkpoint generation before a long TensorRT build."""

    snapshots_root = output.parent / "planner_sources"
    snapshots_root.mkdir(parents=True, exist_ok=True)
    initial_hash = _sha256(source)
    existing_dir = snapshots_root / f"{source.stem}-{initial_hash[:12]}"
    existing_checkpoint = existing_dir / "model.pt"
    existing_args = existing_dir / "args.json"
    if existing_checkpoint.is_file() and existing_args.is_file():
        if _sha256(existing_checkpoint) != initial_hash:
            raise RuntimeError(f"Existing snapshot hash mismatch: {existing_checkpoint}")
        if existing_args.read_bytes() != args_path.read_bytes():
            raise RuntimeError(f"Existing snapshot args mismatch: {existing_args}")
        print(f"[Snapshot] Reusing {existing_checkpoint} sha256={initial_hash}")
        return existing_checkpoint, existing_args, initial_hash

    for attempt in range(1, 4):
        temporary = snapshots_root / f".{source.stem}.snapshot-{time.time_ns()}-{attempt}.pt"
        shutil.copy2(source, temporary)
        snapshot_hash = _sha256(temporary)
        source_hash = _sha256(source)
        if snapshot_hash != source_hash:
            temporary.unlink()
            print(f"[Snapshot] {source.name} changed during copy; retrying ({attempt}/3)")
            continue

        snapshot_dir = snapshots_root / f"{source.stem}-{snapshot_hash[:12]}"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_snapshot = snapshot_dir / "model.pt"
        args_snapshot = snapshot_dir / "args.json"
        if checkpoint_snapshot.exists():
            if _sha256(checkpoint_snapshot) != snapshot_hash:
                temporary.unlink()
                raise RuntimeError(f"Existing snapshot hash mismatch: {checkpoint_snapshot}")
            temporary.unlink()
        else:
            temporary.replace(checkpoint_snapshot)
            checkpoint_snapshot.chmod(0o444)
        if args_snapshot.exists() and args_snapshot.read_bytes() != args_path.read_bytes():
            raise RuntimeError(f"Existing snapshot args mismatch: {args_snapshot}")
        if not args_snapshot.exists():
            shutil.copy2(args_path, args_snapshot)
            args_snapshot.chmod(0o444)
        print(f"[Snapshot] {checkpoint_snapshot} sha256={snapshot_hash}")
        return checkpoint_snapshot, args_snapshot, snapshot_hash
    raise RuntimeError(f"Checkpoint kept changing while snapshotting: {source}")


class _TextEncoderPlaceholder(nn.Module):
    """Avoid loading frozen HuggingFace weights during denoiser export."""

    encoder_type = "placeholder"
    output_dim = 768

    def forward(self, _texts):
        raise RuntimeError("The TensorRT denoiser expects precomputed text embeddings")


def _patch_text_encoder_loader(output_dim: int) -> None:
    import reactivebfm.model.text_encoder.conditioning as conditioning

    class Placeholder(_TextEncoderPlaceholder):
        pass

    Placeholder.output_dim = int(output_dim)
    conditioning.load_text_encoder = lambda _name: Placeholder()


def _supports_decomposition(model: nn.Module) -> bool:
    """The decomposed DiT forward requires plain nn.MultiheadAttention blocks."""
    blocks = getattr(model, "blocks", None)
    return (
        getattr(model, "arch", None) == "dit"
        and blocks is not None
        and len(blocks) > 0
        and isinstance(blocks[0].self_attention, nn.MultiheadAttention)
    )


class PlannerDenoiserWrapper(nn.Module):
    """Tensor-only planner forward with optional fused text CFG."""

    def __init__(self, model: nn.Module, *, model_type: str, cfg: bool):
        super().__init__()
        self.model = model
        self.model_type = model_type
        self.cfg = bool(cfg)
        # The manually decomposed DiT forward only supports plain
        # nn.MultiheadAttention blocks. Models trained with the RoPE /
        # text-KV-cache recipe use the custom _DiTAttention modules; for those
        # we run the native forward with a tensor-only conditioning dict.
        self.decomposed = _supports_decomposition(model)

    def _forward_model(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        prefix: torch.Tensor,
        mask: torch.Tensor,
        text_embed: torch.Tensor,
        text_mask: torch.Tensor,
        continuous_time: torch.Tensor,
    ) -> torch.Tensor:
        if self.decomposed:
            return self._forward_dit(
                x, timesteps, prefix, text_embed, text_mask, continuous_time
            )
        y = {"prefix": prefix, "mask": mask, "text_embed": (text_embed, text_mask)}
        if self.model_type == "flow":
            y["flow_time"] = continuous_time
        return self.model(x, timesteps, y)

    def _forward_dit(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        prefix: torch.Tensor,
        text_embed: torch.Tensor,
        text_mask: torch.Tensor,
        continuous_time: torch.Tensor,
    ) -> torch.Tensor:
        model = self.model
        motion = torch.cat((prefix, x), dim=-1)
        motion_tokens = model.input_process(motion)
        motion_tokens = model.sequence_pos_encoder(motion_tokens).transpose(0, 1)
        time_condition = (
            model.embed_flow_timestep(continuous_time).squeeze(0)
            if self.model_type == "flow"
            else model.embed_timestep(timesteps).squeeze(0)
        )
        text_tokens = model.project_text_tokens(text_embed).transpose(0, 1)
        for block in model.blocks:
            modulation = block.ada_ln(time_condition).chunk(9, dim=-1)
            self_shift, self_scale, self_gate = modulation[:3]
            cross_shift, cross_scale, cross_gate = modulation[3:6]
            mlp_shift, mlp_scale, mlp_gate = modulation[6:]

            normalized = block.self_norm(motion_tokens)
            normalized = normalized * (1.0 + self_scale[:, None, :]) + self_shift[:, None, :]
            attended = self._attention(block.self_attention, normalized, normalized, None)
            motion_tokens = motion_tokens + self_gate[:, None, :] * attended

            query = block.cross_norm(motion_tokens)
            query = query * (1.0 + cross_scale[:, None, :]) + cross_shift[:, None, :]
            attended = self._attention(block.cross_attention, query, text_tokens, text_mask)
            motion_tokens = motion_tokens + cross_gate[:, None, :] * attended

            mlp_input = block.mlp_norm(motion_tokens)
            mlp_input = mlp_input * (1.0 + mlp_scale[:, None, :]) + mlp_shift[:, None, :]
            motion_tokens = motion_tokens + mlp_gate[:, None, :] * block.mlp(mlp_input)
        output = model.final_layer(motion_tokens, time_condition)
        output = output[:, model.context_len :].transpose(0, 1)
        return model.output_process(output)

    @staticmethod
    def _attention(
        attention: nn.MultiheadAttention,
        query: torch.Tensor,
        key_value: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        embed_dim = attention.embed_dim
        weight = attention.in_proj_weight
        bias = attention.in_proj_bias
        q = F.linear(query, weight[:embed_dim], None if bias is None else bias[:embed_dim])
        k = F.linear(
            key_value,
            weight[embed_dim : 2 * embed_dim],
            None if bias is None else bias[embed_dim : 2 * embed_dim],
        )
        v = F.linear(
            key_value,
            weight[2 * embed_dim :],
            None if bias is None else bias[2 * embed_dim :],
        )
        batch_size, query_len, _ = q.shape
        key_len = k.shape[1]
        num_heads = attention.num_heads
        head_dim = embed_dim // num_heads
        q = q.reshape(batch_size, query_len, num_heads, head_dim).transpose(1, 2)
        k = k.reshape(batch_size, key_len, num_heads, head_dim).transpose(1, 2)
        v = v.reshape(batch_size, key_len, num_heads, head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (head_dim ** -0.5)
        if padding_mask is not None:
            scores = torch.where(
                padding_mask[:, None, None, :],
                torch.full_like(scores, -1.0e4),
                scores,
            )
        weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(weights, v)
        output = output.transpose(1, 2).reshape(batch_size, query_len, embed_dim)
        return F.linear(output, attention.out_proj.weight, attention.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        prefix: torch.Tensor,
        mask: torch.Tensor,
        text_embed: torch.Tensor,
        text_mask: torch.Tensor,
        continuous_time: torch.Tensor,
        guidance_scale: torch.Tensor,
    ) -> torch.Tensor:
        if not self.cfg:
            return self._forward_model(
                x, timesteps, prefix, mask, text_embed, text_mask, continuous_time
            )

        # Keep the state prefix in both branches and mask text only. This is
        # TextConditionCFGSampleModel semantics, evaluated as one batch.
        x_pair = torch.cat((x, x), dim=0)
        timestep_pair = torch.cat((timesteps, timesteps), dim=0)
        prefix_pair = torch.cat((prefix, prefix), dim=0)
        mask_pair = torch.cat((mask, mask), dim=0)
        time_pair = torch.cat((continuous_time, continuous_time), dim=0)
        text_pair = torch.cat((text_embed, torch.zeros_like(text_embed)), dim=1)
        text_mask_pair = torch.cat((text_mask, text_mask), dim=0)
        output = self._forward_model(
            x_pair,
            timestep_pair,
            prefix_pair,
            mask_pair,
            text_pair,
            text_mask_pair,
            time_pair,
        )
        conditioned, without_text = output.chunk(2, dim=0)
        scale = guidance_scale.view(-1, 1, 1, 1)
        return without_text + scale * (conditioned - without_text)


def _load_args(checkpoint: Path, args_json: Path | None) -> tuple[Namespace, Path]:
    path = args_json or checkpoint.parent / "args.json"
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Planner args.json not found: {path}")
    with path.open(encoding="utf-8") as handle:
        return Namespace(**json.load(handle)), path


def _text_dimension(train_args: Namespace) -> int:
    name = str(getattr(train_args, "text_encoder", "bert")).lower()
    if "t5-small" in name:
        return 512
    if "t5-large" in name:
        return 1024
    if "t5-xl" in name:
        return 2048
    if "t5-xxl" in name:
        return 4096
    if "t5" in name:
        return 768
    return 768


def _example_inputs(
    train_args: Namespace,
    *,
    text_tokens: int,
    text_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    guidance_scale: float,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(1, 36, 1, train_args.pred_len, device=device, dtype=dtype, generator=generator)
    timesteps = torch.zeros(1, device=device, dtype=torch.long)
    prefix = torch.randn(1, 36, 1, train_args.context_len, device=device, dtype=dtype, generator=generator)
    mask = torch.ones(1, 1, 1, train_args.pred_len, device=device, dtype=torch.bool)
    text_embed = torch.randn(text_tokens, 1, text_dim, device=device, dtype=dtype, generator=generator)
    text_mask = torch.zeros(1, text_tokens, device=device, dtype=torch.bool)
    continuous_time = torch.full((1,), 0.5, device=device, dtype=dtype)
    scale = torch.full((1,), guidance_scale, device=device, dtype=dtype)
    return x, timesteps, prefix, mask, text_embed, text_mask, continuous_time, scale


def _benchmark(module, inputs, warmup: int, iterations: int) -> dict[str, float]:
    times = []
    with torch.inference_mode():
        for _ in range(warmup):
            module(*inputs)
        torch.cuda.synchronize()
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            module(*inputs)
            end.record()
            torch.cuda.synchronize()
            times.append(float(start.elapsed_time(end)))
    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "p95_ms": sorted(times)[max(0, int(0.95 * len(times)) - 1)],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--args-json", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="fp32")
    parser.add_argument("--text-tokens", type=int, default=128)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--cfg", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--atol", type=float)
    parser.add_argument("--rtol", type=float)
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument(
        "--strict-verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Refuse to save an engine that exceeds numerical tolerances (default: enabled).",
    )
    parser.add_argument("--require-full-compilation", action="store_true")
    parser.add_argument("--workspace-size", type=int, default=8 << 30)
    parser.add_argument("--benchmark-warmup", type=int, default=10)
    parser.add_argument("--benchmark-iterations", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    cli = _parse_args()
    source_checkpoint = cli.checkpoint.expanduser().resolve()
    if not source_checkpoint.is_file():
        raise FileNotFoundError(f"Planner checkpoint not found: {source_checkpoint}")
    train_args, source_args_path = _load_args(source_checkpoint, cli.args_json)
    output = cli.output or source_checkpoint.with_name(
        f"{source_checkpoint.stem}_planner_trt_{cli.precision}.ts"
    )
    output = output.expanduser().resolve()
    checkpoint = source_checkpoint
    args_path = source_args_path
    checkpoint_sha256 = _sha256(source_checkpoint) if cli.dry_run else ""
    if not cli.dry_run:
        checkpoint, args_path, checkpoint_sha256 = _snapshot_checkpoint(
            source_checkpoint, source_args_path, output
        )
    if getattr(train_args, "motion_tokenizer_ckpt", ""):
        raise NotImplementedError("TensorRT export currently supports raw 36-dim planner checkpoints only")
    if getattr(train_args, "planner_arch", "trans_dec") not in {"dit", "trans_dec"}:
        raise NotImplementedError("TensorRT export currently supports DiT and MDM planners")
    if int(train_args.frame_dim) != 36 or int(train_args.nfeats) != 1:
        raise ValueError("Deployment export requires the raw G1 (36, 1) representation")

    device = torch.device(cli.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Torch-TensorRT export requires a CUDA device")
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[cli.precision]
    text_dim = _text_dimension(train_args)
    _patch_text_encoder_loader(text_dim)
    model, _process = create_model_and_diffusion_smooth(train_args, data=None)
    use_ema = bool(train_args.use_ema if cli.use_ema is None else cli.use_ema)
    load_saved_model(model, str(checkpoint), use_avg=use_ema)
    model = model.to(device=device, dtype=dtype).eval()
    wrapper = PlannerDenoiserWrapper(
        model, model_type=str(train_args.model_type), cfg=cli.cfg
    ).eval()
    inputs = _example_inputs(
        train_args,
        text_tokens=cli.text_tokens,
        text_dim=text_dim,
        device=device,
        dtype=dtype,
        guidance_scale=cli.guidance_scale,
    )
    with torch.inference_mode():
        reference = wrapper(*inputs)
    print(f"[PyTorch] output={tuple(reference.shape)} dtype={reference.dtype}")
    if wrapper.decomposed:
        x, timesteps, prefix, mask, text_embed, text_mask, continuous_time, _scale = inputs
        condition = {
            "prefix": prefix,
            "mask": mask,
            "text_embed": (text_embed, text_mask),
        }
        if train_args.model_type == "flow":
            condition["flow_time"] = continuous_time
        with torch.inference_mode():
            conditioned = model(x, timesteps, condition)
            if cli.cfg:
                unconditioned = model(x, timesteps, {**condition, "text_uncond": True})
                native = unconditioned + _scale.view(-1, 1, 1, 1) * (conditioned - unconditioned)
            else:
                native = conditioned
        decomposition_max_abs = float((native.float() - reference.float()).abs().max())
        decomposition_atol = 1.0e-3 if cli.precision == "fp32" else 0.2
        print(f"[Decomposition] max_abs={decomposition_max_abs:.6g}")
        if not torch.allclose(native.float(), reference.float(), atol=decomposition_atol, rtol=decomposition_atol):
            raise RuntimeError("Decomposed attention does not match the native PyTorch DiT")
    if cli.dry_run:
        print("[Dry run] TensorRT compilation skipped")
        return

    try:
        import torch_tensorrt
    except ImportError as exc:
        raise RuntimeError(
            "torch_tensorrt is required; install a build compatible with the active PyTorch/CUDA stack"
        ) from exc

    started = time.perf_counter()
    with torch.inference_mode():
        compiled = torch_tensorrt.compile(
            wrapper,
            ir="dynamo",
            inputs=inputs,
            require_full_compilation=cli.require_full_compilation,
            truncate_double=True,
            disable_tf32=not cli.allow_tf32,
            workspace_size=cli.workspace_size,
        )
        actual = compiled(*inputs)
    compile_seconds = time.perf_counter() - started
    max_abs = float((reference.float() - actual.float()).abs().max())
    mean_abs = float((reference.float() - actual.float()).abs().mean())
    default_tolerance = 1.0e-3 if cli.precision == "fp32" else 0.2
    atol = default_tolerance if cli.atol is None else cli.atol
    rtol = default_tolerance if cli.rtol is None else cli.rtol
    verified = torch.allclose(reference.float(), actual.float(), atol=atol, rtol=rtol)
    print(f"[Verify] allclose={verified} max_abs={max_abs:.6g} mean_abs={mean_abs:.6g}")
    if cli.verify and cli.strict_verify and not verified:
        raise RuntimeError("TensorRT output exceeded verification tolerance")

    output.parent.mkdir(parents=True, exist_ok=True)
    torch_tensorrt.save(compiled, str(output), output_format="torchscript", inputs=inputs)
    benchmark = {
        "pytorch": _benchmark(wrapper, inputs, cli.benchmark_warmup, cli.benchmark_iterations),
        "tensorrt": _benchmark(compiled, inputs, cli.benchmark_warmup, cli.benchmark_iterations),
    }
    benchmark["speedup"] = benchmark["pytorch"]["mean_ms"] / benchmark["tensorrt"]["mean_ms"]
    metadata = {
        "format_version": 3,
        "planner_input_contract": "reactivebfm_dit_qpos36_v1",
        "checkpoint": str(checkpoint),
        "source_checkpoint": str(source_checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "args_json": str(args_path),
        "planner_arch": str(train_args.planner_arch),
        "model_type": str(train_args.model_type),
        "context_len": int(train_args.context_len),
        "pred_len": int(train_args.pred_len),
        "frame_dim": 36,
        "nfeats": 1,
        "dit_role_embedding": bool(getattr(train_args, "dit_role_embedding", False)),
        "dit_rope": bool(getattr(train_args, "dit_rope", False)),
        "text_encoder": str(train_args.text_encoder),
        "text_dim": text_dim,
        "text_tokens": int(cli.text_tokens),
        "cfg": bool(cli.cfg),
        "precision": cli.precision,
        "use_ema": use_ema,
        "guidance_scale": float(cli.guidance_scale),
        "allow_tf32": bool(cli.allow_tf32),
        "forward_signature": "x,timesteps,prefix,mask,text_embed,text_mask,continuous_time,guidance_scale",
        "compile_seconds": compile_seconds,
        "verification": {"allclose": verified, "max_abs": max_abs, "mean_abs": mean_abs},
        "benchmark": benchmark,
        "torch_version": torch.__version__,
        "torch_tensorrt_version": getattr(torch_tensorrt, "__version__", "unknown"),
        "gpu": torch.cuda.get_device_name(device),
    }
    metadata_path = output.with_suffix(output.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"[Saved] {output}")
    print(f"[Metadata] {metadata_path}")
    print(f"[Benchmark] {json.dumps(benchmark, indent=2)}")


if __name__ == "__main__":
    main()
