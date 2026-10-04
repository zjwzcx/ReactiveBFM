"""Run ReactiveBFM planning and the ScaleBFM tracking policy online.

The planner is deliberately accessed through ``ReactiveBFMPlanner``.  This
module owns only deployment concerns: robot-state conversion, prefix handling,
reference resampling/blending, prompt scheduling, and the real-time control
loop.

Quaternion convention: everything on the planner side of this runner (and the
planner-facing ScaleBridge interfaces it calls) is **xyzw**, matching the
ReactiveBFM qpos36 data layout. ScaleBridge internals are wxyz (MuJoCo order);
the conversion happens inside ``MotionTrackingOnlineEnv`` /
``BaseEnv.reset`` (see ``scalebridge/utils/quaternion.py``). The only place
this runner still converts is when *reading* wxyz state out of the env buffers
to feed the planner.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shlex
import sys
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path


_DEPLOY_ROOT = Path(__file__).resolve().parents[0]
_REPO_ROOT = _DEPLOY_ROOT.parent
for path in (_DEPLOY_ROOT, _REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

DEFAULT_A_POSE_PATH = _DEPLOY_ROOT / "assets" / "a_pose_g1_36dim.json"

import numpy as np
import torch
import yaml
from scalebridge.utils.config import ConfigNode, deep_merge, instantiate, load_yaml, set_dotted, to_config
from scalebridge.utils.logging import logger
from scalebridge.utils.quaternion import wxyz_to_xyzw_np

from reactivebfm.model.motion_planner.inference import (  # noqa: E402
    G1_29DOF_JOINT_NAMES,
    MOTION_DIM,
    ReactiveBFMPlanner,
)
from prompt_scheduler import (  # noqa: E402
    PromptEvent,
    PromptScheduler,
    load_prompt_events,
    load_prompt_pool,
    random_prompt_events,
)


DEFAULT_FUTURE_IDX = "0,1,2,3,4,5"
DEFAULT_GENERATION_FPS = 60.0
ROOT_Z_CLIP = (0.60, 0.85)


class TimingMeter:
    def __init__(self, name: str):
        self.name = name
        self.count = 0
        self.total_s = 0.0

    def update(self, elapsed_s: float) -> None:
        self.count += 1
        self.total_s += float(elapsed_s)

    def format_ms(self) -> str:
        if self.count == 0:
            return f"{self.name}: n=0"
        return f"{self.name}: n={self.count} mean={self.total_s / self.count * 1000.0:.2f}ms"


@dataclass
class PlannerJobResult:
    reference_motion: dict[str, torch.Tensor]
    generated_chunk36: np.ndarray
    elapsed_s: float
    anchor_perf_s: float


@dataclass
class SessionRecorder:
    output_root: Path
    command: str
    enabled: bool = True
    metadata: dict = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)
    state_records: list[dict] = field(default_factory=list)
    reference_records: list[dict] = field(default_factory=list)

    def event(self, kind: str, **payload) -> None:
        if self.enabled:
            self.events.append({"kind": kind, "time_ns": time.time_ns(), **payload})

    def state(self, policy_step: int, motion36: np.ndarray, sample_perf_s: float) -> None:
        if self.enabled:
            self.state_records.append(
                {
                    "policy_step": int(policy_step),
                    "sample_perf_s": float(sample_perf_s),
                    "motion36": np.asarray(motion36, dtype=np.float32).copy(),
                }
            )

    def reference(self, policy_step: int, motion36: np.ndarray, sample_perf_s: float) -> None:
        if self.enabled:
            self.reference_records.append(
                {
                    "policy_step": int(policy_step),
                    "sample_perf_s": float(sample_perf_s),
                    "motion36": np.asarray(motion36, dtype=np.float32).copy(),
                }
            )

    def save(self, stop_reason: str) -> None:
        if not self.enabled:
            return
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = self.output_root / f"online_motion_{stamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        state = np.stack([item["motion36"] for item in self.state_records]) if self.state_records else np.empty((0, MOTION_DIM), np.float32)
        reference = np.stack([item["motion36"] for item in self.reference_records]) if self.reference_records else np.empty((0, MOTION_DIM), np.float32)
        np.savez_compressed(
            output_dir / "motion_buffers.npz",
            real_robot_motion36=state,
            real_robot_policy_step=np.asarray(
                [item["policy_step"] for item in self.state_records], dtype=np.int64
            ),
            real_robot_sample_perf_s=np.asarray(
                [item["sample_perf_s"] for item in self.state_records], dtype=np.float64
            ),
            active_plan_motion36=reference,
            active_plan_policy_step=np.asarray(
                [item["policy_step"] for item in self.reference_records], dtype=np.int64
            ),
            active_plan_sample_perf_s=np.asarray(
                [item["sample_perf_s"] for item in self.reference_records], dtype=np.float64
            ),
        )
        with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump({**self.metadata, "command": self.command, "stop_reason": stop_reason, "events": self.events}, handle, indent=2)
        logger.info("[Recorder] Saved online session to {}", output_dir)


def _parse_future_idx(text: str) -> list[int]:
    values = [item.strip() for item in str(text).split(",") if item.strip()]
    if not values:
        raise ValueError("future_idx cannot be empty")
    result = [int(item) for item in values]
    if min(result) < 0 or len(set(result)) != len(result):
        raise ValueError("future_idx must contain unique non-negative integers")
    return result


def _reorder_columns(values: np.ndarray, src_names: list[str], dst_names: list[str]) -> np.ndarray:
    indices = {name: idx for idx, name in enumerate(src_names)}
    missing = [name for name in dst_names if name not in indices]
    if missing:
        raise KeyError(f"Missing joint names in source order: {missing}")
    return np.asarray(values)[:, [indices[name] for name in dst_names]]


# NOTE: quaternion interpolation below (normalize/slerp/resample) is built from
# dot products and linear blends only, so it is order-agnostic: the same code is
# correct for the xyzw convention used on the planner side of this runner.
def _normalize_quat(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    return quat / np.maximum(norm, 1.0e-8)


def _slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = _normalize_quat(q0)
    q1 = _normalize_quat(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return _normalize_quat((1.0 - alpha) * q0 + alpha * q1)
    theta = math.acos(np.clip(dot, -1.0, 1.0))
    sin_theta = math.sin(theta)
    return (math.sin((1.0 - alpha) * theta) * q0 + math.sin(alpha * theta) * q1) / sin_theta


def _resample_linear(values: np.ndarray, source_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    source_times = np.asarray(source_times, dtype=np.float64)
    target_times = np.asarray(target_times, dtype=np.float64)
    if values.shape[0] == 1:
        return np.repeat(values, target_times.shape[0], axis=0).astype(np.float32)
    return np.stack([np.interp(target_times, source_times, values[:, dim]) for dim in range(values.shape[1])], axis=1).astype(np.float32)


def _resample_quat(values: np.ndarray, source_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    values = _normalize_quat(values)
    result = []
    for target in target_times:
        right = int(np.searchsorted(source_times, target, side="right"))
        if right <= 0:
            result.append(values[0])
            continue
        if right >= len(source_times):
            result.append(values[-1])
            continue
        left = right - 1
        span = source_times[right] - source_times[left]
        alpha = 0.0 if span <= 1.0e-9 else float((target - source_times[left]) / span)
        result.append(_slerp(values[left], values[right], alpha))
    return np.asarray(result, dtype=np.float32)


def _load_init_motion_context(path: str, context_len: int, fps: float) -> np.ndarray:
    """Load a qpos36 NPZ and resample its first ``context_len`` frames to ``fps``.

    Returns xyzw qpos36 regardless of the stored quaternion order.
    """
    motion_path = Path(path).expanduser().resolve()
    if not motion_path.is_file():
        raise FileNotFoundError(f"Initialization motion not found: {motion_path}")
    with np.load(motion_path, allow_pickle=False) as payload:
        if "qpos" not in payload:
            raise KeyError(f"Initialization motion has no qpos array: {motion_path}")
        qpos = np.asarray(payload["qpos"], dtype=np.float32)
        source_fps = float(payload["frequency"]) if "frequency" in payload else 60.0
        quat_order = str(payload["quat_order"].item()).lower() if "quat_order" in payload else "xyzw"
    if qpos.ndim != 2 or qpos.shape[1] != MOTION_DIM or len(qpos) == 0:
        raise ValueError(f"Expected initialization qpos shape (T, {MOTION_DIM}), got {qpos.shape}")
    if not np.isfinite(qpos).all() or source_fps <= 0.0:
        raise ValueError(f"Initialization motion must be finite and have positive FPS: {motion_path}")
    if quat_order == "xyzw":
        root_quat_xyzw = qpos[:, 3:7]
    elif quat_order == "wxyz":
        root_quat_xyzw = wxyz_to_xyzw_np(qpos[:, 3:7])
    else:
        raise ValueError(f"Unsupported initialization quaternion order {quat_order!r}")

    source_times = np.arange(len(qpos), dtype=np.float64) / source_fps
    target_times = np.minimum(
        np.arange(context_len, dtype=np.float64) / float(fps),
        source_times[-1],
    )
    root_pos = _resample_linear(qpos[:, :3], source_times, target_times)
    root_quat = _resample_quat(root_quat_xyzw, source_times, target_times)
    dof_pos = _resample_linear(qpos[:, 7:], source_times, target_times)
    return np.concatenate([root_pos, root_quat, dof_pos], axis=-1).astype(np.float32)


def _load_a_pose(path: str) -> np.ndarray:
    pose_path = Path(path).expanduser().resolve()
    if not pose_path.is_file():
        raise FileNotFoundError(f"A_pose file not found: {pose_path}")
    with pose_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    pose = np.asarray(payload.get("qpos36"), dtype=np.float32)
    if pose.shape != (MOTION_DIM,):
        raise ValueError(f"Expected A_pose qpos36 shape ({MOTION_DIM},), got {pose.shape}")
    if str(payload.get("quat_order", "xyzw")).lower() != "xyzw":
        raise ValueError("A_pose JSON must use xyzw quaternion order")
    if not np.isfinite(pose).all():
        raise ValueError(f"A_pose contains NaN or Inf: {pose_path}")
    return pose


def _motion36_to_init_state(
    motion36: np.ndarray,
    generator_joint_names: list[str],
    policy_joint_names: list[str],
) -> dict[str, np.ndarray]:
    """Build the planner-facing init state dict (xyzw) for ``BaseEnv.reset``."""
    motion36 = np.asarray(motion36, dtype=np.float32)
    if motion36.shape != (MOTION_DIM,):
        raise ValueError(f"Expected one ({MOTION_DIM},) initialization pose, got {motion36.shape}")
    return {
        "root_pos": motion36[:3].copy(),
        # xyzw at this boundary; BaseEnv.reset converts to simulator-internal wxyz.
        "root_quat_xyzw": motion36[3:7].copy(),
        "dof_pos": _reorder_columns(
            motion36[None, 7:], generator_joint_names, policy_joint_names
        )[0],
    }


def _seed_motion_history(
    context: np.ndarray,
    anchor_s: float,
    history_len: int,
    fps: float,
) -> deque[tuple[float, np.ndarray]]:
    history: deque[tuple[float, np.ndarray]] = deque(maxlen=history_len)
    padding = max(0, history_len - len(context))
    for index in range(history_len):
        context_index = max(0, index - padding)
        timestamp = anchor_s - (history_len - 1 - index) / float(fps)
        history.append((timestamp, context[context_index].copy()))
    return history


def _resample_history(history: deque[tuple[float, np.ndarray]], anchor_s: float, context_len: int, fps: float) -> np.ndarray:
    """Resample timestamped xyzw motion36 history onto a uniform fps grid."""
    if not history:
        raise ValueError("Motion history is empty")
    times = np.asarray([item[0] for item in history], dtype=np.float64)
    motion = np.stack([item[1] for item in history]).astype(np.float32)
    targets = anchor_s - np.arange(context_len - 1, -1, -1, dtype=np.float64) / float(fps)
    root = _resample_linear(motion[:, :3], times, targets)
    quat = _resample_quat(motion[:, 3:7], times, targets)
    dof = _resample_linear(motion[:, 7:MOTION_DIM], times, targets)
    return np.concatenate([root, quat, dof], axis=-1).astype(np.float32)


def _build_hybrid_prefix(clean: np.ndarray, real: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if clean.shape != real.shape or clean.shape[1] != MOTION_DIM:
        raise ValueError(f"Hybrid prefixes must both have shape (T, {MOTION_DIM})")
    hybrid = clean.copy()
    root_residual = np.clip(real[:, :3] - clean[:, :3], -args.hybrid_root_pos_residual_clip, args.hybrid_root_pos_residual_clip)
    root_residual = _lowpass(root_residual, args.hybrid_residual_lowpass)
    hybrid[:, :3] += args.hybrid_real_alpha_root_pos * root_residual
    for index in range(len(hybrid)):
        hybrid[index, 3:7] = _slerp(clean[index, 3:7], real[index, 3:7], args.hybrid_real_alpha_root_rot)
    dof_residual = np.clip(real[:, 7:] - clean[:, 7:], -args.hybrid_dof_residual_clip, args.hybrid_dof_residual_clip)
    dof_residual = _lowpass(dof_residual, args.hybrid_residual_lowpass)
    hybrid[:, 7:] += args.hybrid_real_alpha_dof * dof_residual
    return hybrid.astype(np.float32)


def _lowpass(values: np.ndarray, coefficient: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if len(values) <= 1 or coefficient <= 0.0:
        return values.copy()
    output = values.copy()
    for index in range(1, len(output)):
        output[index] = coefficient * output[index - 1] + (1.0 - coefficient) * values[index]
    return output


def _build_prefix_bundle(real_history, clean_history, anchor_s, context_len, fps, args):
    clean = _resample_history(clean_history, anchor_s, context_len, fps)
    real = _resample_history(real_history, anchor_s, context_len, fps)
    result = {"clean": clean, "real": real}
    if args.replan_prefix_mode == "hybrid":
        result["hybrid"] = _build_hybrid_prefix(clean, real, args)
    return result


def _env_to_motion36(env, generator_joint_names: list[str], reference: bool = False) -> np.ndarray:
    """Read the current env state (or active reference head) as xyzw motion36.

    The env state buffers are wxyz (ScaleBridge-internal convention), so the
    root quaternion is converted wxyz -> xyzw here before crossing back to the
    planner side.
    """
    if reference:
        root_pos = env.state_buffer["ref_root_pos_future"][0, 0]
        root_quat = env.state_buffer["ref_root_rot_future"][0, 0]
        dof = env.state_buffer["ref_dof_pos_future"][0, 0]
    else:
        root_pos = env.state_buffer["root_pos_buffer"][0, -1]
        root_quat = env.state_buffer["root_quat_wxyz_buffer"][0, -1]
        dof = env.state_buffer["dof_pos_buffer"][0, -1]
    root_pos_np = root_pos.detach().cpu().numpy()
    root_quat_np = root_quat.detach().cpu().numpy()
    dof_np = dof.detach().cpu().numpy()
    dof_generator = _reorder_columns(dof_np[None], list(env.metadata_dict["joint_names"]), generator_joint_names)[0]
    return np.concatenate([root_pos_np, wxyz_to_xyzw_np(root_quat_np), dof_generator]).astype(np.float32)


def _pump_communication(env, max_messages: int = 64) -> int:
    pump = getattr(getattr(env, "simulator", None), "pump_communication", None)
    if not callable(pump):
        return 0
    try:
        return int(pump(timeout=0.0, max_messages=max_messages))
    except Exception:
        logger.exception("[Online] Communication pump failed")
        return 0


def _robot_state_age(env) -> float | None:
    getter = getattr(getattr(env, "simulator", None), "robot_state_age_s", None)
    return getter() if callable(getter) else None


def _safe_hold(env, repeats: int, interval_s: float) -> bool:
    if repeats <= 0:
        return True
    simulator = getattr(env, "simulator", None)
    dof_pos = getattr(simulator, "dof_pos_tmp", None)
    joint_index = getattr(simulator, "sim_to_env_joint_idx", None)
    if dof_pos is None or joint_index is None:
        return False
    try:
        hold = np.asarray(dof_pos, dtype=np.float32).copy()[joint_index]
        for _ in range(repeats):
            simulator.apply_action(hold)
            time.sleep(max(0.0, interval_s))
        return True
    except Exception:
        logger.exception("[Safety] Failed to publish safe hold")
        return False


def _compose_config(args: argparse.Namespace, future_idx: list[int]):
    config_dir = _DEPLOY_ROOT / "scalebridge" / "config"
    asset = load_yaml(config_dir / "asset" / "g1_29dof.yaml")
    asset.xml_path = str((_DEPLOY_ROOT / asset.xml_path).resolve())

    simulator_base = load_yaml(config_dir / "simulator" / "base_simulator.yaml")
    simulator_specific = load_yaml(config_dir / "simulator" / f"{args.simulator}.yaml")
    simulator_config = deep_merge(simulator_base.config, simulator_specific.config)
    simulator_config.asset = asset
    simulator = to_config(
        {
            "_target_": simulator_specific["_target_"],
            "config": simulator_config,
        }
    )

    agent_base = load_yaml(config_dir / "agent" / "base_agent.yaml")
    agent_specific = load_yaml(config_dir / "agent" / "bfm_agent.yaml")
    agent_config = deep_merge(agent_base.config, agent_specific.config)
    agent_config.checkpoint = str(Path(args.policy_checkpoint).expanduser().resolve())
    if args.control_mode is not None:
        agent_config.control_mode = args.control_mode
    agent = to_config(
        {
            "_target_": agent_specific["_target_"],
            "config": agent_config,
            "device": args.deploy_device,
        }
    )

    env_base = load_yaml(config_dir / "env" / "base_env.yaml")
    env_specific = load_yaml(config_dir / "env" / "motion_tracking_online.yaml")
    env_config = deep_merge(env_base.config, env_specific.config)
    env_config.simulator = simulator
    env_config.observation = load_yaml(config_dir / "observation" / "bfm.yaml")
    env_config.future_idx = future_idx
    env_config.reference_forcing = args.reference_forcing
    env = to_config(
        {
            "_target_": env_specific["_target_"],
            "config": env_config,
            "device": args.deploy_device,
        }
    )
    cfg = ConfigNode(agent=agent, simulator=simulator, env=env, device=args.deploy_device)

    if args.simulator == "mujoco_simulator":
        simulator.config.headless = args.headless
        simulator.config.record_video = args.record_video
        if args.video_path:
            simulator.config.video_path = str(Path(args.video_path).expanduser().resolve())
    for override in args.scalebridge_override:
        if "=" not in override:
            raise ValueError(f"ScaleBridge override must be PATH=VALUE, got {override!r}")
        path, raw_value = override.split("=", 1)
        set_dotted(cfg, path.strip(), yaml.safe_load(raw_value))
    if args.simulator == "mujoco_simulator" and args.record_video:
        cfg.simulator.config.video_text = args.text_prompt
        cfg.simulator.config.video_total_frames = max(0, args.max_policy_steps)
    return cfg


def _run_planner_job(planner: ReactiveBFMPlanner, prefix: np.ndarray, prompt: str, generation_fps: float, policy_fps: float, root_z_clip: tuple[float, float], joint_names: list[str], policy_joint_names: list[str], anchor_perf_s: float | None = None) -> PlannerJobResult:
    start = time.perf_counter()
    chunk = planner.generate_chunk(prefix, prompt=prompt)
    chunk = chunk.copy()
    chunk[:, 2] = np.clip(chunk[:, 2], root_z_clip[0], root_z_clip[1])
    prefix_last = prefix[-1:]
    motion = np.concatenate([prefix_last, chunk], axis=0)
    source_times = np.arange(len(motion), dtype=np.float64) / float(generation_fps or 1.0)
    policy_dt = 1.0 / float(policy_fps)
    target_times = np.arange(0.0, source_times[-1] + 0.5 * policy_dt, policy_dt)
    target_times[-1] = min(target_times[-1], source_times[-1])
    root_pos = _resample_linear(motion[:, :3], source_times, target_times)
    # Planner output stays xyzw all the way into env.set_reference_motion;
    # quaternion resampling is order-agnostic (see _resample_quat).
    root_quat = _resample_quat(motion[:, 3:7], source_times, target_times)
    dof = _resample_linear(motion[:, 7:], source_times, target_times)
    dof_policy = _reorder_columns(dof, joint_names, policy_joint_names)
    reference = {
        "root_pos": torch.from_numpy(root_pos),
        "root_quat_xyzw": torch.from_numpy(root_quat),
        "dof_pos_policy": torch.from_numpy(dof_policy),
    }
    return PlannerJobResult(reference, chunk, time.perf_counter() - start, start if anchor_perf_s is None else anchor_perf_s)


def _slice_reference(reference: dict[str, torch.Tensor], elapsed_s: float, policy_dt: float) -> dict[str, torch.Tensor] | None:
    length = int(reference["root_pos"].shape[0])
    offset = max(0.0, float(elapsed_s) / policy_dt)
    left = int(math.floor(offset))
    alpha = offset - left
    if left >= length - 1:
        return None
    if alpha < 1.0e-4:
        return {key: value[left:].clone() for key, value in reference.items()}
    result = {}
    for key, value in reference.items():
        if key == "root_quat_xyzw":
            # Hemisphere-aligned nlerp; order-agnostic, valid for xyzw.
            right = value[left + 1]
            if float(torch.dot(value[left], right)) < 0.0:
                right = -right
            first = torch.nn.functional.normalize((1.0 - alpha) * value[left] + alpha * right, dim=0)
        else:
            first = (1.0 - alpha) * value[left] + alpha * value[left + 1]
        result[key] = torch.cat([first[None], value[left + 1:].clone()])
    return result


def _applied_policy_steps(applied_frame: int, generation_fps: float, policy_fps: float) -> int:
    """Convert a planner-frame application window to control-loop steps."""
    if applied_frame <= 0:
        raise ValueError("applied_frame must be positive")
    return max(1, int(math.ceil(applied_frame / generation_fps * policy_fps)))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ReactiveBFM with the ScaleBFM tracking policy.")
    parser.add_argument("--model_path", required=True, help="ReactiveBFM checkpoint (.pt) containing args.json.")
    parser.add_argument("--policy_checkpoint", required=True, help="ScaleBFM TensorRT tracking policy checkpoint.")
    parser.add_argument("--text_prompt", default="", help="Text condition passed to ReactiveBFM.")
    parser.add_argument(
        "--prompt_schedule", "--prompt-schedule", type=Path, default=None,
        help="JSON prompt events [{time_s, prompt}] relative to control-loop start.",
    )
    parser.add_argument(
        "--random_prompt_csv", "--random-prompt-csv", type=Path, default=None,
        help="CSV with a caption column used for deterministic random prompt switches.",
    )
    parser.add_argument(
        "--random_switch_interval_s", "--random-switch-interval-s", type=float, default=0.0,
        help="Seconds between random prompt events; requires --random_switch_count > 0.",
    )
    parser.add_argument(
        "--random_switch_count", "--random-switch-count", type=int, default=0,
        help="Number of random prompt events to schedule (0 disables random switching).",
    )
    parser.add_argument(
        "--random_start_s", "--random-start-s", type=float, default=5.0,
        help="Time of the first random prompt event, relative to control-loop start.",
    )
    parser.add_argument(
        "--interactive_prompts", "--interactive-prompts", action="store_true",
        help="Read user prompts from stdin while running; user commands have highest priority.",
    )
    parser.add_argument(
        "--user_prompt_hold_s", "--user-prompt-hold-s", type=float, default=10.0,
        help="Seconds during which scripted/random events cannot overwrite a user prompt.",
    )
    parser.add_argument(
        "--prompt_clock", "--prompt-clock", choices=("wall", "sim"), default="wall",
        help="Clock for scripted/random events: wall time (real robot default) or policy-step simulation time.",
    )
    parser.add_argument("--planner_device", default="cuda")
    parser.add_argument("--planner_dataset", default="", help="Override checkpoint dataset used for normalization stats.")
    parser.add_argument("--planner_data_dir", default="", help="Optional ReactiveBFM dataset root override.")
    parser.add_argument("--planner_stats_dataset", default="", help="Stats component for a composite checkpoint dataset.")
    parser.add_argument("--planner_hml_type", default="")
    parser.add_argument("--planner_use_ema", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument("--planner_compile", action="store_true", help="Compile the ReactiveBFM denoiser with torch.compile.")
    parser.add_argument("--planner_tensorrt", default="", help="Torch-TensorRT planner engine (.ts) and adjacent metadata JSON.")
    parser.add_argument(
        "--init_motion_path",
        default="",
        help="Optional qpos36 NPZ. Its first planner context seeds the prefix; MuJoCo starts at its final pose.",
    )
    parser.add_argument(
        "--init_pose",
        required=True,
        choices=("zero_pose", "gt_context", "A_pose"),
        help="Required initialization strategy for the online MuJoCo rollout.",
    )
    parser.add_argument("--a_pose_path", default=str(DEFAULT_A_POSE_PATH), help="JSON qpos36 asset used by --init_pose A_pose.")
    parser.add_argument("--deploy_device", default="cuda")
    parser.add_argument(
        "--simulator",
        choices=("mujoco_simulator", "real_world"),
        default="mujoco_simulator",
        help="Control backend. MuJoCo is the safe default; real_world must be explicit.",
    )
    parser.add_argument("--headless", action="store_true", help="Run MuJoCo without an interactive viewer.")
    parser.add_argument("--record_video", action="store_true", help="Record the MuJoCo rollout through ScaleBridge.")
    parser.add_argument("--video_path", default="", help="Explicit MP4 output path for MuJoCo recording.")
    parser.add_argument("--control_mode", type=int, default=None)
    parser.add_argument("--future_idx", default=DEFAULT_FUTURE_IDX)
    parser.add_argument("--generation_fps", type=float, default=DEFAULT_GENERATION_FPS)
    parser.add_argument(
        "--applied_frame",
        type=int,
        default=0,
        help="Replan after this many raw generated frames; 0 keeps the normal lookahead scheduler.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional planner sampling seed.")
    parser.add_argument("--reference_forcing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--replan_prefix_mode", choices=("hybrid", "real", "clean"), default="hybrid")
    parser.add_argument("--hybrid_real_alpha_root_pos", type=float, default=0.0)
    parser.add_argument("--hybrid_real_alpha_root_rot", type=float, default=0.25)
    parser.add_argument("--hybrid_real_alpha_dof", type=float, default=0.35)
    parser.add_argument("--hybrid_residual_lowpass", type=float, default=0.7)
    parser.add_argument("--hybrid_root_pos_residual_clip", type=float, default=0.15)
    parser.add_argument("--hybrid_dof_residual_clip", type=float, default=0.35)
    parser.add_argument("--async_planner_lookahead_s", type=float, default=0.30)
    parser.add_argument("--regenerate_threshold", type=int, default=-1)
    parser.add_argument("--sync_generation", action="store_true")
    parser.add_argument("--reference_blend_frames", type=int, default=5)
    parser.add_argument("--robot_state_stale_timeout_s", type=float, default=0.5)
    parser.add_argument("--safe_hold_repeats", type=int, default=0)
    parser.add_argument("--safe_hold_interval_s", type=float, default=0.02)
    parser.add_argument("--frame_drop_warn_threshold_s", type=float, default=0.05)
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--max_policy_steps", type=int, default=-1)
    parser.add_argument("--timing_log_interval", type=int, default=50)
    parser.add_argument("--output_root", default=str(_DEPLOY_ROOT / "outputs"))
    parser.add_argument("--disable_online_recording", action="store_true")
    parser.add_argument("--validate_only", action="store_true")
    parser.add_argument("--scalebridge_override", action="append", default=[])
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    future_idx = _parse_future_idx(args.future_idx)
    if args.generation_fps <= 0.0 or args.reference_blend_frames < 0:
        raise ValueError("generation_fps must be positive and reference_blend_frames non-negative")
    if args.speed <= 0.0 or args.warmup_steps < 0:
        raise ValueError("speed must be positive and warmup_steps non-negative")
    if args.applied_frame < 0:
        raise ValueError("applied_frame must be non-negative")
    if args.random_switch_count < 0:
        raise ValueError("random_switch_count must be non-negative")
    if args.random_start_s < 0.0:
        raise ValueError("random_start_s must be non-negative")
    if args.random_switch_count and args.random_switch_interval_s <= 0.0:
        raise ValueError("random_switch_interval_s must be positive when random switching is enabled")
    if not args.random_switch_count and args.random_switch_interval_s < 0.0:
        raise ValueError("random_switch_interval_s must be non-negative")
    if args.random_switch_count and args.random_prompt_csv is None:
        raise ValueError("--random_prompt_csv is required when random switching is enabled")
    if args.user_prompt_hold_s < 0.0:
        raise ValueError("user_prompt_hold_s must be non-negative")
    if not 0.0 <= args.hybrid_residual_lowpass < 1.0:
        raise ValueError("hybrid_residual_lowpass must be in [0, 1)")
    if not 0.0 <= args.hybrid_real_alpha_root_pos <= 1.0 or not 0.0 <= args.hybrid_real_alpha_root_rot <= 1.0 or not 0.0 <= args.hybrid_real_alpha_dof <= 1.0:
        raise ValueError("hybrid alpha values must be in [0, 1]")

    if args.seed is not None:
        normalized_seed = args.seed % (2**32)
        random.seed(normalized_seed)
        np.random.seed(normalized_seed)
        torch.manual_seed(normalized_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(normalized_seed)

    planner = ReactiveBFMPlanner.from_checkpoint(
        args.model_path,
        device=args.planner_device,
        data_dir=args.planner_data_dir or None,
        dataset=args.planner_dataset or None,
        hml_type=args.planner_hml_type or None,
        stats_dataset=args.planner_stats_dataset or None,
        use_ema=args.planner_use_ema,
        guidance_scale=args.guidance_scale,
        compile_model=args.planner_compile,
        tensorrt_engine=args.planner_tensorrt or None,
    )
    cfg = _compose_config(args, future_idx)
    agent = instantiate(cfg.agent)
    metadata = agent.get_meta_data()
    env = instantiate(cfg.env, metadata_dict=metadata)
    generator_joint_names = list(G1_29DOF_JOINT_NAMES)
    policy_joint_names = list(metadata["joint_names"])
    policy_fps = 1.0 / float(env.dt)
    if args.applied_frame > planner.raw_pred_len:
        raise ValueError(
            f"applied_frame ({args.applied_frame}) exceeds planner raw prediction length "
            f"({planner.raw_pred_len})"
        )
    applied_policy_steps = (
        _applied_policy_steps(args.applied_frame, args.generation_fps, policy_fps)
        if args.applied_frame > 0
        else 0
    )
    if applied_policy_steps:
        logger.info(
            "[Planner] Applying {} raw frames per chunk ({} policy steps at {:.2f} FPS)",
            args.applied_frame,
            applied_policy_steps,
            policy_fps,
        )
    init_context = None
    init_state_dict = None
    if args.init_pose == "gt_context":
        if args.simulator != "mujoco_simulator":
            raise ValueError("--init_pose gt_context is only supported by the MuJoCo simulator")
        if not args.init_motion_path:
            raise ValueError("--init_pose gt_context requires --init_motion_path")
        init_context = _load_init_motion_context(
            args.init_motion_path,
            planner.raw_context_len,
            args.generation_fps,
        )
        init_state_dict = _motion36_to_init_state(
            init_context[-1], generator_joint_names, policy_joint_names
        )
    elif args.init_pose == "A_pose":
        a_pose = _load_a_pose(args.a_pose_path)
        init_state_dict = _motion36_to_init_state(a_pose, generator_joint_names, policy_joint_names)
        init_context = np.repeat(a_pose[None], planner.raw_context_len, axis=0)
    recorder = SessionRecorder(Path(args.output_root), shlex.join(sys.argv), enabled=not args.disable_online_recording)
    scripted_events: list[PromptEvent] = []
    if args.prompt_schedule is not None:
        scripted_events = load_prompt_events(args.prompt_schedule)
    random_events: list[PromptEvent] = []
    if args.random_switch_count:
        random_pool = load_prompt_pool(args.random_prompt_csv)
        random_events = random_prompt_events(
            random_pool,
            start_s=args.random_start_s,
            interval_s=args.random_switch_interval_s,
            count=args.random_switch_count,
            seed=args.seed,
            initial_prompt=args.text_prompt,
        )
    prompt_scheduler = PromptScheduler(
        args.text_prompt,
        [*scripted_events, *random_events],
        interactive=args.interactive_prompts,
        user_hold_s=args.user_prompt_hold_s,
    )
    recorder.metadata.update(
        {
            "planner_checkpoint": str(Path(args.model_path).expanduser()),
            "future_idx": future_idx,
            "generation_fps": args.generation_fps,
            "policy_fps": policy_fps,
            "planner_context_len": planner.context_len,
            "planner_pred_len": planner.pred_len,
            "planner_raw_context_len": planner.raw_context_len,
            "planner_raw_pred_len": planner.raw_pred_len,
            "applied_frame": args.applied_frame,
            "applied_policy_steps": applied_policy_steps,
            "seed": args.seed,
            "init_pose": args.init_pose,
            "init_motion_path": (
                str(Path(args.init_motion_path).expanduser().resolve())
                if args.init_motion_path
                else ""
            ),
            "a_pose_path": (
                str(Path(args.a_pose_path).expanduser().resolve())
                if args.init_pose == "A_pose"
                else ""
            ),
            "initial_prompt": args.text_prompt,
            "prompt_schedule": str(args.prompt_schedule.expanduser().resolve()) if args.prompt_schedule else "",
            "random_prompt_csv": str(args.random_prompt_csv.expanduser().resolve()) if args.random_prompt_csv else "",
            "random_switch_interval_s": args.random_switch_interval_s,
            "random_switch_count": args.random_switch_count,
            "random_start_s": args.random_start_s,
            "interactive_prompts": bool(args.interactive_prompts),
            "user_prompt_hold_s": args.user_prompt_hold_s,
            "prompt_clock": args.prompt_clock,
        }
    )
    planner_timer = TimingMeter("planner")
    tracking_timer = TimingMeter("tracking")
    planner_executor: ThreadPoolExecutor | None = None
    planner_future: Future | None = None
    planner_future_generation: int | None = None
    prompt_generation = 0
    prompt_replan_pending = False
    prompt_requested_control_s: float | None = None
    prompt_request_source = ""
    stop_reason = "completed"

    try:
        obs = env.reset(init_state_dict=init_state_dict)
        now = time.perf_counter()
        initial_motion = _env_to_motion36(env, generator_joint_names)
        history_len = max(planner.raw_context_len + 4, int(math.ceil((planner.raw_context_len / args.generation_fps + 1.0) * policy_fps)))
        if init_context is None:
            init_context = np.repeat(initial_motion[None], planner.raw_context_len, axis=0)
        real_history = _seed_motion_history(init_context, now, history_len, args.generation_fps)
        clean_history = _seed_motion_history(init_context, now, history_len, args.generation_fps)

        # Apply events explicitly scheduled at t=0 before the first planner
        # call.  Events are recorded as initialization-time changes and do not
        # require a second, redundant generation.
        for event in prompt_scheduler.poll(0.0):
            prompt_generation += 1
            recorder.event(
                "prompt_switch",
                policy_step=-1,
                prompt=event.prompt,
                source=event.source,
                priority=event.priority,
                requested_time_s=event.time_s,
                applied_time_s=0.0,
                latency_s=0.0,
                generation_id=prompt_generation,
                phase="initialization",
            )
        active_prompt = prompt_scheduler.current_prompt
        recorder.metadata["initial_prompt"] = active_prompt
        initial_prefix = _resample_history(real_history, now, planner.raw_context_len, args.generation_fps)
        initial_result = _run_planner_job(planner, initial_prefix, active_prompt, args.generation_fps, policy_fps, ROOT_Z_CLIP, generator_joint_names, policy_joint_names, now)
        planner_timer.update(initial_result.elapsed_s)
        env.set_reference_motion(initial_result.reference_motion)
        clean_history.append((time.perf_counter(), _env_to_motion36(env, generator_joint_names, reference=True)))
        recorder.event("control_started", initial_prompt=active_prompt)

        with torch.inference_mode():
            for _ in range(args.warmup_steps):
                agent.get_action(env.get_observation())

        if args.validate_only:
            stop_reason = "validate_only"
            return

        control_start = time.perf_counter()

        fixed_application_window = args.applied_frame > 0
        async_generation = not args.sync_generation and not fixed_application_window
        planner_executor = ThreadPoolExecutor(max_workers=1) if async_generation else None
        regenerate_threshold = args.regenerate_threshold
        if regenerate_threshold < 0:
            regenerate_threshold = max(
                max(future_idx) + 2,
                int(math.ceil(args.async_planner_lookahead_s * policy_fps)),
            )
        if not fixed_application_window and regenerate_threshold <= max(future_idx):
            raise ValueError("regenerate_threshold must be greater than max(future_idx)")
        playback_interval = float(env.dt) / args.speed
        policy_step = 0
        plan_policy_steps = 0
        while args.max_policy_steps < 0 or policy_step < args.max_policy_steps:
            loop_start = time.perf_counter()
            pumped = _pump_communication(env)
            obs = env.refresh_observation()
            sample_time = time.perf_counter()
            elapsed_control_s = (
                sample_time - control_start
                if args.prompt_clock == "wall"
                else policy_step * float(env.dt)
            )
            prompt_events = prompt_scheduler.poll(elapsed_control_s)
            if prompt_events:
                previous_prompt = active_prompt
                active_prompt = prompt_scheduler.current_prompt
                prompt_generation += 1
                prompt_replan_pending = True
                prompt_requested_control_s = elapsed_control_s
                prompt_request_source = prompt_events[-1].source
                if planner_future is not None:
                    cancelled = planner_future.cancel()
                    if cancelled:
                        planner_future = None
                        planner_future_generation = None
                    else:
                        recorder.event(
                            "planner_invalidated",
                            policy_step=policy_step,
                            generation_id=prompt_generation,
                            reason="prompt_switch_while_generation_running",
                        )
                for event in prompt_events:
                    recorder.event(
                        "prompt_switch",
                        policy_step=policy_step,
                        prompt=event.prompt,
                        source=event.source,
                        priority=event.priority,
                        requested_time_s=event.time_s,
                        applied_time_s=elapsed_control_s,
                        latency_s=max(0.0, elapsed_control_s - event.time_s),
                        generation_id=prompt_generation,
                        previous_prompt=previous_prompt,
                    )
            real_motion = _env_to_motion36(env, generator_joint_names)
            reference_motion = _env_to_motion36(env, generator_joint_names, reference=True)
            real_history.append((sample_time, real_motion.copy()))
            clean_history.append((sample_time, reference_motion.copy()))
            recorder.state(policy_step, real_motion, sample_time)
            recorder.reference(policy_step, reference_motion, sample_time)

            age = _robot_state_age(env)
            if args.robot_state_stale_timeout_s > 0.0 and age is not None and age > args.robot_state_stale_timeout_s:
                stop_reason = "robot_state_stream_stale"
                recorder.event(stop_reason, age_s=float(age))
                break
            if getattr(env.simulator, "right_upper_switch_pressed", False):
                env.simulator.right_upper_switch_pressed = False
                stop_reason = "r1_recording_stop"
                break

            if planner_future is not None and planner_future.done():
                future_generation = planner_future_generation
                result = planner_future.result()
                planner_timer.update(result.elapsed_s)
                if future_generation == prompt_generation:
                    aligned = _slice_reference(result.reference_motion, time.perf_counter() - result.anchor_perf_s, float(env.dt))
                    activated = False
                    if aligned is not None and int(aligned["root_pos"].shape[0]) > max(future_idx):
                        env.set_reference_motion(aligned, blend_frames=args.reference_blend_frames)
                        obs = env.get_observation()
                        activated = True
                        activation_time_s = (
                            policy_step * float(env.dt)
                            if args.prompt_clock == "sim"
                            else time.perf_counter() - control_start
                        )
                        recorder.event(
                            "planner_activated",
                            policy_step=policy_step,
                            prompt=active_prompt,
                            source=prompt_request_source,
                            generation_id=prompt_generation,
                            activation_time_s=activation_time_s,
                            command_to_activation_latency_s=(
                                activation_time_s - prompt_requested_control_s
                                if prompt_replan_pending and prompt_requested_control_s is not None
                                else None
                            ),
                            planner_elapsed_s=result.elapsed_s,
                        )
                    if activated:
                        prompt_replan_pending = False
                        prompt_requested_control_s = None
                        prompt_request_source = ""
                    else:
                        recorder.event(
                            "planner_result_expired",
                            policy_step=policy_step,
                            prompt=active_prompt,
                            generation_id=prompt_generation,
                        )
                else:
                    recorder.event(
                        "stale_planner_result_discarded",
                        policy_step=policy_step,
                        result_generation_id=future_generation,
                        active_generation_id=prompt_generation,
                    )
                planner_future = None
                planner_future_generation = None

            if fixed_application_window and (prompt_replan_pending or plan_policy_steps >= applied_policy_steps):
                bundle = _build_prefix_bundle(
                    real_history,
                    clean_history,
                    sample_time,
                    planner.raw_context_len,
                    args.generation_fps,
                    args,
                )
                prefix = bundle[args.replan_prefix_mode]
                result = _run_planner_job(
                    planner,
                    prefix,
                    active_prompt,
                    args.generation_fps,
                    policy_fps,
                    ROOT_Z_CLIP,
                    generator_joint_names,
                    policy_joint_names,
                    sample_time,
                )
                planner_timer.update(result.elapsed_s)
                env.set_reference_motion(result.reference_motion)
                obs = env.get_observation()
                plan_policy_steps = 0
                was_prompt_replan = prompt_replan_pending
                prompt_replan_pending = False
                activation_time_s = (
                    policy_step * float(env.dt)
                    if args.prompt_clock == "sim"
                    else time.perf_counter() - control_start
                )
                recorder.event(
                    "planner_activated" if was_prompt_replan else "fixed_window_replan",
                    policy_step=policy_step,
                    applied_frame=args.applied_frame,
                    applied_policy_steps=applied_policy_steps,
                    prompt=active_prompt,
                    generation_id=prompt_generation,
                    activation_time_s=activation_time_s,
                    command_to_activation_latency_s=(
                        activation_time_s - prompt_requested_control_s
                        if was_prompt_replan and prompt_requested_control_s is not None
                        else None
                    ),
                    planner_elapsed_s=result.elapsed_s,
                )
                if was_prompt_replan:
                    prompt_requested_control_s = None
                    prompt_request_source = ""
            elif not fixed_application_window and planner_future is None and (prompt_replan_pending or env.reference_length <= regenerate_threshold):
                bundle = _build_prefix_bundle(real_history, clean_history, sample_time, planner.raw_context_len, args.generation_fps, args)
                prefix = bundle[args.replan_prefix_mode]
                if async_generation:
                    planner_future_generation = prompt_generation
                    planner_future = planner_executor.submit(_run_planner_job, planner, prefix, active_prompt, args.generation_fps, policy_fps, ROOT_Z_CLIP, generator_joint_names, policy_joint_names, sample_time)
                    recorder.event(
                        "planner_submitted",
                        policy_step=policy_step,
                        prompt=active_prompt,
                        generation_id=prompt_generation,
                        reason="prompt_switch" if prompt_replan_pending else "lookahead",
                    )
                else:
                    result = _run_planner_job(planner, prefix, active_prompt, args.generation_fps, policy_fps, ROOT_Z_CLIP, generator_joint_names, policy_joint_names, sample_time)
                    planner_timer.update(result.elapsed_s)
                    aligned = _slice_reference(
                        result.reference_motion,
                        time.perf_counter() - result.anchor_perf_s,
                        float(env.dt),
                    )
                    if aligned is not None and int(aligned["root_pos"].shape[0]) > max(future_idx):
                        env.set_reference_motion(aligned, blend_frames=args.reference_blend_frames)
                        obs = env.get_observation()
                    if prompt_replan_pending and aligned is not None and int(aligned["root_pos"].shape[0]) > max(future_idx):
                        activation_time_s = (
                            policy_step * float(env.dt)
                            if args.prompt_clock == "sim"
                            else time.perf_counter() - control_start
                        )
                        recorder.event(
                            "planner_activated",
                            policy_step=policy_step,
                            prompt=active_prompt,
                            generation_id=prompt_generation,
                            activation_time_s=activation_time_s,
                            command_to_activation_latency_s=(
                                activation_time_s - prompt_requested_control_s
                                if prompt_requested_control_s is not None
                                else None
                            ),
                            planner_elapsed_s=result.elapsed_s,
                        )
                        prompt_replan_pending = False
                        prompt_requested_control_s = None
                        prompt_request_source = ""
                    elif prompt_replan_pending:
                        recorder.event(
                            "planner_result_expired",
                            policy_step=policy_step,
                            prompt=active_prompt,
                            generation_id=prompt_generation,
                        )

            agent.before_step(obs)
            infer_start = time.perf_counter()
            with torch.inference_mode():
                tgt_dof_pos, action = agent.get_action(obs)
            tracking_timer.update(time.perf_counter() - infer_start)
            env.apply_control(tgt_dof_pos, action)
            agent.after_step(obs, action)
            policy_step += 1
            plan_policy_steps += 1

            elapsed = time.perf_counter() - loop_start
            if elapsed < playback_interval:
                time.sleep(playback_interval - elapsed)
            elif elapsed - playback_interval >= args.frame_drop_warn_threshold_s:
                logger.warning("[Loop] frame overrun {:.3f}s, pumped={} age={}", elapsed - playback_interval, pumped, age)
            if args.timing_log_interval and policy_step % args.timing_log_interval == 0:
                logger.info("[Timing] {} | {}", tracking_timer.format_ms(), planner_timer.format_ms())
        else:
            stop_reason = "max_policy_steps_reached"
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
    except Exception:
        stop_reason = "exception"
        logger.exception("[Safety] Online control failed")
        _safe_hold(env, args.safe_hold_repeats, args.safe_hold_interval_s)
    finally:
        if planner_executor is not None:
            planner_executor.shutdown(wait=False, cancel_futures=True)
        prompt_scheduler.close()
        localization = getattr(getattr(env, "simulator", None), "localization_module", None)
        if localization is not None and hasattr(localization, "stop"):
            localization.stop()
        simulator = getattr(env, "simulator", None)
        if simulator is not None and hasattr(simulator, "close"):
            simulator.close()
        recorder.event("runtime_stop", reason=stop_reason)
        recorder.save(stop_reason)
        logger.info("[Timing] {} | {}", tracking_timer.format_ms(), planner_timer.format_ms())


if __name__ == "__main__":
    main()
