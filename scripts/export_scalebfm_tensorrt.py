#!/usr/bin/env python3
"""Compile the official ScaleBFM policy for ReactiveBFM qpos references.

The ScaleBFM Transformer is compiled into TensorRT. ScaleBridge keeps the small
G1 forward-kinematics adapter in eager PyTorch so the public agent interface can
accept future ``root_xyz + root_quat_wxyz + 29 dof`` references directly.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
_DEPLOY_ROOT = ROOT / "deploy"
for path in (ROOT, _DEPLOY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

DEFAULT_XML = _DEPLOY_ROOT / "scalebridge" / "data" / "robot" / "g1_29dof" / "g1_29dof.xml"
# Vendored verbatim from ScaleBFM (see deploy/third_party/scalebfm/README.md),
# so no external ScaleBFM checkout is required.
NETWORK_SOURCE = _DEPLOY_ROOT / "third_party" / "scalebfm" / "humanoid_transformer.py"


def _load_network_module():
    spec = importlib.util.spec_from_file_location("scalebfm_humanoid_transformer", NETWORK_SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load ScaleBFM network definitions from {NETWORK_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HumanoidTransformer, module.TaskEmbedder


from scalebridge.agent.qpos_adapter import (  # noqa: E402
    ScaleBFMQposTaskAdapter,
    quat_apply_inverse,
)


POLICY_JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint", "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint", "left_ankle_roll_joint",
    "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint", "left_wrist_roll_joint",
    "right_wrist_roll_joint", "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]
SELECTED_BODY_NAMES = [
    "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
]
MODE_BODY_NAMES = [
    ["pelvis"],
    ["left_wrist_yaw_link", "right_wrist_yaw_link"],
    ["pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link"],
    ["left_wrist_yaw_link", "right_wrist_yaw_link", "left_ankle_roll_link", "right_ankle_roll_link"],
    ["pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link", "left_ankle_roll_link", "right_ankle_roll_link"],
    ["left_shoulder_roll_link", "right_shoulder_roll_link", "left_elbow_link", "right_elbow_link", "left_wrist_yaw_link", "right_wrist_yaw_link"],
    ["pelvis", "left_shoulder_roll_link", "right_shoulder_roll_link", "left_elbow_link", "right_elbow_link", "left_wrist_yaw_link", "right_wrist_yaw_link"],
    SELECTED_BODY_NAMES,
]
FUTURE_IDX = [0, 1, 2, 3, 4, 5]
MODE_FEATURE_DIMS = [3, 3, 6, 6]


def _mode_table(device: torch.device) -> torch.Tensor:
    table = torch.zeros(len(MODE_BODY_NAMES), len(SELECTED_BODY_NAMES), device=device)
    for mode_index, names in enumerate(MODE_BODY_NAMES):
        for name in names:
            table[mode_index, SELECTED_BODY_NAMES.index(name)] = 1.0
    return table


def _mode_mappings(table: torch.Tensor) -> torch.Tensor:
    parts = [table.unsqueeze(-1).expand(-1, -1, dim).flatten(1) for dim in MODE_FEATURE_DIMS]
    parts.append(torch.ones(table.shape[0], 1, device=table.device))
    return torch.cat(parts, dim=-1)


def _vector_by_joint(values: dict[str, float]) -> list[float]:
    return [float(values.get(name, 0.0)) for name in POLICY_JOINT_NAMES]


def _robot_parameters() -> tuple[list[float], list[float], list[float], list[float], list[float]]:
    armature_5020, armature_7520_14, armature_7520_22, armature_4010 = (
        0.003609725, 0.010177520, 0.025101925, 0.00425
    )
    omega = 10.0 * 2.0 * math.pi
    stiffness = {
        "hip_pitch": armature_7520_14 * omega**2,
        "hip_roll": armature_7520_22 * omega**2,
        "hip_yaw": armature_7520_14 * omega**2,
        "knee": armature_7520_22 * omega**2,
        "ankle": 2.0 * armature_5020 * omega**2,
        "arm": armature_5020 * omega**2,
        "wrist_small": armature_4010 * omega**2,
    }
    damping = {
        "hip_pitch": 4.0 * armature_7520_14 * omega,
        "hip_roll": 4.0 * armature_7520_22 * omega,
        "hip_yaw": 4.0 * armature_7520_14 * omega,
        "knee": 4.0 * armature_7520_22 * omega,
        "ankle": 8.0 * armature_5020 * omega,
        "arm": 4.0 * armature_5020 * omega,
        "wrist_small": 4.0 * armature_4010 * omega,
    }
    default, scale, kp, kd, effort = {}, {}, {}, {}, {}
    for name in POLICY_JOINT_NAMES:
        if "hip_pitch" in name:
            key, limit = "hip_pitch", 88.0
            default[name] = -0.312
        elif "hip_roll" in name:
            key, limit = "hip_roll", 139.0
        elif "hip_yaw" in name:
            key, limit = "hip_yaw", 88.0
        elif "knee" in name:
            key, limit = "knee", 139.0
            default[name] = 0.669
        elif "ankle" in name:
            key, limit = "ankle", 50.0
            if "pitch" in name:
                default[name] = -0.363
        elif name == "waist_yaw_joint":
            kp[name], kd[name], effort[name], scale[name] = 100.0, 2.0, 88.0, 0.25
            continue
        elif name in ("waist_roll_joint", "waist_pitch_joint"):
            kp[name], kd[name], effort[name], scale[name] = 300.0, 5.0, 50.0, 0.25
            continue
        elif "wrist_pitch" in name or "wrist_yaw" in name:
            key, limit = "wrist_small", 5.0
        else:
            key, limit = "arm", 25.0
            if "elbow" in name:
                default[name] = 0.6
            elif "shoulder_pitch" in name:
                default[name] = 0.2
            elif name == "left_shoulder_roll_joint":
                default[name] = 0.2
            elif name == "right_shoulder_roll_joint":
                default[name] = -0.2
        kp[name], kd[name], effort[name] = stiffness[key], damping[key], limit
        scale[name] = 0.25 * limit / stiffness[key]
    return tuple(_vector_by_joint(item) for item in (default, scale, kp, kd, effort))


def _parse_kinematics(xml_path: Path, device: torch.device):
    root = ET.parse(xml_path).getroot().find("worldbody")
    if root is None or root.find("body") is None:
        raise ValueError(f"MJCF has no root body: {xml_path}")
    root_body = root.find("body")
    body_names = [root_body.attrib["name"]]
    joint_names: list[str] = []
    parents = [-1]
    translations: list[np.ndarray] = []
    rotations: list[np.ndarray] = []
    axes: list[np.ndarray] = []

    def visit(node: ET.Element, parent_index: int) -> None:
        for child in node.findall("body"):
            joints = child.findall("joint")
            next_parent = parent_index
            if joints:
                if len(joints) != 1:
                    raise ValueError(f"Expected one joint in body {child.attrib.get('name')}")
                joint = joints[0]
                body_names.append(child.attrib["name"])
                joint_names.append(joint.attrib["name"])
                parents.append(parent_index)
                translations.append(np.fromstring(child.attrib.get("pos", "0 0 0"), sep=" "))
                rotations.append(np.fromstring(child.attrib.get("quat", "1 0 0 0"), sep=" "))
                axes.append(np.fromstring(joint.attrib.get("axis", "0 0 1"), sep=" "))
                next_parent = len(body_names) - 1
            visit(child, next_parent)

    visit(root_body, 0)
    if set(joint_names) != set(POLICY_JOINT_NAMES):
        missing = sorted(set(POLICY_JOINT_NAMES) - set(joint_names))
        extra = sorted(set(joint_names) - set(POLICY_JOINT_NAMES))
        raise ValueError(f"MJCF joint mismatch; missing={missing}, extra={extra}")
    selected = torch.tensor([body_names.index(name) for name in SELECTED_BODY_NAMES], device=device)
    policy_to_xml = torch.tensor([POLICY_JOINT_NAMES.index(name) for name in joint_names], device=device)
    return (
        body_names,
        joint_names,
        torch.tensor(parents, dtype=torch.long, device=device),
        torch.tensor(np.asarray(translations), dtype=torch.float32, device=device),
        torch.nn.functional.normalize(torch.tensor(np.asarray(rotations), dtype=torch.float32, device=device), dim=-1),
        torch.tensor(np.asarray(axes), dtype=torch.float32, device=device),
        selected,
        policy_to_xml,
    )


class ScaleBFMTransformerCore(nn.Module):
    def __init__(
        self,
        actor: nn.Module,
        task_embedder: nn.Module,
        default_dof: torch.Tensor,
        action_scale: torch.Tensor,
        context_len: int,
    ) -> None:
        super().__init__()
        self.task_embedder = task_embedder
        self.prop_embedder = actor.prop_projection
        self.action_embedder = actor.action_projection
        self.transformer_blocks = actor.transformer_blocks
        self.final_norm = actor.final_norm
        self.projection_head = actor.projection_head
        self.register_buffer("empty_embedding", actor.empty_embedding.detach().clone())
        mask = torch.zeros(2 * context_len, 2 * context_len, dtype=torch.bool, device=default_dof.device)
        mask[:-1, -1] = True
        self.register_buffer("self_attn_mask", mask)
        self.register_buffer("gravity", torch.tensor([0.0, 0.0, -1.0], device=default_dof.device).view(1, 1, 3))
        self.register_buffer("default_dof", default_dof)
        self.register_buffer("action_scale", action_scale)

    def forward(
        self,
        root_quat_buffer: torch.Tensor,
        base_ang_vel_buffer: torch.Tensor,
        dof_pos_buffer: torch.Tensor,
        dof_vel_buffer: torch.Tensor,
        action_buffer: torch.Tensor,
        task_input: torch.Tensor,
    ):
        gravity = self.gravity.expand(root_quat_buffer.shape[0], root_quat_buffer.shape[1], -1)
        prop = torch.cat(
            (
                quat_apply_inverse(root_quat_buffer, gravity),
                base_ang_vel_buffer,
                dof_pos_buffer - self.default_dof,
                dof_vel_buffer * 0.05,
            ),
            dim=-1,
        )
        prop_token = self.prop_embedder(prop)
        action_token = self.action_embedder(action_buffer)
        tokens = torch.empty(
            prop_token.shape[0], 2 * prop_token.shape[1], prop_token.shape[2],
            dtype=prop_token.dtype, device=prop_token.device,
        )
        tokens[:, ::2] = prop_token
        tokens[:, 1:-1:2] = action_token[:, 1:]
        tokens[:, -1:] = self.empty_embedding

        condition = self.task_embedder(task_input)
        for block in self.transformer_blocks:
            tokens = block(tokens, condition, self_attn_mask=self.self_attn_mask)
        action = self.projection_head(self.final_norm(tokens)[:, -1])
        return action * self.action_scale + self.default_dof, action


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_dict(checkpoint: Path) -> dict[str, torch.Tensor]:
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = loaded.get("model_state_dict", loaded)
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    return state


def _infer_architecture(state: dict[str, torch.Tensor], num_heads: int | None) -> dict:
    prop_weight = state["actor.prop_projection.weight"]
    action_weight = state["actor.action_projection.weight"]
    output_weight = state["actor.projection_head.weight"]
    embed_dim = int(prop_weight.shape[0])
    layers = 1 + max(
        int(match.group(1))
        for key in state
        if (match := re.match(r"actor\.transformer_blocks\.(\d+)\.", key))
    )
    task_weights = sorted(
        ((key, value) for key, value in state.items() if key.startswith("actor_task_embedder.task_projection") and key.endswith("weight")),
        key=lambda item: item[0],
    )
    if not task_weights:
        raise KeyError("Checkpoint has no actor_task_embedder.task_projection weights")
    if num_heads is None:
        presets = {256: 4, 384: 6}
        if embed_dim not in presets:
            raise ValueError(f"Cannot infer attention heads for embed_dim={embed_dim}; pass --num-heads")
        num_heads = presets[embed_dim]
    reduced = state.get("actor_task_embedder.W")
    return {
        "prop_obs_dim": int(prop_weight.shape[1]),
        "action_dim": int(action_weight.shape[1]),
        "output_dim": int(output_weight.shape[0]),
        "embedding_dim": embed_dim,
        "num_heads": int(num_heads),
        "ff_dim": int(state["actor.transformer_blocks.0.feed_forward.w.weight"].shape[0]),
        "num_layers": layers,
        "task_obs_dim": int(task_weights[0][1].shape[1]),
        "reduced_task_dim": int(reduced.shape[-1]) if reduced is not None else None,
        "task_embedder_hidden_dims": [int(value.shape[0]) for _, value in task_weights[:-1]],
    }


def _build_models(state: dict[str, torch.Tensor], architecture: dict, device: torch.device, network_classes=None):
    HumanoidTransformer, TaskEmbedder = network_classes
    actor = HumanoidTransformer(
        prop_obs_dim=architecture["prop_obs_dim"], action_dim=architecture["action_dim"],
        output_dim=architecture["output_dim"], embed_dim=architecture["embedding_dim"],
        num_heads=architecture["num_heads"], ff_dim=architecture["ff_dim"],
        num_layers=architecture["num_layers"],
    )
    task = TaskEmbedder(
        task_obs_dim=architecture["task_obs_dim"], embedding_dim=architecture["embedding_dim"],
        reduced_task_dim=architecture["reduced_task_dim"],
        hidden_dims=architecture["task_embedder_hidden_dims"],
    )
    actor.load_state_dict({key.removeprefix("actor."): value for key, value in state.items() if key.startswith("actor.")})
    task.load_state_dict(
        {key.removeprefix("actor_task_embedder."): value for key, value in state.items() if key.startswith("actor_task_embedder.")}
    )
    return actor.eval().to(device), task.eval().to(device)


def _sample_inputs(device: torch.device, default_dof: torch.Tensor, context_len: int):
    batch, future = 1, len(FUTURE_IDX)
    root_pos = torch.tensor([[0.0, 0.0, 0.76]], device=device)
    root_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    dof = default_dof.expand(batch, context_len, -1).contiguous()
    return [
        root_pos,
        root_quat[:, None].expand(batch, context_len, 4).contiguous(),
        torch.zeros(batch, context_len, 3, device=device),
        dof,
        torch.zeros_like(dof),
        torch.zeros(batch, context_len, len(POLICY_JOINT_NAMES), device=device),
        root_pos[:, None].expand(batch, future, 3).contiguous(),
        root_quat[:, None].expand(batch, future, 4).contiguous(),
        default_dof.expand(batch, future, -1).contiguous(),
        torch.tensor([7], dtype=torch.long, device=device),
        torch.tensor(FUTURE_IDX, dtype=torch.long, device=device).view(1, future, 1),
    ]


def _benchmark(module: nn.Module, inputs: list[torch.Tensor], warmup: int, iterations: int) -> float:
    # Inputs may be produced by an inference-mode adapter. Keep the complete
    # benchmark in inference mode so PyTorch does not try to save them for
    # autograd (PyTorch 2.9+ rejects inference tensors in normal mode).
    with torch.inference_mode():
        for _ in range(warmup):
            module(*inputs)
        if inputs[0].device.type != "cuda":
            start_s = time.perf_counter()
            for _ in range(iterations):
                module(*inputs)
            return (time.perf_counter() - start_s) * 1000.0 / iterations
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            module(*inputs)
        end.record()
        end.synchronize()
    return float(start.elapsed_time(end) / iterations)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="ScaleBFM tracking policy checkpoint (.pt).")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--xml-path", type=Path, default=DEFAULT_XML)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-heads", type=int)
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--optimization-level", type=int, choices=range(0, 6), default=3)
    parser.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--atol", type=float)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true", help="Rebuild and run PyTorch on CPU without importing TensorRT.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    xml_path = args.xml_path.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"ScaleBFM checkpoint not found: {checkpoint}")
    if not xml_path.is_file():
        raise FileNotFoundError(f"G1 MJCF not found: {xml_path}")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("--warmup must be non-negative and --iterations positive")
    device = torch.device("cpu" if args.dry_run else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --dry-run for a CPU reconstruction check")

    state = _state_dict(checkpoint)
    architecture = _infer_architecture(state, args.num_heads)
    network_classes = _load_network_module()
    actor, task = _build_models(state, architecture, device, network_classes)
    body_names, _, parents, translations, rotations, axes, selected, policy_to_xml = _parse_kinematics(xml_path, device)
    default_values, action_scale_values, stiffness, damping, effort = _robot_parameters()
    default_dof = torch.tensor([default_values], dtype=torch.float32, device=device)
    action_scale = torch.tensor([action_scale_values], dtype=torch.float32, device=device)
    context_len = 3
    table = _mode_table(device)
    mappings = _mode_mappings(table)
    adapter = ScaleBFMQposTaskAdapter(
        table, mappings, parents.detach().cpu().tolist(), translations, rotations,
        axes, selected, policy_to_xml, len(FUTURE_IDX),
    ).eval()
    core = ScaleBFMTransformerCore(actor, task, default_dof, action_scale, context_len).eval()
    qpos_inputs = _sample_inputs(device, default_dof, context_len)
    with torch.inference_mode():
        task_input = adapter(
            qpos_inputs[0], qpos_inputs[1], qpos_inputs[3], qpos_inputs[6],
            qpos_inputs[7], qpos_inputs[8], qpos_inputs[9], qpos_inputs[10],
        )
        core_inputs = [*qpos_inputs[1:6], task_input]
        reference = core(*core_inputs)
    expected_task_dim = len(SELECTED_BODY_NAMES) * sum(MODE_FEATURE_DIMS) + 1 + len(SELECTED_BODY_NAMES)
    if architecture["task_obs_dim"] != expected_task_dim:
        raise ValueError(
            f"Policy task_obs_dim={architecture['task_obs_dim']} does not match G1 mode layout {expected_task_dim}"
        )
    print(f"[Model] architecture={architecture}")
    print(f"[Contract] qpos_inputs=11 engine_inputs=6 output={[tuple(item.shape) for item in reference]}")
    if args.dry_run:
        print("[Dry run] Checkpoint reconstruction and qpos-policy forward passed on CPU")
        return

    import torch_tensorrt

    precision = torch.float16 if args.precision == "fp16" else torch.float32
    output = (args.output or checkpoint.with_name(f"{checkpoint.stem}_tensorrt.pt")).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"[Compile] precision={args.precision} gpu={torch.cuda.get_device_name(device)}")
    with torch.inference_mode():
        compiled = torch_tensorrt.compile(
            core,
            ir="dynamo",
            inputs=core_inputs,
            enabled_precisions={precision},
            optimization_level=args.optimization_level,
            truncate_double=True,
        )
        actual = compiled(*core_inputs)
    tolerance = args.atol if args.atol is not None else (2.0e-2 if precision == torch.float16 else 5.0e-4)
    max_error = max(float((ref.float() - got.float()).abs().max()) for ref, got in zip(reference, actual))
    matched = all(torch.allclose(ref.float(), got.float(), rtol=tolerance, atol=tolerance) for ref, got in zip(reference, actual))
    print(f"[Verify] matched={matched} max_abs={max_error:.6g} tolerance={tolerance}")
    if args.verify and not matched:
        raise RuntimeError("TensorRT output exceeded tolerance; artifact was not saved")
    torch_tensorrt.save(compiled, str(output), output_format="torchscript", inputs=core_inputs)

    metadata_path = output.with_name(f"{output.stem}_metadata.json")
    metadata = {
        "schema_version": 2,
        "input_contract": "reactivebfm_qpos_v1",
        "engine_input_contract": "scalebfm_transformer_core_v1",
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": _sha256(checkpoint),
        "precision": args.precision,
        "policy_architecture": architecture,
        "selected_body_names": SELECTED_BODY_NAMES,
        "body_names": body_names,
        "joint_names": POLICY_JOINT_NAMES,
        "action_names": POLICY_JOINT_NAMES,
        "target_joint_names": POLICY_JOINT_NAMES,
        "history_buffer_size": context_len,
        "future_idx": FUTURE_IDX,
        "mode_feature_dims": MODE_FEATURE_DIMS,
        "mode_mapping_with_time": True,
        "qpos_adapter": {
            "mode_table": table.detach().cpu().tolist(),
            "mode_mappings": mappings.detach().cpu().tolist(),
            "parents": parents.detach().cpu().tolist(),
            "translations": translations.detach().cpu().tolist(),
            "rotations": rotations.detach().cpu().tolist(),
            "axes": axes.detach().cpu().tolist(),
            "selected": selected.detach().cpu().tolist(),
            "policy_to_xml": policy_to_xml.detach().cpu().tolist(),
        },
        "default_dof_pos": default_values,
        "action_scale": action_scale_values,
        "stiffness": stiffness,
        "damping": damping,
        "torque_limit": effort,
        "compile_device": torch.cuda.get_device_name(device),
        "verify_max_abs": max_error,
        "verify_tolerance": tolerance,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"[Saved] {output}")
    print(f"[Metadata] {metadata_path}")
    if args.benchmark:
        adapter_ms = _benchmark(adapter, [qpos_inputs[index] for index in (0, 1, 3, 6, 7, 8, 9, 10)], args.warmup, args.iterations)
        pytorch_ms = _benchmark(core, core_inputs, args.warmup, args.iterations)
        tensorrt_ms = _benchmark(compiled, core_inputs, args.warmup, args.iterations)
        print(
            f"[Benchmark] adapter={adapter_ms:.4f}ms PyTorch-core={pytorch_ms:.4f}ms "
            f"TensorRT-core={tensorrt_ms:.4f}ms core-speedup={pytorch_ms / tensorrt_ms:.2f}x"
        )


if __name__ == "__main__":
    main()
