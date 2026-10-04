#!/usr/bin/env python3
"""Play a qpos36 motion NPZ on the Unitree G1 in MuJoCo and/or save an MP4.

Adapted from the ReactiveBFM ``play_rbfm_npz_mujoco`` tool for the deploy/
layout. Supported inputs:

- Raw qpos36 clips (from the retargeting pipeline):
    qpos: (T, 36) = [root_pos(3), root_quat_xyzw(4), dof_pos(29)]
    frequency: scalar (optional; defaults to 60)
    text: caption (optional, used as the overlay label)
- Online session recordings written by ``run_online_generation.py``
  (``motion_buffers.npz`` with ``real_robot_motion36`` /
  ``active_plan_motion36`` at the 50 Hz policy rate) — select with ``--stream``.

Quaternion convention: input files follow the planner/data convention (xyzw
by default; ``--quat-order auto`` detects the layout heuristically). MuJoCo
qpos is wxyz, so the quaternion is reordered exactly once when writing into
``data.qpos`` (see scalebridge/utils/quaternion.py).

Examples:
    # render an MP4 (headless servers: prefix MUJOCO_GL=egl)
    python deploy/tools/play_rbfm_npz_mujoco.py --input-npz /path/to/motion.npz

    # replay an online session recording
    python deploy/tools/play_rbfm_npz_mujoco.py \
        --input-npz deploy/outputs/online_motion_xxx/motion_buffers.npz \
        --stream active_plan_motion36

    # interactive viewer (SPACE pause, R restart, C camera follow)
    python deploy/tools/play_rbfm_npz_mujoco.py --input-npz /path/to/motion.npz --interactive
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Sequence

import imageio.v2 as imageio
import numpy as np


_DEPLOY_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _DEPLOY_ROOT.parent
for path in (_DEPLOY_ROOT, _REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scalebridge.utils.logging import logger  # noqa: E402
from scalebridge.utils.overlay import load_font, overlay_header  # noqa: E402
from scalebridge.utils.quaternion import xyzw_to_wxyz_np  # noqa: E402

# Headless Linux server fallback: if no DISPLAY and MUJOCO_GL is not set,
# default to EGL. macOS uses the default (glfw) backend.
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY") and not os.environ.get("MUJOCO_GL"):
    os.environ["MUJOCO_GL"] = "egl"

try:
    import mujoco
except ImportError as exc:
    raise ImportError(
        "Failed to import `mujoco`. Please run in an environment with MuJoCo. "
        "For headless servers, try prefixing with `MUJOCO_GL=egl`."
    ) from exc


DEFAULT_MODEL_CANDIDATES = (
    _DEPLOY_ROOT / "scalebridge" / "data" / "robot" / "g1_29dof" / "g1_29dof.xml",
)
DEFAULT_VIS_DIR = _DEPLOY_ROOT / "outputs" / "vis"

DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
SESSION_STREAMS = ("real_robot_motion36", "active_plan_motion36")
SESSION_FPS = 50.0
STEM_BASE_PATTERN = re.compile(r"^(?P<base>.+?)_000_smplx(?:_\d+Hz_29dof)?$")


def _open_mp4_writer(output_path: Path, fps: float):
    """Open an MP4 writer via ffmpeg explicitly.

    We force `format="FFMPEG"` to avoid imageio accidentally selecting the TIFF
    writer backend (which does not accept `fps` and fails on append_data).
    """
    try:
        return imageio.get_writer(
            str(output_path),
            format="FFMPEG",
            mode="I",
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=1,
            pixelformat="yuv420p",
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to open MP4 writer with imageio FFMPEG backend. "
            "Please install ffmpeg plugin first, e.g.:\n"
            "  pip install imageio-ffmpeg\n"
            "or:\n"
            "  pip install \"imageio[ffmpeg]\""
        ) from exc


def get_g1_29dof_joint_names() -> list[str]:
    return [
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
    ]


def _resolve_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path.resolve()


def _resolve_robot_model(explicit_path: str) -> Path:
    if explicit_path:
        path = _resolve_path(explicit_path)
        if not path.is_file():
            raise FileNotFoundError(f"Robot model not found: {path}")
        return path

    for candidate in DEFAULT_MODEL_CANDIDATES:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not find G1 MuJoCo model. Please pass --robot-model explicitly."
    )


def _normalize_quat(quat: np.ndarray, scalar_first: bool) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0] if scalar_first else [0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return quat / norm


def _resolve_frame_window(total_frames: int, start_frame: int, end_frame: int) -> tuple[int, int]:
    if total_frames <= 0:
        raise ValueError("Input motion has zero frames.")

    start = max(0, min(start_frame, total_frames - 1))
    if end_frame < 0:
        end = total_frames
    else:
        end = max(start + 1, min(end_frame, total_frames))

    if start >= end:
        raise ValueError(
            f"Invalid frame range: start={start_frame}, end={end_frame}, total={total_frames}"
        )
    return start, end


def _resolve_image_size(
    model: mujoco.MjModel,
    width: int,
    height: int,
) -> tuple[int, int]:
    max_w = int(model.vis.global_.offwidth)
    max_h = int(model.vis.global_.offheight)
    if width <= 0 or height <= 0:
        raise ValueError("width/height must be positive")
    scale = min(1.0, max_w / width if max_w > 0 else 1.0, max_h / height if max_h > 0 else 1.0)
    return max(1, int(np.floor(width * scale))), max(1, int(np.floor(height * scale)))


def _build_joint_qpos_addresses(model: mujoco.MjModel, joint_names: Sequence[str]) -> list[int]:
    addresses: list[int] = []
    missing: list[str] = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            missing.append(name)
            continue
        qpos_adr = int(model.jnt_qposadr[joint_id])
        next_adr = int(model.jnt_qposadr[joint_id + 1]) if joint_id + 1 < model.njnt else int(model.nq)
        if next_adr - qpos_adr != 1:
            raise ValueError(f"Joint `{name}` is not a 1-DoF hinge in model.")
        addresses.append(qpos_adr)
    if missing:
        raise KeyError(f"Joints not found in model: {', '.join(missing)}")
    return addresses


def _set_qpos36_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos36: np.ndarray,
    frame_idx: int,
    joint_qpos_addresses: Sequence[int],
    root_pos_offset: np.ndarray,
    quat_order: str,
) -> None:
    frame = qpos36[frame_idx]
    data.qpos[:] = 0.0
    data.qpos[0:3] = frame[0:3] + root_pos_offset
    # Boundary conversion: planner/data convention -> MuJoCo wxyz.
    if quat_order == "xyzw":
        data.qpos[3:7] = xyzw_to_wxyz_np(_normalize_quat(frame[3:7], scalar_first=False))
    elif quat_order == "wxyz":
        data.qpos[3:7] = _normalize_quat(frame[3:7], scalar_first=True)
    else:
        raise ValueError(f"Unsupported quat order: {quat_order}")
    for value, adr in zip(frame[7:36], joint_qpos_addresses):
        data.qpos[adr] = value
    if data.qvel.size:
        data.qvel[:] = 0.0
    if data.ctrl.size:
        data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)


def _make_camera(
    model: mujoco.MjModel,
    distance: float,
    azimuth: float,
    elevation: float,
) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam


def _follow_camera(cam: mujoco.MjvCamera, data: mujoco.MjData, z_offset: float) -> None:
    cam.lookat[0] = data.qpos[0]
    cam.lookat[1] = data.qpos[1]
    cam.lookat[2] = data.qpos[2] + z_offset


def _stem_to_base(stem: str) -> str:
    match = STEM_BASE_PATTERN.match(stem)
    return match.group("base") if match else stem


def _maybe_str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip()
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        if value.ndim == 0:
            return _maybe_str(value.item())
        return _maybe_str(value.reshape(-1)[0])
    return str(value).strip()


def _load_caption_from_meta(raw_root: Path, shard: str, base: str) -> str:
    meta_path = raw_root / shard / f"{base}_meta.json"
    if not meta_path.is_file():
        return ""
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return ""
    rewrites = payload.get("text_rewrite")
    if isinstance(rewrites, list) and rewrites:
        text = _maybe_str(rewrites[0])
        if text:
            return text
    return _maybe_str(payload.get("text"))


def _resolve_text_label(
    explicit_text: str,
    npz_text: str,
    input_npz: Path,
    raw_root: Path | None,
) -> tuple[str, str]:
    if explicit_text.strip():
        return explicit_text.strip(), "cli"
    if npz_text.strip():
        return npz_text.strip(), "npz:text"
    if raw_root is not None and raw_root.is_dir():
        shard = input_npz.parent.name
        base = _stem_to_base(input_npz.stem)
        from_meta = _load_caption_from_meta(raw_root, shard, base)
        if from_meta:
            return from_meta, "raw_meta"
    return input_npz.stem, "filename"


def _infer_quat_order(qpos36: np.ndarray) -> str:
    """Heuristic detection for source quaternion layout in qpos[:, 3:7]."""
    q = np.asarray(qpos36[:, 3:7], dtype=np.float64)
    score_wxyz = float(np.mean(np.abs(q[:, 0])))
    score_xyzw = float(np.mean(np.abs(q[:, 3])))
    if score_wxyz > score_xyzw + 0.05:
        return "wxyz"
    if score_xyzw > score_wxyz + 0.05:
        return "xyzw"

    q0 = q[0]
    # Compare closeness to identity under both interpretations.
    # wxyz identity: [1, 0, 0, 0]
    c_wxyz = abs(abs(q0[0]) - 1.0) + float(np.linalg.norm(q0[1:4]))
    q0_as_wxyz = np.roll(q0, 1)  # interpret source as xyzw
    c_xyzw = abs(abs(q0_as_wxyz[0]) - 1.0) + float(np.linalg.norm(q0_as_wxyz[1:4]))
    return "wxyz" if c_wxyz <= c_xyzw else "xyzw"


def _load_motion(npz_path: Path, stream: str) -> tuple[np.ndarray, float, str, str]:
    """Return (qpos36, fps, text, quat_order_hint)."""
    with np.load(npz_path, allow_pickle=True) as payload:
        keys = set(payload.files)
        if "qpos" in keys:
            qpos = np.asarray(payload["qpos"], dtype=np.float64)
            fps = float(payload["frequency"]) if "frequency" in payload else 60.0
            text = _maybe_str(payload["text"]) if "text" in payload else ""
            quat_order = _maybe_str(payload["quat_order"]).lower() if "quat_order" in payload else ""
        elif stream in keys:
            # Session recording from run_online_generation.py (50 Hz policy rate).
            qpos = np.asarray(payload[stream], dtype=np.float64)
            fps = SESSION_FPS
            text = f"{npz_path.parent.name} [{stream}]"
            quat_order = "xyzw"
        else:
            raise KeyError(
                f"{npz_path} has neither a 'qpos' array nor a session stream "
                f"{SESSION_STREAMS}; found {sorted(keys)}"
            )
    if qpos.ndim != 2 or qpos.shape[1] != 36:
        raise ValueError(f"Expected qpos shape (T,36), got {qpos.shape}")
    if fps <= 0:
        fps = 60.0
    return qpos, fps, text, quat_order


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Play a qpos36 motion npz in MuJoCo and/or save an MP4."
    )
    parser.add_argument("--input-npz", type=str, required=True, help="Input motion npz path.")
    parser.add_argument(
        "--stream",
        default="real_robot_motion36",
        choices=SESSION_STREAMS,
        help="Which stream to play when given a session recording (motion_buffers.npz).",
    )
    parser.add_argument(
        "--robot-model",
        type=str,
        default="",
        help="Path to G1 29-DoF MJCF/XML. If empty, auto-resolve.",
    )
    parser.add_argument(
        "--vis-dir",
        type=str,
        default=str(DEFAULT_VIS_DIR),
        help="Output video directory.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="",
        help="Optional output mp4 filename; default uses input stem.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="",
        help="Text label shown at top of video. Default: npz text/raw meta/input stem.",
    )
    parser.add_argument(
        "--raw-root",
        type=str,
        default="",
        help="Optional raw data root used to auto-load caption from *_meta.json.",
    )
    parser.add_argument("--fps", type=float, default=0.0, help="Output FPS. <=0 uses motion frequency.")
    parser.add_argument(
        "--quat-order",
        type=str,
        default="auto",
        choices=("auto", "xyzw", "wxyz"),
        help="Quaternion order in qpos[:,3:7] of input npz.",
    )
    parser.add_argument("--font-size", type=int, default=30, help="Top overlay font size.")
    parser.add_argument("--start-frame", type=int, default=0, help="Inclusive start frame.")
    parser.add_argument("--end-frame", type=int, default=-1, help="Exclusive end frame; -1 means all.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Render every N-th frame.")
    parser.add_argument("--max-frames", type=int, default=-1, help="Render at most N output frames.")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="Output width.")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="Output height.")
    parser.add_argument("--camera-distance", type=float, default=3.1, help="Camera distance.")
    parser.add_argument("--camera-azimuth", type=float, default=135.0, help="Camera azimuth (deg).")
    parser.add_argument("--camera-elevation", type=float, default=-12.0, help="Camera elevation (deg).")
    parser.add_argument("--camera-height-offset", type=float, default=0.18, help="Follow camera z offset.")
    parser.add_argument(
        "--root-pos-offset",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="Offset applied to root_pos every frame.",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip MP4 rendering (use with --interactive).",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Open the interactive viewer (SPACE pause, R restart, C camera follow).",
    )
    parser.add_argument("--loop", action="store_true", help="Loop playback (interactive mode).")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed (interactive mode).")
    return parser


def _render_video(args, model, qpos36, out_fps, text_label, quat_order, frame_indices, output_path) -> None:
    data = mujoco.MjData(model)
    render_width, render_height = _resolve_image_size(model, args.width, args.height)
    try:
        renderer = mujoco.Renderer(model, width=render_width, height=render_height)
    except mujoco.FatalError as exc:
        raise RuntimeError(
            "Failed to create MuJoCo offscreen renderer. "
            "For headless servers, run with `MUJOCO_GL=egl` (or `MUJOCO_GL=osmesa` if EGL is unavailable)."
        ) from exc
    camera = _make_camera(
        model,
        distance=args.camera_distance,
        azimuth=args.camera_azimuth,
        elevation=args.camera_elevation,
    )
    joint_qpos_addresses = _build_joint_qpos_addresses(model, get_g1_29dof_joint_names())
    root_pos_offset = np.asarray(args.root_pos_offset, dtype=np.float64)
    font = load_font(args.font_size)

    logger.info("[render] output: {}", output_path)
    writer = _open_mp4_writer(output_path=output_path, fps=out_fps)
    try:
        for idx, frame_idx in enumerate(frame_indices, start=1):
            _set_qpos36_frame(
                model=model,
                data=data,
                qpos36=qpos36,
                frame_idx=frame_idx,
                joint_qpos_addresses=joint_qpos_addresses,
                root_pos_offset=root_pos_offset,
                quat_order=quat_order,
            )
            _follow_camera(camera, data, z_offset=args.camera_height_offset)
            renderer.update_scene(data, camera=camera)
            rgb = renderer.render()
            rgb = overlay_header(
                rgb,
                frame_idx=frame_idx,
                total_frames=qpos36.shape[0],
                fps=out_fps,
                text_label=text_label,
                font=font,
            )
            writer.append_data(rgb)

            if idx == 1 or idx == len(frame_indices) or idx % 100 == 0:
                logger.info("[render] wrote {}/{} frames", idx, len(frame_indices))
    finally:
        writer.close()
        renderer.close()
    logger.info("[render] done: {}", output_path)


def _play_interactive(args, model, qpos36, motion_fps, quat_order) -> None:
    import mujoco.viewer as mjv

    data = mujoco.MjData(model)
    joint_qpos_addresses = _build_joint_qpos_addresses(model, get_g1_29dof_joint_names())
    root_pos_offset = np.asarray(args.root_pos_offset, dtype=np.float64)

    paused = False
    camera_follow = True
    frame_idx = 0
    frame_dt = 1.0 / motion_fps / max(args.speed, 1.0e-6)

    def key_callback(keycode) -> None:
        nonlocal paused, camera_follow, frame_idx
        key = chr(keycode)
        if key == " ":
            paused = not paused
        elif key == "R":
            frame_idx = 0
        elif key == "C":
            camera_follow = not camera_follow

    viewer = mjv.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False,
        key_callback=key_callback,
    )
    try:
        while viewer.is_running():
            if not paused:
                _set_qpos36_frame(
                    model=model,
                    data=data,
                    qpos36=qpos36,
                    frame_idx=frame_idx,
                    joint_qpos_addresses=joint_qpos_addresses,
                    root_pos_offset=root_pos_offset,
                    quat_order=quat_order,
                )
                if camera_follow:
                    viewer.cam.lookat = data.qpos[:3].copy()
                viewer.sync()
                frame_idx += 1
                if frame_idx >= qpos36.shape[0]:
                    if args.loop:
                        frame_idx = 0
                    else:
                        break
                time.sleep(frame_dt)
            else:
                time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()


def main() -> None:
    args = build_argparser().parse_args()
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be > 0.")
    if args.speed <= 0.0:
        raise ValueError("--speed must be positive")
    if not args.interactive and args.no_video:
        raise ValueError("Nothing to do: pass --interactive and/or drop --no-video")

    input_npz = _resolve_path(args.input_npz)
    if not input_npz.is_file():
        raise FileNotFoundError(f"Input npz not found: {input_npz}")

    model_path = _resolve_robot_model(args.robot_model)
    qpos36, motion_fps, npz_text, quat_order_hint = _load_motion(input_npz, args.stream)
    out_fps = motion_fps if args.fps <= 0 else float(args.fps)
    if out_fps <= 0:
        out_fps = 60.0

    raw_root = _resolve_path(args.raw_root) if args.raw_root else None
    text_label, text_source = _resolve_text_label(args.text, npz_text, input_npz, raw_root)
    output_path = None
    if not args.no_video:
        vis_dir = _resolve_path(args.vis_dir)
        vis_dir.mkdir(parents=True, exist_ok=True)
        output_name = args.output_name.strip() if args.output_name.strip() else f"{input_npz.stem}_mujoco.mp4"
        if not output_name.endswith(".mp4"):
            output_name += ".mp4"
        output_path = vis_dir / output_name

    model = mujoco.MjModel.from_xml_path(str(model_path))
    if args.quat_order == "auto":
        quat_order = quat_order_hint or _infer_quat_order(qpos36)
    else:
        quat_order = args.quat_order

    start_frame, end_frame = _resolve_frame_window(qpos36.shape[0], args.start_frame, args.end_frame)
    frame_indices = list(range(start_frame, end_frame, args.frame_stride))
    if args.max_frames > 0:
        frame_indices = frame_indices[: args.max_frames]
    if not frame_indices:
        raise RuntimeError("No frames selected for playback.")

    logger.info("[play] input: {}", input_npz)
    logger.info("[play] model: {}", model_path)
    logger.info("[play] frames: {} (window [{}, {}), stride={})", len(frame_indices), start_frame, end_frame, args.frame_stride)
    logger.info("[play] fps: {:.2f}", out_fps)
    logger.info("[play] quat-order: {}", quat_order)
    logger.info("[play] motion text (source: {}): {}", text_source, text_label)

    if args.interactive:
        _play_interactive(args, model, qpos36, motion_fps, quat_order)
    if output_path is not None:
        _render_video(args, model, qpos36, out_fps, text_label, quat_order, frame_indices, output_path)


if __name__ == "__main__":
    main()
