from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np
import torch
from scalebridge.utils.logging import logger
from copy import deepcopy

from scalebridge.env.base_env import BaseEnv
from scalebridge.utils.torch_utils import calc_heading_quat, calc_heading_quat_inv, quat_apply, quat_mul
from scalebridge.utils.xsens_dataloader import XsensOnlineClient

def _prompt(msg: str) -> None:
    logger.info(msg)
    input()

class MotionTrackingXsensEnv(BaseEnv):

    def _setup_metadata(self):
        sim_cfg = self.cfg.simulator.config
        self.policy_dt = float(sim_cfg.low_dt * sim_cfg.decimation)
        
        self.prefill_duration_s = self.policy_dt * (self.cfg.future_idx[-1])
        self.xsens_port = int(self.cfg.get("xsens_port", 9763))
        self.xsens_host = str(self.cfg.get("xsens_host", "0.0.0.0"))
        self.scale_factor = float(self.cfg.get("xsens_scale_factor", 0.75))

        self._xsens_client: Optional[XsensOnlineClient] = None
        self._ref_stream_t0 = 0.0
        self._xsens_ready = False

        self.future_idx = self.cfg.future_idx
        self.future_frame_offset = torch.as_tensor(self.cfg.future_idx, dtype=torch.long, device=self.device)

        self.has_hand = self.cfg.get('has_hand', False)
        if self.has_hand:
            logger.info(f"[Env] Using dexterous hands: {self.cfg.hand_type}")
            target_joint_names = deepcopy(self.metadata_dict["action_names"])
            target_joint_names.extend(self.cfg.hand_joint_names)
            
            self.metadata_dict["target_joint_names"] = target_joint_names
            self.metadata_dict["stiffness"].extend(self.cfg.hand_stiffness)
            self.metadata_dict["damping"].extend(self.cfg.hand_damping)
            self.metadata_dict["torque_limit"].extend(self.cfg.hand_torque_limit)

        self.reference_forcing = self.cfg.get('reference_forcing', True)
        logger.info(f"[Env] Using reference root position to apply forcing: {self.reference_forcing}")
        self.metadata_dict["enable_root_localization"] = not self.reference_forcing

    def _setup_state_manager(self):
        super()._setup_state_manager()

        # Here the xsens client may not be ready
        num_future_frames = int(self.future_frame_offset.shape[0])
        num_bodies = len(self.metadata_dict["selected_body_names"])
        self.body_pos_w_future = torch.zeros(1, num_future_frames, num_bodies, 3, device=self.device)
        self.body_quat_w_wxyz_future = torch.zeros(1, num_future_frames, num_bodies, 4, device=self.device)

        self._pos_offset = torch.zeros(3, device=self.device)
        self._quat_offset = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        self._quat_offset = self._quat_offset.view(1,1,1,4).repeat(1, num_future_frames, num_bodies, 1)

        self.state_buffer.update({
            "body_pos_w_future": self.body_pos_w_future,
            "body_quat_w_wxyz_future": self.body_quat_w_wxyz_future,
            "future_frame_offset": self.future_frame_offset[None, :, None],
        }) # placeholder
        
    def _update_state_manager(self):
        super()._update_state_manager()
        self._gather_reference_state()

    def _gather_reference_state(self):
        cur_timestep = time.monotonic()

        max_offset = self.future_idx[-1]
        query_times = np.array(
            [cur_timestep - (max_offset - offset) * self.policy_dt for offset in self.future_idx]
        )
        last_query_time = min(query_times[-1], self._xsens_client.buffer.latest()[0])
        last_idx = (last_query_time - query_times[0]) // self.policy_dt
        last_idx = np.clip(last_idx, self.future_idx[-2] + 1, self.future_idx[-1]) # ensure the last index is within the range of future_idx
        query_times[-1] = query_times[0] + last_idx * self.policy_dt
        self.future_frame_offset[-1] = int(last_idx)

        pos_batch, quat_batch = self._xsens_client.buffer.interpolate_batch(query_times)
        # pos_batch: (K, B, 3),  quat_batch: (K, B, 4)
        self.body_pos_w_future[0].copy_(torch.from_numpy(pos_batch).float(), non_blocking=True)
        self.body_quat_w_wxyz_future[0].copy_(torch.from_numpy(quat_batch).float(), non_blocking=True)
        
        
        self.body_pos_w_future[:] = quat_apply(self._quat_offset, self.body_pos_w_future - self._pos_offset)
        self.body_quat_w_wxyz_future[:] = quat_mul(self._quat_offset, self.body_quat_w_wxyz_future)

        self.state_buffer.update({
            "body_pos_w_future": self.body_pos_w_future,
            "body_quat_w_wxyz_future": self.body_quat_w_wxyz_future,
            "future_frame_offset": self.future_frame_offset[None, :, None],
        })

        if self.reference_forcing:
            self.state_buffer["root_pos_buffer"][:, -1] = self.state_buffer["body_pos_w_future"][:, 0, 0]

    def _calibrate(self):
        if (not self._xsens_ready):
            self._launch_xsens()
            self._xsens_ready = True

        root_pos, root_quat = self.simulator.calibrate()
        root_pos = torch.from_numpy(root_pos).to(self.device)
        root_quat = torch.from_numpy(root_quat).to(self.device)

        logger.info("[Env] Aligning streamed mocap ...")

        lastest_info = self._xsens_client.buffer.latest()
        if lastest_info is None:
            raise RuntimeError("Xsens buffer is empty during calibration; no latest frame available.")
        
        self._ref_stream_t0 = lastest_info[0]
        ref_root_pos_np = lastest_info[1][0]
        ref_root_pos_np[..., -1] = 0
        ref_root_quat_np = lastest_info[2][0]

        self._pos_offset.copy_(torch.from_numpy(ref_root_pos_np), non_blocking=True)
        
        target_heading = calc_heading_quat(root_quat)
        source_q0 = torch.from_numpy(ref_root_quat_np).to(self.device)
        source_heading_inv = calc_heading_quat_inv(source_q0)
        self._quat_offset.copy_(quat_mul(target_heading, source_heading_inv))

        logger.info("[Xsens] Waiting for buffer to be stuffed ...")
        while self._xsens_client.buffer.latest()[0] - self._ref_stream_t0 < self.prefill_duration_s:
            time.sleep(0.01)
        logger.info(
            f"[Xsens] Buffer ready; Start tracking ..."
        )

    def _launch_xsens(self) -> None:
        if self._xsens_client is not None:
            self._xsens_client.stop()

        self._xsens_client = XsensOnlineClient(
            host=self.xsens_host,
            port=self.xsens_port,
            scale_factor=self.scale_factor,
            has_hand=self.has_hand
        )

        _prompt(
            "[Xsens] Press Enter to open the TCP server and wait for MVN to connect "
            f"(listening on {self.xsens_host}:{self.xsens_port})..."
        )
        self._xsens_client.listen()
        logger.info("[Xsens] Waiting for MVN TCP connection...")
        self._xsens_client.accept_blocking()
        self._xsens_client.start()
        if not self._xsens_client.wait_first_frame(timeout_s=45.0):
            raise RuntimeError("Xsens: no frames received within timeout.")
        logger.info(
            f"[Xsens] Streaming ready."
        )

    def step(self, tgt_dof_pos, action):

        self.action.copy_(action)
        
        tgt_dof_pos = tgt_dof_pos.detach().cpu().numpy()
        if self.has_hand:
            hand_qpos = self._xsens_client.buffer.latest()[3][None, :]
            tgt_dof_pos = np.concatenate([tgt_dof_pos, hand_qpos], axis=-1)

        self.simulator.apply_action(tgt_dof_pos)

        self.episode_length_buf += 1
        
        obs_dict = self._compute_observation()

        obs_last = obs_dict["body_pos_w_future"][0,-1].cpu()
        self.simulator.update_marker_pos(obs_last)

        return obs_dict

