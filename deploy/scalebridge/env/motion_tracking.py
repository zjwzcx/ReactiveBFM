import os
import torch
import numpy as np
from scalebridge.utils.logging import logger
from scalebridge.env.base_env import BaseEnv
from scalebridge.utils.torch_utils import calc_heading_quat, calc_heading_quat_inv, quat_apply, quat_mul

class MotionTrackingEnv(BaseEnv):

    def _setup_metadata(self):
        # weishuai: We explicitly pass motion dict as arguments to support heritage on motion data processing
        assert os.path.exists(self.cfg.motion_path), f"You have to ensure the correct path to motion file. Current motion path is: {self.cfg.motion_path}."
        logger.info(f"[Env] Loading offline motion trajectory from {self.cfg.motion_path}.")
        motion_data = np.load(self.cfg.motion_path)
        assert motion_data['fps'] == 50, "You should first process the data to ensure the same format and FPS with IsaacLab compatible format."
        self._setup_motion(motion_data)

        self.reference_forcing = self.cfg.get('reference_forcing', True)
        logger.info(f"[Env] Using reference root position to apply forcing: {self.reference_forcing}")
        self.metadata_dict["enable_root_localization"] = not self.reference_forcing

    def _global_time_indices(self, frame_offsets):
        return torch.clamp(
            self.episode_length_buf.unsqueeze(-1) + frame_offsets,
            min=0, max=self.motion_len - 1
        )
    
    def _setup_motion(self, motion_data):
        motion_keys = set(motion_data.files)

        def pick_key(*candidates):
            for key in candidates:
                if key in motion_keys:
                    return key
            raise KeyError(f"Missing motion keys. Tried: {candidates}")

        dof_pos_key = pick_key("dof_pos", "joint_pos")
        dof_vel_key = None
        for key in ("dof_vel", "joint_vel"):
            if key in motion_keys:
                dof_vel_key = key
                break

        body_indexes = None
        if "body_pos_w" in motion_keys and "body_quat_w" in motion_keys:
            selected_links = self.metadata_dict["selected_body_names"]
            complete_links = self.metadata_dict["body_names"]
            body_indexes = np.array([complete_links.index(link) for link in selected_links])

        if "root_pos" in motion_keys:
            root_pos = motion_data["root_pos"]
        elif body_indexes is not None:
            root_pos = motion_data["body_pos_w"][:, body_indexes[0]]
        else:
            raise KeyError("Motion must provide root_pos or body_pos_w/body_quat_w")

        root_rot_key = next((key for key in ("root_rot", "root_quat_w", "root_quat") if key in motion_keys), None)
        if root_rot_key is not None:
            root_rot = motion_data[root_rot_key]
        elif body_indexes is not None:
            root_rot = motion_data["body_quat_w"][:, body_indexes[0]]
        else:
            raise KeyError("Motion must provide root rotation or body_pos_w/body_quat_w")

        self.ref_root_pos = torch.from_numpy(root_pos).to(self.device, dtype=torch.float32)
        self.ref_root_rot = torch.from_numpy(root_rot).to(self.device, dtype=torch.float32)
        self.ref_dof_pos = torch.from_numpy(motion_data[dof_pos_key]).to(self.device, dtype=torch.float32)
        if self.ref_root_pos.ndim != 2 or self.ref_root_pos.shape[-1] != 3:
            raise ValueError(f"root_pos must have shape (T, 3), got {tuple(self.ref_root_pos.shape)}")
        if self.ref_root_rot.ndim != 2 or self.ref_root_rot.shape[-1] != 4:
            raise ValueError(f"root rotation must have shape (T, 4), got {tuple(self.ref_root_rot.shape)}")
        if self.ref_dof_pos.ndim != 2 or self.ref_dof_pos.shape[-1] != len(self.metadata_dict["joint_names"]):
            raise ValueError(
                "dof_pos must have shape "
                f"(T, {len(self.metadata_dict['joint_names'])}), got {tuple(self.ref_dof_pos.shape)}"
            )
        if not (len(self.ref_root_pos) == len(self.ref_root_rot) == len(self.ref_dof_pos)):
            raise ValueError("Reference root and joint arrays must have the same length")
        self.motion_len = len(self.ref_root_pos)
        self.future_frame_offset = torch.as_tensor(self.cfg.future_idx, dtype=torch.long, device=self.device)
        self.joint_pos = motion_data[dof_pos_key]
        if dof_vel_key is not None:
            self.joint_vel = motion_data[dof_vel_key]
        else:
            self.joint_vel = np.zeros_like(self.joint_pos)

        self.body_pos_w = None
        self.body_quat_w = None
        self.body_lin_vel_w = None
        self.body_ang_vel_w = None
        if body_indexes is not None:
            self.body_pos_w = torch.from_numpy(motion_data["body_pos_w"][:, body_indexes]).to(self.device)
            self.body_quat_w = torch.from_numpy(motion_data["body_quat_w"][:, body_indexes]).to(self.device) # wxyz
            if "body_lin_vel_w" in motion_keys:
                self.body_lin_vel_w = motion_data["body_lin_vel_w"][:, body_indexes]
            if "body_ang_vel_w" in motion_keys:
                self.body_ang_vel_w = motion_data["body_ang_vel_w"][:, body_indexes]

    def _gather_reference_state(self):
        temporal_index = self._global_time_indices(self.future_frame_offset).reshape(-1)
        self.state_buffer.update({
            "ref_root_pos_future": self.ref_root_pos.index_select(0, temporal_index).unsqueeze(0),
            "ref_root_rot_future": self.ref_root_rot.index_select(0, temporal_index).unsqueeze(0),
            "ref_dof_pos_future": self.ref_dof_pos.index_select(0, temporal_index).unsqueeze(0),
            "future_frame_offset": self.future_frame_offset[None, :, None]
        })

        if self.body_pos_w is not None and self.body_quat_w is not None:
            self.state_buffer.update({
                "body_pos_w_future": self.body_pos_w.index_select(0, temporal_index).unsqueeze(0),
                "body_quat_w_wxyz_future": self.body_quat_w.index_select(0, temporal_index).unsqueeze(0),
            })
        
        if self.reference_forcing:
            self.state_buffer["root_pos_buffer"][:, -1] = self.state_buffer["ref_root_pos_future"][:, 0]

    def _setup_state_manager(self):
        super()._setup_state_manager()
        self._gather_reference_state()
        
    def _update_state_manager(self):
        super()._update_state_manager()
        self._gather_reference_state()
        
    def _calibrate(self):
        
        use_rsi = self.cfg.get('rsi', False)
        
        if use_rsi:
            logger.warning("Reference State Initialization activated! This should only be used in simulator!")
            if self.body_lin_vel_w is None or self.body_ang_vel_w is None:
                raise RuntimeError(
                    "RSI requires motion data with body_lin_vel_w and body_ang_vel_w."
                )
            init_state_dict = {
                "root_pos": self.ref_root_pos[0].cpu().numpy(),
                "root_quat": self.ref_root_rot[0].cpu().numpy(),
                "root_lin_vel": self.body_lin_vel_w[0,0],
                "root_ang_vel": self.body_ang_vel_w[0,0],
                "dof_pos": self.joint_pos[0],
                "dof_vel": self.joint_vel[0],
            }
        else:
            init_state_dict = {}

        root_pos, root_quat = self.simulator.calibrate(init_state_dict) # weishuai: We default to not applying RSI to pure motion tracking
        
        if not use_rsi:
            self._update_state_manager() # weishuai: This would not affect the initial model context; Later it would get overwritten in reset
            
            # weishuai: We do not use RSI but adjust motion based on the current state;
            logger.info("[Env] Adjusting the xy-offset of offline trajectories ...")
            pos_offset = self.ref_root_pos[0].clone()
            pos_offset[..., -1] = 0

            logger.info("[Env] Adjusting the heading direction of offline trajectories ...")
            root_quat_wxyz = torch.from_numpy(root_quat).float().to(self.device)
            target_heading = calc_heading_quat(root_quat_wxyz)
            source_q0 = self.ref_root_rot[0]
            source_heading_inv = calc_heading_quat_inv(source_q0)
            
            q_align = quat_mul(target_heading, source_heading_inv) # (4)
            q_align_expand = q_align[None,:].expand(self.ref_root_rot.shape[0], -1)

            self.ref_root_rot = quat_mul(q_align_expand, self.ref_root_rot)
            self.ref_root_pos = quat_apply(q_align_expand, self.ref_root_pos - pos_offset)

            if self.body_quat_w is not None and self.body_pos_w is not None:
                q_align_expand = q_align[None,None,:].expand(self.body_quat_w.shape[0], self.body_quat_w.shape[1], -1)
                self.body_quat_w = quat_mul(q_align_expand, self.body_quat_w)
                self.body_pos_w = quat_apply(q_align_expand, self.body_pos_w - pos_offset)

    def step(self, tgt_dof_pos, action):
        obs_dict = super().step(tgt_dof_pos, action)
        if "body_pos_w_future" in obs_dict:
            self.simulator.update_marker_pos(obs_dict["body_pos_w_future"][0,0])
        else:
            self.simulator.update_marker_pos(obs_dict["ref_root_pos_future"][0,0].unsqueeze(0))
        return obs_dict
