import torch
from scalebridge.utils.logging import logger
from scalebridge.utils.quaternion import xyzw_to_wxyz_torch

from scalebridge.env.base_env import BaseEnv


class MotionTrackingOnlineEnv(BaseEnv):
    def _setup_metadata(self):
        self.future_frame_offset = torch.as_tensor(self.cfg.future_idx, dtype=torch.long, device=self.device)
        self.reference_forcing = self.cfg.get("reference_forcing", True)
        self.default_root_height = float(self.cfg.get("root_height", self.cfg.simulator.config.asset.get("root_height", 0.0)))
        self._reference_ready = False
        self._reference_motion = None
        self.metadata_dict["enable_root_localization"] = not self.reference_forcing
        logger.info(f"[Env] Using online generated reference; future_idx={list(self.cfg.future_idx)}")
        logger.info(f"[Env] Using reference root position to apply forcing: {self.reference_forcing}")

    def _setup_state_manager(self):
        super()._setup_state_manager()
        self._sim_has_root_pos = "root_pos" in self.simulator.refresh_sim()
        num_future_frames = int(self.future_frame_offset.shape[0])
        num_joints = len(self.metadata_dict["joint_names"])
        self.state_buffer.update(
            {
                "ref_root_pos_future": torch.zeros(1, num_future_frames, 3, dtype=torch.float, device=self.device),
                "ref_root_rot_future": torch.zeros(1, num_future_frames, 4, dtype=torch.float, device=self.device),
                "ref_dof_pos_future": torch.zeros(1, num_future_frames, num_joints, dtype=torch.float, device=self.device),
                "future_frame_offset": self.future_frame_offset[None, :, None],
            }
        )
        self.state_buffer["ref_root_rot_future"][..., 0] = 1.0

    def _apply_default_root_pos_if_needed(self):
        if not self._sim_has_root_pos:
            self.state_buffer["root_pos_buffer"][..., 2] = self.default_root_height

    @property
    def reference_length(self):
        if self._reference_motion is None:
            return 0
        return int(self._reference_motion["root_pos"].shape[0])

    def _prepare_reference_motion(self, reference_motion):
        required_keys = ("root_pos", "root_quat_xyzw", "dof_pos_policy")
        missing = [key for key in required_keys if key not in reference_motion]
        if missing:
            raise KeyError(f"Generated reference is missing keys: {missing}")

        root_pos = reference_motion["root_pos"].to(self.device, dtype=torch.float32)
        # Boundary conversion: planner/data convention is xyzw, ScaleBridge
        # internals are wxyz (see scalebridge.utils.quaternion).
        root_rot = xyzw_to_wxyz_torch(
            reference_motion["root_quat_xyzw"].to(self.device, dtype=torch.float32)
        )
        dof_pos = reference_motion["dof_pos_policy"].to(self.device, dtype=torch.float32)
        if root_pos.ndim != 2 or root_pos.shape[-1] != 3:
            raise ValueError(f"`root_pos` should have shape (T, 3), got {tuple(root_pos.shape)}.")
        if root_rot.ndim != 2 or root_rot.shape[-1] != 4:
            raise ValueError(f"`root_quat_xyzw` should have shape (T, 4), got {tuple(root_rot.shape)}.")
        if dof_pos.ndim != 2 or dof_pos.shape[-1] != len(self.metadata_dict["joint_names"]):
            raise ValueError(
                "`dof_pos_policy` should have shape "
                f"(T, {len(self.metadata_dict['joint_names'])}), got {tuple(dof_pos.shape)}."
            )
        if not (root_pos.shape[0] == root_rot.shape[0] == dof_pos.shape[0]):
            raise ValueError("Generated reference arrays must have the same length.")

        return {
            "root_pos": root_pos,
            "root_rot_wxyz": root_rot,
            "dof_pos_policy": dof_pos,
        }

    def _blend_reference_prefix(self, current_reference, new_reference, blend_frames):
        blend_frames = int(blend_frames)
        if current_reference is None or blend_frames <= 0:
            return new_reference

        num_blend = min(
            blend_frames,
            int(current_reference["root_pos"].shape[0]),
            int(new_reference["root_pos"].shape[0]),
        )
        if num_blend <= 0:
            return new_reference

        weights = torch.linspace(
            0.0,
            1.0,
            num_blend + 2,
            dtype=torch.float32,
            device=self.device,
        )[1:-1, None]

        blended_root_pos = (1.0 - weights) * current_reference["root_pos"][:num_blend] + weights * new_reference[
            "root_pos"
        ][:num_blend]
        blended_dof_pos = (1.0 - weights) * current_reference["dof_pos_policy"][:num_blend] + weights * new_reference[
            "dof_pos_policy"
        ][:num_blend]

        q0 = current_reference["root_rot_wxyz"][:num_blend]
        q1 = new_reference["root_rot_wxyz"][:num_blend]
        same_hemisphere = (q0 * q1).sum(dim=-1, keepdim=True) >= 0.0
        q1 = torch.where(same_hemisphere, q1, -q1)
        blended_root_rot = (1.0 - weights) * q0 + weights * q1
        blended_root_rot = torch.nn.functional.normalize(blended_root_rot, dim=-1)

        return {
            "root_pos": torch.cat([blended_root_pos, new_reference["root_pos"][num_blend:]], dim=0),
            "root_rot_wxyz": torch.cat([blended_root_rot, new_reference["root_rot_wxyz"][num_blend:]], dim=0),
            "dof_pos_policy": torch.cat([blended_dof_pos, new_reference["dof_pos_policy"][num_blend:]], dim=0),
        }

    def set_reference_motion(self, reference_motion, blend_frames=0):
        """Install a planner-generated reference trajectory.

        Interface convention (planner-facing): ``reference_motion`` follows the
        ReactiveBFM qpos36 data layout, i.e. the root quaternion is **xyzw**
        under the key ``root_quat_xyzw``. ScaleBridge internals (MuJoCo qpos,
        tracking policy, state buffers) use **wxyz**; the xyzw -> wxyz
        conversion happens exactly once here, at the interface boundary.

        Required keys, all torch tensors of equal length T:
            root_pos        (T, 3)
            root_quat_xyzw  (T, 4)   # xyzw, converted to internal wxyz below
            dof_pos_policy  (T, num_policy_joints)
        """
        prepared_reference = self._prepare_reference_motion(reference_motion)
        prepared_reference = self._blend_reference_prefix(self._reference_motion, prepared_reference, blend_frames)

        self._reference_motion = {
            key: value.contiguous() for key, value in prepared_reference.items()
        }
        self._reference_ready = True
        self._gather_reference_state()

    def get_observation(self):
        return self._update_observation_manager()

    def refresh_observation(self):
        return self._compute_observation()

    def _advance_reference(self):
        if self.reference_length > 1:
            self._reference_motion = {
                key: value[1:] for key, value in self._reference_motion.items()
            }

    def _gather_reference_state(self):
        if not self._reference_ready:
            return

        temporal_index = torch.clamp(self.future_frame_offset, min=0, max=self.reference_length - 1)
        self.state_buffer.update(
            {
                "ref_root_pos_future": self._reference_motion["root_pos"].index_select(0, temporal_index).unsqueeze(0),
                "ref_root_rot_future": self._reference_motion["root_rot_wxyz"].index_select(0, temporal_index).unsqueeze(0),
                "ref_dof_pos_future": self._reference_motion["dof_pos_policy"].index_select(0, temporal_index).unsqueeze(0),
                "future_frame_offset": self.future_frame_offset[None, :, None],
            }
        )

        if self.reference_forcing:
            self.state_buffer["root_pos_buffer"][:, -1] = self.state_buffer["ref_root_pos_future"][:, 0]

    def _update_state_manager(self):
        super()._update_state_manager()
        self._gather_reference_state()

    def reset(self, init_state_dict=None):
        super().reset(init_state_dict=init_state_dict)
        self._apply_default_root_pos_if_needed()
        self._gather_reference_state()
        return self._update_observation_manager()

    def step(self, tgt_dof_pos, action):
        self.apply_control(tgt_dof_pos, action)
        return self._compute_observation()

    def apply_control(self, tgt_dof_pos, action):
        self.action.copy_(action)
        self.simulator.update_marker_pos(self.state_buffer["ref_root_pos_future"][0, 0].unsqueeze(0))
        self.simulator.apply_action(tgt_dof_pos.detach().cpu().numpy())

        self.episode_length_buf += 1
        self._advance_reference()
