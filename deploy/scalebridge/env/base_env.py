import torch
from scalebridge.utils.config import instantiate
from scalebridge.utils.logging import logger
from scalebridge.utils.quaternion import xyzw_to_wxyz_np

class BaseEnv:
    def __init__(self, config, metadata_dict, device):
        self.cfg = config
        self.metadata_dict = metadata_dict
        assert "joint_names" in metadata_dict, f"Metadata dict should contain the joint names to indicate the joint order for perceiving joint state on env side."
        assert "action_names" in metadata_dict, f"Metadata dict should contain the action names to indicate the action order for applying action from env side." # weishuai: Sometimes the action names may be less than the joint names
        self.device = device
        self.episode_length_buf = torch.zeros((1,), dtype=torch.int64, device=self.device)

        self._setup_metadata()
        self._setup_simulator()
        self._setup_state_manager()
        self._setup_observation_manager()

    def _setup_metadata(self):
        pass

    def _setup_simulator(self):
        self.simulator = instantiate(self.cfg.simulator, metadata_dict=self.metadata_dict)
        self.dt = self.simulator.high_dt
    
    def _setup_state_buffer(self):
        self.buffer_size = self.metadata_dict.get("history_buffer_size", 1)
        logger.info(f"[Env] Initializing state buffer with history length: {self.buffer_size}")
        self.state_buffer = {
            "root_pos_buffer": torch.zeros(1, self.buffer_size, 3, dtype=torch.float, device=self.device),
            "root_quat_wxyz_buffer": torch.zeros(1, self.buffer_size, 4, dtype=torch.float, device=self.device),
            "base_ang_vel_buffer": torch.zeros(1, self.buffer_size, 3, dtype=torch.float, device=self.device),
            "dof_pos_buffer": torch.zeros(1, self.buffer_size, len(self.metadata_dict["joint_names"]), dtype=torch.float, device=self.device),
            "dof_vel_buffer": torch.zeros(1, self.buffer_size, len(self.metadata_dict["joint_names"]), dtype=torch.float, device=self.device),
            "action_buffer": torch.zeros(1, self.buffer_size, len(self.metadata_dict["action_names"]), dtype=torch.float, device=self.device),
        }

    def _setup_state_manager(self):
        self._setup_state_buffer()

        state_dict_template = self.simulator.refresh_sim()
        for name, term in state_dict_template.items():
            assert f"{name}_buffer" in self.state_buffer and term.shape[-1] == self.state_buffer[f"{name}_buffer"].shape[-1], f"To ensure effcient parsing of the state dict, you may initialize the {name} buffer in state_buffer and pre-allocate memory on desired devices."

        self.action = torch.zeros(1, len(self.metadata_dict["action_names"]), dtype=torch.float, device=self.device)

    def _update_state_manager(self):
        cur_state_dict = self.simulator.refresh_sim()

        for key in cur_state_dict:
            self.state_buffer[f"{key}_buffer"] = torch.roll(self.state_buffer[f"{key}_buffer"], -1, 1)
            self.state_buffer[f"{key}_buffer"][:, -1].copy_(cur_state_dict[key], non_blocking=True)
        
        self.state_buffer["action_buffer"] = torch.roll(self.state_buffer["action_buffer"], -1, 1)
        self.state_buffer["action_buffer"][:, -1].copy_(self.action)

    def _setup_observation_manager(self):
        self.obs_func_names = []
        self.obs_func_list = []
        
        for name, cfg in self.cfg.observation.items():
            obs_func = instantiate(cfg.func)
            dummy_obs = obs_func(state_buffer=self.state_buffer)
            logger.info(f"[Env] Observation Term: {name}; Shape: {dummy_obs.shape}.")
            
            self.obs_func_names.append(name)
            self.obs_func_list.append(obs_func)
        
    def _update_observation_manager(self):
        return {name: func(state_buffer=self.state_buffer) for name, func in zip(self.obs_func_names, self.obs_func_list)}

    def _compute_observation(self):
        self._update_state_manager()
        return self._update_observation_manager()
    
    def _calibrate(self):
        self.simulator.calibrate()

    def reset(self, init_state_dict=None):
        """Reset the env, optionally from an explicit initial robot state.

        Interface convention (planner-facing): ``init_state_dict`` follows the
        ReactiveBFM data layout, so the root quaternion is passed as
        ``root_quat_xyzw`` in **xyzw** order. Simulators consume **wxyz**
        (MuJoCo qpos order); the conversion happens here, at the interface
        boundary. Recognized keys: ``root_pos``, ``root_quat_xyzw``,
        ``dof_pos``, and optionally the ``*_vel`` counterparts understood by
        the simulator's ``calibrate``.
        """
        self.episode_length_buf *=0
        self.action *=0
        self.state_buffer["action_buffer"][:] *= 0

        if init_state_dict is None:
            self._calibrate()
        else:
            init_state_dict = dict(init_state_dict)
            if "root_quat_xyzw" in init_state_dict:
                # Boundary conversion: xyzw (planner/data) -> wxyz (simulator).
                init_state_dict["root_quat"] = xyzw_to_wxyz_np(
                    init_state_dict.pop("root_quat_xyzw")
                )
            self.simulator.calibrate(init_state_dict)

        cur_state_dict = self.simulator.refresh_sim()
        for key in cur_state_dict:
            self.state_buffer[f"{key}_buffer"].copy_(
                torch.broadcast_to(cur_state_dict[key], self.state_buffer[f"{key}_buffer"].shape), non_blocking=True
            )

        return self._update_observation_manager()
    
    def step(self, tgt_dof_pos, action):
        self.action.copy_(action)
        self.simulator.apply_action(tgt_dof_pos.detach().cpu().numpy())

        self.episode_length_buf += 1
        return self._compute_observation()
