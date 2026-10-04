import time
import torch
import select
import threading
import numpy as np
from scalebridge.simulator.base_simulator import BaseSimulator
from scalebridge.utils.config import instantiate
from scalebridge.utils.logging import logger

try:
    import lcm
except ImportError:  # defer the failure to RealWorld instantiation
    lcm = None


class RealWorld(BaseSimulator):
    
    def __init__(self, config, metadata_dict):
        self.use_joystick = config.get('joystick', False)
        super().__init__(config, metadata_dict)
        self._init_communication()

    def _setup_backbone(self):
        super()._setup_backbone()
        if lcm is None:
            raise RuntimeError(
                "Real-world deployment requires the `lcm` Python package "
                "(from the Unitree LCM toolchain)."
            )
        self.lcm = lcm.LCM('udpm://239.255.76.67:7667?ttl=255')
    
    def _setup_asset(self):
        super()._setup_asset()
        
        # remote controller metadata
        self.mode = 0
        self.ctrlmode_left = 0
        self.ctrlmode_right = 0
        self.left_stick = [0, 0]
        self.right_stick = [0, 0]
        self.left_upper_switch = 0
        self.left_lower_left_switch = 0
        self.left_lower_right_switch = 0
        self.right_upper_switch = 0
        self.right_lower_left_switch = 0
        self.right_lower_right_switch = 0
        self.left_upper_switch_pressed = 0
        self.left_lower_left_switch_pressed = 0
        self.left_lower_right_switch_pressed = 0
        self.right_upper_switch_pressed = 0
        self.right_lower_left_switch_pressed = 0
        self.right_lower_right_switch_pressed = 0
        self.commands_tmp = np.zeros((3,), dtype=np.float32)

        self.rc_decoder = instantiate(self.cfg.asset.rc_decoder)
        self.state_decoder = instantiate(self.cfg.asset.state_decoder)
        self.command_encoder = instantiate(self.cfg.asset.command_encoder)

        if "enable_root_localization" in self.metadata_dict and self.metadata_dict["enable_root_localization"]:
            self.localization_module = instantiate(self.cfg.asset.localization_module)
        else:
            self.localization_module = None

    def _get_joint_names(self):
        return self.cfg.asset.real_joint_names # weishuai: FIXME maybe it should read from SDK or something else instead of manual assignment

    def _init_communication(self):
        self.firstReceiveRobotState = False
        self.firstReceiveRemoteController = False
        self._last_robot_state_time_s = None

        self._init_time = time.time()
        self._lcm_lock = threading.Lock()
        
        self.robot_state_subscriber = self.lcm.subscribe('robot_state_data', self._robot_state_handler)
        self.remote_controller_subscriber = self.lcm.subscribe('rc_command_data', self._remote_controller_handler)
        
        self.run_thread = threading.Thread(target=self._poll, daemon=False)
        self.run_thread.start()

        logger.info("[Simulator] Waiting for the first robot state signal to arrive ...")
        while not self.firstReceiveRobotState:
            time.sleep(1)          

        if self.localization_module:
            self.localization_module.listen()
            logger.info("[Simulator] Waiting for the localization module client ...")
            addr = self.localization_module.accept_blocking()
            logger.info(f"[Simulator] Localization module client connected: {addr}")
            self.localization_module.start()
            while not len(self.localization_module.buffer) > 0:
                time.sleep(0.02)
            logger.info("[Simulator] First signal from localization module arrives.")
    
    def _robot_state_handler(self, channel, data):
        robot_state = self.state_decoder.decode(data)
        self._last_robot_state_time_s = time.perf_counter()

        self.dof_pos_tmp = np.array(robot_state.q)
        self.dof_vel_tmp = np.array(robot_state.qd)
        self.base_ang_vel_tmp = np.array(robot_state.omegaBody)
        self.root_quat_tmp = np.array(robot_state.quat)

        if not self.firstReceiveRobotState:
            self.time_delay = time.time() - self._init_time
            self.firstReceiveRobotState = True
            logger.info('[Simulator] Communication build successfully between the policy and the transition layer!')
            logger.info(f'[Simulator] First signal arrives after {self.time_delay}s!')
    
    def _remote_controller_handler(self, channel, data):
        msg = self.rc_decoder.decode(data)
        if not self.firstReceiveRemoteController:
            self.firstReceiveRemoteController = True
            logger.info('[Simulator] Communication build successfully between the policy and the remote controller!')
        
        self.left_upper_switch_pressed = ((msg.left_upper_switch and not self.left_upper_switch) or self.left_upper_switch_pressed)
        self.left_lower_left_switch_pressed = ((msg.left_lower_left_switch and not self.left_lower_left_switch) or self.left_lower_left_switch_pressed)
        self.left_lower_right_switch_pressed = ((msg.left_lower_right_switch and not self.left_lower_right_switch) or self.left_lower_right_switch_pressed)
        self.right_upper_switch_pressed = ((msg.right_upper_switch and not self.right_upper_switch) or self.right_upper_switch_pressed)
        self.right_lower_left_switch_pressed = ((msg.right_lower_left_switch and not self.right_lower_left_switch) or self.right_lower_left_switch_pressed)
        self.right_lower_right_switch_pressed = ((msg.right_lower_right_switch) and not self.right_lower_right_switch) or self.right_lower_right_switch_pressed

        self.mode = msg.mode
        self.right_stick = msg.right_stick
        self.left_stick = msg.left_stick
        self.left_upper_switch = msg.left_upper_switch
        self.left_lower_left_switch = msg.left_lower_left_switch
        self.left_lower_right_switch = msg.left_lower_right_switch
        self.right_upper_switch = msg.right_upper_switch
        self.right_lower_left_switch = msg.right_lower_left_switch
        self.right_lower_right_switch = msg.right_lower_right_switch

        self.commands_tmp = np.array([
            self.left_stick[1],
            self.left_stick[0] * -1,
            self.right_stick[0] * -1
        ], dtype=np.float32)
        self.commands_tmp = np.where(np.abs(self.commands_tmp) < 0.05, 0, self.commands_tmp)

    def _poll(self, cb=None):
        try:
            while True:
                self.pump_communication(timeout=0.01, max_messages=8)
        except KeyboardInterrupt:
            pass

    def pump_communication(self, timeout=0.0, max_messages=32):
        """Drain queued LCM packets so robot_state_data is fresh before policy sampling."""
        handled = 0
        with self._lcm_lock:
            for _ in range(max(1, int(max_messages))):
                rfds, _, _ = select.select([self.lcm.fileno()], [], [], timeout if handled == 0 else 0.0)
                if not rfds:
                    break
                self.lcm.handle()
                handled += 1
        return handled

    def refresh_sim(self):

        state_dict = {
            "root_quat_wxyz": self.root_quat_tmp.copy(),
            "base_ang_vel": self.base_ang_vel_tmp.copy(),
            "dof_pos": self.dof_pos_tmp.copy()[self.sim_to_env_joint_idx],
            "dof_vel": self.dof_vel_tmp.copy()[self.sim_to_env_joint_idx],
        }

        if self.localization_module:
            state_dict.update({"root_pos": self.localization_module.get_root_pos()})

        if self.use_joystick:
            state_dict.update({"commands": self.commands_tmp.copy()})

        return {k:torch.from_numpy(v).float() for k,v in state_dict.items()}
    
    def calibrate(self, init_state_dict={}):
        assert not init_state_dict, "Current code does not support reference state initialization for real-world deployment."

        logger.info('[Simulator] Calibraiting..., Press R2 to continue')
        while True:
            if self.right_lower_right_switch_pressed:
                logger.info('[Simulator] R2 button pressed, Start Calibrating...')
                self.right_lower_right_switch_pressed = False
                break
        
        root_quat = self.root_quat_tmp.copy()
        if self.localization_module:
            self.localization_module.calibrate(root_quat)
            root_pos = self.localization_module.get_root_pos()
        else:
            root_pos = np.zeros((3,), dtype=np.float32)
        
        logger.info('[Simulator] Calibration Done. Press R2 to continue')
        while True:
            if self.right_lower_right_switch_pressed:
                logger.info('[Simulator] R2 pressed again, Communication built between policy layer and transition layer!')
                self.right_lower_right_switch_pressed =  False
                break

        return root_pos, root_quat

    def apply_action(self, tgt_dof_pos):
        tgt_dof_pos = tgt_dof_pos.squeeze()

        target_dof_pos_in_sim = np.zeros((self.num_joints,), dtype=np.float32)
        target_dof_pos_in_sim[self.env_action_to_sim_idx] = tgt_dof_pos

        cmd = self.command_encoder
        cmd.q_des = target_dof_pos_in_sim.copy()
        cmd.qd_des = np.zeros_like(target_dof_pos_in_sim)
        cmd.kp = self.stiffness.copy()
        cmd.kd = self.damping.copy()
        cmd.tau_ff = np.zeros_like(target_dof_pos_in_sim)
        cmd.se_contactState = np.zeros(2)
        cmd.timestamp_us = int(time.time()*10**6)

        self.lcm.publish("pd_plustau_targets", cmd.encode())

    def robot_state_age_s(self):
        if self._last_robot_state_time_s is None:
            return None
        return time.perf_counter() - self._last_robot_state_time_s
