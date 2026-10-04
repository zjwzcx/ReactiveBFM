import numpy as np
from scalebridge.utils.logging import logger

class BaseSimulator:

    def __init__(self, config, metadata_dict):

        self.cfg = config
        self.metadata_dict = metadata_dict
        self._setup_backbone()
        self._setup_asset()

    def _setup_backbone(self):
        self.low_dt = self.cfg.low_dt
        self.decimation = self.cfg.decimation
        self.high_dt = self.low_dt * self.decimation

        logger.info(f'[Simulator] Robot-level Control Frequency Set to {1/self.low_dt}HZ')
        logger.info(f'[Simulator] Policy-level Control Frequency Set to {1/self.high_dt}HZ')

    def _setup_asset(self):
        sim_end_joint_names = self._get_joint_names()
        self.num_joints = len(sim_end_joint_names)

        env_end_joint_names = self.metadata_dict["joint_names"] # weishuai: This is for indexing dof_pos and dof_vel
        env_end_action_names = self.metadata_dict.get("target_joint_names", self.metadata_dict["action_names"]) # weishuai: sometimes target joints do not equal to actions
        assert len(env_end_action_names) == len(sim_end_joint_names), f"You have to ensure that sim end and env end have the same number of joints to control."

        self.sim_to_env_joint_idx = np.array([sim_end_joint_names.index(name) for name in env_end_joint_names], dtype=np.int64) # right index
        self.env_action_to_sim_idx = np.array([sim_end_joint_names.index(name) for name in env_end_action_names], dtype=np.int64) # left index
        
        self.stiffness = np.zeros((len(sim_end_joint_names),), dtype=np.float32)
        self.stiffness[self.env_action_to_sim_idx] = np.array(self.metadata_dict["stiffness"], dtype=np.float32)
        
        self.damping = np.zeros((len(sim_end_joint_names),), dtype=np.float32)
        self.damping[self.env_action_to_sim_idx] = np.array(self.metadata_dict["damping"], dtype=np.float32)

        # self.torque_limit = np.zeros((len(sim_end_joint_names),), dtype=np.float32)
        # self.torque_limit[self.env_action_to_sim_idx] = np.array(self.metadata_dict["torque_limit"], dtype=np.float32)

    def _get_joint_names(self):
        raise NotImplementedError
    
    def update_marker_pos(self, marker_pos):
        pass
    
    def refresh_sim(self):
        raise NotImplementedError

    def calibrate(self, init_state_dict={}):
        raise NotImplementedError
    
    def apply_action(self, tgt_dof_pos):
        raise NotImplementedError
