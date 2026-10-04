import os
from scalebridge.utils.logging import logger

class BaseAgent:

    def __init__(self, config, device):
        self.config = config
        self.device = device
        logger.info(f"[Agent] Using device {self.device}")

        self.checkpoint = self.config.checkpoint
        assert os.path.exists(self.checkpoint), f"You should assign the correct path to the checkpoint; Current checkpoint is {self.checkpoint}."
        logger.info(f"[Agent] Loading Checkpoint from {self.checkpoint}")

        self._load_policy()

    def _load_policy(self):
        raise NotImplementedError
    
    def get_action(self, obs_dict):
        raise NotImplementedError
    
    def get_meta_data(self): 
        raise NotImplementedError

    def before_step(self, obs_dict):
        pass

    def after_step(self, next_obs_dict, action):
        pass
