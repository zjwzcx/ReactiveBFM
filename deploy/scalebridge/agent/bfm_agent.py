import os
import json
import torch
from scalebridge.utils.logging import logger
from scalebridge.agent.base_agent import BaseAgent
from scalebridge.agent.qpos_adapter import ScaleBFMQposTaskAdapter

CONTROL_MODE_DICT = {
    0: "Pelvis",
    1: "Double Hands",
    2: "Pelvis and Double Hands",
    3: "Double Hands and Feet",
    4: "Pelvis and Double Hands and Feet",
    5: "Left and Right Shoulders, Elbows and Hands",
    6: "Pelvis and Left and Right Shoulders, Elbows and Hands",
    7: "Pelvis, Torso, Left and Right Shoulders, Elbows and Hands, Left and Right Hips, Knees and Feet"
}

class BFMAgent(BaseAgent):
    def __init__(self, config, device):

        super().__init__(config, device)

        assert self.device == "cuda", "Compiled tensorrt model should only run on CUDA devices!"
        self.control_mode = torch.tensor([self.config.control_mode], dtype=torch.long, device=self.device)
        logger.info(f"[Agent] Activating control mode {self.config.control_mode}: {CONTROL_MODE_DICT[self.config.control_mode]}")

    def _load_policy(self):
        path_base, _ = os.path.splitext(self.checkpoint)
        meta_path = path_base + "_metadata.json"
        logger.info(f"[Agent] Loading metadata from: {meta_path}")

        with open(meta_path, "r") as f:
            self.meta_data_dict = json.load(f)
        contract = self.meta_data_dict.get("input_contract")
        engine_contract = self.meta_data_dict.get("engine_input_contract")
        if contract != "reactivebfm_qpos_v1" or engine_contract != "scalebfm_transformer_core_v1":
            raise ValueError(
                "ScaleBFM policy is incompatible with online qpos references: "
                f"metadata input_contract={contract!r}, engine_input_contract={engine_contract!r}. "
                "Compile the original checkpoint with "
                "scripts/export_scalebfm_tensorrt.py."
            )

        try:
            import torch_tensorrt  # noqa: F401 - registers TensorRT TorchScript operators before loading
        except ImportError as exc:
            raise RuntimeError(
                "Loading the TensorRT tracking policy requires torch_tensorrt; "
                "install the wheel matching your deployment platform."
            ) from exc

        logger.info(f"[Agent] Loading checkpoint from {self.checkpoint}")
        self.policy = torch.jit.load(self.checkpoint).to(self.device)
        self.policy.eval()
        self.qpos_adapter = ScaleBFMQposTaskAdapter.from_metadata(
            self.meta_data_dict, self.device
        )

        for key, item in self.meta_data_dict.items():
            logger.debug(f"Metadata {key}: {item}")
        
    def get_meta_data(self):
        return self.meta_data_dict

    def get_action(self, obs_dict):
        task_input = self.qpos_adapter(
            obs_dict["root_pos"],
            obs_dict["root_quat_buffer"],
            obs_dict["dof_pos_buffer"],
            obs_dict["ref_root_pos_future"],
            obs_dict["ref_root_rot_future"],
            obs_dict["ref_dof_pos_future"],
            self.control_mode,
            obs_dict["future_time_offsets"],
        )
        return self.policy(
            obs_dict["root_quat_buffer"],
            obs_dict["base_ang_vel_buffer"],
            obs_dict["dof_pos_buffer"],
            obs_dict["dof_vel_buffer"],
            obs_dict["actions_buffer"],
            task_input,
        )
