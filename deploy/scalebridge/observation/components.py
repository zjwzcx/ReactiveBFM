import torch
from torch import Tensor
from typing import Dict
from scalebridge.utils.torch_utils import quat_apply_inverse, quat_mul_inverse_left

def root_pos(state_buffer: Dict[str, Tensor]):
    return state_buffer["root_pos_buffer"][:, -1]

def root_quat_buffer(state_buffer: Dict[str, Tensor]):
    return state_buffer["root_quat_wxyz_buffer"]

def base_ang_vel_buffer(state_buffer: Dict[str, Tensor]):
    return state_buffer["base_ang_vel_buffer"]

def dof_pos_buffer(state_buffer: Dict[str, Tensor]):
    return state_buffer["dof_pos_buffer"]

def dof_vel_buffer(state_buffer: Dict[str, Tensor]):
    return state_buffer["dof_vel_buffer"]

def actions_buffer(state_buffer: Dict[str, Tensor]):
    return state_buffer["action_buffer"]

def ref_root_pos_future(state_buffer: Dict[str, Tensor]):
    return state_buffer["ref_root_pos_future"]

def ref_root_rot_future(state_buffer: Dict[str, Tensor]):
    return state_buffer["ref_root_rot_future"]

def ref_dof_pos_future(state_buffer: Dict[str, Tensor]):
    return state_buffer["ref_dof_pos_future"]

def body_pos_w_future(state_buffer: Dict[str, Tensor]):
    return state_buffer["body_pos_w_future"]

def body_quat_w_future(state_buffer: Dict[str, Tensor]):
    return state_buffer["body_quat_w_wxyz_future"]

@torch.jit.script
def target_body_pos_future_to_robot_base(state_buffer: Dict[str, Tensor]):
    return quat_apply_inverse(state_buffer["root_quat_wxyz_buffer"][:,-1], state_buffer["body_pos_w_future"] - state_buffer["root_pos_buffer"][:, -1][:, None, None, :])

@torch.jit.script
def target_body_rot_future_to_robot_base(state_buffer: Dict[str, Tensor]):
    body_quat_w_future = state_buffer["body_quat_w_wxyz_future"]
    root_quat_expand = state_buffer["root_quat_wxyz_buffer"][:,-1][:, None, None, :].expand(-1, body_quat_w_future.shape[1], body_quat_w_future.shape[2], -1)
    return quat_mul_inverse_left(root_quat_expand, body_quat_w_future)

def future_time_offsets(state_buffer: Dict[str, Tensor]):
    return state_buffer["future_frame_offset"]

def cur_object_pos(state_buffer: Dict[str, Tensor]):
    return state_buffer["object_root_pos_buffer"][:, -1, 0]

def cur_object_quat(state_buffer: Dict[str, Tensor]):
    return state_buffer["object_root_quat_wxyz_buffer"][:, -1, 0]

def object_pos_w_future(state_buffer: Dict[str, Tensor]):
    return state_buffer["object_pos_w_future"][:,:,0]

def object_quat_w_future(state_buffer: Dict[str, Tensor]):
    return state_buffer["object_quat_w_wxyz_future"][:, :, 0]

def object_bps_feature(state_buffer: Dict[str, Tensor]):
    return state_buffer["object_bps_feature"][:, 0] # weishuai: All the object operations default to single object so we just extract the 0-dim