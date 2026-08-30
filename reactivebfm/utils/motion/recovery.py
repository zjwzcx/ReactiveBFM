"""Motion representation recovery utilities."""

import torch

from reactivebfm.utils.motion.quaternion import (
    euler2quat,
    qinv,
    qmul,
    qrot,
    quaternion_to_cont6d,
)


def recover_root_rot_pos(data, hml_type=None):
    if hml_type == "global":
        raise ValueError("Function only applicable for None and global_root representations.")
    if hml_type == "global_root":
        r_rot_ang = data[..., 0]
    else:
        rot_vel = data[..., 0]
        r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
        r_rot_ang[..., 1:] = rot_vel[..., :-1]
        r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    if hml_type == "global_root":
        r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
        r_pos[..., [0, 2]] = data[..., 1:3]
    else:
        r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
        r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
        r_pos = qrot(qinv(r_rot_quat), r_pos)
        r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_from_rot(data, joints_num, skeleton):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)

    start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
    end_indx = start_indx + (joints_num - 1) * 6
    cont6d_params = data[..., start_indx:end_indx]
    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
    cont6d_params = cont6d_params.view(-1, joints_num, 6)
    return skeleton.forward_kinematics_cont6d(cont6d_params, r_pos)


def recover_rot(data):
    joints_num = 22 if data.shape[-1] == 263 else 21
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    r_pos_pad = torch.cat([r_pos, torch.zeros_like(r_pos)], dim=-1).unsqueeze(-2)
    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)
    start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
    end_indx = start_indx + (joints_num - 1) * 6
    cont6d_params = data[..., start_indx:end_indx]
    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
    cont6d_params = cont6d_params.view(-1, joints_num, 6)
    cont6d_params = torch.cat([cont6d_params, r_pos_pad], dim=-2)
    return cont6d_params


def recover_from_ric(data, joints_num, hml_type=None):
    r_rot_quat, r_pos = recover_root_rot_pos(data, hml_type)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]
    return torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)


def traj_global2vel(traj_positions, traj_rot):
    bs, _, seqlen = traj_positions.shape
    traj_positions = traj_positions.permute(0, 2, 1)
    euler = torch.zeros([bs, 3, seqlen], dtype=traj_rot.dtype, device=traj_rot.device)
    euler[:, 1:2] = traj_rot
    euler = euler.permute(0, 2, 1).contiguous()
    traj_rot_quat = euler2quat(euler, "yxz", deg=False)

    r_rot = traj_rot_quat.clone()
    velocity = torch.zeros_like(euler[:, 1:, :])
    velocity[:, :, [0, 2]] = (traj_positions[:, 1:, :] - traj_positions[:, :-1, :]).clone()
    velocity = qrot(r_rot[:, 1:], velocity)
    r_velocity = qmul(r_rot[:, 1:].contiguous(), qinv(r_rot[:, :-1]))

    r_velocity = torch.arcsin(r_velocity[:, :, 2:3])
    l_velocity = velocity[:, :, [0, 2]]
    return torch.cat([r_velocity, l_velocity], axis=-1).permute(0, 2, 1)[:, :, None]

__all__ = [
    "qrot",
    "recover_rot",
    "recover_from_ric",
    "recover_from_rot",
    "recover_root_rot_pos",
    "traj_global2vel",
]
