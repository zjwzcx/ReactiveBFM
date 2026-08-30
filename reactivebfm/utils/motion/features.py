"""Feature extraction for T2M-compatible motion representations."""

# Source: https://github.com/EricGuo5513/text-to-motion/blob/main/scripts/motion_process.py
import numpy as np
import torch

from reactivebfm.data.motion import t2m_kinematic_chain, t2m_raw_offsets
from reactivebfm.utils.motion.quaternion import *  # noqa: F403
from reactivebfm.utils.motion.skeleton import Skeleton


def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    r_pos = qrot(qinv(r_rot_quat), r_pos)  # noqa: F405

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_from_rot(data, joints_num, skeleton):
    r_rot_quat, r_pos = recover_root_rot_pos(data)

    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)  # noqa: F405

    start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
    end_indx = start_indx + (joints_num - 1) * 6
    cont6d_params = data[..., start_indx:end_indx]
    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
    cont6d_params = cont6d_params.view(-1, joints_num, 6)

    return skeleton.forward_kinematics_cont6d(cont6d_params, r_pos)


def recover_from_ric(data, joints_num):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)  # noqa: F405
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]
    return torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)


def extract_features_t2m(
    positions,
    feet_thre=0.002,
    n_raw_offsets=t2m_raw_offsets,
    kinematic_chain=t2m_kinematic_chain,
    face_joint_indx=None,
    fid_r=None,
    fid_l=None,
    fix_ik_bug=False,
):
    face_joint_indx = [2, 1, 17, 16] if face_joint_indx is None else face_joint_indx
    fid_r = [8, 11] if fid_r is None else fid_r
    fid_l = [7, 10] if fid_l is None else fid_l
    return extract_features_torch(
        positions,
        feet_thre,
        torch.from_numpy(n_raw_offsets),
        kinematic_chain,
        face_joint_indx,
        fid_r,
        fid_l,
        fix_ik_bug=fix_ik_bug,
    )


def extract_features(positions, feet_thre, n_raw_offsets, kinematic_chain, face_joint_indx, fid_r, fid_l):
    global_positions = positions.clone()
    feet_l, feet_r = foot_detect(positions, feet_thre, fid_l, fid_r)

    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(
        positions, n_raw_offsets, kinematic_chain, face_joint_indx
    )
    positions = get_rifke(positions, r_rot)

    root_y = positions[:, 0, 1:2]
    r_velocity = np.arcsin(r_velocity[:, 2:3])
    l_velocity = velocity[:, [0, 2]]
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)
    ric_data = positions[:, 1:].reshape(len(positions), -1)

    local_vel = qrot_np(  # noqa: F405
        np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
        global_positions[1:] - global_positions[:-1],
    )
    local_vel = local_vel.reshape(len(local_vel), -1)

    data = root_data
    data = np.concatenate([data, ric_data[:-1]], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    data = np.concatenate([data, local_vel], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)
    return data


def extract_features_torch(
    positions,
    feet_thre,
    n_raw_offsets,
    kinematic_chain,
    face_joint_indx,
    fid_r,
    fid_l,
    fix_ik_bug=False,
):
    bs, n_frames, _, _ = positions.shape
    global_positions = positions.clone()
    positions = positions.clone()
    feet_l, feet_r = foot_detect_torch(positions, feet_thre, fid_l, fid_r)

    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params_torch(
        positions, n_raw_offsets, kinematic_chain, face_joint_indx, fix_ik_bug=fix_ik_bug
    )

    recon_data = {
        "r_rot": r_rot[:, -2].clone(),
        "r_pos": positions[:, -2, 0].clone(),
    }

    positions = get_rifke_torch(positions, r_rot)

    root_y = positions[:, :, 0, 1:2]
    r_velocity = torch.arcsin(r_velocity[:, :, 2:3])
    l_velocity = velocity[:, :, [0, 2]]
    root_data = torch.cat([r_velocity, l_velocity, root_y[:, :-1]], axis=-1)

    rot_data = cont_6d_params[:, :, 1:].reshape(bs, n_frames, -1)
    ric_data = positions[:, :, 1:].reshape(bs, n_frames, -1)

    local_vel = qrot(  # noqa: F405
        torch.repeat_interleave(r_rot[:, :-1, None], global_positions.shape[2], axis=2),
        global_positions[:, 1:] - global_positions[:, :-1],
    )
    local_vel = local_vel.reshape(bs, n_frames - 1, -1)

    data = root_data
    data = torch.cat([data, ric_data[:, :-1]], axis=-1)
    data = torch.cat([data, rot_data[:, :-1]], axis=-1)
    data = torch.cat([data, local_vel], axis=-1)
    data = torch.cat([data, feet_l, feet_r], axis=-1)
    return data, recon_data


def foot_detect(positions, thres, fid_l, fid_r):
    velfactor = np.array([thres, thres])

    feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
    feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
    feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
    feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(float)

    feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
    feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
    feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
    feet_r = ((feet_r_x + feet_r_y + feet_r_z) < velfactor).astype(float)
    return feet_l, feet_r


def foot_detect_torch(positions, thres, fid_l, fid_r):
    velfactor = torch.tensor([thres, thres]).to(positions.device)

    feet_l_x = (positions[:, 1:, fid_l, 0] - positions[:, :-1, fid_l, 0]) ** 2
    feet_l_y = (positions[:, 1:, fid_l, 1] - positions[:, :-1, fid_l, 1]) ** 2
    feet_l_z = (positions[:, 1:, fid_l, 2] - positions[:, :-1, fid_l, 2]) ** 2
    feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).float()

    feet_r_x = (positions[:, 1:, fid_r, 0] - positions[:, :-1, fid_r, 0]) ** 2
    feet_r_y = (positions[:, 1:, fid_r, 1] - positions[:, :-1, fid_r, 1]) ** 2
    feet_r_z = (positions[:, 1:, fid_r, 2] - positions[:, :-1, fid_r, 2]) ** 2
    feet_r = ((feet_r_x + feet_r_y + feet_r_z) < velfactor).float()
    return feet_l, feet_r


def get_rifke(positions, r_rot):
    positions[..., 0] -= positions[:, 0:1, 0]
    positions[..., 2] -= positions[:, 0:1, 2]
    return qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)  # noqa: F405


def get_rifke_torch(positions, r_rot):
    positions[..., 0] -= positions[..., 0:1, 0].clone()
    positions[..., 2] -= positions[..., 0:1, 2].clone()
    return qrot(  # noqa: F405
        torch.repeat_interleave(r_rot[:, :, None], positions.shape[2], axis=2), positions
    )


def get_quaternion(positions, n_raw_offsets, kinematic_chain, face_joint_indx):
    skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
    quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=False)

    quat_params = qfix(quat_params)  # noqa: F405
    r_rot = quat_params[:, 0].copy()
    velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
    velocity = qrot_np(r_rot[1:], velocity)  # noqa: F405
    r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))  # noqa: F405
    quat_params[1:, 0] = r_velocity
    return quat_params, r_velocity, velocity, r_rot


def get_quaternion_torch(positions, n_raw_offsets, kinematic_chain, face_joint_indx):
    skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
    quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=False)

    quat_params = qfix(quat_params)  # noqa: F405
    r_rot = quat_params[:, 0].copy()
    velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
    velocity = qrot_np(r_rot[1:], velocity)  # noqa: F405
    r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))  # noqa: F405
    quat_params[1:, 0] = r_velocity
    return quat_params, r_velocity, velocity, r_rot


def get_cont6d_params(positions, n_raw_offsets, kinematic_chain, face_joint_indx):
    skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
    quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)

    cont_6d_params = quaternion_to_cont6d_np(quat_params)  # noqa: F405
    r_rot = quat_params[:, 0].copy()
    velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
    velocity = qrot_np(r_rot[1:], velocity)  # noqa: F405
    r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))  # noqa: F405
    return cont_6d_params, r_velocity, velocity, r_rot


def get_cont6d_params_torch(positions, n_raw_offsets, kinematic_chain, face_joint_indx, fix_ik_bug):
    skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
    bs, n_frames, n_joints, n_dim = positions.shape
    quat_params = skel.inverse_kinematics_np(
        positions.reshape(-1, n_joints, n_dim).cpu().numpy(),
        face_joint_indx,
        smooth_forward=False,
        fix_bug=fix_ik_bug,
    )
    quat_params = torch.from_numpy(quat_params).reshape(bs, n_frames, n_joints, -1)
    quat_params = quat_params.float().to(positions.device)

    cont_6d_params = quaternion_to_cont6d(quat_params)  # noqa: F405
    r_rot = quat_params[:, :, 0].contiguous().clone()
    velocity = (positions[:, 1:, 0] - positions[:, :-1, 0]).clone()
    velocity = qrot(r_rot[:, 1:], velocity)  # noqa: F405
    r_velocity = qmul(r_rot[:, 1:], qinv(r_rot[:, :-1]))  # noqa: F405
    return cont_6d_params, r_velocity, velocity, r_rot


__all__ = [
    "extract_features",
    "extract_features_t2m",
    "extract_features_torch",
    "foot_detect",
    "foot_detect_torch",
    "get_cont6d_params",
    "get_cont6d_params_torch",
    "get_quaternion",
    "get_quaternion_torch",
    "get_rifke",
    "get_rifke_torch",
    "recover_from_ric",
    "recover_from_rot",
    "recover_root_rot_pos",
]
