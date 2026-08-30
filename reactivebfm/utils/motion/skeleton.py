"""Skeleton kinematics helpers for motion representation conversion."""

import numpy as np
import scipy.ndimage.filters as filters
import torch

from reactivebfm.utils.motion.quaternion import *  # noqa: F403


class Skeleton(object):
    def __init__(self, offset, kinematic_tree, device):
        self.device = device
        self._raw_offset_np = offset.numpy()
        self._raw_offset = offset.clone().detach().to(device).float()
        self._kinematic_tree = kinematic_tree
        self._offset = None
        self._parents = [0] * len(self._raw_offset)
        self._parents[0] = -1
        for chain in self._kinematic_tree:
            for j in range(1, len(chain)):
                self._parents[chain[j]] = chain[j - 1]

    def njoints(self):
        return len(self._raw_offset)

    def offset(self):
        return self._offset

    def set_offset(self, offsets):
        self._offset = offsets.clone().detach().to(self.device).float()

    def kinematic_tree(self):
        return self._kinematic_tree

    def parents(self):
        return self._parents

    def get_offsets_joints_batch(self, joints):
        assert len(joints.shape) == 3
        offsets = self._raw_offset.expand(joints.shape[0], -1, -1).clone()
        for i in range(1, self._raw_offset.shape[0]):
            offsets[:, i] = (
                torch.norm(joints[:, i] - joints[:, self._parents[i]], p=2, dim=1)[:, None]
                * offsets[:, i]
            )

        self._offset = offsets.detach()
        return offsets

    def get_offsets_joints(self, joints):
        assert len(joints.shape) == 2
        offsets = self._raw_offset.clone()
        for i in range(1, self._raw_offset.shape[0]):
            offsets[i] = torch.norm(joints[i] - joints[self._parents[i]], p=2, dim=0) * offsets[i]

        self._offset = offsets.detach()
        return offsets

    def inverse_kinematics_np(self, joints, face_joint_idx, smooth_forward=False, fix_bug=False):
        assert len(face_joint_idx) == 4
        if fix_bug:
            r_hip, l_hip, sdr_r, sdr_l = face_joint_idx
        else:
            l_hip, r_hip, sdr_r, sdr_l = face_joint_idx
        across1 = joints[:, r_hip] - joints[:, l_hip]
        across2 = joints[:, sdr_r] - joints[:, sdr_l]
        across = across1 + across2
        across = across / np.sqrt((across**2).sum(axis=-1))[:, np.newaxis]

        forward = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
        if smooth_forward:
            forward = filters.gaussian_filter1d(forward, 20, axis=0, mode="nearest")
        forward = forward / np.sqrt((forward**2).sum(axis=-1))[..., np.newaxis]

        target = np.array([[0, 0, 1]]).repeat(len(forward), axis=0)
        root_quat = qbetween_np(forward, target)  # noqa: F405

        quat_params = np.zeros(joints.shape[:-1] + (4,))
        quat_params[:, 0] = root_quat
        for chain in self._kinematic_tree:
            rotation = root_quat
            for j in range(len(chain) - 1):
                u = self._raw_offset_np[chain[j + 1]][np.newaxis, ...].repeat(len(joints), axis=0)
                v = joints[:, chain[j + 1]] - joints[:, chain[j]]
                v = v / np.sqrt((v**2).sum(axis=-1))[:, np.newaxis]
                rot_u_v = qbetween_np(u, v)  # noqa: F405

                local_rotation = qmul_np(qinv_np(rotation), rot_u_v)  # noqa: F405

                quat_params[:, chain[j + 1], :] = local_rotation
                rotation = qmul_np(rotation, local_rotation)  # noqa: F405

        return quat_params

    def forward_kinematics(self, quat_params, root_pos, skel_joints=None, do_root_R=True):
        if skel_joints is not None:
            offsets = self.get_offsets_joints_batch(skel_joints)
        if len(self._offset.shape) == 2:
            offsets = self._offset.expand(quat_params.shape[0], -1, -1)
        joints = torch.zeros(quat_params.shape[:-1] + (3,)).to(self.device)
        joints[:, 0] = root_pos
        for chain in self._kinematic_tree:
            if do_root_R:
                rotation = quat_params[:, 0]
            else:
                rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(len(quat_params), -1)
                rotation = rotation.detach().to(self.device)
            for i in range(1, len(chain)):
                rotation = qmul(rotation, quat_params[:, chain[i]])  # noqa: F405
                offset_vec = offsets[:, chain[i]]
                joints[:, chain[i]] = qrot(rotation, offset_vec) + joints[:, chain[i - 1]]  # noqa: F405
        return joints

    def forward_kinematics_np(self, quat_params, root_pos, skel_joints=None, do_root_R=True):
        if skel_joints is not None:
            skel_joints = torch.from_numpy(skel_joints)
            offsets = self.get_offsets_joints_batch(skel_joints)
        if len(self._offset.shape) == 2:
            offsets = self._offset.expand(quat_params.shape[0], -1, -1)
        offsets = offsets.numpy()
        joints = np.zeros(quat_params.shape[:-1] + (3,))
        joints[:, 0] = root_pos
        for chain in self._kinematic_tree:
            if do_root_R:
                rotation = quat_params[:, 0]
            else:
                rotation = np.array([[1.0, 0.0, 0.0, 0.0]]).repeat(len(quat_params), axis=0)
            for i in range(1, len(chain)):
                rotation = qmul_np(rotation, quat_params[:, chain[i]])  # noqa: F405
                offset_vec = offsets[:, chain[i]]
                joints[:, chain[i]] = qrot_np(rotation, offset_vec) + joints[:, chain[i - 1]]  # noqa: F405
        return joints

    def forward_kinematics_cont6d_np(self, cont6d_params, root_pos, skel_joints=None, do_root_R=True):
        if skel_joints is not None:
            skel_joints = torch.from_numpy(skel_joints)
            offsets = self.get_offsets_joints_batch(skel_joints)
        if len(self._offset.shape) == 2:
            offsets = self._offset.expand(cont6d_params.shape[0], -1, -1)
        offsets = offsets.numpy()
        joints = np.zeros(cont6d_params.shape[:-1] + (3,))
        joints[:, 0] = root_pos
        for chain in self._kinematic_tree:
            if do_root_R:
                matR = cont6d_to_matrix_np(cont6d_params[:, 0])  # noqa: F405
            else:
                matR = np.eye(3)[np.newaxis, :].repeat(len(cont6d_params), axis=0)
            for i in range(1, len(chain)):
                matR = np.matmul(matR, cont6d_to_matrix_np(cont6d_params[:, chain[i]]))  # noqa: F405
                offset_vec = offsets[:, chain[i]][..., np.newaxis]
                joints[:, chain[i]] = np.matmul(matR, offset_vec).squeeze(-1) + joints[:, chain[i - 1]]
        return joints

    def forward_kinematics_cont6d(self, cont6d_params, root_pos, skel_joints=None, do_root_R=True):
        if skel_joints is not None:
            offsets = self.get_offsets_joints_batch(skel_joints)
        if len(self._offset.shape) == 2:
            offsets = self._offset.expand(cont6d_params.shape[0], -1, -1)
        joints = torch.zeros(cont6d_params.shape[:-1] + (3,)).to(cont6d_params.device)
        joints[..., 0, :] = root_pos
        for chain in self._kinematic_tree:
            if do_root_R:
                matR = cont6d_to_matrix(cont6d_params[:, 0])  # noqa: F405
            else:
                matR = torch.eye(3).expand((len(cont6d_params), -1, -1))
                matR = matR.detach().to(cont6d_params.device)
            for i in range(1, len(chain)):
                matR = torch.matmul(matR, cont6d_to_matrix(cont6d_params[:, chain[i]]))  # noqa: F405
                offset_vec = offsets[:, chain[i]].unsqueeze(-1)
                joints[:, chain[i]] = torch.matmul(matR, offset_vec).squeeze(-1) + joints[:, chain[i - 1]]
        return joints


__all__ = ["Skeleton"]
