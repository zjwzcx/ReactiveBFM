"""Convert ReactiveBFM G1 qpos references into ScaleBFM task tokens."""

from __future__ import annotations

import torch
import torch.nn as nn


def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    shape = vec.shape
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    xyz = quat[:, 1:]
    cross = xyz.cross(vec, dim=-1)
    return (vec + 2.0 * (quat[:, :1] * cross + xyz.cross(cross, dim=-1))).view(shape)


def quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    return quat_apply(torch.cat((quat[..., :1], -quat[..., 1:]), dim=-1), vec)


def quat_mul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_xyz, right_xyz = left[..., 1:], right[..., 1:]
    scalar = left[..., :1] * right[..., :1] - (left_xyz * right_xyz).sum(dim=-1, keepdim=True)
    vector = left[..., :1] * right_xyz + right[..., :1] * left_xyz + left_xyz.cross(right_xyz, dim=-1)
    return torch.cat((scalar, vector), dim=-1)


def quat_mul_inverse_left(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return quat_mul(torch.cat((left[..., :1], -left[..., 1:]), dim=-1), right)


def quat_mul_inverse_right(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return quat_mul(left, torch.cat((right[..., :1], -right[..., 1:]), dim=-1))


class ScaleBFMQposTaskAdapter(nn.Module):
    """Small eager GPU adapter; the Transformer remains in TensorRT."""

    def __init__(
        self,
        mode_table: torch.Tensor,
        mode_mappings: torch.Tensor,
        parents: list[int],
        translations: torch.Tensor,
        rotations: torch.Tensor,
        axes: torch.Tensor,
        selected: torch.Tensor,
        policy_to_xml: torch.Tensor,
        future_len: int,
    ) -> None:
        super().__init__()
        self.parent_indices = [int(value) for value in parents]
        self.register_buffer("mode_table", mode_table)
        self.register_buffer("mode_mappings", mode_mappings)
        self.register_buffer("translations", translations)
        self.register_buffer("rotations", rotations)
        self.register_buffer("axes", axes)
        self.register_buffer("selected", selected)
        self.register_buffer("policy_to_xml", policy_to_xml)
        tangent = torch.zeros(1, future_len, int(selected.shape[0]), 3, device=mode_table.device)
        normal = torch.zeros_like(tangent)
        tangent[..., 0] = 1.0
        normal[..., 2] = 1.0
        self.register_buffer("tangent", tangent)
        self.register_buffer("normal", normal)

    @classmethod
    def from_metadata(cls, metadata: dict, device: str | torch.device):
        cfg = metadata["qpos_adapter"]
        target = torch.device(device)
        tensor = lambda value, dtype=torch.float32: torch.tensor(value, dtype=dtype, device=target)
        return cls(
            mode_table=tensor(cfg["mode_table"]),
            mode_mappings=tensor(cfg["mode_mappings"]),
            parents=cfg["parents"],
            translations=tensor(cfg["translations"]),
            rotations=tensor(cfg["rotations"]),
            axes=tensor(cfg["axes"]),
            selected=tensor(cfg["selected"], torch.long),
            policy_to_xml=tensor(cfg["policy_to_xml"], torch.long),
            future_len=len(metadata["future_idx"]),
        ).eval()

    def _fk(self, root_pos: torch.Tensor, root_quat: torch.Tensor, dof: torch.Tensor):
        dof_xml = dof.index_select(-1, self.policy_to_xml)
        half = dof_xml.unsqueeze(-1) * 0.5
        joint_quat = torch.cat((torch.cos(half), self.axes * torch.sin(half)), dim=-1)
        body_pos = [root_pos]
        body_quat = [root_quat]
        for joint_index, parent in enumerate(self.parent_indices[1:]):
            parent_pos, parent_quat = body_pos[parent], body_quat[parent]
            translation = self.translations[joint_index].expand_as(parent_pos)
            rotation = self.rotations[joint_index].expand_as(parent_quat)
            body_pos.append(parent_pos + quat_apply(parent_quat, translation))
            body_quat.append(quat_mul(parent_quat, quat_mul(rotation, joint_quat[..., joint_index, :])))
        return torch.stack(body_pos, dim=-2), torch.stack(body_quat, dim=-2)

    def forward(
        self,
        root_pos: torch.Tensor,
        root_quat_buffer: torch.Tensor,
        dof_pos_buffer: torch.Tensor,
        ref_root_pos_future: torch.Tensor,
        ref_root_rot_future: torch.Tensor,
        ref_dof_pos_future: torch.Tensor,
        mode_index: torch.Tensor,
        time_offsets: torch.Tensor,
    ) -> torch.Tensor:
        current_root_quat = root_quat_buffer[:, -1]
        identity = torch.cat((torch.ones_like(current_root_quat[:, :1]), torch.zeros_like(current_root_quat[:, 1:])), dim=-1)
        current_pos, current_quat = self._fk(torch.zeros_like(root_pos), identity, dof_pos_buffer[:, -1])
        future_pos, future_quat = self._fk(ref_root_pos_future, ref_root_rot_future, ref_dof_pos_future)
        current_pos = current_pos.index_select(-2, self.selected)
        current_quat = current_quat.index_select(-2, self.selected)
        future_pos = future_pos.index_select(-2, self.selected)
        future_quat = future_quat.index_select(-2, self.selected)

        root_quat = current_root_quat[:, None, None, :].expand_as(future_quat)
        target_pos = quat_apply_inverse(root_quat, future_pos - root_pos[:, None, None, :])
        target_pos_relative = target_pos - current_pos[:, None]
        target_quat = quat_mul_inverse_left(root_quat, future_quat)
        target_quat_relative = quat_mul_inverse_right(target_quat, current_quat[:, None].expand_as(target_quat))
        target_rot = torch.cat((quat_apply(target_quat, self.tangent), quat_apply(target_quat, self.normal)), dim=-1)
        target_rot_relative = torch.cat(
            (quat_apply(target_quat_relative, self.tangent), quat_apply(target_quat_relative, self.normal)), dim=-1
        )
        task = torch.cat(
            (
                target_pos.flatten(2), target_pos_relative.flatten(2),
                target_rot.flatten(2), target_rot_relative.flatten(2), time_offsets,
            ),
            dim=-1,
        )
        mapping = self.mode_mappings.index_select(0, mode_index)
        mode = self.mode_table.index_select(0, mode_index)
        return torch.cat(
            (task * mapping.unsqueeze(1), mode.unsqueeze(1).expand(-1, task.shape[1], -1)), dim=-1
        )
