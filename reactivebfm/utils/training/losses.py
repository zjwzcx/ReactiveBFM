from reactivebfm.model.motion_planner.objectives.diffusion.nn import mean_flat, sum_flat
import torch


ROOT_POS_SLICE = slice(0, 3)
ROOT_ROT_SLICE = slice(3, 7)
DOF_SLICE = slice(7, 36)

def angle_l2(angle1, angle2):
    a = angle1 - angle2
    a = (a + (torch.pi/2)) % torch.pi - (torch.pi/2)
    return a ** 2

def diff_l2(a, b):
    return (a - b) ** 2

def masked_l2(a, b, mask, loss_fn=diff_l2, epsilon=1e-8, entries_norm=True):
    # assuming a.shape == b.shape == bs, J, Jdim, seqlen
    # assuming mask.shape == bs, 1, 1, seqlen
    loss = loss_fn(a, b)
    loss = sum_flat(loss * mask.float())  # gives \sigma_euclidean over unmasked elements
    n_entries = a.shape[1]
    if len(a.shape) > 3:
        n_entries *= a.shape[2]
    non_zero_elements = sum_flat(mask)
    if entries_norm:
        # In cases the mask is per frame, and not specifying the number of entries per frame, this normalization is needed,
        # Otherwise set it to False
        non_zero_elements *= n_entries
    mse_loss_val = loss / (non_zero_elements + epsilon)  # Add epsilon to avoid division by zero
    return mse_loss_val


def _masked_l2_channels(pred, target, mask, channel_slice):
    return masked_l2(pred[:, channel_slice], target[:, channel_slice], mask)


def masked_motion_metrics(pred, target, mask, prefix=None):
    """Motion quality metrics for WandB logging, not for optimization."""
    metrics = {}

    with torch.no_grad():
        metrics["metrics/mse"] = masked_l2(pred, target, mask).detach()
        if pred.shape[1] >= 36:
            metrics["metrics/mse_root_pos"] = _masked_l2_channels(
                pred, target, mask, ROOT_POS_SLICE
            ).detach()
            metrics["metrics/mse_root_rot"] = _masked_l2_channels(
                pred, target, mask, ROOT_ROT_SLICE
            ).detach()
            metrics["metrics/mse_dof"] = _masked_l2_channels(
                pred, target, mask, DOF_SLICE
            ).detach()

        if pred.shape[-1] > 0:
            metrics["metrics/mse_first_frame"] = masked_l2(
                pred[..., :1], target[..., :1], mask[..., :1]
            ).detach()
            valid_frames = mask.reshape(mask.shape[0], -1, mask.shape[-1]).any(dim=1).sum(dim=-1)
            mid = torch.div(valid_frames, 2, rounding_mode="floor").clamp(
                min=0, max=pred.shape[-1] - 1
            )
            final = (valid_frames - 1).clamp(min=0, max=pred.shape[-1] - 1)
            gather_shape = (-1, *pred.shape[1:-1], 1)
            mid_index = mid.view(-1, *([1] * (pred.dim() - 2)), 1).expand(gather_shape)
            final_index = final.view(-1, *([1] * (pred.dim() - 2)), 1).expand(gather_shape)
            mid_pred = pred.gather(-1, mid_index)
            mid_target = target.gather(-1, mid_index)
            final_pred = pred.gather(-1, final_index)
            final_target = target.gather(-1, final_index)
            frame_mask = (valid_frames > 0).view(-1, 1, 1, 1)
            metrics["metrics/mse_mid_frame"] = masked_l2(
                mid_pred, mid_target, frame_mask
            ).detach()
            metrics["metrics/mse_final_frame"] = masked_l2(
                final_pred, final_target, frame_mask
            ).detach()

        if pred.shape[-1] > 1:
            pred_vel = pred[..., 1:] - pred[..., :-1]
            target_vel = target[..., 1:] - target[..., :-1]
            vel_mask = mask[..., 1:] & mask[..., :-1]
            metrics["metrics/mse_velocity"] = masked_l2(pred_vel, target_vel, vel_mask).detach()
            if pred.shape[1] >= 36:
                metrics["metrics/mse_velocity_root_pos"] = _masked_l2_channels(
                    pred_vel, target_vel, vel_mask, ROOT_POS_SLICE
                ).detach()
                metrics["metrics/mse_velocity_dof"] = _masked_l2_channels(
                    pred_vel, target_vel, vel_mask, DOF_SLICE
                ).detach()

        if pred.shape[-1] > 2:
            pred_acc = pred[..., 2:] - 2.0 * pred[..., 1:-1] + pred[..., :-2]
            target_acc = target[..., 2:] - 2.0 * target[..., 1:-1] + target[..., :-2]
            acc_mask = mask[..., 2:] & mask[..., 1:-1] & mask[..., :-2]
            metrics["metrics/mse_acc"] = masked_l2(
                pred_acc, target_acc, acc_mask
            ).detach()
            if pred.shape[1] >= 36:
                metrics["metrics/mse_acc_dof"] = _masked_l2_channels(
                    pred_acc, target_acc, acc_mask, DOF_SLICE
                ).detach()

        if prefix is not None and pred.shape[-1] > 0 and prefix.shape[-1] > 0 and pred.shape[1] >= 36:
            pred_first_delta = pred[..., :1] - prefix[..., -1:]
            target_first_delta = target[..., :1] - prefix[..., -1:]
            first_mask = mask[..., :1]
            metrics["metrics/smooth_first_root_pos"] = _masked_l2_channels(
                pred_first_delta, target_first_delta, first_mask, ROOT_POS_SLICE
            ).detach()
            metrics["metrics/smooth_first_dof"] = _masked_l2_channels(
                pred_first_delta, target_first_delta, first_mask, DOF_SLICE
            ).detach()

    return metrics
