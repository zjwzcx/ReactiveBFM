# This code is based on https://github.com/openai/guided-diffusion
"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import numpy as np
import torch
import torch as th
from reactivebfm.utils.training.losses import masked_l2, masked_motion_metrics
from .gaussian_diffusion import GaussianDiffusion, LossType, ModelMeanType, ModelVarType


class GaussianDiffusionSmooth(GaussianDiffusion):
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
        lambda_rcxyz=0.,
        lambda_vel=0.,
        lambda_pose=1.,
        lambda_orient=1.,
        lambda_loc=1.,
        data_rep='rot6d',
        lambda_root_vel=0.,
        lambda_vel_rcxyz=0.,
        lambda_fc=0.,
        lambda_velocity=0.,
        lambda_acceleration=0.,
        lambda_velocity_prefix=0.,
        **kargs,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps
        self.data_rep = data_rep

        if data_rep != 'rot_vel' and lambda_pose != 1.:
            raise ValueError('lambda_pose is relevant only when training on velocities!')
        self.lambda_pose = lambda_pose
        self.lambda_orient = lambda_orient
        self.lambda_loc = lambda_loc

        self.lambda_rcxyz = lambda_rcxyz
        self.lambda_vel = lambda_vel
        self.lambda_root_vel = lambda_root_vel
        self.lambda_vel_rcxyz = lambda_vel_rcxyz
        self.lambda_fc = lambda_fc
        self.lambda_velocity = lambda_velocity
        self.lambda_acceleration = lambda_acceleration
        self.lambda_velocity_prefix = lambda_velocity_prefix

        if self.lambda_rcxyz > 0. or self.lambda_vel > 0. or self.lambda_root_vel > 0. or \
                self.lambda_vel_rcxyz > 0. or self.lambda_fc > 0. or \
                self.lambda_velocity > 0. or self.lambda_acceleration > 0. or self.lambda_velocity_prefix > 0.:
            assert self.loss_type == LossType.MSE, 'Geometric losses are supported by MSE loss type only!'

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

        # self.l2_loss = lambda a, b: (a - b) ** 2  # th.nn.MSELoss(reduction='none')  # must be None for handling mask later on.
        self.masked_l2 = masked_l2

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None, dataset=None, return_model_output=False):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs. [bs, n_joints, 1, pred_len]
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :param return_model_output: if True, include detached model_output (x_0 prediction)
            in returned terms dict under key 'model_output'. Used by self-rollout training.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """

        # enc = model.model._modules['module']
        # enc = model.model
        mask = model_kwargs['y']['mask']
        loss_mask = model_kwargs['y'].get('loss_mask', mask)
        loss_entries_norm = loss_mask.shape[1] == 1

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)

        # NOTE: add noise to x_start (gt_predicted_motion)
        x_t = self.q_sample(x_start, t, noise=noise, model_kwargs=model_kwargs) # [bs, njoints, nfeats, pred_len]
        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE: # <-
            # NOTE: model.forward(), [bs, njoints, nfeats, pred_len]
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

            if self.model_var_type in [ # FIXED_SMALL
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            # model_mean_type: usually use START_X in MDM
            target = {
                ModelMeanType.PREVIOUS_X: 
                self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start, # original gt motion
                ModelMeanType.EPSILON: noise,   # added noise
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape  # [bs, njoints, nfeats, nframes]

            terms["rot_mse"] = self.masked_l2(
                target, model_output, loss_mask,
                entries_norm=loss_entries_norm) # [bs], mean_flat(rot_mse)

            # NOTE: Compute velocity (1st-order difference) and acceleration (2nd-order difference) losses
            # They will only be added to the final loss if lambda_velocity > 0 or lambda_acceleration > 0
            # Velocity loss: MSE between predicted and ground truth first-order differences
            pred_velocity = None
            if 'velocity_gt' in model_kwargs['y']:
                assert self.model_mean_type == ModelMeanType.START_X, 'Velocity loss supports only X_start prediction for now!'
                # Compute predicted velocity: model_output[t+1] - model_output[t]
                smooth_prediction = model_output
                pred_velocity = smooth_prediction[:, :, :, 1:] - smooth_prediction[:, :, :, :-1]  # [bs, njoints, 1, seqlen-1]
                velocity_gt = model_kwargs['y']['velocity_gt']  # [bs, njoints, 1, seqlen-1]
                velocity_mask = model_kwargs['y'].get(
                    'velocity_loss_mask',
                    model_kwargs['y']['velocity_mask'])  # [bs, 1, 1, seqlen-1]
                terms["velocity_loss"] = self.masked_l2(
                    pred_velocity, velocity_gt, velocity_mask,
                    entries_norm=(velocity_mask.shape[1] == 1))

            # Acceleration loss: MSE between predicted and ground truth second-order differences
            if 'acceleration_gt' in model_kwargs['y']:
                assert self.model_mean_type == ModelMeanType.START_X, 'Acceleration loss supports only X_start prediction for now!'
                # Compute predicted velocity first (reuse if already computed above)
                if pred_velocity is None:
                    smooth_prediction = model_output
                    pred_velocity = smooth_prediction[:, :, :, 1:] - smooth_prediction[:, :, :, :-1]  # [bs, njoints, 1, seqlen-1]
                # Then compute predicted acceleration: pred_velocity[t+1] - pred_velocity[t]
                pred_acceleration = pred_velocity[:, :, :, 1:] - pred_velocity[:, :, :, :-1]  # [bs, njoints, 1, seqlen-2]
                acceleration_gt = model_kwargs['y']['acceleration_gt']  # [bs, njoints, 1, seqlen-2]
                acceleration_mask = model_kwargs['y'].get(
                    'acceleration_loss_mask',
                    model_kwargs['y']['acceleration_mask'])  # [bs, 1, 1, seqlen-2]
                terms["acceleration_loss"] = self.masked_l2(
                    pred_acceleration, acceleration_gt, acceleration_mask,
                    entries_norm=(acceleration_mask.shape[1] == 1))

            # Velocity loss between first predicted frame and last prefix frame
            if self.lambda_velocity_prefix > 0. and 'prefix' in model_kwargs['y']:
                assert self.model_mean_type == ModelMeanType.START_X, 'Velocity prefix loss supports only X_start prediction for now!'
                prefix = model_kwargs['y']['prefix']  # [bs, njoints, nfeats, context_len]
                # Compute predicted velocity: model_output[first_frame] - prefix[last_frame]
                pred_velocity_prefix = model_output[:, :, :, 0:1] - prefix[:, :, :, -1:]  # [bs, njoints, nfeats, 1]
                # Compute ground truth velocity: x_start[first_frame] - prefix[last_frame]
                gt_velocity_prefix = x_start[:, :, :, 0:1] - prefix[:, :, :, -1:]  # [bs, njoints, nfeats, 1]
                # Use mask for first frame
                prefix_velocity_mask = model_kwargs['y'].get(
                    'prefix_velocity_loss_mask',
                    mask[:, :, :, 0:1])  # [bs, 1, 1, 1]
                terms["velocity_prefix_loss"] = self.masked_l2(
                    pred_velocity_prefix, gt_velocity_prefix, prefix_velocity_mask,
                    entries_norm=(prefix_velocity_mask.shape[1] == 1))

            terms["loss"] = terms["rot_mse"] + \
                            (self.lambda_velocity * terms.get('velocity_loss', 0.)) + \
                            (self.lambda_acceleration * terms.get('acceleration_loss', 0.)) + \
                            (self.lambda_velocity_prefix * terms.get('velocity_prefix_loss', 0.)) + \
                            terms.get('vb', 0.) + \
                            (self.lambda_vel * terms.get('vel_mse', 0.)) + \
                            (self.lambda_rcxyz * terms.get('rcxyz_mse', 0.)) + \
                            (self.lambda_fc * terms.get('fc', 0.))

            if return_model_output:
                terms['model_output'] = model_output.detach()
        else:
            raise NotImplementedError(self.loss_type)

        return terms


class GaussianDiffusionSmoothStandard(GaussianDiffusionSmooth):
    """Smooth diffusion loss for the standard planner."""

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None, dataset=None, return_model_output=False):
        if model_kwargs is None:
            model_kwargs = {}

        mask = model_kwargs['y']['mask']
        loss_mask = model_kwargs['y'].get('loss_mask', mask)
        loss_entries_norm = loss_mask.shape[1] == 1

        if noise is None:
            noise = th.randn_like(x_start)

        x_t = self.q_sample(x_start, t, noise=noise, model_kwargs=model_kwargs)
        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                batch_size, channels = x_t.shape[:2]
                assert model_output.shape == (batch_size, channels * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, channels, dim=1)
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape

            terms["rot_mse"] = self.masked_l2(
                target, model_output, loss_mask, entries_norm=loss_entries_norm
            )

            pred_velocity = None
            if 'velocity_gt' in model_kwargs['y']:
                assert self.model_mean_type == ModelMeanType.START_X
                smooth_prediction = model_output
                pred_velocity = smooth_prediction[:, :, :, 1:] - smooth_prediction[:, :, :, :-1]
                velocity_mask = model_kwargs['y'].get(
                    'velocity_loss_mask',
                    model_kwargs['y']['velocity_mask'],
                )
                terms["velocity_loss"] = self.masked_l2(
                    pred_velocity,
                    model_kwargs['y']['velocity_gt'],
                    velocity_mask,
                    entries_norm=(velocity_mask.shape[1] == 1),
                )

            if 'acceleration_gt' in model_kwargs['y']:
                assert self.model_mean_type == ModelMeanType.START_X
                if pred_velocity is None:
                    smooth_prediction = model_output
                    pred_velocity = smooth_prediction[:, :, :, 1:] - smooth_prediction[:, :, :, :-1]
                pred_acceleration = pred_velocity[:, :, :, 1:] - pred_velocity[:, :, :, :-1]
                acceleration_mask = model_kwargs['y'].get(
                    'acceleration_loss_mask',
                    model_kwargs['y']['acceleration_mask'],
                )
                terms["acceleration_loss"] = self.masked_l2(
                    pred_acceleration,
                    model_kwargs['y']['acceleration_gt'],
                    acceleration_mask,
                    entries_norm=(acceleration_mask.shape[1] == 1),
                )

            if self.lambda_velocity_prefix > 0. and 'prefix' in model_kwargs['y']:
                assert self.model_mean_type == ModelMeanType.START_X
                prefix = model_kwargs['y']['prefix']
                pred_velocity_prefix = model_output[:, :, :, 0:1] - prefix[:, :, :, -1:]
                gt_velocity_prefix = x_start[:, :, :, 0:1] - prefix[:, :, :, -1:]
                prefix_velocity_mask = model_kwargs['y'].get(
                    'prefix_velocity_loss_mask',
                    mask[:, :, :, 0:1],
                )
                terms["velocity_prefix_loss"] = self.masked_l2(
                    pred_velocity_prefix,
                    gt_velocity_prefix,
                    prefix_velocity_mask,
                    entries_norm=(prefix_velocity_mask.shape[1] == 1),
                )

            terms["loss"] = (
                terms["rot_mse"]
                + self.lambda_velocity * terms.get("velocity_loss", 0.)
                + self.lambda_acceleration * terms.get("acceleration_loss", 0.)
                + self.lambda_velocity_prefix * terms.get("velocity_prefix_loss", 0.)
                + terms.get("vb", 0.)
            )
            terms.update(
                masked_motion_metrics(
                    model_output,
                    x_start,
                    loss_mask,
                    prefix=model_kwargs['y'].get('prefix'),
                )
            )

            if return_model_output:
                terms['model_output'] = model_output.detach()
        else:
            raise NotImplementedError(self.loss_type)

        return terms
