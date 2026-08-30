"""Conditional flow-matching objective for autoregressive motion chunks."""

import torch
import torch as th

from reactivebfm.utils.training.losses import masked_l2, masked_motion_metrics


class FlowMatchingSmoothStandard:
    """Flow-matching process with the smooth losses used by the planner.

    The convention follows pi0/OpenPI: tau=0 is clean data and tau=1 is noise.
    The model predicts the velocity field ``noise - x_start`` and sampling
    integrates from noise to data with a negative time step.
    """

    def __init__(
        self,
        *,
        num_timesteps,
        lambda_velocity=0.0,
        lambda_acceleration=0.0,
        lambda_velocity_prefix=0.0,
        sampler="euler",
        train_time_sampler="beta",
        time_min=0.001,
        time_beta_alpha=1.5,
        time_beta_beta=1.0,
    ):
        if num_timesteps < 2:
            raise ValueError("Flow matching requires at least 2 timesteps.")
        if sampler not in {"euler", "heun"}:
            raise ValueError(f"Unknown flow sampler: {sampler}")
        if train_time_sampler not in {"beta", "uniform", "uniform_discrete"}:
            raise ValueError(f"Unknown flow train time sampler: {train_time_sampler}")
        if not (0.0 <= time_min < 0.5):
            raise ValueError(f"time_min must be in [0, 0.5), got {time_min}")

        self.num_timesteps = int(num_timesteps)
        self.lambda_velocity = lambda_velocity
        self.lambda_acceleration = lambda_acceleration
        self.lambda_velocity_prefix = lambda_velocity_prefix
        self.sampler = sampler
        self.train_time_sampler = train_time_sampler
        self.time_min = float(time_min)
        self.time_beta_alpha = float(time_beta_alpha)
        self.time_beta_beta = float(time_beta_beta)
        self.masked_l2 = masked_l2

    def _scale_timesteps(self, t):
        return t.long()

    def _tau_from_indices(self, t, x):
        tau = t.to(device=x.device, dtype=x.dtype) / float(self.num_timesteps - 1)
        return tau.view(-1, *([1] * (x.dim() - 1)))

    def _indices_from_tau(self, tau, batch_size, device):
        tau_tensor = torch.full((batch_size,), tau, device=device)
        return torch.round(tau_tensor * (self.num_timesteps - 1)).long()

    def _indices_from_tau_tensor(self, tau):
        tau_flat = tau.reshape(tau.shape[0], -1)[:, 0]
        return torch.round(tau_flat * (self.num_timesteps - 1)).long()

    def _sample_train_tau(self, t, x):
        if self.train_time_sampler == "uniform_discrete":
            tau = self._tau_from_indices(t, x)
            if self.time_min > 0.0:
                tau = tau.clamp(self.time_min, 1.0 - self.time_min)
            return tau, self._scale_timesteps(t)

        batch_size = x.shape[0]
        device = x.device
        dtype = x.dtype
        if self.train_time_sampler == "beta":
            alpha = torch.tensor(self.time_beta_alpha, device=device, dtype=torch.float32)
            beta = torch.tensor(self.time_beta_beta, device=device, dtype=torch.float32)
            tau_flat = torch.distributions.Beta(alpha, beta).sample((batch_size,))
        else:
            tau_flat = torch.rand(batch_size, device=device, dtype=torch.float32)

        tau_flat = tau_flat * (1.0 - 2.0 * self.time_min) + self.time_min
        tau = tau_flat.to(dtype=dtype).view(-1, *([1] * (x.dim() - 1)))
        return tau, self._scale_timesteps(self._indices_from_tau_tensor(tau))

    def _set_flow_time(self, model_kwargs, tau, batch_size, device):
        y = model_kwargs.setdefault("y", {})
        if torch.is_tensor(tau):
            flow_time = tau.reshape(batch_size, -1)[:, 0]
            flow_time = flow_time.to(device=device, dtype=torch.float32)
        else:
            flow_time = torch.full((batch_size,), float(tau), device=device, dtype=torch.float32)
        y["flow_time"] = flow_time
        return model_kwargs

    def training_losses(
        self,
        model,
        x_start,
        t,
        model_kwargs=None,
        noise=None,
        dataset=None,
        return_model_output=False,
    ):
        if model_kwargs is None:
            model_kwargs = {}

        mask = model_kwargs["y"]["mask"]
        loss_mask = model_kwargs["y"].get("loss_mask", mask)
        loss_entries_norm = loss_mask.shape[1] == 1

        if noise is None:
            noise = th.randn_like(x_start)

        tau, model_t = self._sample_train_tau(t, x_start)

        x_t = tau * noise + (1.0 - tau) * x_start
        target_velocity = noise - x_start

        model_kwargs = self._set_flow_time(model_kwargs, tau, x_start.shape[0], x_start.device)
        model_output = model(x_t, model_t, **model_kwargs)
        assert model_output.shape == target_velocity.shape == x_start.shape

        terms = {}
        terms["flow/time_mean"] = tau.detach().flatten(1).mean(dim=1)
        terms["flow_mse"] = self.masked_l2(
            target_velocity, model_output, loss_mask, entries_norm=loss_entries_norm
        )

        x0_pred = x_t - tau * model_output
        pred_velocity = None
        if "velocity_gt" in model_kwargs["y"]:
            pred_velocity = x0_pred[:, :, :, 1:] - x0_pred[:, :, :, :-1]
            velocity_mask = model_kwargs["y"].get(
                "velocity_loss_mask",
                model_kwargs["y"]["velocity_mask"],
            )
            terms["velocity_loss"] = self.masked_l2(
                pred_velocity,
                model_kwargs["y"]["velocity_gt"],
                velocity_mask,
                entries_norm=(velocity_mask.shape[1] == 1),
            )

        if "acceleration_gt" in model_kwargs["y"]:
            if pred_velocity is None:
                pred_velocity = x0_pred[:, :, :, 1:] - x0_pred[:, :, :, :-1]
            pred_acceleration = pred_velocity[:, :, :, 1:] - pred_velocity[:, :, :, :-1]
            acceleration_mask = model_kwargs["y"].get(
                "acceleration_loss_mask",
                model_kwargs["y"]["acceleration_mask"],
            )
            terms["acceleration_loss"] = self.masked_l2(
                pred_acceleration,
                model_kwargs["y"]["acceleration_gt"],
                acceleration_mask,
                entries_norm=(acceleration_mask.shape[1] == 1),
            )

        if self.lambda_velocity_prefix > 0.0 and "prefix" in model_kwargs["y"]:
            prefix = model_kwargs["y"]["prefix"]
            pred_velocity_prefix = x0_pred[:, :, :, 0:1] - prefix[:, :, :, -1:]
            gt_velocity_prefix = x_start[:, :, :, 0:1] - prefix[:, :, :, -1:]
            prefix_velocity_mask = model_kwargs["y"].get(
                "prefix_velocity_loss_mask",
                mask[:, :, :, 0:1],
            )
            terms["velocity_prefix_loss"] = self.masked_l2(
                pred_velocity_prefix,
                gt_velocity_prefix,
                prefix_velocity_mask,
                entries_norm=(prefix_velocity_mask.shape[1] == 1),
            )

        terms["loss"] = (
            terms["flow_mse"]
            + self.lambda_velocity * terms.get("velocity_loss", 0.0)
            + self.lambda_acceleration * terms.get("acceleration_loss", 0.0)
            + self.lambda_velocity_prefix * terms.get("velocity_prefix_loss", 0.0)
        )
        terms.update(
            masked_motion_metrics(
                x0_pred,
                x_start,
                loss_mask,
                prefix=model_kwargs["y"].get("prefix"),
            )
        )

        if return_model_output:
            terms["model_output"] = x0_pred.detach()
        return terms

    @th.no_grad()
    def ode_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        **unused_kwargs,
    ):
        if model_kwargs is None:
            model_kwargs = {}
        if device is None:
            device = next(model.parameters()).device
        if noise is None:
            x_t = th.randn(*shape, device=device)
        else:
            x_t = noise.to(device)

        steps = range(self.num_timesteps)
        if progress:
            from tqdm.auto import tqdm

            steps = tqdm(steps)

        dt = -1.0 / float(self.num_timesteps)
        for step_idx in steps:
            tau = 1.0 - step_idx / float(self.num_timesteps)
            t = self._indices_from_tau(tau, shape[0], device)
            model_kwargs = self._set_flow_time(model_kwargs, tau, shape[0], device)
            velocity = model(x_t, self._scale_timesteps(t), **model_kwargs)

            if self.sampler == "heun":
                x_euler = x_t + dt * velocity
                tau_next = max(tau + dt, 0.0)
                t_next = self._indices_from_tau(tau_next, shape[0], device)
                model_kwargs = self._set_flow_time(model_kwargs, tau_next, shape[0], device)
                velocity_next = model(x_euler, self._scale_timesteps(t_next), **model_kwargs)
                x_t = x_t + 0.5 * dt * (velocity + velocity_next)
            else:
                x_t = x_t + dt * velocity

        if denoised_fn is not None:
            x_t = denoised_fn(x_t)
        return x_t

    def sample_loop(self, *args, **kwargs):
        return self.ode_sample_loop(*args, **kwargs)

    def p_sample_loop(self, *args, **kwargs):
        sample = self.ode_sample_loop(*args, **kwargs)
        model_kwargs = kwargs.get("model_kwargs")
        if model_kwargs is not None and "prefix" in model_kwargs.get("y", {}):
            return torch.cat([model_kwargs["y"]["prefix"], sample], dim=-1)
        return sample

    def p_sample_loop_gt_prefix(self, *args, **kwargs):
        return self.ode_sample_loop(*args, **kwargs)
