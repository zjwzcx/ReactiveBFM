"""DiT-style motion planner with adaptive normalization conditioning."""

import torch
import torch.nn as nn

from reactivebfm.model.text_encoder.conditioning import FrozenTextEncoderMixin
from reactivebfm.model.utils import (
    ContinuousTimestepEmbedder,
    InputProcess,
    OutputProcess,
    PositionalEncoding,
    TimestepEmbedder,
)


def _modulate(hidden_states, shift, scale):
    if shift.ndim == 2:
        shift = shift[:, None, :]
        scale = scale[:, None, :]
    return hidden_states * (1.0 + scale) + shift


class DiTBlock(nn.Module):
    """AdaLN-Zero DiT block with text cross-attention.

    Motion tokens first communicate through self-attention, then attend to the
    frozen text-encoder tokens. Flow/diffusion time modulates and gates all
    three residual branches. Zero-initialized modulation makes every block an
    identity map at initialization, following the standard DiT recipe.
    """

    def __init__(
        self,
        latent_dim,
        num_heads,
        ff_size,
        dropout=0.0,
        activation="gelu",
    ):
        super().__init__()
        if num_heads <= 0 or latent_dim % num_heads:
            raise ValueError(
                f"latent_dim ({latent_dim}) must be divisible by "
                f"num_heads ({num_heads})."
            )
        if ff_size <= 0:
            raise ValueError(f"ff_size must be positive, got {ff_size}.")

        self.self_norm = nn.LayerNorm(latent_dim, elementwise_affine=False)
        self.self_attention = nn.MultiheadAttention(
            latent_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(latent_dim, elementwise_affine=False)
        self.cross_attention = nn.MultiheadAttention(
            latent_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(latent_dim, elementwise_affine=False)
        activation_layer = {
            "gelu": nn.GELU(approximate="tanh"),
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),
        }.get(activation)
        if activation_layer is None:
            raise ValueError(f"Unsupported DiT activation: {activation!r}.")
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, ff_size),
            activation_layer,
            nn.Dropout(dropout),
            nn.Linear(ff_size, latent_dim),
            nn.Dropout(dropout),
        )

        # shift, scale, and residual gate for self-attention, cross-attention,
        # and the MLP respectively.
        self.ada_ln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(latent_dim, 9 * latent_dim),
        )
        nn.init.zeros_(self.ada_ln[-1].weight)
        nn.init.zeros_(self.ada_ln[-1].bias)

    def forward(
        self,
        hidden_states,
        time_condition,
        text_tokens,
        motion_padding_mask=None,
        text_padding_mask=None,
    ):
        modulation = self.ada_ln(time_condition).chunk(9, dim=-1)
        (
            self_shift,
            self_scale,
            self_gate,
            cross_shift,
            cross_scale,
            cross_gate,
            mlp_shift,
            mlp_scale,
            mlp_gate,
        ) = modulation

        normalized = _modulate(
            self.self_norm(hidden_states), self_shift, self_scale
        )
        attended = self.self_attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=motion_padding_mask,
            need_weights=False,
        )[0]
        gate = self_gate[:, None, :] if self_gate.ndim == 2 else self_gate
        hidden_states = hidden_states + gate * attended

        query = _modulate(
            self.cross_norm(hidden_states), cross_shift, cross_scale
        )
        attended = self.cross_attention(
            query,
            text_tokens,
            text_tokens,
            key_padding_mask=text_padding_mask,
            need_weights=False,
        )[0]
        gate = cross_gate[:, None, :] if cross_gate.ndim == 2 else cross_gate
        hidden_states = hidden_states + gate * attended

        mlp_input = _modulate(
            self.mlp_norm(hidden_states), mlp_shift, mlp_scale
        )
        gate = mlp_gate[:, None, :] if mlp_gate.ndim == 2 else mlp_gate
        return hidden_states + gate * self.mlp(mlp_input)


class DiTFinalLayer(nn.Module):
    """Final time-modulated normalization before motion projection."""

    def __init__(self, latent_dim):
        super().__init__()
        self.norm = nn.LayerNorm(latent_dim, elementwise_affine=False)
        self.ada_ln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(latent_dim, 2 * latent_dim),
        )
        nn.init.zeros_(self.ada_ln[-1].weight)
        nn.init.zeros_(self.ada_ln[-1].bias)

    def forward(self, hidden_states, time_condition):
        shift, scale = self.ada_ln(time_condition).chunk(2, dim=-1)
        return _modulate(self.norm(hidden_states), shift, scale)


class DiTMotionPlanner(FrozenTextEncoderMixin, nn.Module):
    """Text-conditioned DiT action head using ReactiveBFM's planner API.

    The observed prefix and noisy prediction chunk form one bidirectional
    motion-token sequence. Each DiT block uses flow/diffusion time through
    AdaLN-Zero and cross-attends to frozen text-encoder tokens. Only prediction-token
    outputs are returned.
    """

    def __init__(
        self,
        modeltype,
        njoints,
        nfeats,
        num_actions,
        translation,
        pose_rep,
        glob,
        glob_rot,
        latent_dim=512,
        ff_size=2048,
        num_layers=16,
        num_heads=8,
        dropout=0.0,
        ablation=None,
        activation="gelu",
        legacy=False,
        data_rep="rot6d",
        dataset="amass",
        clip_dim=512,
        arch="dit",
        emb_trans_dec=False,
        clip_version=None,
        **kargs,
    ):
        super().__init__()
        if arch != "dit":
            raise ValueError(
                f"DiTMotionPlanner expects arch='dit', got {arch!r}."
            )
        if latent_dim % 2:
            raise ValueError(f"latent_dim must be even, got {latent_dim}.")
        if num_heads <= 0 or latent_dim % num_heads:
            raise ValueError(
                f"latent_dim ({latent_dim}) must be divisible by "
                f"num_heads ({num_heads})."
            )
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}.")
        if ff_size <= 0:
            raise ValueError(f"ff_size must be positive, got {ff_size}.")

        # Public attributes retained for training, sampling, and logging code.
        self.legacy = legacy
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.data_rep = data_rep
        self.dataset = dataset
        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.ablation = ablation
        self.activation = activation
        self.input_feats = self.njoints * self.nfeats
        self.cond_mode = "text"
        self.cond_mask_prob = kargs.get("cond_mask_prob", 0.0)
        self.mask_frames = kargs.get("mask_frames", False)
        self.arch = arch

        self.context_len = kargs.get("context_len", 0)
        self.pred_len = kargs.get("pred_len", 0)
        if self.context_len <= 0 or self.pred_len <= 0:
            raise ValueError(
                "ReactiveBFM DiT planner requires context_len > 0 and pred_len > 0."
            )

        self.input_process = InputProcess(
            self.data_rep, self.input_feats, self.latent_dim
        )
        self.sequence_pos_encoder = PositionalEncoding(
            self.latent_dim,
            self.dropout,
            max_len=kargs.get("pos_embed_max_len", 256),
        )
        self.embed_timestep = TimestepEmbedder(
            self.latent_dim, self.sequence_pos_encoder
        )
        self.embed_flow_timestep = ContinuousTimestepEmbedder(self.latent_dim)

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    latent_dim=self.latent_dim,
                    num_heads=self.num_heads,
                    ff_size=self.ff_size,
                    dropout=self.dropout,
                    activation=self.activation,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.final_layer = DiTFinalLayer(self.latent_dim)

        self.init_text_encoder(
            kargs.get("text_encoder", "bert"),
            projection_dim=self.latent_dim,
        )
        self.output_process = OutputProcess(
            self.data_rep,
            self.input_feats,
            self.latent_dim,
            self.njoints,
            self.nfeats,
        )
        # A zero output projection is part of the AdaLN-Zero DiT
        # initialization and gives both diffusion and flow a stable start.
        nn.init.zeros_(self.output_process.poseFinal.weight)
        nn.init.zeros_(self.output_process.poseFinal.bias)

    def mask_cond(self, condition, force_mask=False):
        _, batch_size, _ = condition.shape
        if force_mask:
            return torch.zeros_like(condition)
        if self.training and self.cond_mask_prob > 0.0:
            mask = torch.bernoulli(
                torch.ones(batch_size, device=condition.device)
                * self.cond_mask_prob
            ).view(1, batch_size, 1)
            return condition * (1.0 - mask)
        return condition

    def forward(self, x, timesteps, y=None):
        """Predict a future motion or flow field shaped like ``x``."""
        if y is None:
            raise ValueError(
                "DiTMotionPlanner forward expects conditioning dict y."
            )

        batch_size, _, _, action_length = x.shape
        if action_length != self.pred_len:
            raise ValueError(
                f"Expected pred_len={self.pred_len}, got x shape {tuple(x.shape)}."
            )
        prefix = self._motion_prefix(y["prefix"])
        motion = torch.cat([prefix, x], dim=-1)
        motion_tokens = self.input_process(motion)
        motion_tokens = self.sequence_pos_encoder(motion_tokens).transpose(0, 1)

        time_condition = self._time_condition(x, timesteps, y)
        text_tokens, text_padding_mask = self._text_condition(y, batch_size)
        motion_padding_mask = self._motion_padding_mask(
            y["mask"], batch_size, action_length, x.device
        )

        for block in self.blocks:
            motion_tokens = block(
                motion_tokens,
                time_condition,
                text_tokens,
                motion_padding_mask=motion_padding_mask,
                text_padding_mask=text_padding_mask,
            )

        output = self.final_layer(motion_tokens, time_condition)
        output = output[:, self.context_len :].transpose(0, 1)
        output = self.output_process(output)
        action_valid = ~motion_padding_mask[:, self.context_len :]
        return output.masked_fill(
            ~action_valid[:, None, None, :],
            0.0,
        )

    def _time_condition(self, x, timesteps, y):
        if "flow_time" in y:
            embedding = self.embed_flow_timestep(
                y["flow_time"].to(device=x.device)
            )
        else:
            embedding = self.embed_timestep(timesteps)
        return embedding.squeeze(0)

    def _text_condition(self, y, batch_size):
        encoded = (
            y["text_embed"] if "text_embed" in y else self.encode_text(y["text"])
        )
        text_tokens, text_padding_mask = encoded
        if text_padding_mask.shape[0] == 1 and batch_size > 1:
            text_padding_mask = torch.repeat_interleave(
                text_padding_mask, batch_size, dim=0
            )
        text_tokens = self.mask_cond(
            text_tokens,
            force_mask=y.get("text_uncond", False),
        )
        text_tokens = self.project_text_tokens(text_tokens).transpose(0, 1)
        return text_tokens, text_padding_mask.to(
            device=text_tokens.device, dtype=torch.bool
        )

    def _motion_prefix(self, prefix):
        if prefix.shape[-1] != self.context_len:
            raise ValueError(
                f"Expected context_len={self.context_len}, "
                f"got prefix shape {tuple(prefix.shape)}."
            )
        return prefix

    def _motion_padding_mask(
        self, pred_mask, batch_size, action_length, device
    ):
        context_valid = torch.ones(
            batch_size,
            self.context_len,
            dtype=torch.bool,
            device=device,
        )
        if not self.mask_frames or pred_mask.shape[-1] <= 1:
            action_valid = torch.ones(
                batch_size,
                action_length,
                dtype=torch.bool,
                device=device,
            )
        else:
            action_valid = (
                pred_mask[..., :action_length]
                .squeeze(1)
                .squeeze(1)
                .to(device=device, dtype=torch.bool)
            )
        return ~torch.cat([context_valid, action_valid], dim=1)
