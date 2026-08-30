"""Shared frozen-text-encoder behavior for motion planners."""

import torch.nn as nn

from reactivebfm.model.text_encoder.factory import load_text_encoder


class FrozenTextEncoderMixin:
    def init_text_encoder(self, text_encoder, projection_dim):
        self.text_encoder_name = text_encoder
        print(f"Loading text encoder: {text_encoder}")
        # Keep the historical attribute name so existing checkpoint filters work.
        self.clip_model = load_text_encoder(text_encoder)
        self.text_encoder_type = self.clip_model.encoder_type
        self.clip_dim = self.clip_model.output_dim
        self.embed_text = nn.Linear(self.clip_dim, projection_dim)
        self.encode_text = self.text_encode_text

    def train(self, mode=True):
        super().train(mode)
        self.clip_model.eval()
        return self

    def parameters_wo_clip(self):
        return [
            parameter
            for name, parameter in self.named_parameters()
            if not name.startswith("clip_model.")
        ]

    def text_encode_text(self, raw_text):
        encoded_text, valid_mask = self.clip_model(raw_text)
        return encoded_text.permute(1, 0, 2), ~valid_mask

    def bert_encode_text(self, raw_text):
        """Backward-compatible name for the generic text encoding path."""
        return self.text_encode_text(raw_text)

    def project_text_tokens(self, text_tokens):
        return self.embed_text(
            text_tokens.to(dtype=self.embed_text.weight.dtype)
        )


__all__ = ["FrozenTextEncoderMixin"]
