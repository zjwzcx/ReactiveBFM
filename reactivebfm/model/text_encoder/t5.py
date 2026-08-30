"""Frozen T5 encoder for text-conditioned motion planning."""

import os
from pathlib import Path

import torch.nn as nn


DEFAULT_T5_MODEL = "t5-base"
DEFAULT_T5_LOCAL_ROOT = Path(__file__).resolve().parent
T5_MODEL_ALIASES = {
    "t5-small": "google-t5/t5-small",
    "t5-base": "google/t5-v1_1-base",
    "t5-large": "google-t5/t5-large",
    "t5-xl": "google/t5-v1_1-xl",
    "t5-xxl": "google/flan-t5-xxl",
}


def _validate_local_t5(path):
    path = Path(path)
    has_tokenizer = (path / "spiece.model").is_file() or (
        path / "tokenizer.json"
    ).is_file()
    has_weights = any(path.glob("model*.safetensors")) or any(
        path.glob("pytorch_model*.bin")
    )
    if not (path.joinpath("config.json").is_file() and has_tokenizer and has_weights):
        raise FileNotFoundError(
            f"Incomplete local T5 model at {path}. Expected config.json, "
            "spiece.model or tokenizer.json, and SafeTensors or PyTorch weights."
        )
    return str(path)


def resolve_t5_model_path(model_path=None, local_root=DEFAULT_T5_LOCAL_ROOT):
    requested = model_path or DEFAULT_T5_MODEL
    explicit_path = Path(requested).expanduser()
    if explicit_path.is_dir():
        print(f"Loading T5 from local path: {explicit_path}")
        return _validate_local_t5(explicit_path)

    if requested in T5_MODEL_ALIASES:
        preset_path = Path(local_root).expanduser() / requested
        if preset_path.is_dir():
            print(f"Loading T5 preset from local path: {preset_path}")
            return _validate_local_t5(preset_path)

    remote_repo = T5_MODEL_ALIASES.get(requested, requested)
    print(f"Loading T5 from HuggingFace: {remote_repo}")
    return remote_repo


def load_t5(model_path=None, max_length=128):
    t5 = T5(
        resolve_t5_model_path(model_path),
        max_length=max_length,
    )
    t5.eval()
    t5.text_model.training = False
    for parameter in t5.parameters():
        parameter.requires_grad = False
    return t5


class T5(nn.Module):
    """Encoder-only T5 wrapper returning token features and a valid-token mask."""

    def __init__(
        self,
        modelpath,
        max_length=128,
        tokenizer=None,
        text_model=None,
    ):
        super().__init__()

        from transformers import AutoTokenizer, T5EncoderModel, logging

        logging.set_verbosity_error()
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.max_length = int(max_length) if max_length else None
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(modelpath)
        if text_model is None:
            text_model = T5EncoderModel.from_pretrained(modelpath)
        self.text_model = text_model
        self.output_dim = int(self.text_model.config.d_model)

    def forward(self, texts):
        tokenizer_kwargs = {"return_tensors": "pt", "padding": True}
        if self.max_length is not None:
            tokenizer_kwargs.update(
                {"truncation": True, "max_length": self.max_length}
            )
        encoded_inputs = self.tokenizer(texts, **tokenizer_kwargs)
        model_inputs = encoded_inputs.to(self.text_model.device)
        output = self.text_model(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
        ).last_hidden_state
        valid_mask = model_inputs.attention_mask.to(device=output.device, dtype=bool)
        return output, valid_mask


__all__ = [
    "DEFAULT_T5_LOCAL_ROOT",
    "DEFAULT_T5_MODEL",
    "T5",
    "T5_MODEL_ALIASES",
    "load_t5",
    "resolve_t5_model_path",
]
