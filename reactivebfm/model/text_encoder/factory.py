"""Single-parameter text encoder selection."""

import json
from pathlib import Path

from reactivebfm.model.text_encoder.bert import load_bert
from reactivebfm.model.text_encoder.t5 import load_t5


DEFAULT_TEXT_ENCODER = "bert"
PLANNER_TEXT_MAX_TOKENS = 128
TEXT_ENCODER_PRESETS = {
    "bert": ("bert", None),
    "t5": ("t5", "t5-base"),
    "t5-small": ("t5", "t5-small"),
    "t5-base": ("t5", "t5-base"),
    "t5-large": ("t5", "t5-large"),
    "t5-xl": ("t5", "t5-xl"),
    "t5-xxl": ("t5", "t5-xxl"),
}


def _local_model_type(path):
    config_path = Path(path).expanduser() / "config.json"
    if not config_path.is_file():
        return None
    with config_path.open() as handle:
        return json.load(handle).get("model_type", "").lower()


def resolve_text_encoder(text_encoder=DEFAULT_TEXT_ENCODER):
    name = str(text_encoder or DEFAULT_TEXT_ENCODER).strip()
    preset = TEXT_ENCODER_PRESETS.get(name.lower())
    if preset is not None:
        return preset

    model_type = _local_model_type(name)
    normalized = (model_type or name).lower()
    if "t5" in normalized:
        return "t5", name
    remote_name = normalized.rsplit("/", 1)[-1]
    if model_type in {"bert", "distilbert"} or remote_name.startswith(
        ("bert-", "distilbert-")
    ):
        return "bert", name
    raise ValueError(
        f"Cannot infer a supported text encoder from {name!r}. Use 'bert', a "
        "T5 preset such as 't5-xl' or a T5/BERT HuggingFace ID, "
        "or a local model directory. Other encoder architectures require "
        "explicit approval."
    )


def load_text_encoder(text_encoder=DEFAULT_TEXT_ENCODER):
    encoder_type, model_path = resolve_text_encoder(text_encoder)
    if encoder_type == "bert":
        encoder = load_bert(
            model_path=model_path, max_length=PLANNER_TEXT_MAX_TOKENS
        )
    else:
        encoder = load_t5(
            model_path=model_path, max_length=PLANNER_TEXT_MAX_TOKENS
        )
    encoder.encoder_type = encoder_type
    encoder.encoder_name = str(text_encoder or DEFAULT_TEXT_ENCODER)
    return encoder


__all__ = [
    "DEFAULT_TEXT_ENCODER",
    "PLANNER_TEXT_MAX_TOKENS",
    "TEXT_ENCODER_PRESETS",
    "load_text_encoder",
    "resolve_text_encoder",
]
