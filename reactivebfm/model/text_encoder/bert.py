"""DistilBERT text encoder used by the ReactiveBFM planner."""

import os
from pathlib import Path

import torch.nn as nn


DEFAULT_BERT_LOCAL_PATH = (
    Path(__file__).resolve().parent / "distilbert-base-uncased"
)
DEFAULT_BERT_REMOTE_REPO = "distilbert/distilbert-base-uncased"


def _bert_model_available(path):
    if not os.path.isdir(path):
        return False
    required_files = ["config.json", "tokenizer_config.json", "vocab.txt"]
    weight_files = ["model.safetensors", "pytorch_model.bin"]
    if not all(os.path.isfile(os.path.join(path, name)) for name in required_files):
        return False
    return any(os.path.isfile(os.path.join(path, name)) for name in weight_files)


def resolve_bert_model_path(
    local_path=DEFAULT_BERT_LOCAL_PATH,
    remote_repo=DEFAULT_BERT_REMOTE_REPO,
):
    if _bert_model_available(local_path):
        print(f"Loading BERT from local path: {local_path}")
        return local_path
    print(f"Local BERT not found at {local_path}, downloading from HuggingFace: {remote_repo}")
    return remote_repo


def load_bert(model_path=None, max_length=128):
    model_path = resolve_bert_model_path() if model_path is None else model_path
    bert = BERT(model_path, max_length=max_length)
    bert.eval()
    bert.text_model.training = False
    for param in bert.parameters():
        param.requires_grad = False
    return bert


class BERT(nn.Module):
    def __init__(
        self,
        modelpath: str,
        max_length=128,
        tokenizer=None,
        text_model=None,
    ):
        super().__init__()

        from transformers import AutoModel, AutoTokenizer, logging

        logging.set_verbosity_error()
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.max_length = int(max_length)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(modelpath)
        if text_model is None:
            text_model = AutoModel.from_pretrained(modelpath)
        self.text_model = text_model
        self.output_dim = int(self.text_model.config.hidden_size)

    def forward(self, texts):
        encoded_inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        output = self.text_model(**encoded_inputs.to(self.text_model.device)).last_hidden_state
        mask = encoded_inputs.attention_mask.to(device=output.device, dtype=bool)
        return output, mask
