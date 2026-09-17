import torch.nn as nn

from reactivebfm.utils.training.common import wrapped_getattr


class TextConditionCFGSampleModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        if model.cond_mask_prob <= 0:
            raise ValueError("Text CFG requires training with cond_mask_prob > 0.")
        self.model = model

    def forward(self, x, timesteps, y=None):
        conditioned = self.model(x, timesteps, y)
        state_baseline = dict(y)
        state_baseline["text_uncond"] = True
        without_text = self.model(x, timesteps, state_baseline)
        scale = y["scale"].view(-1, 1, 1, 1)
        return without_text + scale * (conditioned - without_text)

    def __getattr__(self, name, default=None):
        return wrapped_getattr(self, name, default=default)
