import torch

from reactivebfm.model.motion_planner.factory import (
    create_flow_matching_smooth,
    create_gaussian_diffusion_smooth,
    create_model_and_diffusion_smooth,
    create_motion_planner_model,
    get_model_args,
)


def load_model_wo_clip(model, state_dict):
    model_state = model.state_dict()
    expected_keys = {
        key for key in model_state if not key.startswith("clip_model.")
    }
    checkpoint_keys = set(state_dict)
    missing_keys = sorted(expected_keys - checkpoint_keys)
    unexpected_keys = sorted(checkpoint_keys - expected_keys)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Checkpoint does not match the current planner structure: "
            f"missing={missing_keys}, unexpected={unexpected_keys}"
        )
    model_state.update(state_dict)
    model.load_state_dict(model_state, strict=True)


def load_saved_model(model, model_path, use_avg: bool = False):
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    if use_avg and "model_avg" in state_dict.keys():
        print("loading avg model")
        state_dict = state_dict["model_avg"]
    else:
        if "model" in state_dict:
            state_dict = state_dict["model"]
        else:
            print("checkpoint has no avg model, loading as usual.")
    load_model_wo_clip(model, state_dict)
    return model


__all__ = [
    "create_flow_matching_smooth",
    "create_gaussian_diffusion_smooth",
    "create_model_and_diffusion_smooth",
    "create_motion_planner_model",
    "get_model_args",
    "load_model_wo_clip",
    "load_saved_model",
]
