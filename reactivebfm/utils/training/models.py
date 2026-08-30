import torch

from reactivebfm.model.motion_planner.factory import (
    create_flow_matching_smooth,
    create_gaussian_diffusion,
    create_gaussian_diffusion_smooth,
    create_model_and_diffusion,
    create_model_and_diffusion_smooth,
    create_motion_planner_model,
    get_model_args,
)


def load_model_wo_clip(model, state_dict):
    # assert (state_dict['sequence_pos_encoder.pe'][:model.sequence_pos_encoder.pe.shape[0]] == model.sequence_pos_encoder.pe).all()  # TEST
    # assert (state_dict['embed_timestep.sequence_pos_encoder.pe'][:model.embed_timestep.sequence_pos_encoder.pe.shape[0]] == model.embed_timestep.sequence_pos_encoder.pe).all()  # TEST
    state_dict.pop(
        "sequence_pos_encoder.pe", None
    )  # fixed buffer; may size-mismatch older models
    state_dict.pop(
        "embed_timestep.sequence_pos_encoder.pe", None
    )  # fixed buffer; may size-mismatch older models
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0
    assert all(
        [
            k.startswith("clip_model.")
            or k.startswith("embed_flow_timestep.")
            or "sequence_pos_encoder" in k
            for k in missing_keys
        ]
    )


def load_saved_model(model, model_path, use_avg: bool = False):  # use_avg_model
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    # Use average model when possible
    if use_avg and "model_avg" in state_dict.keys():
        # if use_avg_model:
        print("loading avg model")
        state_dict = state_dict["model_avg"]
    else:
        if "model" in state_dict:
            # print('loading model without avg')
            state_dict = state_dict["model"]
        else:
            print("checkpoint has no avg model, loading as usual.")
    load_model_wo_clip(model, state_dict)
    return model


__all__ = [
    "create_flow_matching_smooth",
    "create_gaussian_diffusion",
    "create_gaussian_diffusion_smooth",
    "create_model_and_diffusion",
    "create_model_and_diffusion_smooth",
    "create_motion_planner_model",
    "get_model_args",
    "load_model_wo_clip",
    "load_saved_model",
]
