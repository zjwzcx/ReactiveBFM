# third_party/scalebfm

`humanoid_transformer.py` is vendored verbatim from the **ScaleBFM** project:

- Source: <https://github.com/zengweishuai/ScaleBFM>,
  `ScaleTrack/source/my_rsl_rl/my_rsl_rl/networks/humanoid_transformer.py`
- Paper: *Scaling Behavior Foundation Model for Humanoid Robots*
  (Zeng et al., arXiv:2607.15163)

It provides the `HumanoidTransformer` / `TaskEmbedder` network definitions
needed by `scripts/export_scalebfm_tensorrt.py` to reconstruct the official
tracking policy from its checkpoint. It is vendored (with credit) so that this
repository is self-contained and does not require a separate ScaleBFM
checkout or submodule. If you use this file, please cite the ScaleBFM paper.
