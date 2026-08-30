# Utilities

Utilities are grouped by the system area that owns them:

- `motion/`: representation recovery, feature extraction, rotations, skeletons,
  and motion filtering.
- `training/`: losses, conditioning, model loading, sampling, and normalization.
- `runtime/`: arguments, distributed execution, seeds, paths, and integrations.
- `simulation/`: Isaac Gym configuration, tasks, state, tensor, and motion-library
  helpers.
- `visualization/`: drawing and motion plotting.

Historical flat module paths remain available through lazy aliases in
`utils.__init__`; new code should import from the owning subpackage.
