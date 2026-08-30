# Data

Data-facing modules for ReactiveBFM.

## Motion data contract

The public DiT training recipes use the **36-dimensional G1 motion
representation**. A clip has shape `(T, 36)` and is stored as `motion` in
`full_train.pkl`.

| Index | Dim | Field | Description |
| :---: | :-: | ----- | ----------- |
| `[0:3]` | 3 | `root_pos` | Pelvis position in world coordinates (meters, Z-up). |
| `[3:7]` | 4 | `root_rot` | Pelvis quaternion in **xyzw** order. |
| `[7:36]` | 29 | `dof_pos` | G1 joint positions in actuator order. |

The 29-DoF order is:

```text
left_hip_pitch, left_hip_roll, left_hip_yaw, left_knee,
left_ankle_pitch, left_ankle_roll,
right_hip_pitch, right_hip_roll, right_hip_yaw, right_knee,
right_ankle_pitch, right_ankle_roll,
waist_yaw, waist_roll, waist_pitch,
left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw, left_elbow,
left_wrist_roll, left_wrist_pitch, left_wrist_yaw,
right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw, right_elbow,
right_wrist_roll, right_wrist_pitch, right_wrist_yaw
```

All bundled `*_g1_36dim` datasets use 60 Hz, world-frame Z-up coordinates, and
per-dataset normalization `(motion - Mean) / Std`.

Each dataset directory should contain:

```text
<dataset_name>/
├── full_train.pkl          # motion (T, 36), length, caption list
├── Mean.npy / Std.npy      # normalization statistics
├── train.txt / val.txt / test.txt
├── split_summary.json      # optional split/leakage audit
└── texts.csv               # optional caption index
```

Compose registered datasets at runtime with, for example,
`--dataset bones_seed_v3_g1_36dim,hymotion_v3_g1_36dim`. With `--data_dir`, use
a common parent containing one directory per registered dataset; without it,
the registry paths are used directly.

- `datasets/`: the single planner-data path: motion-text samples, dataloader
  construction, collation, registry, corpus tools, and assets. Select a dataset
  and a split; there are no `train`/`eval`/`gt` mode strings.
- `motion.py`: joint names, masks, raw offsets, and kinematic topology.
- `datasets/assets/`: small dataset config and split assets. Large datasets
  should still live outside the repo and be pointed to with
  `REACTIVEBFM_DATASET_ROOT`.

Motion joint metadata and kinematic topology live in `data.motion`. Motion
recovery, feature extraction, quaternion, and skeleton operations live in
`reactivebfm.utils.motion`. New code should import planner data from
`reactivebfm.data.datasets` directly.

Some motion/text processing utilities were adapted from
https://github.com/EricGuo5513/text-to-motion.git.
