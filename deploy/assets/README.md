# Deployment assets

- `a_pose_g1_36dim.json` — the G1 A-pose used by
  `run_online_generation.py --init_pose A_pose`. Format:

```json
{
  "format": "reactivebfm_qpos36_v1",
  "quat_order": "xyzw",
  "joint_order": ["root_x", "..."],
  "qpos36": [/* 36 floats: root_xyz (3) + root_quat_xyzw (4) + 29 DoF */]
}
```
