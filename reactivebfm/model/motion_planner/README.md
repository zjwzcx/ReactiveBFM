# Motion planner package

This package separates the motion network from the generative objective used to
train and sample it:

```text
motion_planner/
├── architectures/          # Text-conditioned DiT motion planner
│   └── dit.py               # AdaLN-Zero DiT with text cross-attention
├── objectives/              # Training losses and sampling processes
│   ├── diffusion/           # Gaussian diffusion and schedule utilities
│   └── flow/                # Conditional flow matching
└── factory.py               # Configuration-to-component composition
```

Use `motion_planner.factory` to construct the DiT and the selected diffusion or
flow-matching objective. The planner keeps the historical motion/text tensor
shapes so checkpoints can be trained and sampled by either public training
entrypoint.
