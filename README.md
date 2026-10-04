<p align="center">
<h1 align="center"><strong>ReactiveBFM: Reactive Closed-Loop Motion Planning Towards Universal Humanoid Whole-Body Control</strong></h1>
  <p align="center">
    <strong>CoRL 2026</strong><br>
    <a href='https://xiao-chen.tech/' target='_blank'>Xiao Chen</a><sup>1,2</sup>&emsp;
    <a href='https://zengweishuai.github.io/' target='_blank'>Weishuai Zeng</a><sup>2*</sup>&emsp;
    <a href='https://scholar.google.com/citations?user=R42aU6gAAAAJ&hl=en' target='_blank'>Xiaojie Niu</a><sup>2*</sup>&emsp;
    <a href='https://openreview.net/profile?id=%7EZiRui_Wang4' target='_blank'>Zirui Wang</a><sup>2*</sup>&emsp;
    <a href='https://www.researchgate.net/scientific-contributions/Jianan-Li-2257559456' target='_blank'>Jianan Li</a><sup>1*</sup>
    <br>
    <a href='https://why618188.github.io/' target='_blank'>Huayi Wang</a><sup>2</sup>&emsp;
    <a href='https://openreview.net/profile?id=~Furui_Xu1' target='_blank'>Furui Xu</a><sup>2</sup>&emsp;
    <a href='https://jiahe-chen.cn/' target='_blank'>Jiahe Chen</a><sup>2</sup>&emsp;
    <a href='#' target='_blank'>Weixiang Zhong</a><sup>2</sup>&emsp;
    <a href='https://dinglihe.github.io/' target='_blank'>Lihe Ding</a><sup>1</sup>
    <br>
    <a href='https://kailinli.top/' target='_blank'>Kailin Li</a><sup>2</sup>&emsp;
    <a href='https://oceanpang.github.io/' target='_blank'>Jiangmiao Pang</a><sup>2</sup>&emsp;
    <a href='https://tai-wang.github.io/' target='_blank'>Tai Wang</a><sup>2</sup>&emsp;
    <a href='https://tianfan.info/' target='_blank'>Tianfan Xue</a><sup>1,2†</sup>&emsp;
    <a href='https://wangjingbo1219.github.io/' target='_blank'>Jingbo Wang</a><sup>2†</sup>
    <br>
    <sup>1</sup>The Chinese University of Hong Kong&emsp;
    <sup>2</sup>Shanghai AI Laboratory
    <br>
    <sup>*</sup>Core Contributors (Random Order)&emsp;
    <sup>†</sup>Corresponding Authors
  </p>
</p>


<div id="top" align="center">

<a href='https://arxiv.org/abs/2606.30362' style='padding-left: 0.5rem;'><img src='https://img.shields.io/badge/arXiv-2606.30362-A42C25?style=flat&logo=arXiv&logoColor=A42C25'></a>
<a href='https://xiao-chen.tech/reactivebfm/' style='padding-left: 0.5rem;'><img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Project Page'></a>

</div>


## 🛠️ Installation

We test our code under the following environment:
- Ubuntu 24.04.2 LTS
- NVIDIA Driver 595.84
- CUDA 12.8
- Python 3.12

1. Clone this repository.

```bash
git clone https://github.com/zjwzcx/ReactiveBFM.git
cd ReactiveBFM
```

2. Create an environment.

```bash
conda create -n reactivebfm python=3.12 -y
conda activate reactivebfm
```

3. Install the Python dependencies.

```bash
python -m pip install "setuptools<81" wheel
python -m pip install --no-build-isolation -r requirements.txt
python -m spacy download en_core_web_sm
```

4. Optional: log training to [Weights & Biases](https://wandb.ai/).

```bash
wandb login
export WANDB_PROJECT=reactivebfm
export WANDB_ENTITY=<your_wandb_username>   # optional; omit to use your wandb login default
```


## 📁 Structure

```text
ReactiveBFM/
├── README.md
├── pyproject.toml
├── deploy/            # deployment stack (online runner + ScaleBridge tracking layer)
└── reactivebfm/
    ├── data/       # datasets, collators, registries, HumanML utilities
    ├── model/      # DiT motion planner and frozen text encoders
    ├── utils/      # shared training and runtime utilities
    ├── train/      # public training entrypoints
    └── eval/       # planned: evaluation tools, to be released later
```


## 📊 Data Preparation

The full train/val/test data is available on
[Google Drive](https://drive.google.com/drive/folders/1cGUKce0g1lSTkebFhKxEFaDuytIeA5P0?usp=sharing),
including our re-made G1 corpus built from [AMASS](https://amass.is.tue.mpg.de/)
motions with [HumanML3D](https://github.com/EricGuo5513/HumanML3D) text
annotations, the [100STYLE](https://ianxmason.github.io/100style/) dataset, and
the [KungfuBot](https://github.com/TeleHuman/PBHC) dataset.

To customize your own dataset, register it in
`reactivebfm/data/datasets/registry.py` and combine registered names with a
comma-separated `--dataset` argument. The complete data contract—frame layout,
joint order, normalization files, splits, and directory examples—is documented
in [`reactivebfm/data/README.md`](reactivebfm/data/README.md).


## 🏋️ Training

The scheduled-forcing entrypoint handles both stages in one run: it uses pure
teacher forcing for the first 400k steps, then ramps three-primitive
self-rollout to 1M steps.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  -m reactivebfm.train.train_planner_scheduled_forcing \
  --save_dir save/reactivebfm_tf400k_sr3_1m_bs128x8 \
  --dataset amass_100style_kungfu_g1_36dim \
  --model_type flow \
  --num_warmup_steps 400000 \
  --num_steps 1000000 \
  --batch_size_local 128 \
  --train_platform_type WandBPlatform
```

- `--cross_prob` defaults to `0`. To learn streaming instruction transitions,
  use `--cross_prob 0.1`.

The default `amass_100style_kungfu_g1_36dim` is the registered full training
set built from the three released data sources (AMASS/HumanML3D + 100STYLE +
Kungfu). You can also pass any single registered name or a comma-separated
composition (e.g. `amass_g1_36dim,100style_g1_36dim,kungfu_g1_36dim`).

For a standalone teacher-forcing baseline, use
`reactivebfm.train.train_planner_teacher_forcing` with `--num_steps 1000000`.


## 🤖 Deployment

```bash
python deploy/run_online_generation.py \
  --model_path /path/to/reactivebfm/modelXXXXXXXX.pt \
  --policy_checkpoint /path/to/tracking_policy_tensorrt.pt \
  --text_prompt "walk forward" \
  --init_pose A_pose \
  --simulator mujoco_simulator --headless
```

This closes the loop between the ReactiveBFM planner and the ScaleBFM tracking
policy on the Unitree G1 (MuJoCo Sim2Sim; Unitree LCM for Sim2Real). Setup,
TensorRT compilation, interfaces, prompt switching, and real-world deployment
are documented in [`deploy/README.md`](deploy/README.md).


## 📝 TODO List

- [x] Release the arXiv paper and project page in June, 2026.
- [x] Release the training code in August, 2026.
- [x] Release all training data in September, 2026.
- [x] Release the deployment code for Unitree G1 in September, 2026.


## 🔗 Citation

If you find our work helpful, please cite it:

```bibtex
@article{chen2026reactivebfm,
  title={ReactiveBFM: Reactive Closed-Loop Motion Planning Towards Universal Humanoid Whole-Body Control},
  author={Chen, Xiao and Zeng, Weishuai and Niu, Xiaojie and Wang, Zirui and Li, Jianan and Wang, Huayi and Xu, Furui and Chen, Jiahe and Zhong, Weixiang and Ding, Lihe and Li, Kailin and Pang, Jiangmiao and Wang, Tai and Xue, Tianfan and Wang, Jingbo},
  journal={arXiv preprint arXiv:2606.30362},
  year={2026}
}
```


We acknowledge that our work builds upon the following awesome projects.

[ScaleBFM](https://github.com/zengweishuai/ScaleBFM) provides the deployment
codebase and the tracking-policy checkpoint, and
[CLoSD](https://github.com/GuyTevet/CLoSD) provides the earliest codebase of
our motion planner. Our training data is built upon
[AMASS](https://amass.is.tue.mpg.de/),
[MotionMillion](https://arxiv.org/abs/2507.07095),
[HumanML3D](https://github.com/EricGuo5513/HumanML3D),
[KungfuBot](https://github.com/TeleHuman/PBHC),
[100STYLE](https://ianxmason.github.io/100style/), and
[PhysHSI](https://arxiv.org/abs/2510.11072), whose data sources and processing
scripts we gratefully acknowledge.

## 📄 License
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a>
<br />
This work is under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
