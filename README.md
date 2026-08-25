<p align="center">
<h1 align="center"><strong>ReactiveBFM: Reactive Closed-Loop Motion Planning Towards Universal Humanoid Whole-Body Control</strong></h1>
  <p align="center">
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
git clone https://github.com/zjwzcx/ReactiveBFM
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
python -m pip install --no-build-isolation -r requirement.txt
python -m spacy download en_core_web_sm
```

4. Optional: log training to [Weights & Biases](https://wandb.ai/).

```bash
wandb login
export WANDB_PROJECT=reactivebfm
export WANDB_ENTITY=<your_wandb_username>   # optional; omit to use your wandb login default
```



## 📝 TODO List

- [x] Release the arXiv paper and project page in June.
- [ ] Release the training and sim2sim inference code in August. (🚧 Currently under refactoring)
- [ ] Release the deployment code in September. (🚧 Currently under refactoring)






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


## 📄 License
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a>
<br />
This work is under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
