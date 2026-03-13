<p align="center">
  <h1 align="center"><i>I’m a Map!</i> Interpretable Motion-Attentive Maps: Spatio-Temporally Localizing Concepts in Video Diffusion Transformers</h1>
  <h3 align="center"><b>CVPR 2026</b></h3>
  <p align="center">
    <h3 align="center">
      <a href="https://github.com/youngjun-jun"><strong>Youngjun Jun<sup>1</sup></strong></a> · 
      <a href="https://seilk.github.io/"><strong>Seil Kang<sup>2</sup></strong></a> · 
      <a href="https://dnwjddl.github.io/"><strong>Woojung Han<sup>2</sup></strong></a> · 
      <a href="https://micv.yonsei.ac.kr/seongjae"><strong>Seong Jae Hwang<sup>1</sup></strong></a>
    </h3>
    <h3 align="center">
      <sup>1</sup>Department of Artificial Intelligence
      <sup>2</sup>Department of Computer Science
    </h3>
    <h3 align="center">
      Yonsei University
    </h3>
  </p>
  <p align="center">
    <a href="https://arxiv.org/abs/2603.02919"><img alt='arXiv' src="https://img.shields.io/badge/arXiv-2603.02919-b31b1b.svg"></a>
    <a href="https://youngjun-jun.github.io/imap-project/"><img alt='Project Page' src="https://img.shields.io/badge/Project-Website-orange"></a>
    <a href="https://github.com/youngjun-jun/IMAP"><img alt='Demo (Soon)' src="https://img.shields.io/badge/%F0%9F%A4%97%20Demo (Soon)-IMAP-yellow"></a>
  </p>
  <br>
</p>
 
<be>
<img width="1775" alt="Image" src="assets/fig1.png" />

> #### **Interpretable Motion-Attentive Maps: Spatio-Temporally Localizing Concepts in Video Diffusion Transformers**<be>  
>IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2026  
>Youngjun Jun, Seil Kang, Woojung Han, Seong Jae Hwang  
>Yonsei University
### Abstract
Video Diffusion Transformers (DiTs) have been synthesizing high-quality video with high fidelity from given text descriptions involving motion. However, understanding how Video DiTs convert motion words into video remains insufficient. Furthermore, while prior studies on interpretable saliency maps primarily target objects, motion-related behavior in Video DiTs remains largely unexplored. In this paper, we investigate concrete motion features that specify when and which object moves for a given motion concept. First, to spatially localize, we introduce GramCol, which adaptively produces per-frame saliency maps for any text concept, including both motion and non-motion. Second, we propose a motion-feature selection algorithm to obtain an Interpretable Motion-Attentive Map (IMAP) that localizes motion spatially and temporally. Our method discovers concept saliency maps without the need for any gradient calculation or parameter update. Experimentally, our method shows outstanding localization capability on the motion localization task and zero-shot video semantic segmentation, providing interpretable and clearer saliency maps for both motion and non-motion concepts.

## Prerequisites

- Python > 3.10
- CUDA > 12.4

## Installation

Please install the appropriate conda environment corresponding to the model you intend to use.

**For CogVideoX:**
<!-- To generate IMAP using CogVideoX, install the environment from `imap-cogvideox.yaml`. -->

```bash
conda env create -f EnvCreate.yml
conda activate imap
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install -r requirements-cogvideox.txt
```

**For Wan2.1 or HunyuanVideo:**
<!-- To generate IMAP using Wan2.1 or HunyuanVideo, install the environment from `imap-wanhunyuan.yaml`. -->

```bash
conda env create -f EnvCreate.yml
conda activate imap
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install -r requirements-wan-hunyuan.txt
```

## Usage

### Generating IMAP (Generation Process)

To create an IMAP while generating a video, run the sampling script.

> **Note:** The appearance of the visualized map can vary significantly depending on the visualization method used. We highly recommend saving and using the `.npy` files for visualization.

Run the following command:

```bash
python main_Sampling.py
```

Key arguments and variables used in the paper can be found in the `scripts` directory.

### Extracting IMAP from Real Videos (Renoising Process)

To extract an IMAP from an existing real video, run the renoising script.

> **Note:** The appearance of the visualized map can vary significantly depending on the visualization method used. We highly recommend saving and using the `.npy` files for visualization.

> **Important - Input Resolution & Frames:**
> * **CogVideoX** and **HunyuanVideo**: Require 49-frame videos with a resolution of **480x720**.
> * **Wan2.1**: Requires 49-frame videos with a resolution of **480x832**.

Run the following command:

```bash
python main_Renoising.py
```

Key arguments and variables used in the paper can be found in the `scripts` directory.

## Acknowledgement

Our implementation partially builds on [helblazer811/ConceptAttention](https://github.com/helblazer811/ConceptAttention).