# Mask2IV: Interaction-Centric Video Generation via Mask Trajectories

[![arXiv](https://img.shields.io/badge/arXiv-2510.03135-b31b1b.svg)](https://arxiv.org/abs/2510.03135)
[![GitHub](https://img.shields.io/website?label=Project%20&up_message=website&url=https://reagan1311.github.io/mask2iv/)](https://reagan1311.github.io/mask2iv/)

## Abstract

Generating interaction-centric videos, such as those depicting humans or robots interacting with objects, is crucial for embodied intelligence, as they provide rich and diverse visual priors for robot learning, manipulation policy training, and affordance reasoning. However, existing methods often struggle to model such complex and dynamic interactions. While recent studies show that masks can serve as effective control signals and enhance generation quality, obtaining dense and precise mask annotations remains a major challenge for real-world use.
To overcome this limitation, we introduce Mask2IV, a novel framework specifically designed for interaction-centric video generation. It adopts a decoupled two-stage pipeline that first predicts plausible motion trajectories for both actor and object, then generates a video conditioned on these trajectories. This design eliminates the need for dense mask inputs from users while preserving the flexibility to manipulate the interaction process. Furthermore, Mask2IV supports versatile and intuitive control, allowing users to specify the target object of interaction and guide the motion trajectory through action descriptions or spatial position cues.
To support systematic training and evaluation, we curate two benchmarks covering diverse action and object categories across both human-object interaction and robotic manipulation scenarios.
Extensive experiments demonstrate that our method achieves superior visual realism and controllability compared to existing baselines.

<p align="center">
 <img src="./assets/teaser.png" align=center>
</p>

## Setup
### 1. Environment Installation
Install environment following [DynamiCrafter](https://github.com/Doubiiu/DynamiCrafter).


### 2. Datasets Preparation
(1) Download datasets of [HOI4D](https://hoi4d.github.io/) and [BridgeDataV2](https://rail-berkeley.github.io/bridgedata/)

(2) Update `data_dir` in config files

(3) Update `meta_path` (data csv files, in `/data_csv` folder) in config files

PS: Please refer to scripts in `/func` folder for details on data preprocessing.


## Training & Evaluation
Download pretrained models [DynamiCrafter_512](https://huggingface.co/Doubiiu/DynamiCrafter_512/blob/main/model.ckpt) and [DynamiCrafter512_interp](https://huggingface.co/Doubiiu/DynamiCrafter_512_Interp/blob/main/model.ckpt), and update `pretrained_checkpoint` in config files.

I. Stage 1 training & evaluation
```
sh run_fist.sh hoi4d | bdv2
```

II. Stage 2 training & evaluation
```
sh run_second.sh hoi4d | bdv2
```

Pretrained models of Mask2IV can be downloaded here: [Huggingface](https://huggingface.co/Gen1113/Mask2IV-pretrained-models)

## Inference
```
sh scripts/run_hoi4d_2tage.sh  # HOI4D
sh scripts/run_bdv2_2stage.sh  # BridgeDataV2 
```

## Citation
```
@inproceedings{li2026mask2iv,
      title     = {Mask2IV: Interaction-Centric Video Generation via Mask Trajectories}, 
      author    = {Li, Gen and Zhao, Bo and Yang, Jianfei and Sevilla-Lara, Laura},
      journal   = {Proceedings of the AAAI Conference on Artificial Intelligence},
      year      = {2026},
    }
```

## Anckowledgement
A large portion of the code is based on [DynamiCrafter](https://github.com/Doubiiu/DynamiCrafter). Thanks for their great work!