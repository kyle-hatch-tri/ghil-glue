# GHIL-Glue

Code for the paper [GHIL-Glue: Hierarchical Control with Filtered Subgoal Images](https://ghil-glue.github.io/).


This repository contains the code for evaluating GHIL-Glue with both image-generative and video-generative high-level policies, and implements 
our subgoal filtering method. 

The code for training the image-generative high-level policies can be found in 
the original [SuSIE repo](https://github.com/kvablack/susie.git). We use the same checkpoints for the image-generative high-level 
policies that are used in SuSIE.
The code for training the video-generative high level policies can be found in
our [fork of the DynamiCrafter repo](https://github.com/kyle-hatch-tri/DynamiCrafter_ghil-glue).
The code for training the low-level policies and the subgoal classifier networks can be found in our [fork of the BridgeData V2 repo](https://github.com/kyle-hatch-tri/bridge_data_v2_ghil-glue).
All checkpoints can be downloaded from https://huggingface.co/kyle-hatch-tri/ghil-glue-checkpoints.

## Installation
1. ```git clone --recurse-submodules https://github.com/kyle-hatch-tri/ghil-glue.git```
2. ```conda create -n ghil-glue python=3.8```
3. Install [tensorflow](https://www.tensorflow.org/install/pip) and [JAX](https://jax.readthedocs.io/en/latest/installation.html)
4. ```bash install.sh``` (for troubleshooting see https://github.com/mees/calvin)

## Evaluation

To evaluate GHIL-Glue on the CALVIN simulator benchmark,

1. Set the values of the environment variables in ```eval_ghilglue.sh``` to the paths to your downloaded checkpoints.
2. Run ```bash eval_ghilglue.sh```

To evaluate GHIL-Glue on a physical BridgeData V2 WidowX environment, 

1. Set up your WidowX server following these [instructions](https://github.com/rail-berkeley/bridge_data_v2?tab=readme-ov-file#evaluation) from the [BridgeData V2 repo](https://github.com/rail-berkeley/bridge_data_v2).
2. Convert the low-level policy checkpoints to hlo format using ```external/susie/scripts/policy_ckpt_to_hlo.py```
3. Set the values of the environment variables in ```external/susie/scripts/robot/eval_diffusion.sh``` to the paths to your downloaded checkpoints.
4. Run ```bash external/susie/scripts/robot/eval_diffusion.sh```

## Cite

This code is based on [calvin-sim](https://github.com/pranavatreya/calvin-sim) from Pranav Atreya.

If you use this code and/or GHIL-Glue in your work, please cite the paper with:

```
@misc{hatch2024ghilgluehierarchicalcontrolfiltered,
      title={GHIL-Glue: Hierarchical Control with Filtered Subgoal Images}, 
      author={Kyle B. Hatch and Ashwin Balakrishna and Oier Mees and Suraj Nair and Seohong Park and Blake Wulfe and Masha Itkina and Benjamin Eysenbach and Sergey Levine and Thomas Kollar and Benjamin Burchfiel},
      year={2024},
      eprint={2410.20018},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2410.20018}, 
}
```
