# GHIL-Glue

Code for the paper [GHIL-Glue: Hierarchical Control with Filtered Subgoal Images](https://ghil-glue.github.io/).


This repository contains the code for evaluating GHIL-Glue with both image-generative and video-generative high level policies, and implements 
our subgoal filtering method. 

The code for training the image-generative high level policies, as well as instructions for downloaded the checkpoints, can be found in 
the [SuSIE](https://github.com/kvablack/susie.git) repo. 
The code for training the video-generative high level policies, as well as instructions for downloaded the checkpoints, can be found in
our fork of the [DynamiCrafter]() repo.
The code for training the low-level policies and the subgoal classifier networks, as well as instructions for downloaded the checkpoints, can be found in our fork of the [BridgeData V2]() repo.

## Installation
1. ```git clone --recurse-submodules https://github.com/pranavatreya/calvin-sim.git```
2. ```conda create -n susie-calvin python=3.8```
3. Install [tensorflow](https://www.tensorflow.org/install/pip) and [JAX](https://jax.readthedocs.io/en/latest/installation.html)
4. ```bash install.sh``` (for troubleshooting see https://github.com/mees/calvin)

## Evaluation

To evaluate GHIL-Glue on the CALVIN simulator benchmark,

1. Set the values of the environment variables in ```eval_ghilglue.sh``` to the paths to your downloaded checkpoints.
2. Run ```bash eval_ghilglue.sh```

To see our evaluation procedure of GHIL-Glue on our physical BridgeData V2 WidowX environment, 

1. Convert the downloaded checkpoints to hlo format using ```external/susie/scripts/policy_ckpt_to_hlo.py```
2. Set the values of the environment variables in ```external/susie/scripts/robot/eval_diffusion.sh``` to the paths to your downloaded checkpoints.
3. Run ```bash external/susie/scripts/robot/eval_diffusion.sh```