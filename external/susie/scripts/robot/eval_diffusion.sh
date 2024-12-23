#!/bin/bash


export PYTHONPATH="/PATH/TO/calvin-sim/external/susie:$PYTHONPATH"
export PYTHONPATH="/PATH/TO/calvin-sim/external/jaxrl_m:$PYTHONPATH"


# SUSIE with low level policy trained with unsynced image augmentations, high level task sampling with 8 goal images, seed_0
python3 -u scripts/robot/eval_diffusion.py \
--diffusion_checkpoint kvablack/susie \
--policy_checkpoint /PATH/TO/bridge_susie_checkpoints/gcdiffusion/auggoaldiff/seed_0/20240602_011058/checkpoint_150000/serialized_policy_ckpt \
--diffusion_pretrained_path runwayml/stable-diffusion-v1-5:flax \
--diffusion_num_steps 50 \
--prompt_w 7.5 \
--context_w 2 \
--num_timesteps 15 \
--num_samples 8 \
--vf_agent_type lcgcprogressvf \
--vf_policy_checkpoint /PATH/TO/bridge_susie_checkpoints/lcgcprogressvf/auggoaldiff/seed_0/20240602_010253/checkpoint_100000 \
--vf_agent_config_path /PATH/TO/bridge_susie_checkpoints/lcgcprogressvf/auggoaldiff/seed_0/20240602_010253/config.yaml \
--dummy_run=True \
--video_save_path /PATH/TO/bridge_saved_videos


# # SUSIE with high level task sampling with 8 goal images, seed_0
# python3 -u scripts/robot/eval_diffusion.py \
# --diffusion_checkpoint kvablack/susie \
# --policy_checkpoint /PATH/TO/bridge_susie_checkpoints/gcdiffusion/default/seed_0/20240602_010259/checkpoint_150000/serialized_policy_ckpt \
# --diffusion_pretrained_path runwayml/stable-diffusion-v1-5:flax \
# --diffusion_num_steps 50 \
# --prompt_w 7.5 \
# --context_w 2 \
# --num_timesteps 15 \
# --num_samples 8 \
# --vf_agent_type lcgcprogressvf \
# --vf_policy_checkpoint /PATH/TO/bridge_susie_checkpoints/lcgcprogressvf/default/seed_0/20240602_011054/checkpoint_100000 \
# --vf_agent_config_path /PATH/TO/bridge_susie_checkpoints/lcgcprogressvf/default/seed_0/20240602_011054/config.yaml \
# --dummy_run=True 


# # SUSIE with low level policy trained with unsynced image augmentations, no high level task filtering, seed_0
# python3 -u scripts/robot/eval_diffusion.py \
# --diffusion_checkpoint kvablack/susie \
# --policy_checkpoint /PATH/TO/bridge_susie_checkpoints/gcdiffusion/auggoaldiff/seed_0/20240602_011058/checkpoint_150000/serialized_policy_ckpt \
# --diffusion_pretrained_path runwayml/stable-diffusion-v1-5:flax \
# --diffusion_num_steps 50 \
# --prompt_w 7.5 \
# --context_w 2 \
# --num_timesteps 15 \
# --num_samples 1 \
# --vf_agent_type lcgcprogressvf \
# --vf_policy_checkpoint /PATH/TO/bridge_susie_checkpoints/lcgcprogressvf/auggoaldiff/seed_0/20240602_010253/checkpoint_100000 \
# --vf_agent_config_path /PATH/TO/bridge_susie_checkpoints/lcgcprogressvf/auggoaldiff/seed_0/20240602_010253/config.yaml \
# --dummy_run=True 


# # SUSIE, no high level task filtering, seed_0
# python3 -u scripts/robot/eval_diffusion.py \
# --diffusion_checkpoint kvablack/susie \
# --policy_checkpoint /PATH/TO/bridge_susie_checkpoints/gcdiffusion/default/seed_0/20240602_010259/checkpoint_150000/serialized_policy_ckpt \
# --diffusion_pretrained_path runwayml/stable-diffusion-v1-5:flax \
# --diffusion_num_steps 50 \
# --prompt_w 7.5 \
# --context_w 2 \
# --num_timesteps 15 \
# --num_samples 1 \
# --vf_agent_type lcgcprogressvf \
# --vf_policy_checkpoint /PATH/TO/bridge_susie_checkpoints/lcgcprogressvf/default/seed_0/20240602_011054/checkpoint_100000 \
# --vf_agent_config_path /PATH/TO/bridge_susie_checkpoints/lcgcprogressvf/default/seed_0/20240602_011054/config.yaml \
# --dummy_run=True 


