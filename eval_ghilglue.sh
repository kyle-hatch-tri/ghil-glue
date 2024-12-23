#!/bin/bash

export PYTHONPATH="/<path_to>/calvin-sim/external/susie:$PYTHONPATH"
export PYTHONPATH="/<path_to>/calvin-sim/external/jaxrl_m:$PYTHONPATH"
export WANDB_API_KEY="<wandb_api_key>"
export WANDB_ENTITY="<wandb_entity>" 

S3_SAVE_URI="s3://<s3_save_uri>"


export NUM_DENOISING_STEPS=50
export NUM_EVAL_SEQUENCES=100
export EP_LEN=360

export DIFFUSION_MODEL_CHECKPOINT="/<path_to_checkpoints>/susie_test/public_model/checkpoint_only/params_ema"

export UNIPI_MODEL_CONFIG="dynamicrafer_configs/inference_256_v1.0.yaml"
export UNIPI_MODEL_CHECKPOINT="/<path_to_checkpoints>/DynamiCrafter/training_256_v1.0/2024.09.02_17.41.42/checkpoints/epoch=63-step=36000.ckpt"

export GC_POLICY_CHECKPOINT="/<path_to_checkpoints>/susie_low_level/calvin/gcdiffusion/auggoaldiff/seed_0/20240227_194024/checkpoint_150000"
agent_config_string="calvin_gcdiffusion_noactnorm-sagemaker-auggoaldiff"
export HIGH_LEVEL_VF_CHECKPOINT="/<path_to_checkpoints>/susie_low_level/calvinlcbc/lcgcprogressvf/auggoaldiff/seed_0/20240510_005751/checkpoint_100000"
vf_agent_config_string="calvinlcbc_lcgcprogressvf_noactnorm-auggoaldiff"


FLAT_POLICY=0
PROMPT_W=7.5
CONTEXT_W=1.5

UNIPI=0
export SUBGOAL_MAX=20
USE_TEMPORAL_ENSEMBLING=0
FILTERING_METHOD="high_level_vf"

# UNIPI=1
# export SUBGOAL_MAX=1
# USE_TEMPORAL_ENSEMBLING=0
# FILTERING_METHOD="high_level_video_vf"

# NUM_SAMPLES=4
NUM_SAMPLES=1

export XLA_PYTHON_CLIENT_MEM_FRACTION=.1
export DEBUG=0
export CUDA_VISIBLE_DEVICES=0,1

export HF_HOME="/<path_to_cache>" 


echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

python3 -u evaluate_policy_subgoal_diffusion.py \
--agent_config_string $agent_config_string \
--vf_agent_config_string $vf_agent_config_string \
--dataset_path mini_dataset \
--diffusion_model_checkpoint_path "$DIFFUSION_MODEL_CHECKPOINT" \
--gc_policy_checkpoint_path "$GC_POLICY_CHECKPOINT" \
--diffusion_model_framework jax \
--save_to_s3 0 \
--s3_save_uri $S3_SAVE_URI \
--use_temporal_ensembling $USE_TEMPORAL_ENSEMBLING \
--num_denoising_steps $NUM_DENOISING_STEPS \
--num_samples $NUM_SAMPLES \
--filtering_method $FILTERING_METHOD \
--flat_policy $FLAT_POLICY \
--prompt_w $PROMPT_W \
--context_w $CONTEXT_W \
--unipi $UNIPI