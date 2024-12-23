from pathlib import Path
import os 
import hydra
import numpy as np
from omegaconf import OmegaConf

from calvin_agent.evaluation import jax_diffusion_model
from calvin_agent.evaluation import diffusion_gc_policy
from calvin_env.envs.play_table_env import get_env
from calvin_agent.evaluation.utils import get_env_state_for_initial_condition
from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.gcbc_train_config import get_config

from tqdm import tqdm, trange

"""
Example usage:

export PYTHONPATH="/<path_to>/calvin-sim/external/susie:$PYTHONPATH"
export PYTHONPATH="/<path_to>/calvin-sim/external/jaxrl_m:$PYTHONPATH"
export PYTHONPATH="/<path_to>/calvin-sim/external/urdfpy:$PYTHONPATH"
export PYTHONPATH="/<path_to>/calvin-sim/external/networkx:$PYTHONPATH"
export DIFFUSION_MODEL_CHECKPOINT=/<path_to_susie-calvin-checkpoints>/public_model/checkpoint_only/params_ema
export GC_POLICY_CHECKPOINT=/<path_to_susie-calvin-checkpoints>/susie_low_level/calvin/gcdiffusion/auggoaldiff/seed_0/20240227_194024/checkpoint_150000
python3 inference_minimal.py 
"""

class MinimalModel:
    def __init__(self, agent_config, env_name, agent_type):
        self.diffusion_model = jax_diffusion_model.DiffusionModel(50, num_samples=1)

        self.gc_policy = diffusion_gc_policy.GCPolicy(agent_config, env_name, agent_type, os.getenv("GC_POLICY_CHECKPOINT"), normalize_actions=False, use_temporal_ensembling=True, text_processor=None)

        self.subgoal_max = 20
        self.episode_counter = None
        self.language_task = None
        self.sub_task = None

        # Other necessary variables for running rollouts
        self.goal_image = None
        self.subgoal_counter = 0

    def reset(self):
        if self.episode_counter is None: # this is the first time reset has been called
            self.episode_counter = 0
        else:
            # Update/reset all the variables
            self.episode_counter += 1

            # Reset the GC policy
            self.gc_policy.reset()

    def step(self, obs, goal, subtask, ep_idx=0):
        """
        Args:
            obs: environment observations
            goal: embedded language goal
        Returns:
            action: predicted action
        """
        rgb_obs = obs["rgb_obs"]["rgb_static"]
        self.language_task = goal
        self.subtask = subtask
 
        # If we need to, generate a new goal image
        if self.goal_image is None or self.subgoal_counter >= self.subgoal_max:
            goal_images = self.diffusion_model.generate(self.language_task, rgb_obs)
            
            self.subgoal_counter = 0
            assert goal_images.shape[0] == 1, f"goal_images.shape: {goal_images.shape}"
            self.goal_image = goal_images[0]

        # Query the behavior cloning model
        action_cmd = self.gc_policy.predict_action(rgb_obs, self.goal_image)

        # Update variables
        self.subgoal_counter += 1

        return action_cmd

            


EP_LEN = 360
NUM_SEQUENCES = 2
dataset_path = "mini_dataset"
agent_config_string = "calvin_gcdiffusion_noactnorm-sagemaker-auggoaldiff"

agent_config, env_name, agent_type, _ = get_config(agent_config_string)

model = MinimalModel( agent_config, env_name, agent_type)



val_folder = Path(dataset_path) / "validation"
env = get_env(val_folder, show_gui=False)


conf_dir = Path(__file__).absolute().parents[0] / "calvin_models" / "conf"
task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
task_oracle = hydra.utils.instantiate(task_cfg)
val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

eval_sequences = get_sequences(NUM_SEQUENCES)

for initial_state, eval_sequence in eval_sequences:
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    for subtask in eval_sequence:
        obs = env.get_obs()

        # get lang annotation for subtask
        lang_annotation = val_annotations[subtask][0]
        model.reset()
        start_info = env.get_info()

        print("lang_annotation:", lang_annotation)

        for step in trange(EP_LEN):
            action = model.step(obs, lang_annotation, subtask)
            obs, _, _, current_info = env.step(action)

            current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
            if len(current_task_info) > 0:
                print(f"Success for subtask {subtask} after {step + 1} timesteps")
                break 
