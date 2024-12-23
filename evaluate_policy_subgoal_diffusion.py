import argparse
from collections import defaultdict
import logging
import os
from pathlib import Path
import sys
import time
import requests
import json
import cv2
from tqdm import tqdm, trange
import tensorflow as tf
import base64
import io
from calvin_agent.evaluation import unipi_dynamicrafter_inference

from calvin_agent.evaluation import jax_diffusion_model
from calvin_agent.evaluation import diffusion_gc_policy
from calvin_agent.evaluation import gc_policy

import datetime
from s3_save import S3SyncCallback
import random 
from copy import deepcopy
import shutil
import pickle

from jaxrl_m.data.text_processing import text_processors

from calvin_agent.evaluation.gcbc_train_config import get_config

# This is for using the locally installed repo clone when using slurm
from calvin_agent.models.calvin_base_model import CalvinBaseModel

sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())

from calvin_agent.evaluation.multistep_sequences import get_sequences, tasks
from calvin_agent.evaluation.utils import (
    collect_plan,
    count_success,
    create_tsne,
    get_env_state_for_initial_condition,
    get_log_dir,
    join_vis_lang,
    print_and_save,
)

import hydra
import numpy as np
from omegaconf import OmegaConf
from termcolor import colored
import torch
from tqdm.auto import tqdm
from PIL import Image
from glob import glob

from multiprocessing import Pool, Manager
from itertools import combinations

from calvin_env.envs.play_table_env import get_env
from calvin_agent.evaluation.libero_env import LiberoEnv

logger = logging.getLogger(__name__)

DEBUG = int(os.getenv("DEBUG"))

if DEBUG: 
    EP_LEN = 10
    NUM_SEQUENCES = 1
else:
    EP_LEN = int(os.getenv("EP_LEN"))
    NUM_SEQUENCES = int(os.getenv("NUM_EVAL_SEQUENCES"))

def make_env(env_name, dataset_path):
    if env_name == "calvin" or env_name == "calvinlcbc":
        val_folder = Path(dataset_path) / "validation"
        env = get_env(val_folder, show_gui=False)
        return env
    else:
        raise ValueError(f"Unsupported env_name: \"{env_name}\".")

def save_video(output_video_file, frames, fps=30):
     # Extract frame dimensions
    height, width, _ = frames.shape[1:]

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also use other codecs such as 'XVID'
    os.makedirs(os.path.dirname(output_video_file), exist_ok=True)
    video_writer = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height))

    # Write each frame to the video file
    for frame in frames:
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(bgr_frame)

    # Release the video writer object
    video_writer.release()

def get_oracle_goals_calvin(oracle_goals_dir, oracle_goals_type):
    oracle_goals = {}
    language_task_files = glob(os.path.join(oracle_goals_dir, "**", "language_task.txt"), recursive=True)
    for language_task_file in language_task_files:
        with open(language_task_file, "r") as f:
            subtask = f.readline().strip() 
            f.readline()
            line = f.readline().strip()
            if line.split(":")[-1].strip() == "True":
                if subtask not in oracle_goals:
                    oracle_goals_file = os.path.join(os.path.dirname(language_task_file), f"{oracle_goals_type}_goals.npy")
                    print(f"Loading oracle goals from \"{oracle_goals_file}\".")
                    goals = np.load(oracle_goals_file)
                    goals = np.concatenate([goals, goals[-1][None], goals[-1][None], goals[-1][None], goals[-1][None], goals[-1][None]])

                    oracle_goals[subtask] = goals

    return oracle_goals


def high_level_vf_filter(vf_agent, language_goal, rgb_obs, goal_images):
    v = vf_agent.value_function_ranking_lcgc(rgb_obs, goal_images, language_goal)
    sorted_idxs = np.argsort(v)[::-1]
    ordered_goal_images = goal_images[sorted_idxs]
    ordered_vs = v[sorted_idxs]

    best_goal_idx = sorted_idxs[0]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (0, 255, 0) 
    line_type = 2  # Line thickness

    frame = []
    for i in range(ordered_goal_images.shape[0]):
        img = ordered_goal_images[i]
        orig_idx = np.arange(goal_images.shape[0])[i]
        img = cv2.putText(img, f'[{i}] v: {ordered_vs[i]:.3f}', (25, 20), font, font_scale, font_color, line_type)
        frame.append(img)

    assert len(frame) % 4 == 0, f"len(frame): {len(frame)}"
    frame_rows = []
    for row_idx in range(len(frame) // 4):
        start = row_idx * 4
        end = start + 4
        frame_row = np.concatenate([rgb_obs] + frame[start:end], axis=1)
        frame_rows.append(frame_row)

    query_frame = np.concatenate(frame_rows, axis=0)

    mode = "okay"
    return best_goal_idx, {"query_frame":query_frame}, mode


def high_level_video_vf_filter(vf_agent, language_goal, rgb_obs, goal_videos):

    goal_images = goal_videos[:, -1, ...]

    v = vf_agent.value_function_ranking_lcgc(rgb_obs, goal_images, language_goal)
    sorted_idxs = np.argsort(v)[::-1]
    ordered_goal_videos = goal_videos[sorted_idxs]
    ordered_vs = v[sorted_idxs]

    best_goal_idx = sorted_idxs[0]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (0, 255, 0) 
    line_type = 2  # Line thickness

    vid = []
    for i in range(ordered_goal_videos.shape[0]):
        video = ordered_goal_videos[i]
        new_video = []
        for frame in video:
            frame = cv2.putText(frame, f'[{i}] v: {ordered_vs[i]:.3f}', (25, 20), font, font_scale, font_color, line_type)
            new_video.append(frame)
        new_video = np.stack(new_video, axis=0)
        vid.append(new_video)

    
    assert len(vid) % 4 == 0, f"len(vid): {len(vid)}"
    video_rows = []
    for row_idx in range(len(vid) // 4):
        start = row_idx * 4
        end = start + 4
        video_row = np.concatenate([np.tile(rgb_obs, [16, 1, 1, 1])] + vid[start:end], axis=2)
        video_rows.append(video_row)

    query_video = np.concatenate(video_rows, axis=1)

    query_frame = query_video[-1]

    mode = "okay"
    return best_goal_idx, {"query_frame":query_frame}, mode



class CustomModel(CalvinBaseModel):
    def __init__(self, agent_config, env_name, agent_type, oracle_goals_dir, oracle_goals_type, use_temporal_ensembling, diffusion_model_checkpoint_path, gc_policy_checkpoint_path, gc_vf_checkpoint_path, num_denoising_steps, num_samples=1, prompt_w=7.5, context_w=1.5, filtering_method=None, flat_policy=False, diffusion_model_framework="jax", vf_agent_config=None, vf_agent_type=None):
        # Initialize diffusion model

        self.num_samples = num_samples
        self.filtering_method = filtering_method
        self.flat_policy = flat_policy
        self.prompt_w = prompt_w
        self.context_w = context_w

        self.oracle_goals_type = oracle_goals_type
    
        if not self.flat_policy:
            if diffusion_model_framework == "jax":
                self.diffusion_model = jax_diffusion_model.DiffusionModel(num_denoising_steps, num_samples=self.num_samples)
            else:
                raise ValueError(f"Unsupported diffusion model framework: \"{diffusion_model_framework}\".")

        if self.flat_policy:
            text_processor = text_processors["muse_embedding"]()
        else:
            text_processor = None

        if agent_type == "gcbc":
            assert not use_temporal_ensembling # Don't want to use temporal ensembling for the gcbc policy for SuSIE
            self.gc_policy = gc_policy.GCPolicy(os.getenv("GC_POLICY_CHECKPOINT"), three_layers=True, use_temporal_ensembling=use_temporal_ensembling)
        else:
            assert use_temporal_ensembling # Want to use temporal ensembling for the gcdiffusion policy for SuSIE
            normalize_actions = False ### TODO better way of handling this. 
            self.gc_policy = diffusion_gc_policy.GCPolicy(agent_config, env_name, agent_type, os.getenv("GC_POLICY_CHECKPOINT"), normalize_actions=normalize_actions, use_temporal_ensembling=use_temporal_ensembling, text_processor=text_processor)


        if self.filtering_method == "high_level_vf":
            assert vf_agent_config is not None
            assert vf_agent_type is not None
            self.vf_agent = diffusion_gc_policy.GCPolicy(vf_agent_config, env_name, vf_agent_type, os.getenv("HIGH_LEVEL_VF_CHECKPOINT"), normalize_actions=False, use_temporal_ensembling=False, text_processor=text_processors["muse_embedding"]())
        elif self.filtering_method == "low_level_vf":
            assert vf_agent_config is not None
            assert vf_agent_type is not None
            self.vf_agent = diffusion_gc_policy.GCPolicy(vf_agent_config, env_name, vf_agent_type, os.getenv("LOW_LEVEL_VF_CHECKPOINT"), normalize_actions=False, use_temporal_ensembling=False, text_processor=None)

        timestamp = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
        
        self.log_dir = "results"
        if DEBUG:
            self.log_dir = os.path.join(self.log_dir, "el_trasho")

        self.oracle_goals = None
        if oracle_goals_dir is not None:
            if "calvin" in env_name:
                self.oracle_goals = get_oracle_goals_calvin(oracle_goals_dir, self.oracle_goals_type)

        if self.num_samples > 1:
            assert self.filtering_method is not None
        else:
            assert self.num_samples == 1, f"self.num_samples: {self.num_samples}"


        self.log_dir = os.path.join(self.log_dir, env_name, *diffusion_model_checkpoint_path.strip("/").split("/")[-3:-1], *gc_policy_checkpoint_path.strip("/").split("/")[-6:], timestamp)

        print(f"Logging to \"{self.log_dir}\"...")
        os.makedirs(self.log_dir, exist_ok=True)
        self.episode_counter = None
        self.language_task = None
        self.sub_task = None
        self.obs_image_seq = None
        self.goal_image_seq = None
        self.vranking_save_freq = 1
        self.action_seq = None
        self.combined_images = None

        # Other necessary variables for running rollouts
        self.goal_image = None
        self.subgoal_counter = 0
        self.subgoal_max = int(os.getenv("SUBGOAL_MAX"))
        self.pbar = None

        shutil.copy2("eval_susie.sh", os.path.join(self.log_dir, "eval_susie.sh"))

    def save_info(self, success, initial_images=None):
        episode_log_dir = os.path.join(self.log_dir, "ep" + str(self.episode_counter))
        if not os.path.exists(episode_log_dir):
            os.makedirs(episode_log_dir)

        # Log the language task
        with open(os.path.join(episode_log_dir, "language_task.txt"), "w") as f:
            f.write(self.subtask + "\n")
            f.write(self.language_task + "\n")
            f.write(f"success: {success}\n")
        

        if not self.flat_policy:
            # Log the combined image
            size = (400, 200)
            out = cv2.VideoWriter(os.path.join(episode_log_dir, "combined.mp4"), cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
            for i in range(len(self.combined_images)):
                rgb_img = cv2.cvtColor(self.combined_images[i], cv2.COLOR_RGB2BGR)
                out.write(rgb_img)
            out.release()

        if initial_images is not None:
            cv2.imwrite(os.path.join(episode_log_dir, "initial_images.png"), initial_images[..., ::-1])

        if "unfiltered_goal_images_frames" in self.goal_images_info:
            size = (self.goal_images_info["unfiltered_goal_images_frames"][0].shape[1], 200)
            out = cv2.VideoWriter(os.path.join(episode_log_dir, "unfiltered_goal_images.mp4"), cv2.VideoWriter_fourcc(*'DIVX'), 2, size)
            for i in range(len(self.goal_images_info["unfiltered_goal_images_frames"])):
                img = self.goal_images_info["unfiltered_goal_images_frames"][i]
                rgb_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                out.write(rgb_img)
            out.release()

        if "unfiltered_goal_images" in self.goal_images_info:
            for t in range(len(self.goal_images_info["unfiltered_goal_images"])):
                goal_images = self.goal_images_info["unfiltered_goal_images"][t]
                save_dir = os.path.join(episode_log_dir, "unfiltered_goal_images", f"timestep_{t * self.subgoal_max:03}")
                os.makedirs(save_dir, exist_ok=True)
                for i, goal_image in enumerate(goal_images):
                    cv2.imwrite(os.path.join(save_dir, f"goal_image_{i:03}.png"), goal_image[..., ::-1])


                rgb_obs = self.goal_images_info["rgb_obs_at_filtering"][t]
                cv2.imwrite(os.path.join(save_dir, f"rgb_obs.png"), rgb_obs[..., ::-1])


        if "unfiltered_goal_images_and_vpreds" in self.goal_images_info:
            for t in range(len(self.goal_images_info["unfiltered_goal_images_and_vpreds"])):
                goal_images_and_vpreds = self.goal_images_info["unfiltered_goal_images_and_vpreds"][t]
                save_dir = os.path.join(episode_log_dir, "unfiltered_goal_images_and_vpreds", f"timestep_{t * self.subgoal_max:03}")
                os.makedirs(save_dir, exist_ok=True)
                for v_pred, goal_image in goal_images_and_vpreds.items():
                    cv2.imwrite(os.path.join(save_dir, f"goal_image_vpred_{v_pred:.4f}.png"), goal_image[..., ::-1])

                rgb_obs = self.goal_images_info["rgb_obs_at_filtering"][t]
                cv2.imwrite(os.path.join(save_dir, f"rgb_obs.png"), rgb_obs[..., ::-1])


        if "chat_gpt_answer" in self.goal_images_info:
            with open(os.path.join(episode_log_dir, "chat_gpt_answers.txt"), "w") as f:
                for i in range(len(self.goal_images_info["chat_gpt_answer"])):
                    chat_gpt_answer = self.goal_images_info["chat_gpt_answer"][i]
                    f.write(f"[{i}] {chat_gpt_answer}\n")

        if "query_infos" in self.goal_images_info:
            save_dir = os.path.join(episode_log_dir, "chat_gpt_answers")
            os.makedirs(save_dir, exist_ok=True)
            for i in range(len(self.goal_images_info["query_infos"])):
                query_info = self.goal_images_info["query_infos"][i]
                win_counts = self.goal_images_info["win_counts"][i]
                total_query_time = self.goal_images_info["total_query_time"][i]

                with open(os.path.join(save_dir, f"chat_gpt_answers_{i * self.subgoal_max:03}.txt"), "w") as f:
                    f.write(f"WIN COUNTS\n")
                    for i, count in win_counts.items():
                        f.write(f"\tgoal image {i}: {count}\n")

                    f.write(f"\\nTotal query time: {total_query_time:.3f}s.\n\n")

                    f.write("\n\n\n")
                    

                    for (i, j), info in query_info.items():
                        f.write("=" * 30 + f" goal images ({i}, {j}) " + "=" * 30 + "\n")
                        f.write(f"timestep: {i * self.subgoal_max:03}\n")
                        f.write(f"Query time: {info['query1 time']:.3f}s, {info['query2 time']:.3f}s.\n")
                        answer = info["answer"]
                        f.write(f"{answer}\n\n\n")

        if "query_frame" in self.goal_images_info:
            save_video(os.path.join(episode_log_dir, "query_frames.mp4"), np.stack(self.goal_images_info["query_frame"], axis=0), fps=10)

    def reset(self):
        if self.episode_counter is None: # this is the first time reset has been called
            self.episode_counter = 0
            self.obs_image_seq = []
            self.goal_image_seq = []
            self.action_seq = []
            self.combined_images = []

            self.generated_goals = []
            self.retrospective_true_goals = []
            self.hardcoded_goal_idx_counter = 0

            self.goal_images_info = defaultdict(list)
        else:
            # Update/reset all the variables
            self.episode_counter += 1
            self.obs_image_seq = []
            self.goal_image_seq = []
            self.action_seq = []
            self.goal_image = None
            self.combined_images = []
            self.subgoal_counter = 0

            self.generated_goals = []
            self.retrospective_true_goals = []
            self.hardcoded_goal_idx_counter = 0

            self.goal_images_info = defaultdict(list)

            # Reset the GC policy
            self.gc_policy.reset()

        # tqdm progress bar
        if self.pbar is not None:
            self.pbar.close()
        self.pbar = tqdm(total=EP_LEN)

    def goal_filtering_fn(self, goal, rgb_obs, goal_images):
        if self.filtering_method == "high_level_vf":
            return high_level_vf_filter(self.vf_agent, goal, rgb_obs, goal_images)
        else:
            raise NotImplementedError(f"Goal images filtering method \"{self.filtering_method}\" is not implemented.")


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

        mode = "okay"

        if not self.flat_policy:
 
            # If we need to, generate a new goal image
            if self.goal_image is None or self.subgoal_counter >= self.subgoal_max:
                if self.oracle_goals is None:
                    t0 = time.time()
                    goal_images = self.diffusion_model.generate(self.language_task, rgb_obs, prompt_w=self.prompt_w, context_w=self.context_w)
                    t1 = time.time()
                    print(f"[generation] t1 - t0: {t1 - t0:.3f}") 
                else:
                    if self.hardcoded_goal_idx_counter >= self.oracle_goals[subtask][ep_idx].shape[0]:
                        return -1
                
                    goal_images = self.oracle_goals[subtask][ep_idx][self.hardcoded_goal_idx_counter][None]
                    self.hardcoded_goal_idx_counter += 1


                self.subgoal_counter = 0
                

                if self.num_samples > 1:
                    t0 = time.time()
                    goal_idx, goal_images_info, mode = self.goal_filtering_fn(goal, rgb_obs, goal_images)
                    t1 = time.time()
                    print(f"[filtering] t1 - t0: {t1 - t0:.3f}") 

                    self.goal_image = goal_images[goal_idx]

                    for key, val in goal_images_info.items():
                        self.goal_images_info[key].append(val)
                else:
                    assert goal_images.shape[0] == 1, f"goal_images.shape: {goal_images.shape}"
                    self.goal_image = goal_images[0]


                self.generated_goals.append(self.goal_image.copy())

            self.retrospective_true_goals.append(rgb_obs.copy())

        # Log the image observation and the goal image
        self.obs_image_seq.append(rgb_obs)

        if not self.flat_policy:
            self.goal_image_seq.append(self.goal_image)
            self.combined_images.append(np.concatenate([rgb_obs, self.goal_image], axis=1))
            assert self.combined_images[-1].shape == (200, 400, 3)


        if self.flat_policy:
            action_cmd = self.gc_policy.predict_action_lc(rgb_obs, self.language_task)
        else:
            # Query the behavior cloning model
            action_cmd = self.gc_policy.predict_action(rgb_obs, self.goal_image)

        # Log the predicted action
        self.action_seq.append(action_cmd)

        # Update variables
        self.subgoal_counter += 1

        # Update progress bar
        self.pbar.update(1)

        return action_cmd, mode

class CustomModelUniPi(CalvinBaseModel):
    def __init__(self, agent_config, env_name, agent_type, use_temporal_ensembling, gc_policy_checkpoint_path, num_denoising_steps, num_samples=1, filtering_method=None, vf_agent_config=None, vf_agent_type=None):
        # Initialize diffusion model

        self.num_samples = num_samples
        self.filtering_method = filtering_method
        self.num_denoising_steps = num_denoising_steps

        config_path = os.getenv('UNIPI_MODEL_CONFIG')
        ckpt_path = os.getenv('UNIPI_MODEL_CHECKPOINT')
        self.video_diffusion_model = unipi_dynamicrafter_inference.VideoDiffusionModel(config_path, ckpt_path)

        assert not use_temporal_ensembling # Don't want to use temporal ensembling for either the gcbc or the gcdiffusion policy for unipi
        if agent_type == "gcbc":
            self.gc_policy = gc_policy.GCPolicy(os.getenv("GC_POLICY_CHECKPOINT"), three_layers=True, use_temporal_ensembling=use_temporal_ensembling)
        else:
            normalize_actions = False ### TODO better way of handling this.
            self.gc_policy = diffusion_gc_policy.GCPolicy(agent_config, env_name, agent_type, os.getenv("GC_POLICY_CHECKPOINT"), normalize_actions=normalize_actions, use_temporal_ensembling=use_temporal_ensembling, text_processor=None)
            

        if self.filtering_method == "high_level_video_vf":
            assert vf_agent_config is not None
            assert vf_agent_type is not None
            self.vf_agent = diffusion_gc_policy.GCPolicy(vf_agent_config, env_name, vf_agent_type, os.getenv("HIGH_LEVEL_VF_CHECKPOINT"), normalize_actions=False, use_temporal_ensembling=False, text_processor=text_processors["muse_embedding"]())


        timestamp = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
        
        self.log_dir = "results"
        if DEBUG:
            self.log_dir = os.path.join(self.log_dir, "el_trasho")

        if self.num_samples > 1:
            assert self.filtering_method is not None
        else:
            assert self.num_samples == 1, f"self.num_samples: {self.num_samples}"


        self.subgoal_max = int(os.getenv("SUBGOAL_MAX"))

        self.log_dir = os.path.join(self.log_dir, env_name, f"unipi", *gc_policy_checkpoint_path.strip("/").split("/")[-6:], f"actionrepeat_{self.subgoal_max}_" + timestamp)

        print(f"Logging to \"{self.log_dir}\"...")
        os.makedirs(self.log_dir, exist_ok=True)
        self.episode_counter = None
        self.language_task = None
        self.sub_task = None
        self.obs_image_seq = None
        self.goal_image_seq = None
        self.action_seq = None
        self.combined_images = None

        # Other necessary variables for running rollouts
        self.video_buffer = None
        self.action_buffer = None
        self.pbar = None

        
        self.step_counter = 0
        

        shutil.copy2("eval_susie.sh", os.path.join(self.log_dir, "eval_susie.sh"))

    def save_info(self, success, initial_images=None):
        episode_log_dir = os.path.join(self.log_dir, "ep" + str(self.episode_counter))
        if not os.path.exists(episode_log_dir):
            os.makedirs(episode_log_dir)

        # Log the language task
        with open(os.path.join(episode_log_dir, "language_task.txt"), "w") as f:
            f.write(self.subtask + "\n")
            f.write(self.language_task + "\n")
            f.write(f"success: {success}\n")
        

        # Log the combined image
        size = (400, 200)
        out = cv2.VideoWriter(os.path.join(episode_log_dir, "combined.mp4"), cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
        for i in range(len(self.combined_images)):
            rgb_img = cv2.cvtColor(self.combined_images[i], cv2.COLOR_RGB2BGR)
            out.write(rgb_img)
        out.release()
    

        if initial_images is not None:
            cv2.imwrite(os.path.join(episode_log_dir, "initial_images.png"), initial_images[..., ::-1])

        if "unfiltered_goal_images_frames" in self.goal_images_info:
            size = (self.goal_images_info["unfiltered_goal_images_frames"][0].shape[1], 200)
            out = cv2.VideoWriter(os.path.join(episode_log_dir, "unfiltered_goal_images.mp4"), cv2.VideoWriter_fourcc(*'DIVX'), 2, size)
            for i in range(len(self.goal_images_info["unfiltered_goal_images_frames"])):
                img = self.goal_images_info["unfiltered_goal_images_frames"][i]
                rgb_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                out.write(rgb_img)
            out.release()

        if "unfiltered_goal_images" in self.goal_images_info:
            for t in range(len(self.goal_images_info["unfiltered_goal_images"])):
                goal_images = self.goal_images_info["unfiltered_goal_images"][t]
                save_dir = os.path.join(episode_log_dir, "unfiltered_goal_images", f"timestep_{t * self.subgoal_max:03}")
                os.makedirs(save_dir, exist_ok=True)
                for i, goal_image in enumerate(goal_images):
                    cv2.imwrite(os.path.join(save_dir, f"goal_image_{i:03}.png"), goal_image[..., ::-1])


                rgb_obs = self.goal_images_info["rgb_obs_at_filtering"][t]
                cv2.imwrite(os.path.join(save_dir, f"rgb_obs.png"), rgb_obs[..., ::-1])

        if "chat_gpt_answer" in self.goal_images_info:
            with open(os.path.join(episode_log_dir, "chat_gpt_answers.txt"), "w") as f:
                for i in range(len(self.goal_images_info["chat_gpt_answer"])):
                    chat_gpt_answer = self.goal_images_info["chat_gpt_answer"][i]
                    f.write(f"[{i}] {chat_gpt_answer}\n")

        if "query_infos" in self.goal_images_info:
            save_dir = os.path.join(episode_log_dir, "chat_gpt_answers")
            os.makedirs(save_dir, exist_ok=True)
            for i in range(len(self.goal_images_info["query_infos"])):
                query_info = self.goal_images_info["query_infos"][i]
                win_counts = self.goal_images_info["win_counts"][i]
                total_query_time = self.goal_images_info["total_query_time"][i]

                with open(os.path.join(save_dir, f"chat_gpt_answers_{i * self.subgoal_max:03}.txt"), "w") as f:
                    f.write(f"WIN COUNTS\n")
                    for i, count in win_counts.items():
                        f.write(f"\tgoal image {i}: {count}\n")

                    f.write(f"\\nTotal query time: {total_query_time:.3f}s.\n\n")

                    f.write("\n\n\n")
                    

                    for (i, j), info in query_info.items():
                        f.write("=" * 30 + f" goal images ({i}, {j}) " + "=" * 30 + "\n")
                        f.write(f"timestep: {i * self.subgoal_max:03}\n")
                        # f.write(f"language goal: \"{language_goal}\".\n")
                        f.write(f"Query time: {info['query1 time']:.3f}s, {info['query2 time']:.3f}s.\n")
                        answer = info["answer"]
                        f.write(f"{answer}\n\n\n")

        if "query_frame" in self.goal_images_info:
            save_video(os.path.join(episode_log_dir, "query_frames.mp4"), np.stack(self.goal_images_info["query_frame"], axis=0), fps=10)



    def reset(self):
        if self.episode_counter is None: # this is the first time reset has been called
            self.episode_counter = 0
            self.obs_image_seq = []
            self.goal_image_seq = []
            self.action_seq = []
            self.combined_images = []
            self.video_buffer = []
            self.action_buffer = []
            self.step_counter = 0

            self.goal_images_info = defaultdict(list)
        else:
            # Update/reset all the variables
            self.episode_counter += 1
            self.obs_image_seq = []
            self.goal_image_seq = []
            self.action_seq = []
            self.video_buffer = []
            self.action_buffer = []
            self.combined_images = []
            self.step_counter = 0

            self.goal_images_info = defaultdict(list)

            # Reset the GC policy
            self.gc_policy.reset()

        # tqdm progress bar
        if self.pbar is not None:
            self.pbar.close()
        self.pbar = tqdm(total=EP_LEN)

    def goal_filtering_fn(self, goal, rgb_obs, goal_images):
        if self.filtering_method == "high_level_video_vf":
            return high_level_video_vf_filter(self.vf_agent, goal, rgb_obs, goal_images)
        else:
            raise NotImplementedError(f"Goal images filtering method \"{self.filtering_method}\" is not implemented.")


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

        mode = "okay"

        # If we need to, generate a new video sequence
        if len(self.video_buffer) == 0:
            t0 = time.time()
            video_generations = self.video_diffusion_model.predict_video_sequence(self.language_task, rgb_obs, n_samples=self.num_samples, sampling_timesteps=self.num_denoising_steps)
            t1 = time.time()
            print(f"[generation] t1 - t0: {t1 - t0:.3f}") 
            if self.num_samples > 1:
                t0 = time.time()
                goal_idx, goal_images_info, mode = self.goal_filtering_fn(goal, rgb_obs, video_generations)
                t1 = time.time()
                print(f"[filtering] t1 - t0: {t1 - t0:.3f}") 
                
                video_generation = video_generations[goal_idx]

                for key, val in goal_images_info.items():
                    self.goal_images_info[key].append(val)
            else:
                assert video_generations.shape[0] == 1, f"video_generations.shape: {video_generations.shape}"
                video_generation = video_generations[0]
            
            for img, next_img in zip(video_generation[:-1], video_generation[1:]):
                # Query the behavior cloning model
                action_cmd = self.gc_policy.predict_action(img, next_img)
                for _ in range(self.subgoal_max):
                    self.action_buffer.append(deepcopy(action_cmd))


            # For a 16-timestep video, we only get 15 actions
            for i, frame in enumerate(video_generation):
                if i != 0:
                    for _ in range(self.subgoal_max):
                        self.video_buffer.append(deepcopy(frame))


        # Extract the next action and image generation
        action_cmd = self.action_buffer[0]
        image_generation = self.video_buffer[0]

        self.action_buffer = self.action_buffer[1:]
        self.video_buffer = self.video_buffer[1:]

        # Log the image observation and the goal image
        self.obs_image_seq.append(rgb_obs)
        self.goal_image_seq.append(image_generation)
        self.combined_images.append(np.concatenate([rgb_obs, image_generation], axis=1))
        assert self.combined_images[-1].shape == (200, 400, 3)

        # Log the predicted action
        self.action_seq.append(action_cmd)

        # Update progress bar
        self.pbar.update(1)

        self.step_counter += 1

        

        return action_cmd, mode
    

def evaluate_policy(model, env, epoch=0, eval_log_dir=None, debug=False, create_plan_tsne=False):
    """
    Run this function to evaluate a model on the CALVIN challenge.

    Args:
        model: Must implement methods of CalvinBaseModel.
        env: (Wrapped) calvin env.
        epoch:
        eval_log_dir: Path where to log evaluation results. If None, logs to /tmp/evaluation/
        debug: If True, show camera view and debug info.
        create_plan_tsne: Collect data for TSNE plots of latent plans (does not work for your custom model)

    Returns:
        Dictionary with results
    """
    conf_dir = Path(__file__).absolute().parents[0] / "calvin_models" / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    eval_log_dir = get_log_dir(eval_log_dir)

    eval_sequences = get_sequences(NUM_SEQUENCES)

    pickle_object(os.path.join(eval_log_dir, "saved_state", "eval_sequences.pkl"), eval_sequences)



    results = []
    plans = defaultdict(list) 

    if not debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    eval_idx = 0
    for initial_state, eval_sequence in eval_sequences:
        result, _ = evaluate_sequence(env, model, task_oracle, initial_state, eval_sequence, val_annotations, plans, debug)
        results.append(result)
        if not debug:
            eval_sequences.set_description(
                " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(count_success(results))]) + "|"
            )
        
        pickle_object(os.path.join(eval_log_dir, "saved_state", f"results_{eval_idx}.pkl"), results)
        pickle_object(os.path.join(eval_log_dir, "saved_state", f"plans_{eval_idx}.pkl"), plans)
        eval_idx += 1

    if create_plan_tsne:
        create_tsne(plans, eval_log_dir, epoch)
    print_and_save(results, eval_sequences, eval_log_dir, epoch)

    return results


def pickle_object(filepath, obj):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'wb') as file:
        pickle.dump(obj, file)



def evaluate_sequence(env, model, task_checker, initial_state, eval_sequence, val_annotations, plans, debug):
    """
    Evaluates a sequence of language instructions.
    """
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    if debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    for subtask in eval_sequence:
        success, end_early = rollout(env, model, task_checker, subtask, val_annotations, plans, debug)
        if success:
            success_counter += 1
        else:
            return success_counter, end_early
    return success_counter, end_early 


def rollout(env, model, task_oracle, subtask, val_annotations, plans, debug):
    """
    Run the actual rollout on one subtask (which is one natural language instruction).
    """
    if debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)

    obs = env.get_obs()

    # get lang annotation for subtask
    lang_annotation = val_annotations[subtask][0]
    model.reset()
    start_info = env.get_info()


    for step in range(EP_LEN):
        action, _ = model.step(obs, lang_annotation, subtask)

        if isinstance(action, int) and action == -1:
            print("Ran out of hard coded goal images")
            break 

        obs, _, _, current_info = env.step(action)
        rgb_obs = obs["rgb_obs"]["rgb_static"]
        if debug:
            img = env.render(mode="rgb_array")
            join_vis_lang(img, lang_annotation)
        if step == 0:
            # for tsne plot, only if available
            collect_plan(model, plans, subtask)

        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            if debug:
                print(colored("success", "green"), end=" ")
            print(colored("success", "green"), end=" ")
            print("step:", step)
            print("current_task_info:", current_task_info)
            model.save_info(True)
            return True, False
    if debug:
        print(colored("fail", "red"), end=" ")
    model.save_info(False)
    print(colored("fail", "red"), end=" ")
    return False, False

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on multistep sequences with language goals.")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset root directory.")

    # arguments for loading default model
    parser.add_argument(
        "--train_folder", type=str, help="If calvin_agent was used to train, specify path to the log dir."
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default=None,
        help="Comma separated list of epochs for which checkpoints will be loaded",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path of the checkpoint",
    )
    parser.add_argument(
        "--last_k_checkpoints",
        type=int,
        help="Specify the number of checkpoints you want to evaluate (starting from last). Only used for calvin_agent.",
    )


    parser.add_argument(
        "--diffusion_model_framework",
        type=str,
        default="jax",
        choices=["jax", "pytorch"],
        help="Comma separated list of epochs for which checkpoints will be loaded",
    )

    parser.add_argument(
        "--diffusion_model_checkpoint_path", type=str, help=""
    )

    parser.add_argument(
        "--gc_policy_checkpoint_path", type=str, help=""
    )


    parser.add_argument(
        "--gc_vf_checkpoint_path", type=str, help=""
    )

    parser.add_argument(
        "--s3_save_uri", type=str, help=""
    )

    parser.add_argument(
        "--save_to_s3", type=int, default=1, help=""
    )


    parser.add_argument(
        "--num_denoising_steps", type=int, default=200, help=""
    )

    parser.add_argument(
        "--use_temporal_ensembling", type=int, default=1, help=""
    )

    parser.add_argument(
        "--num_samples", type=int, default=1, help=""
    )


    parser.add_argument(
        "--filtering_method", type=str, default=None, help=""
    )

    parser.add_argument(
        "--flat_policy", type=int, default=0, help=""
    )


    parser.add_argument(
        "--prompt_w", type=float, default=7.5, help=""
    )

    parser.add_argument(
        "--context_w", type=float, default=1.5, help=""
    )

    parser.add_argument("--debug", type=int, default=0)

    parser.add_argument("--eval_log_dir", default=None, type=str, help="")

    parser.add_argument("--oracle_goals_dir", default=None, type=str, help="")

    parser.add_argument("--agent_config_string", default="calvin", type=str, help="")

    parser.add_argument("--vf_agent_config_string", default=None, type=str, help="")

    parser.add_argument("--oracle_goals_type", default=None, choices=[None, "generated", "retrospective_true", "dataset_true"], type=str, help="Where to log the evaluation results.")

    parser.add_argument("--unipi", type=int, default=0, help="")

    parser.add_argument("--device", default=0, type=int, help="CUDA device")
    return parser.parse_args()



def main():
    args = parse_arguments()

    # evaluate a custom model
    agent_config, env_name, agent_type, _ = get_config(args.agent_config_string)

    if args.filtering_method in ["high_level_vf", "high_level_video_vf"]:
        vf_agent_config, _, vf_agent_type, _ = get_config(args.vf_agent_config_string)
    else:
        vf_agent_config, vf_agent_type = None, None


    if args.unipi:
        model = CustomModelUniPi(agent_config, 
                             env_name, 
                             agent_type, 
                             args.use_temporal_ensembling, 
                             args.gc_policy_checkpoint_path, 
                             args.num_denoising_steps, 
                             num_samples=args.num_samples, 
                             filtering_method=args.filtering_method, 
                             vf_agent_config=vf_agent_config, 
                             vf_agent_type=vf_agent_type)
    else:
        model = CustomModel(agent_config, 
                            env_name, 
                            agent_type, 
                            args.oracle_goals_dir, 
                            args.oracle_goals_type, 
                            args.use_temporal_ensembling, 
                            args.diffusion_model_checkpoint_path, 
                            args.gc_policy_checkpoint_path, 
                            args.gc_vf_checkpoint_path, 
                            args.num_denoising_steps, 
                            num_samples=args.num_samples, 
                            prompt_w=args.prompt_w, 
                            context_w=args.context_w, 
                            filtering_method=args.filtering_method, 
                            flat_policy=args.flat_policy, 
                            diffusion_model_framework=args.diffusion_model_framework, 
                            vf_agent_config=vf_agent_config, 
                            vf_agent_type=vf_agent_type)
    

    
    env = make_env(env_name, args.dataset_path)
    

    if env_name == "calvin" or env_name == "calvinlcbc":
        assert not args.debug

        evaluate_policy(model, env, debug=args.debug, eval_log_dir=model.log_dir)
    else:
        raise ValueError(f"Unsupported env_name: \"{env_name}\".")

    if args.save_to_s3:
        s3_callback = S3SyncCallback(model.log_dir, os.path.join(args.s3_save_uri, "/".join(model.log_dir.split("results")[-1].strip("/").split("/"))) + "/")
        print("s3_log_dir:", os.path.join(args.s3_save_uri, "/".join(model.log_dir.split("results")[-1].strip("/").split("/"))))
        s3_callback.on_train_epoch_end()
    

if __name__ == "__main__":
    main()
