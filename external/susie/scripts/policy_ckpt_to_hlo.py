import inspect

import jax
import numpy as np
import orbax.checkpoint
import tensorflow as tf
from absl import app, flags
from jaxrl_m.agents import agents
from jaxrl_m.vision import encoders
import json
import yaml

import wandb
from susie.jax_utils import serialize_jax_fn

FLAGS = flags.FLAGS

flags.DEFINE_string("checkpoint_path", None, "Path to checkpoint file", required=True)

flags.DEFINE_string(
    "wandb_run_name", None, "Name of wandb run to get config from.", required=False # required=True
)

flags.DEFINE_string(
    "outpath", None, "Path to save serialized policy to.", required=True
)

flags.DEFINE_integer(
    "im_size", 256, "Image size, which was unfortunately not saved in config"
)

flags.DEFINE_string("config_path", None, "Path to save video")

def load_dict_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    config_dict = {}
    for key, val in data.items():
        config_dict[key] = val["value"]

    return config_dict


def load_dict_from_yaml(file_path):
    """
    Load a dictionary from a YAML file.

    :param file_path: Path to the YAML file
    :return: Dictionary with the contents of the YAML file
    """
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def load_policy_checkpoint(path, wandb_run_name):
    assert tf.io.gfile.exists(path)

    if wandb_run_name is None:
        bridge_data_config = load_dict_from_json("scripts/robot/gcdiffusion_auggoaldiff.json")
        config = load_dict_from_yaml(FLAGS.config_path)
        # config = load_dict_from_json(FLAGS.config_path)
    else:
        # load information from wandb
        api = wandb.Api()
        run = api.run(wandb_run_name)
        config = run.config

    # create encoder from wandb config
    encoder_def = encoders[config["encoder"]](**config["encoder_kwargs"])

    act_pred_horizon = config["dataset_kwargs"].get("act_pred_horizon")
    obs_horizon = config.get("obs_horizon") or config["dataset_kwargs"].get("obs_horizon")

    if act_pred_horizon is not None:
        example_actions = np.zeros((1, act_pred_horizon, 7), dtype=np.float32)
    else:
        example_actions = np.zeros((1, 7), dtype=np.float32)

    if obs_horizon is not None:
        example_obs = {
            "image": np.zeros(
                (1, obs_horizon, FLAGS.im_size, FLAGS.im_size, 3), dtype=np.uint8
            )
        }
    else:
        example_obs = {
            "image": np.zeros((1, FLAGS.im_size, FLAGS.im_size, 3), dtype=np.uint8)
        }

    example_goal = {
        "image": np.zeros((1, FLAGS.im_size, FLAGS.im_size, 3), dtype=np.uint8)
    }

    example_batch = {
        "observations": example_obs,
        "actions": example_actions,
        "goals": example_goal,
    }

    # create agent from wandb config
    agent = agents[config["agent"]].create(
        rng=jax.random.PRNGKey(0),
        encoder_def=encoder_def,
        observations=example_batch["observations"],
        goals=example_batch["goals"],
        actions=example_batch["actions"],
        **config["agent_kwargs"],
    )


    # load action metadata from wandb
    action_proprio_metadata = bridge_data_config["bridgedata_config"]["action_proprio_metadata"]
    action_mean = np.array(action_proprio_metadata["action"]["mean"])
    action_std = np.array(action_proprio_metadata["action"]["std"])

    # hydrate agent with parameters from checkpoint
    agent = orbax.checkpoint.PyTreeCheckpointer().restore(
        path,
        item=agent,
    )

    def get_action(rng, obs_image, goal_image):
        obs = {"image": obs_image}
        goal_obs = {"image": goal_image}
        # some agents (e.g. DDPM) don't have argmax
        if inspect.signature(agent.sample_actions).parameters.get("argmax"):
            action = agent.sample_actions(obs, goal_obs, seed=rng, argmax=True)
        else:
            action = agent.sample_actions(obs, goal_obs, seed=rng)
        action = action * action_std + action_mean
        return action

    serialized = serialize_jax_fn(
        get_action,
        jax.random.PRNGKey(0),
        example_obs["image"][0],
        example_goal["image"][0],
    )

    return serialized


def main(_):
    serialized = load_policy_checkpoint(FLAGS.checkpoint_path, FLAGS.wandb_run_name)

    with open(FLAGS.outpath, "wb") as f:
        f.write(serialized)


if __name__ == "__main__":
    app.run(main)

"""
export PYTHONPATH="/<path_to>/calvin-sim/external/susie:$PYTHONPATH"
export PYTHONPATH="/<path_to>/calvin-sim/external/jaxrl_m:$PYTHONPATH"
export PYTHONPATH="/<path_to>/calvin-sim/external/urdfpy:$PYTHONPATH"
export PYTHONPATH="/<path_to>/calvin-sim/external/networkx:$PYTHONPATH"

python3 -u scripts/policy_ckpt_to_hlo.py \
--checkpoint_path /<path_to_checkpoints>/susie_low_level/bridge/gcdiffusion/auggoaldiff/seed_0/20240602_011058/checkpoint_1000 \
--config_path /<path_to_checkpoints>/susie_low_level/bridge/gcdiffusion/auggoaldiff/seed_0/20240602_011058/config.yaml \
--outpath /<path_to_checkpoints>/susie_low_level/bridge/gcdiffusion/auggoaldiff/seed_0/20240602_011058/checkpoint_1000/serialized_policy_ckpt \
--im_size 200
"""