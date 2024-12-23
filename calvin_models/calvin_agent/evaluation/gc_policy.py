import json
from jaxrl_m.vision import encoders
from jaxrl_m.data.calvin_dataset import CalvinDataset
import jax
from jaxrl_m.agents import agents
import numpy as np
import os
import orbax.checkpoint


class GCPolicy:
    def __init__(self, resume_path, three_layers=False, use_temporal_ensembling=False):
        self.use_temporal_ensembling = use_temporal_ensembling
        if self.use_temporal_ensembling:
            # Prepare action buffer for temporal ensembling
            self.action_buffer = np.zeros((4, 4, 7))
            self.action_buffer_mask = np.zeros((4, 4), dtype=bool)

        # We need to first create a dataset object to supply to the agent
        train_paths = [[
            "mini_dataset/0.tfrecord",
            "mini_dataset/1.tfrecord"
        ]]

        print("three_layers:", three_layers)

        dataset_kwargs = dict(
            shuffle_buffer_size=25000,
            prefetch_num_batches=20,
            augment=True,
            augment_next_obs_goal_differently=False,
            augment_kwargs=dict(
                random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.1],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            ),
            goal_relabeling_strategy="uniform",
            goal_relabeling_kwargs=dict(reached_proportion=0.0),
            relabel_actions=True,

            normalize_actions=False, 
        )

        ACT_MEAN = [
            2.9842544e-04,
            -2.6099570e-04,
            -1.5863389e-04,
            5.8916201e-05,
            -4.4560504e-05,
            8.2349771e-04,
            9.4075650e-02,
        ]

        ACT_STD = [
            0.27278143,
            0.23548537,
            0.2196189,
            0.15881406,
            0.17537235,
            0.27875036,
            1.0049515,
        ]

        action_metadata = {
            "mean": ACT_MEAN,
            "std": ACT_STD,
        }

        train_data = CalvinDataset(
            train_paths,
            42,
            batch_size=256,
            num_devices=1,
            train=True,
            action_metadata=action_metadata,
            sample_weights=None,
            obs_horizon=None,
            **dataset_kwargs,
        )
        train_data_iter = train_data.iterator()
        example_batch = next(train_data_iter)

        # Next let's initialize the agent
        agent_kwargs = dict(
            network_kwargs=dict(
                hidden_dims=(256, 256),
                dropout_rate=0.1,
            ),
            policy_kwargs=dict(
                tanh_squash_distribution=False,
                fixed_std=[1, 1, 1, 1, 1, 1, 1],
                state_dependent_std=False,
            ),
            early_goal_concat=True,
            shared_goal_encoder=True,
            use_proprio=False,
            learning_rate=3e-4,
            warmup_steps=2000,
            decay_steps=int(2e6),
        )

        if three_layers:
            agent_kwargs["network_kwargs"]["hidden_dims"] = (256, 256, 256)

        encoder_def = encoders["resnetv1-34-bridge"](**{"act" : "swish", "add_spatial_coordinates" : "true", "pooling_method" : "avg"})
        rng = jax.random.PRNGKey(42)
        rng, construct_rng = jax.random.split(rng)
        agent = agents["gc_bc"].create(
            rng=construct_rng,
            observations=example_batch["observations"],
            goals=example_batch["goals"],
            actions=example_batch["actions"],
            encoder_def=encoder_def,
            **agent_kwargs,
        )

        print("Loading checkpoint...") 
        print(f"gcbc_policy resume path: \"{resume_path}\"")
        restored = orbax.checkpoint.PyTreeCheckpointer().restore(resume_path, item=agent)
        if agent is restored:
            raise FileNotFoundError(f"Cannot load checkpoint from {resume_path}")
        print("Checkpoint successfully loaded")
        agent = restored

        # save the loaded agent
        self.agent = agent
        self.action_statistics = action_metadata

    def predict_action(self, image_obs : np.ndarray, goal_image : np.ndarray):
        action = self.agent.sample_actions(
                            {"image" : image_obs}, 
                            {"image" : goal_image}, 
                            seed=jax.random.PRNGKey(42), 
                            temperature=0.0, 
                            argmax=True
                        )
        action = np.array(action.tolist())

        if self.use_temporal_ensembling:
            # Since the t=[1,2,3] action predictions are never used,
            # can use the same buffer machinery for the one step policies 
            # by just tiling the action along a new time 4 times 
            action = action[None, :] * np.ones((4, 7))

            # Shift action buffer
            self.action_buffer[1:, :, :] = self.action_buffer[:-1, :, :]
            self.action_buffer_mask[1:, :] = self.action_buffer_mask[:-1, :]
            self.action_buffer[:, :-1, :] = self.action_buffer[:, 1:, :]
            self.action_buffer_mask[:, :-1] = self.action_buffer_mask[:, 1:]
            self.action_buffer_mask = self.action_buffer_mask * np.array([[True, True, True, True],
                                                                        [True, True, True, False],
                                                                        [True, True, False, False],
                                                                        [True, False, False, False]], dtype=bool)
            
            # Add to action buffer
            self.action_buffer[0] = action
            self.action_buffer_mask[0] = np.array([True, True, True, True], dtype=bool)

            # Ensemble temporally to predict action
            action_prediction = np.sum(self.action_buffer[:, 0, :] * self.action_buffer_mask[:, 0:1], axis=0) / np.sum(self.action_buffer_mask[:, 0], axis=0)

            action = action_prediction



        # Make gripper action either -1 or 1
        if action[-1] < 0:
            action[-1] = -1
        else:
            action[-1] = 1

        return action
    
    def reset(self):
        pass 
