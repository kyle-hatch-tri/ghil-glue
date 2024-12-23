from functools import partial
from typing import Any

import copy 
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict

from jaxrl_m.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from jaxrl_m.common.encoding import LCGCEncodingWrapper
from jaxrl_m.common.typing import Batch, PRNGKey
from jaxrl_m.networks.actor_critic_nets import ValueCritic
from jaxrl_m.networks.mlp import MLP

class LCGCProgressVFAgent(flax.struct.PyTreeNode):
    debug_metrics_rng: PRNGKey
    state: JaxRLTrainState
    lr_schedule: Any = nonpytree_field()
    config: dict = nonpytree_field()
    
    @partial(jax.jit, static_argnames="pmap_axis")
    def update(self, batch: Batch, pmap_axis: str = None):

        def loss_fn(params, rng):
            batch_size = batch["observations"]["image"].shape[0]

            pos_idx_end = int(batch_size * self.config["frac_pos"])

            neg_wrong_lang_goal_idx_start = pos_idx_end
            neg_wrong_lang_goal_idx_end = neg_wrong_lang_goal_idx_start + int(batch_size * self.config["frac_neg_wrong_lang"])

            neg_reverse_direction_idx_start = neg_wrong_lang_goal_idx_end
            neg_reverse_direction_idx_end = neg_reverse_direction_idx_start + int(batch_size * self.config["frac_neg_reverse_direction"])

            neg_wrong_goalimg_idx_start = neg_reverse_direction_idx_end

            pos_obs_images = batch["observations"]["image"][:pos_idx_end]
            pos_goal_images = batch["goals"]["image"][:pos_idx_end]
            pos_goal_language = batch["goals"]["language"][:pos_idx_end]

            neg_wrong_lang_obs_images = batch["observations"]["image"][neg_wrong_lang_goal_idx_start:neg_wrong_lang_goal_idx_end]
            neg_wrong_lang_goal_images = batch["goals"]["image"][neg_wrong_lang_goal_idx_start:neg_wrong_lang_goal_idx_end]
            neg_wrong_lang_goal_language = batch["goals"]["language"][neg_wrong_lang_goal_idx_start:neg_wrong_lang_goal_idx_end]

            neg_reverse_direction_goal_language = batch["goals"]["language"][neg_reverse_direction_idx_start:neg_reverse_direction_idx_end]

            neg_wrong_goalimg_obs_images = batch["observations"]["image"][neg_wrong_goalimg_idx_start:]
            neg_wrong_goalimg_goal_images = batch["goals"]["image"][neg_wrong_goalimg_idx_start:]
            neg_wrong_goalimg_goal_language = batch["goals"]["language"][neg_wrong_goalimg_idx_start:]

            neg_wrong_lang_goal_language_idxs = jnp.arange(neg_wrong_lang_goal_language.shape[0])

            rng, key = jax.random.split(rng)
            neg_wrong_lang_goal_language_idxs = jax.random.permutation(key, neg_wrong_lang_goal_language_idxs, axis=0)
            neg_wrong_lang_goal_language = neg_wrong_lang_goal_language[neg_wrong_lang_goal_language_idxs]

            # create negative examples where the current observation image and the goal image are swapped (backwards progress). (s_t, s_{t+k}, l) --> (s_{t+k}, s_t, l)
            neg_reverse_direction_obs_images = batch["goals"]["image"][neg_reverse_direction_idx_start:neg_reverse_direction_idx_end] # these are deliberately swapped
            neg_reverse_direction_goal_images = batch["observations"]["image"][neg_reverse_direction_idx_start:neg_reverse_direction_idx_end] # these are deliberately swapped



            # create negative examples where the goal imgs are randomly shuffled, so that the goal image does not match the current obs and language instruction (s_t, s_{t+k}, l) --> (s_t, s_{t+k}', l)
            neg_wrong_goalimg_goal_images_idxs = jnp.arange(neg_wrong_goalimg_goal_images.shape[0])
            rng, key = jax.random.split(rng)
            neg_wrong_goalimg_goal_images_idxs = jax.random.permutation(key, neg_wrong_goalimg_goal_images_idxs, axis=0)
            neg_wrong_goalimg_goal_images = neg_wrong_goalimg_goal_images[neg_wrong_goalimg_goal_images_idxs]

            obs_images = jnp.concatenate([pos_obs_images, neg_wrong_lang_obs_images, neg_reverse_direction_obs_images, neg_wrong_goalimg_obs_images], axis=0)
            goal_images = jnp.concatenate([pos_goal_images, neg_wrong_lang_goal_images, neg_reverse_direction_goal_images, neg_wrong_goalimg_goal_images], axis=0)
            goal_language = jnp.concatenate([pos_goal_language, neg_wrong_lang_goal_language, neg_reverse_direction_goal_language, neg_wrong_goalimg_goal_language], axis=0)

            new_batch = {"observations":{"image":obs_images},
                         "goals":{"image":goal_images,
                                  "language":goal_language}}
            
            batch_size_pos = pos_obs_images.shape[0]
            batch_size_neg = batch_size - batch_size_pos
            labels_pos = jnp.ones((batch_size_pos,), dtype=int)
            if self.config["loss_fn"] == "bce":
                labels_neg = jnp.zeros((batch_size_neg,), dtype=int)
            elif self.config["loss_fn"] == "hinge":
                labels_neg = jnp.ones((batch_size_neg,), dtype=int) * -1
            else:
                raise ValueError(f'Unsupported loss_fn: {self.config["loss_fn"]}')
            
            labels = jnp.concatenate([labels_pos, labels_neg], axis=0)
            
            rng, key = jax.random.split(rng)
            logits = self.state.apply_fn(
                {"params": params},
                (new_batch["observations"], new_batch["goals"]),
                train=True,
                rngs={"dropout": key},
                name="value",
            )


            pos_idx_end = int(batch_size * self.config["frac_pos"])

            neg_wrong_lang_goal_idx_start = pos_idx_end
            neg_wrong_lang_goal_idx_end = neg_wrong_lang_goal_idx_start + int(batch_size * self.config["frac_neg_wrong_lang"])

            neg_reverse_direction_idx_start = neg_wrong_lang_goal_idx_end
            neg_reverse_direction_idx_end = neg_reverse_direction_idx_start + int(batch_size * self.config["frac_neg_reverse_direction"])

            neg_wrong_goalimg_idx_start = neg_reverse_direction_idx_end

            if self.config["loss_fn"] == "bce":
                loss = optax.sigmoid_binary_cross_entropy(logits, labels)
            elif self.config["loss_fn"] == "hinge":
                loss = optax.hinge_loss(logits, labels)
            else:
                raise ValueError(f'Unsupported loss_fn: {self.config["loss_fn"]}')

            loss_pos = loss[:pos_idx_end].mean()
            loss_neg_wrong_lang = loss[neg_wrong_lang_goal_idx_start:neg_wrong_lang_goal_idx_end].mean()
            loss_neg_reverse_direction = loss[neg_reverse_direction_idx_start:neg_reverse_direction_idx_end].mean()
            loss_neg_wrong_goalimg = loss[neg_wrong_goalimg_idx_start:].mean()

            logits_pos = logits[:pos_idx_end].mean()
            logits_neg_wrong_lang = logits[neg_wrong_lang_goal_idx_start:neg_wrong_lang_goal_idx_end].mean()
            logits_neg_reverse_direction = logits[neg_reverse_direction_idx_start:neg_reverse_direction_idx_end].mean()
            logits_neg_wrong_goalimg = logits[neg_wrong_goalimg_idx_start:].mean()

            return (
                loss.mean(),
                {
                    "loss": loss.mean(),
                    "loss_pos": loss_pos,
                    "loss_neg_wrong_lang": loss_neg_wrong_lang,
                    "loss_neg_reverse_direction": loss_neg_reverse_direction,
                    "loss_neg_wrong_goalimg": loss_neg_wrong_goalimg,

                    "logits": logits.mean(),
                    "logits_pos": logits_pos,
                    "logits_neg_wrong_lang": logits_neg_wrong_lang,
                    "logits_neg_reverse_direction": logits_neg_reverse_direction,
                    "logits_neg_wrong_goalimg": logits_neg_wrong_goalimg,

                },
            )

        # compute gradients and update params
        new_state, info = self.state.apply_loss_fns(
            loss_fn, pmap_axis=pmap_axis, has_aux=True
        )

        # log learning rates
        info["lr"] = self.lr_schedule(self.state.step)

        return self.replace(state=new_state), info

    @jax.jit
    def get_debug_metrics(self, batch, **kwargs):
        batch_size = batch["observations"]["image"].shape[0]

        pos_idx_end = int(batch_size * self.config["frac_pos"])

        neg_wrong_lang_goal_idx_start = pos_idx_end
        neg_wrong_lang_goal_idx_end = neg_wrong_lang_goal_idx_start + int(batch_size * self.config["frac_neg_wrong_lang"])

        neg_reverse_direction_idx_start = neg_wrong_lang_goal_idx_end
        neg_reverse_direction_idx_end = neg_reverse_direction_idx_start + int(batch_size * self.config["frac_neg_reverse_direction"])

        neg_wrong_goalimg_idx_start = neg_reverse_direction_idx_end

        pos_obs_images = batch["observations"]["image"][:pos_idx_end]
        pos_goal_images = batch["goals"]["image"][:pos_idx_end]
        pos_goal_language = batch["goals"]["language"][:pos_idx_end]

        neg_wrong_lang_obs_images = batch["observations"]["image"][neg_wrong_lang_goal_idx_start:neg_wrong_lang_goal_idx_end]
        neg_wrong_lang_goal_images = batch["goals"]["image"][neg_wrong_lang_goal_idx_start:neg_wrong_lang_goal_idx_end]
        neg_wrong_lang_goal_language = batch["goals"]["language"][neg_wrong_lang_goal_idx_start:neg_wrong_lang_goal_idx_end]

        neg_reverse_direction_goal_language = batch["goals"]["language"][neg_reverse_direction_idx_start:neg_reverse_direction_idx_end]

        neg_wrong_goalimg_obs_images = batch["observations"]["image"][neg_wrong_goalimg_idx_start:]
        neg_wrong_goalimg_goal_images = batch["goals"]["image"][neg_wrong_goalimg_idx_start:]
        neg_wrong_goalimg_goal_language = batch["goals"]["language"][neg_wrong_goalimg_idx_start:]

        # create negative examples where the language goals are randomly shuffled, so that the obs and goal do not match the language instruction (s_t, s_{t+k}, l) --> (s_t, s_{t+k}, l')
        neg_wrong_lang_goal_language_idxs = jnp.arange(neg_wrong_lang_goal_language.shape[0])
        rng, key = jax.random.split(self.debug_metrics_rng)
        neg_wrong_lang_goal_language_idxs = jax.random.permutation(key, neg_wrong_lang_goal_language_idxs, axis=0)
        neg_wrong_lang_goal_language = neg_wrong_lang_goal_language[neg_wrong_lang_goal_language_idxs]

        # create negative examples where the current observation image and the goal image are swapped (backwards progress). (s_t, s_{t+k}, l) --> (s_{t+k}, s_t, l)
        neg_reverse_direction_obs_images = batch["goals"]["image"][neg_reverse_direction_idx_start:neg_reverse_direction_idx_end] # these are deliberately swapped
        neg_reverse_direction_goal_images = batch["observations"]["image"][neg_reverse_direction_idx_start:neg_reverse_direction_idx_end] # these are deliberately swapped



        # create negative examples where the goal imgs are randomly shuffled, so that the goal image does not match the current obs and language instruction (s_t, s_{t+k}, l) --> (s_t, s_{t+k}', l)
        neg_wrong_goalimg_goal_images_idxs = jnp.arange(neg_wrong_goalimg_goal_images.shape[0])
        rng, key = jax.random.split(rng)
        neg_wrong_goalimg_goal_images_idxs = jax.random.permutation(key, neg_wrong_goalimg_goal_images_idxs, axis=0)
        neg_wrong_goalimg_goal_images = neg_wrong_goalimg_goal_images[neg_wrong_goalimg_goal_images_idxs]

        obs_images = jnp.concatenate([pos_obs_images, neg_wrong_lang_obs_images, neg_reverse_direction_obs_images, neg_wrong_goalimg_obs_images], axis=0)
        goal_images = jnp.concatenate([pos_goal_images, neg_wrong_lang_goal_images, neg_reverse_direction_goal_images, neg_wrong_goalimg_goal_images], axis=0)
        goal_language = jnp.concatenate([pos_goal_language, neg_wrong_lang_goal_language, neg_reverse_direction_goal_language, neg_wrong_goalimg_goal_language], axis=0)

        new_batch = {"observations":{"image":obs_images},
                        "goals":{"image":goal_images,
                                "language":goal_language}}
        
        batch_size_pos = pos_obs_images.shape[0]
        batch_size_neg = batch_size - batch_size_pos
        labels_pos = jnp.ones((batch_size_pos,), dtype=int)

        if self.config["loss_fn"] == "bce":
            labels_neg = jnp.zeros((batch_size_neg,), dtype=int)
        elif self.config["loss_fn"] == "hinge":
            labels_neg = jnp.ones((batch_size_neg,), dtype=int) * -1
        else:
            raise ValueError(f'Unsupported loss_fn: {self.config["loss_fn"]}')

        labels = jnp.concatenate([labels_pos, labels_neg], axis=0)

        rng, key = jax.random.split(rng)
        logits = self.state.apply_fn(
            {"params": self.state.params},
            (new_batch["observations"], new_batch["goals"]),
            name="value",
        )


        pos_idx_end = int(batch_size * self.config["frac_pos"])

        neg_wrong_lang_goal_idx_start = pos_idx_end
        neg_wrong_lang_goal_idx_end = neg_wrong_lang_goal_idx_start + int(batch_size * self.config["frac_neg_wrong_lang"])

        neg_reverse_direction_idx_start = neg_wrong_lang_goal_idx_end
        neg_reverse_direction_idx_end = neg_reverse_direction_idx_start + int(batch_size * self.config["frac_neg_reverse_direction"])

        neg_wrong_goalimg_idx_start = neg_reverse_direction_idx_end

        if self.config["loss_fn"] == "bce":
            loss = optax.sigmoid_binary_cross_entropy(logits, labels)
        elif self.config["loss_fn"] == "hinge":
            loss = optax.hinge_loss(logits, labels)
        else:
            raise ValueError(f'Unsupported loss_fn: {self.config["loss_fn"]}')

        loss_pos = loss[:pos_idx_end].mean()
        loss_neg_wrong_lang = loss[neg_wrong_lang_goal_idx_start:neg_wrong_lang_goal_idx_end].mean()
        loss_neg_reverse_direction = loss[neg_reverse_direction_idx_start:neg_reverse_direction_idx_end].mean()
        loss_neg_wrong_goalimg = loss[neg_wrong_goalimg_idx_start:].mean()

        logits_pos = logits[:pos_idx_end].mean()
        logits_neg_wrong_lang = logits[neg_wrong_lang_goal_idx_start:neg_wrong_lang_goal_idx_end].mean()
        logits_neg_reverse_direction = logits[neg_reverse_direction_idx_start:neg_reverse_direction_idx_end].mean()
        logits_neg_wrong_goalimg = logits[neg_wrong_goalimg_idx_start:].mean()
        

        return {   
                "loss": loss.mean(),
                "loss_pos": loss_pos,
                "loss_neg_wrong_lang": loss_neg_wrong_lang,
                "loss_neg_reverse_direction": loss_neg_reverse_direction,
                "loss_neg_wrong_goalimg": loss_neg_wrong_goalimg,

                "logits": logits.mean(),
                "logits_pos": logits_pos,
                "logits_neg_wrong_lang": logits_neg_wrong_lang,
                "logits_neg_reverse_direction": logits_neg_reverse_direction,
                "logits_neg_wrong_goalimg": logits_neg_wrong_goalimg,
            }
    

    @jax.jit
    def value_function(self, observations, goals):
        logits = self.state.apply_fn(
            {"params": self.state.params},
            (observations, goals),
            name="value",
        )

        return logits

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: FrozenDict,
        actions: jnp.ndarray,
        goals: FrozenDict,
        # Model architecture
        encoder_def: nn.Module,
        shared_goal_encoder: bool = True,
        early_goal_concat: bool = False,
        use_proprio: bool = False,
        network_kwargs: dict = {"hidden_dims": [256, 256]},

        # Optimizer
        learning_rate: float = 3e-4,
        warmup_steps: int = 1000,
        decay_steps: int = 1000000,
        frac_pos: float = 0.5,
        frac_neg_wrong_lang: float = 0.2,
        frac_neg_reverse_direction: float = 0.2,
        frac_neg_wrong_goalimg: float = 0.1, # not directly used
        noise_goals: bool = False, 

        loss_fn: str = "bce",
    ):
        if early_goal_concat:
            goal_encoder_def = None
        else:
            if shared_goal_encoder:
                goal_encoder_def = encoder_def
            else:
                goal_encoder_def = copy.deepcopy(encoder_def)

        encoder_def = LCGCEncodingWrapper(
            encoder=encoder_def,
            goal_encoder=goal_encoder_def,
            use_proprio=use_proprio,
            stop_gradient=False,
        )

        network_kwargs["activate_final"] = False 
        networks = {
            "value": ValueCritic(encoder_def, MLP(**network_kwargs)),
        }

        model_def = ModuleDict(networks)

        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=0.0,
        )
        tx = optax.adam(lr_schedule)

        rng, init_rng = jax.random.split(rng)
        params = model_def.init(init_rng, value=[(observations, goals)])["params"]

        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=tx,
            target_params=params,
            rng=create_rng,
        )


        rng, debug_metrics_rng = jax.random.split(rng)

        assert abs(frac_pos + frac_neg_wrong_lang + frac_neg_reverse_direction + frac_neg_wrong_goalimg - 1.0) < 1e-5, f"frac_pos: {frac_pos}, frac_neg_wrong_lang: {frac_neg_wrong_lang}, frac_neg_reverse_direction: {frac_neg_reverse_direction}, frac_neg_wrong_goalimg: {frac_neg_wrong_goalimg} | frac_pos + frac_neg_wrong_lang + frac_neg_reverse_direction + frac_neg_wrong_goalimg = {frac_pos + frac_neg_wrong_lang + frac_neg_reverse_direction + frac_neg_wrong_goalimg}"
        assert loss_fn in ["bce", "hinge"], f"Unsupported loss_fn: {loss_fn}"

        config = flax.core.FrozenDict(
            dict(
                frac_pos=frac_pos,
                frac_neg_wrong_lang=frac_neg_wrong_lang,
                frac_neg_reverse_direction=frac_neg_reverse_direction,
                frac_neg_wrong_goalimg=frac_neg_wrong_goalimg,
                noise_goals=noise_goals,

                loss_fn=loss_fn,
            )
        )

        return cls(debug_metrics_rng, state, lr_schedule, config)

