import copy
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import flax
import flax.linen as nn
import optax

from jaxrl_m.agents.continuous.iql import iql_value_loss
from jaxrl_m.agents.continuous.iql import iql_actor_loss
from jaxrl_m.agents.continuous.iql import iql_critic_loss
from jaxrl_m.agents.continuous.iql import expectile_loss
from flax.core import FrozenDict
from jaxrl_m.common.typing import Batch
from jaxrl_m.common.typing import PRNGKey
from jaxrl_m.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from jaxrl_m.common.encoding import GCEncodingWrapper
from jaxrl_m.networks.actor_critic_nets import ValueCritic
from jaxrl_m.networks.mlp import MLP


class GCDiscriminatorAgent(flax.struct.PyTreeNode):
    state: JaxRLTrainState
    config: dict = nonpytree_field()
    lr_schedules: dict = nonpytree_field()

    @partial(jax.jit, static_argnames="pmap_axis")
    def update(self, batch: Batch, pmap_axis: str = None):
        batch_size = batch["terminals"].shape[0]

        def value_loss_fn(params, rng):
            rng, key = jax.random.split(rng)
            batch_size = batch["observations"]["image"].shape[0]
            
            if self.config["zero_out_obs"]:
                observations = {key:jnp.zeros_like(val) for key, val in batch["observations"].items()}
            else:
                observations = batch["observations"]

            rng, key = jax.random.split(rng)
            v = self.state.apply_fn(
                {"params": params},  # gradient flows through here
                (observations, batch["goals"]),
                train=True,
                rngs={"dropout": key},
                name="value",
            )

            labels = jnp.logical_not(batch["uses_generated_goal"])

            bce_loss = optax.sigmoid_binary_cross_entropy(logits=v, labels=labels)
            overall_loss = bce_loss.mean()

            generated_goal_mask = batch["uses_generated_goal"].astype(bool)
            encode_decode_mask = batch["uses_encode_decode_goal"].astype(bool)
            noised_encode_decode_mask = batch["uses_noised_encode_decode_goal"].astype(bool)
            real_goal_mask = jnp.logical_not(batch["uses_generated_goal"] + batch["uses_encode_decode_goal"] + batch["uses_noised_encode_decode_goal"])

            def masked_mean(values, mask):
                return (values * mask).sum() / mask.sum()

            real_loss = masked_mean(bce_loss, real_goal_mask)
            generated_loss = masked_mean(bce_loss, generated_goal_mask)
            encode_decode_loss = masked_mean(bce_loss, encode_decode_mask)
            noised_encode_decode_loss = masked_mean(bce_loss, noised_encode_decode_mask)

            overall_logits = v.mean()
            real_logits = masked_mean(v, real_goal_mask)
            generated_logits = masked_mean(v, generated_goal_mask)
            encode_decode_logits = masked_mean(v, encode_decode_mask)
            noised_encode_decode_logits = masked_mean(v, noised_encode_decode_mask)


            overall_pred = jax.nn.sigmoid(v).mean()
            real_pred = masked_mean(jax.nn.sigmoid(v), real_goal_mask)
            generated_pred = masked_mean(jax.nn.sigmoid(v), generated_goal_mask)
            encode_decode_pred = masked_mean(jax.nn.sigmoid(v), encode_decode_mask)
            noised_encode_decode_pred = masked_mean(jax.nn.sigmoid(v), noised_encode_decode_mask)
            

            metrics = {
                "overall_loss": overall_loss,
                "real_loss": real_loss,
                "generated_loss": generated_loss,
                "encode_decode_loss": encode_decode_loss,
                "noised_encode_decode_loss": noised_encode_decode_loss,

                "overall_logits": overall_logits,
                "real_logits": real_logits,
                "generated_logits": generated_logits,
                "encode_decode_logits": encode_decode_logits,
                "noised_encode_decode_logits": noised_encode_decode_logits,

                "overall_pred": overall_pred,
                "real_pred": real_pred,
                "generated_pred": generated_pred,
                "encode_decode_pred": encode_decode_pred,
                "noised_encode_decode_pred": noised_encode_decode_pred,
                
            }


            return overall_loss, metrics


        loss_fns = {
            "value": value_loss_fn,
        }

        # compute gradients and update params
        new_state, info = self.state.apply_loss_fns(
            loss_fns, pmap_axis=pmap_axis, has_aux=True
        )

        # update the target params
        info["value_lr"] = self.lr_schedules["value"](self.state.step)

        return self.replace(state=new_state), info
    
    @jax.jit
    def value_function(self, observations, goals):
        v = self.state.apply_fn(
            {"params": self.state.params},
            (observations, goals),
            name="value",
        )

        return v



    @jax.jit
    def get_debug_metrics(self, batch, gripper_close_val=None, **kwargs):
        if self.config["zero_out_obs"]:
            observations = {key:jnp.zeros_like(val) for key, val in batch["observations"].items()}
        else:
            observations = batch["observations"]

        v = self.state.apply_fn(
            {"params": self.state.params},
            (observations, batch["goals"]),
            name="value",
        )
        

        labels = jnp.logical_not(batch["uses_generated_goal"])

        bce_loss = optax.sigmoid_binary_cross_entropy(logits=v, labels=labels)

        overall_loss = bce_loss.mean()

        generated_goal_mask = batch["uses_generated_goal"].astype(bool)
        encode_decode_mask = batch["uses_encode_decode_goal"].astype(bool)
        noised_encode_decode_mask = batch["uses_noised_encode_decode_goal"].astype(bool)
        real_goal_mask = jnp.logical_not(batch["uses_generated_goal"] + batch["uses_encode_decode_goal"] + batch["uses_noised_encode_decode_goal"])

        def masked_mean(values, mask):
            return (values * mask).sum() / mask.sum()



        real_loss = masked_mean(bce_loss, real_goal_mask)
        generated_loss = masked_mean(bce_loss, generated_goal_mask)
        encode_decode_loss = masked_mean(bce_loss, encode_decode_mask)
        noised_encode_decode_loss = masked_mean(bce_loss, noised_encode_decode_mask)

        overall_logits = v.mean()
        real_logits = masked_mean(v, real_goal_mask)
        generated_logits = masked_mean(v, generated_goal_mask)
        encode_decode_logits = masked_mean(v, encode_decode_mask)
        noised_encode_decode_logits = masked_mean(v, noised_encode_decode_mask)

        overall_pred = jax.nn.sigmoid(v).mean()
        real_pred = masked_mean(jax.nn.sigmoid(v), real_goal_mask)
        generated_pred = masked_mean(jax.nn.sigmoid(v), generated_goal_mask)
        encode_decode_pred = masked_mean(jax.nn.sigmoid(v), encode_decode_mask)
        noised_encode_decode_pred = masked_mean(jax.nn.sigmoid(v), noised_encode_decode_mask)
        

        metrics = {
            "overall_loss": overall_loss,
            "real_loss": real_loss,
            "generated_loss": generated_loss,
            "encode_decode_loss": encode_decode_loss,
            "noised_encode_decode_loss": noised_encode_decode_loss,

            "overall_logits": overall_logits,
            "real_logits": real_logits,
            "generated_logits": generated_logits,
            "encode_decode_logits": encode_decode_logits,
            "noised_encode_decode_logits": noised_encode_decode_logits,

            "overall_pred": overall_pred,
            "real_pred": real_pred,
            "generated_pred": generated_pred,
            "encode_decode_pred": encode_decode_pred,
            "noised_encode_decode_pred": noised_encode_decode_pred,
            
        }

        return metrics

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: FrozenDict,
        goals: FrozenDict,
        actions: jnp.ndarray,
        # Model architecture
        encoder_def: nn.Module,
        shared_encoder: bool = True,
        shared_goal_encoder: bool = True,
        early_goal_concat: bool = False,
        use_proprio: bool = False,
        network_kwargs: dict = {"hidden_dims": [256, 256], "dropout": 0.0},
        # Optimizer
        learning_rate: float = 3e-4,
        warmup_steps: int = 2000,
        # Algorithm config
        zero_out_obs: bool = False,
    ):
        if early_goal_concat:
            # passing None as the goal encoder causes early goal concat
            goal_encoder_def = None
        else:
            if shared_goal_encoder:
                goal_encoder_def = encoder_def
            else:
                goal_encoder_def = copy.deepcopy(encoder_def)


        encoder_def = GCEncodingWrapper(
            encoder=encoder_def,
            goal_encoder=goal_encoder_def,
            use_proprio=use_proprio,
            stop_gradient=False,
        )

        if shared_encoder:
            encoders = {
                "value": encoder_def,
            }
        else:
            # I (kvablack) don't think these deepcopies will break
            # shared_goal_encoder, but I haven't tested it.
            encoders = {
                "value": encoder_def,
            }

        network_kwargs["activate_final"] = False 
        
        networks = {
            "value": ValueCritic(encoders["value"], MLP(**network_kwargs)),
        }

        model_def = ModuleDict(networks)

        rng, init_rng = jax.random.split(rng)
        params = model_def.init(
            init_rng,
            value=[(observations, goals)],
        )["params"]

        # no decay
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=warmup_steps + 1,
            end_value=learning_rate,
        )
        lr_schedules = {
            "value": lr_schedule,
        }
        
        txs = {k: optax.adam(v) for k, v in lr_schedules.items()}

        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=txs,
            target_params=params,
            rng=create_rng,
        )

        config = flax.core.FrozenDict(
            dict(
                zero_out_obs=zero_out_obs,
            )
        )
        return cls(state, config, lr_schedules)
