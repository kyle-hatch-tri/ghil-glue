import copy
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import flax
import flax.linen as nn
import optax

from flax.core import FrozenDict
from jaxrl_m.common.typing import Batch
from jaxrl_m.common.typing import PRNGKey
from jaxrl_m.common.common import ModuleDict, JaxRLTrainState, nonpytree_field
from jaxrl_m.common.encoding import LCEncodingWrapper
from jaxrl_m.networks.actor_critic_nets import ContrastiveValue
from jaxrl_m.networks.mlp import MLP


class StableContrastiveRLAgent(flax.struct.PyTreeNode):
    state: JaxRLTrainState
    config: dict = nonpytree_field()
    lr_schedules: dict = nonpytree_field()

    @partial(jax.jit, static_argnames="pmap_axis")
    def update(self, batch: Batch, pmap_axis: str = None):
        def critic_loss_fn(params, rng):
            batch_size = batch["terminals"].shape[0]
            I = jnp.eye(batch_size)


            rng, key = jax.random.split(rng)
            logits = self.state.apply_fn(
                {"params": params},  # gradient flows through here
                (batch["observations"], batch["goals"]),
                train=True,
                rngs={"dropout": key},
                name="critic",
            )

            # the weight of negative term is 1 / (B - 1)
            weights = np.ones((batch_size, batch_size)) / (batch_size - 1) 
            weights[np.arange(batch_size), np.arange(batch_size)] = 1
            if len(logits.shape) == 3:
                # logits.shape = (B, B, 2) with 1 term for positive pair
                # and (B - 1) terms for negative pairs in each row

                critic_loss = jax.vmap(
                    lambda _logits: optax.sigmoid_binary_cross_entropy(
                        logits=_logits, labels=I
                    ),
                    in_axes=-1,
                    out_axes=-1,
                )(logits)
                critic_loss = jnp.mean(critic_loss, axis=-1)

                # Take the mean here so that we can compute the accuracy.
                logits = jnp.mean(logits, axis=-1)
            else:
                critic_loss = optax.sigmoid_binary_cross_entropy(
                    logits=logits, labels=I
                )

            critic_loss = jnp.mean(critic_loss)  # critic loss optimize nothing
            correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
            logits_pos = jnp.sum(logits * I) / jnp.sum(I)
            logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)

            return (
                critic_loss,
                {
                    "critic_loss": critic_loss,
                    "binary_accuracy": jnp.mean((logits > 0) == I),
                    "categorical_accuracy": jnp.mean(correct),
                    "logits_pos": logits_pos,
                    "logits_neg": logits_neg,
                    "logits": logits.mean(),
                },
            )


        loss_fns = {"critic": critic_loss_fn}

        # compute gradients and update params
        new_state, info = self.state.apply_loss_fns(
            loss_fns, pmap_axis=pmap_axis, has_aux=True
        )

        # update the target params
        new_state = new_state.target_update(self.config["target_update_rate"])

        return self.replace(state=new_state), info


    @jax.jit
    def get_debug_metrics(self, batch, **kwargs):
        batch_size = batch["terminals"].shape[0]


        logits = self.state.apply_fn(
            {"params": self.state.params},
            (batch["observations"], batch["goals"]),
            name="critic",
        )
        logits = logits.mean(-1)
        target_logits = self.state.apply_fn(
            {"params": self.state.target_params},
            (batch["observations"], batch["goals"]),
            name="critic",
        )
        target_logits = target_logits.mean(-1)

        I = jnp.eye(batch_size)
        correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = jnp.sum(logits * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)
        target_logits_pos = jnp.sum(target_logits * I) / jnp.sum(I)
        target_logits_neg = jnp.sum(target_logits * (1 - I)) / jnp.sum(1 - I)

        metrics = {
            "logits_pos": logits_pos,
            "logits_neg": logits_neg,
            "target_logits_pos": target_logits_pos,
            "target_logits_neg": target_logits_neg,
            "binary_accuracy": jnp.mean((logits > 0) == I),
            "categorical_accuracy": jnp.mean(correct),
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
        critic_network_kwargs: dict = {"hidden_dims": [256, 256], "dropout_rate": 0.0},
        critic_kwargs: dict = {"init_final": 1e-12, "repr_dim": 16, "twin_q": True},
        policy_network_kwargs: dict = {"hidden_dims": [256, 256], "dropout_rate": 0.0},
        policy_kwargs: dict = {
            "tanh_squash_distribution": False,
            "state_dependent_std": False,
        },
        # Optimizer
        learning_rate: float = 3e-4,
        warmup_steps: int = 2000,
        actor_decay_steps: Optional[int] = None,
        # Algorithm config
        use_td=False,
        gcbc_coef=0.1,
        discount=0.95,
        temperature=1.0,
        target_update_rate=0.002,
        dropout_target_networks=True,
    ):
        if early_goal_concat:
            # passing None as the goal encoder causes early goal concat
            goal_encoder_def = None
        else:
            if shared_goal_encoder:
                goal_encoder_def = encoder_def
            else:
                goal_encoder_def = copy.deepcopy(encoder_def)

        encoder_def = LCEncodingWrapper(
            encoder=encoder_def, use_proprio=use_proprio, stop_gradient=False
        )

        if shared_encoder:
            assert False 
        else:
            # I (kvablack) don't think these deepcopies will break
            # shared_goal_encoder, but I haven't tested it.
            encoders = {
                "critic": encoder_def,
            }


        policy_network_kwargs["activate_final"] = True
        critic_network_kwargs["activate_final"] = True
        networks = {
            "critic": ContrastiveValue(
                encoders["critic"],
                MLP(**critic_network_kwargs),
                MLP(**critic_network_kwargs),
                sa_net2=MLP(**critic_network_kwargs),
                g_net2=MLP(**critic_network_kwargs),
                **critic_kwargs,
            ),
        }

        model_def = ModuleDict(networks)

        rng, init_rng = jax.random.split(rng)
        if len(observations["image"].shape) == 3:
            observations = jax.tree_map(
                lambda x: jnp.expand_dims(x, axis=0), observations
            )
            goals = jax.tree_map(lambda x: jnp.expand_dims(x, axis=0), goals)
        params = model_def.init(
            init_rng,
            critic=[(observations, goals),],
        )["params"]

        # no decay
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=warmup_steps + 1,
            end_value=learning_rate,
        )
        lr_schedules = {"critic": lr_schedule}
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
                use_td=use_td,
                gcbc_coef=gcbc_coef,
                discount=discount,
                temperature=temperature,
                target_update_rate=target_update_rate,
                dropout_target_networks=dropout_target_networks,
            )
        )
        return cls(state, config, lr_schedules)
