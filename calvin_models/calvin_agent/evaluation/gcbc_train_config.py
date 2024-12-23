from ml_collections import ConfigDict
from copy import deepcopy

def get_config(config_string):
    dataset, algo, variant = config_string.split("_")
    variant = variant.split("-")

    base_real_config = dict(
        batch_size=256,
        num_val_batches=8,
        num_steps=350_000, 
        log_interval=1000,
        eval_interval=50_000,
        save_interval=50_000,
        save_dir="<path to save dir>",
        data_path="<path_to_data_dir>/goal_conditioned",
        dataset_name=dataset,
        resume_path=None,
        seed=42,
    )

    if "calvin" in dataset:
        dataset_dir = "calvin_data_processed"
    else:
        raise ValueError(f"Unsupported dataset: \"{dataset}\".")

    if dataset == "calvin":
        base_real_config["data_path"] = f"<path_to_data_dir>/{dataset_dir}/goal_conditioned"
    elif dataset == "calvinlcbc":
        base_real_config["data_path"] = f"<path_to_data_dir>/{dataset_dir}/language_conditioned"
    else:
        base_real_config["data_path"] = f"<path_to_data_dir>/{dataset_dir}"


    base_data_config = dict(
        shuffle_buffer_size=25_000,
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
        normalize_actions=True, 
        use_float64=False, 
    )

    if "libero" in dataset:
        base_data_config["use_float64"] = True 

    # params that need to be specified multiple places
    normalization_type = "normal"
    
    dataset_kwargs = dict(
                    goal_relabeling_strategy="delta_goals",
                    goal_relabeling_kwargs=dict(goal_delta=[0, 24]),
                    relabel_actions=False,
                    act_pred_horizon=None,
                    obs_horizon=None,
                    **base_data_config,
                )


    if algo == "gcdiffusion":
        agent_type = "gc_ddpm_bc"

        dataset_kwargs["obs_horizon"] = 1
        dataset_kwargs["act_pred_horizon"] = 4

        agent_kwargs = dict(
            score_network_kwargs=dict(
                time_dim=32,
                num_blocks=3,
                dropout_rate=0.1,
                hidden_dim=256,
                use_layer_norm=True,
            ),
            early_goal_concat=True,
            shared_goal_encoder=True,
            use_proprio=False,
            beta_schedule="cosine",
            diffusion_steps=20,
            action_samples=1,
            repeat_last_step=0,
            learning_rate=3e-4,
            warmup_steps=2000,
            actor_decay_steps=int(2e6),
        )

        if "unipi" in variant:
            base_real_config["num_steps"] = 325_000
            base_real_config["eval_interval"] = 25_000
            base_real_config["save_interval"] = 25_000

            dataset_kwargs["relabel_actions"] = True

            dataset_kwargs["goal_relabeling_kwargs"] = dict(goal_delta=[1, 1])

            if "uniformgoals" in variant:
                dataset_kwargs["goal_relabeling_strategy"] = "uniform" 
                dataset_kwargs["goal_relabeling_kwargs"] = dict(reached_proportion=1.0) 




    elif algo == "lcdiffusion":
        agent_type = "gc_ddpm_bc"

        
        dataset_kwargs["goal_relabeling_kwargs"] = dict(goal_delta=[0, 20]) # This value doesn't matter since it is always the same language instruction
        dataset_kwargs["load_language"] = True 
        dataset_kwargs["skip_unlabeled"] = True 
        dataset_kwargs["obs_horizon"] = 1
        dataset_kwargs["act_pred_horizon"] = 4

        agent_kwargs = dict(
            score_network_kwargs=dict(
                time_dim=32,
                num_blocks=3,
                dropout_rate=0.1,
                hidden_dim=256,
                use_layer_norm=True,
            ),
            language_conditioned=True,
            early_goal_concat=False,
            shared_goal_encoder=False,
            use_proprio=False,
            beta_schedule="cosine",
            diffusion_steps=20,
            action_samples=1,
            repeat_last_step=0,
            learning_rate=3e-4,
            warmup_steps=2000,
            actor_decay_steps=int(2e6),
        )

    elif algo == "lcgcprogressvf":
        agent_type = "lcgc_progress_vf"

        dataset_kwargs["goal_relabeling_kwargs"] = dict(goal_delta=[16, 24])
        dataset_kwargs["load_language"] = True 
        dataset_kwargs["skip_unlabeled"] = True 

        agent_kwargs = dict(
            network_kwargs=dict(
                dropout_rate=0.1,
                hidden_dims=[256, 256],
                use_layer_norm=True,
            ),
            early_goal_concat=False,
            shared_goal_encoder=False,
            use_proprio=False,
            learning_rate=3e-4,
            warmup_steps=2000,

            frac_pos=0.5,
            frac_neg_wrong_lang=0.2,
            frac_neg_reverse_direction=0.2,
            frac_neg_wrong_goalimg=0.1,

            loss_fn="bce",
        )

        if "hingeloss" in variant:
            agent_kwargs["loss_fn"] = "hinge"

        if "noisegoals" in variant:
            agent_kwargs["noise_goals"] = True
            

    elif algo == "contrastivevf":
        agent_type = "stable_contrastive_rl_vf"

        agent_kwargs = dict(
            critic_network_kwargs=dict(
                dropout_rate=0.1,
                hidden_dims=[1024, 1024],
                use_layer_norm=True,
            ),
            early_goal_concat=True,
            shared_goal_encoder=True,
            use_proprio=False,
            learning_rate=3e-4,
            warmup_steps=2000,
        )
    elif algo == "contrastiverl":
        agent_type = "stable_contrastive_rl"

        agent_kwargs = dict(
            critic_network_kwargs=dict(
                dropout_rate=0.1,
                hidden_dims=[1024, 1024],
                use_layer_norm=True,
            ),
            early_goal_concat=True,
            shared_goal_encoder=True,
            use_proprio=False,
            learning_rate=3e-4,
            warmup_steps=2000,
        )
    elif algo == "gcdiscriminator":
        
        agent_type = "gc_discriminator"

        agent_kwargs = dict(
            network_kwargs=dict(
                dropout_rate=0.1,
                hidden_dims=[256, 256],
                use_layer_norm=True,
            ),
            early_goal_concat=True,
            shared_goal_encoder=True,
            use_proprio=False,
            learning_rate=3e-4,
            warmup_steps=2000,
        )

        if "zeroobs" in variant:
            agent_kwargs["zero_out_obs"] = True 

    elif algo == "gcbc":
        agent_type = "gc_bc"

        agent_kwargs = dict(
            network_kwargs=dict(
                dropout_rate=0.1,
                hidden_dims=[256, 256],
                use_layer_norm=True,
            ),
            policy_kwargs=dict(tanh_squash_distribution=False, 
                               state_dependent_std=False,
                               ),
            early_goal_concat=True,
            shared_goal_encoder=True,
            use_proprio=False,
            learning_rate=3e-4,
            warmup_steps=2000,
        )

        if "susie" in variant:
            base_real_config["num_steps"] = 325_000
            base_real_config["eval_interval"] = 25_000
            base_real_config["save_interval"] = 25_000

            agent_kwargs["network_kwargs"] = dict(
                                                hidden_dims=(256, 256, 256),
                                                dropout_rate=0.1,
                                            )
            agent_kwargs["policy_kwargs"] = dict(
                        tanh_squash_distribution=False,
                        fixed_std=[1, 1, 1, 1, 1, 1, 1],
                        state_dependent_std=False,
                    )
            agent_kwargs["decay_steps"]= int(2e6)

        if "unipi" in variant:
            base_real_config["num_steps"] = 325_000
            base_real_config["eval_interval"] = 25_000
            base_real_config["save_interval"] = 25_000

            agent_kwargs["network_kwargs"] = dict(
                                                hidden_dims=(256, 256, 256),
                                                dropout_rate=0.1,
                                            )
            agent_kwargs["policy_kwargs"] = dict(
                        tanh_squash_distribution=False,
                        fixed_std=[1, 1, 1, 1, 1, 1, 1],
                        state_dependent_std=False,
                    )
            agent_kwargs["decay_steps"]= int(2e6)

            dataset_kwargs["relabel_actions"] = True

            dataset_kwargs["goal_relabeling_kwargs"] = dict(goal_delta=[1, 1])

            if "uniformgoals" in variant:
                dataset_kwargs["goal_relabeling_strategy"] = "uniform" 
                dataset_kwargs["goal_relabeling_kwargs"] = dict(reached_proportion=1.0) 
            
            
            

    elif algo == "lcbc":
        agent_type = "lc_bc"

        dataset_kwargs["goal_relabeling_kwargs"] = dict(goal_delta=[0, 20])
        dataset_kwargs["load_language"] = True 
        dataset_kwargs["skip_unlabeled"] = True 
        
        agent_kwargs = dict(
            network_kwargs=dict(
                dropout_rate=0.1,
                hidden_dims=[256, 256],
                use_layer_norm=True,
            ),
            policy_kwargs=dict(tanh_squash_distribution=False, 
                               state_dependent_std=False,
                               ),
            early_goal_concat=False,
            shared_goal_encoder=False,
            use_proprio=False,
            learning_rate=3e-4,
            warmup_steps=2000,
        )
    elif algo == "gciql":
        agent_type = "gc_iql"
        
        dataset_kwargs["goal_relabeling_strategy"] = "geometric"
        dataset_kwargs["goal_relabeling_kwargs"] = dict(reached_proportion=0.2, discount=0.25)

        agent_kwargs = dict(
            network_kwargs=dict(
                dropout_rate=0.1,
                hidden_dims=[256, 256],
                use_layer_norm=True,
            ),
            policy_kwargs=dict(tanh_squash_distribution=False, 
                               state_dependent_std=False,
                               ),
            early_goal_concat=True,
            shared_goal_encoder=True,
            use_proprio=False,
            learning_rate=3e-4,
            warmup_steps=2000,

            actor_decay_steps=int(2e6),
            negative_proportion=0.0,
            shared_encoder=False,
            discount=0.95,
            expectile=0.9,
            temperature=1.0,
            target_update_rate=0.002,
            dropout_target_networks=True,
        )

        if "hparams2" in variant:
            agent_kwargs["discount"] = 0.99
        elif "hparams2" in variant:
            agent_kwargs["discount"] = 0.99
        elif "hparams3" in variant:
            agent_kwargs["target_update_rate"] = 0.005
        elif "hparams4" in variant:
            agent_kwargs["discount"] = 0.99
            agent_kwargs["expectile"] = 0.7
        elif "hparams5" in variant:
            agent_kwargs["discount"] = 0.99
            agent_kwargs["expectile"] = 0.7
            agent_kwargs["temperature"] = 3
    else:
        raise ValueError(f"Unsupported algo: \"{algo}\".")


    encoder_kwargs=dict(
                    pooling_method="avg",
                    add_spatial_coordinates=True,
                    act="swish",
                )
    

    config = dict(
                agent=agent_type,
                agent_kwargs=agent_kwargs,
                dataset_kwargs=dataset_kwargs,
                encoder="resnetv1-34-bridge", 
                encoder_kwargs=encoder_kwargs,
                language_conditioned=False, 
                **base_real_config,
    )

    if "lc" in algo:
        assert algo[:2] == "lc"
        config["language_conditioned"] = True 
        config["encoder"] = "resnetv1-34-bridge-film"
        config["text_processor"] = "muse_embedding"
        config["text_processor_kwargs"] = dict()
        
    if "generatedencdecgoal" in variant:
        config["dataset_kwargs"]["use_generated_goals"] = True 
        config["dataset_kwargs"]["use_encode_decode_goals"] = True 
        config["dataset_kwargs"]["goal_relabeling_strategy"] = "delta_goals_with_generated_encode_decode"

        config["dataset_kwargs"]["goal_relabeling_kwargs"]["goal_delta"] = [16, 24]
        config["dataset_kwargs"]["goal_relabeling_kwargs"]["zero_goal"] = False

        if "frac0.5" in variant:
            config["dataset_kwargs"]["goal_relabeling_kwargs"]["frac_generated"] = 0.5
            config["dataset_kwargs"]["goal_relabeling_kwargs"]["frac_encode_decode"] = 0.16
            config["dataset_kwargs"]["goal_relabeling_kwargs"]["frac_noised_encode_decode"] = 0.16
        elif "frac0.25" in variant:
            config["dataset_kwargs"]["goal_relabeling_kwargs"]["frac_generated"] = 0.25
            config["dataset_kwargs"]["goal_relabeling_kwargs"]["frac_encode_decode"] = 0.25
            config["dataset_kwargs"]["goal_relabeling_kwargs"]["frac_noised_encode_decode"] = 0.25
        else:
            raise ValueError(f"Need to specify a valid frac_generated")
        
        if "zerogoal" in variant:
            config["dataset_kwargs"]["goal_relabeling_kwargs"]["zero_goal"] = True


    if "goaldelta50" in variant:
        config["dataset_kwargs"]["goal_relabeling_kwargs"]["goal_delta"] = [0, 50]


    if "goaldelta20long" in variant:
        config["dataset_kwargs"]["goal_relabeling_kwargs"]["goal_delta"] = [16, 24]

    if "goaldelta5long" in variant:
        config["dataset_kwargs"]["goal_relabeling_kwargs"]["goal_delta"] = [4, 6]


    if "goaldelta20short" in variant:
        config["dataset_kwargs"]["goal_relabeling_kwargs"]["goal_delta"] = [0, 24]


    if "goaldelta5short" in variant:
        config["dataset_kwargs"]["goal_relabeling_kwargs"]["goal_delta"] = [0, 5]


    if "noactnorm" in variant:
        config["dataset_kwargs"]["normalize_actions"] = False

    if "auggoaldiff" in variant:
        config["dataset_kwargs"]["augment_next_obs_goal_differently"] = True 

    for batch_size in [1024, 2048, 4096, 8192]:
        if f"b{batch_size}" in variant:
            config["batch_size"] = batch_size

    config = ConfigDict(config)
    return config, dataset, algo, variant


if __name__ == "__main__":
    config = get_config("calvin_gcbc_noactnorm-unipi-auggoaldiff")
    print(config)
    
