import argparse
import os
from datetime import datetime

import boto3
import sagemaker
# from sagemaker.pytorch import PyTorch
from sagemaker.tensorflow import TensorFlow
from sagemaker.inputs import FileSystemInput
import yaml

import boto3
import time
import subprocess
import awswrangler as wr
from smart_open import smart_open

def load_yaml(filepath):
    with open(filepath) as f:
        return yaml.safe_load(f)

def get_job_name(base):
    now = datetime.now()
    now_ms_str = f'{now.microsecond // 1000:03d}'
    date_str = f"{now.strftime('%Y-%m-%d-%H-%M-%S')}-{now_ms_str}"
    job_name = '_'.join([base, date_str])
    return job_name


def launch(args):

    launch_config = load_yaml(args.launch_config)
    

    if launch_config["launch_args"]["wandb_api_key"] is None:
        wandb_api_key = os.environ.get('WANDB_API_KEY', None)
        assert wandb_api_key is not None, 'Please provide wandb api key either via --wandb-api-key or env variable WANDB_API_KEY.'
        launch_config["launch_args"]["wandb_api_key"] = wandb_api_key

    if launch_config["launch_args"]["local"]:
        assert launch_config["launch_args"]["instance_count"] == 1, f'Local mode requires 1 instance, now get {args.instance_count}.'
        assert launch_config["launch_args"]["input_source"] not in {'lustre'}
        launch_config["launch_args"]["sagemaker_session"] = sagemaker.LocalSession()
    else:
        assert launch_config["launch_args"]["input_source"] not in {'local'}    
        launch_config["launch_args"]["sagemaker_session"] = sagemaker.Session()



    if launch_config["launch_args"]["input_source"] == 'local':
        from glob import glob 
        input_mode = 'File'
        training_inputs = {}
        training_inputs["high_level"] = 'file:///' + launch_config["high_level_policy_local_path"].strip("/") + f"/params_ema"
        training_inputs["high_level_vf"] = 'file:///' + launch_config["high_level_vf_local_path"].strip("/") + "/"

        for policy_name, policy_dict in launch_config["low_level_policy"].items():
            for seed in policy_dict["seeds"]:
                for checkpoint in policy_dict["checkpoints"]:
                    matching_local_paths = glob(os.path.join(policy_dict["local_path"], f"seed_{seed}", "*", f"checkpoint_{checkpoint}"))
                    for local_path in matching_local_paths:
                        unique_id = local_path.strip("/").split("/")[-2]
                        training_inputs[f"{policy_name}-{seed}-{unique_id}-{checkpoint}"] = 'file:///' + local_path.strip("/") + "/"
    elif launch_config["launch_args"]["input_source"] == 's3':
        input_mode = 'FastFile'
        bucket_name = "kyle-sagemaker-training-outputs"
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(bucket_name)

        training_inputs = {}
        training_inputs["high_level"] = launch_config["high_level_policy_s3_uri"].strip("/") + f"/params_ema"
        training_inputs["high_level_vf"] = launch_config["high_level_vf_s3_uri"].strip("/") + "/"

        for policy_name, policy_dict in launch_config["low_level_policy"].items():
            prefix = launch_config["low_level_policy"][policy_name]["s3_uri"].split(bucket_name)[-1].strip("/")
            for seed in policy_dict["seeds"]:
                for obj in bucket.objects.filter(Prefix=f"{prefix}/seed_{seed}"):
                    for checkpoint in policy_dict["checkpoints"]:
                        if f"checkpoint_{checkpoint}/" in obj.key:
                            print("obj.key:", obj.key)
                            unique_id = obj.key.strip("/").split("/")[-3]
                            training_inputs[f"{policy_name}-{seed}-{unique_id}-{checkpoint}"] = os.path.join(f"s3://{bucket_name}", *obj.key.strip("/").split("/")[:-1]) + "/"
    else:
        raise ValueError(f'Invalid input source {args.input_source}')

    role = '<sagemaker_role>'
    role_name = role.split(['/'][-1])

    session = boto3.session.Session()
    region = session.region_name

    hyperparameters = {}


    subnets = [
        'subnet-<subnet_id1>',
        'subnet-<subnet_id2>',
    ]

    security_group_ids = [
        'sg-<security_group_id1>', 
        'sg-<security_group_id2>', 
        'sg-<security_group_id3>',
    ]



    job_name = get_job_name(launch_config["launch_args"]["base_job_name"])

    if launch_config["launch_args"]["local"]:
        image_uri = f'{launch_config["launch_args"]["base_job_name"]}:latest' 
    else:
        image_uri = f'<docker_image_uri>/{launch_config["launch_args"]["base_job_name"]}:latest'
    
    output_path = os.path.join(f's3://<s3_save_uri>/{launch_config["launch_args"]["user"]}/bridge_data_v2/', job_name)

    checkpoint_s3_uri = None if launch_config["launch_args"]["local"] else output_path
    checkpoint_local_path = None if launch_config["launch_args"]["local"] else '/opt/ml/checkpoints'
    code_location = output_path

    base_job_name = launch_config["launch_args"]["base_job_name"].replace("_", "-")
    instance_count = launch_config["launch_args"]["instance_count"]
    sagemaker_session = launch_config["launch_args"]["sagemaker_session"]

    instance_type = 'local_gpu' if launch_config["launch_args"]["local"] else launch_config["launch_args"]["instance_type"] 
    keep_alive_period_in_seconds = 0
    max_run = 60 * 60 * 24 * 5

    

    environment = {
        'WANDB_API_KEY': launch_config["launch_args"]["wandb_api_key"],
        'WANDB_ENTITY': "tri",
        "AGENT_CONFIG_STRING":launch_config["eval_args"]["agent_config_string"],
        "VF_AGENT_CONFIG_STRING":launch_config["eval_args"]["vf_agent_config_string"],
        'NUM_EVAL_SEQUENCES':str(launch_config["eval_args"]["num_eval_sequences"]),
        "EP_LEN":str(launch_config["eval_args"]["ep_len"]),
        "SUBGOAL_MAX":str(launch_config["eval_args"]["subgoal_max"]),

        'DIFFUSION_MODEL_CHECKPOINT_PATH':training_inputs["high_level"],
        'HIGH_LEVEL_VF_CHECKPOINT_PATH':training_inputs["high_level_vf"],
        "SAVE_TO_S3":str(launch_config["launch_args"]["save_to_s3"]),
        "S3_SAVE_URI":str(launch_config["launch_args"]["s3_save_uri"]),

        "DEBUG":str(launch_config["launch_args"]["debug"]),
        "NUM_DENOISING_STEPS":str(launch_config["eval_args"]["num_denoising_steps"]),
        "NUM_SAMPLES":str(launch_config["eval_args"]["num_samples"]),
        "FLAT_POLICY":str(launch_config["eval_args"]["flat_policy"]),
        "FILTERING_METHOD":str(launch_config["eval_args"]["filtering_method"]),
        "UNIPI":str(launch_config["eval_args"]["unipi"]),
        "USE_TEMPORAL_ENSEMBLING":str(launch_config["eval_args"]["use_temporal_ensembling"]),
    }

    checkpoint_idx = 0
    for key, val in training_inputs.items():
        if "high_level" in key:
            continue 

        environment[f"GC_POLICY_CHECKPOINT_PATH_{checkpoint_idx}"] = f"{key}|{val}"
        checkpoint_idx += 1

    environment["NUM_LOW_LEVEL_CHECKPOINTS"] = str(checkpoint_idx)


    print("environment:", environment)

    

    distribution = {
        'smdistributed': {
            'dataparallel': {
                    'enabled': False,
            },
        },
    }

    print()
    print()
    print('#############################################################')
    print(f'SageMaker Execution Role:       {role}')
    print(f'The name of the Execution role: {role_name[-1]}')
    print(f'AWS region:                     {region}')
    # print(f'Entry point:                    {entry_point}')
    print(f'Image uri:                      {image_uri}')
    print(f'Job name:                       {job_name}')
    # print(f'Configuration file:             {config}')
    print(f'Instance count:                 {instance_count}')
    print(f'Input mode:                     {input_mode}')
    print('#############################################################')
    print()
    print()

    sagemaker_job_tags = [
        {
            "Key": "XXXX",
            "Value": "XXXX"
        },
        {
            "Key": "XXXX",
            "Value": "XXXX"
        }
    ]

    estimator = TensorFlow(
        base_job_name=base_job_name,
        entry_point="train",
        hyperparameters=hyperparameters,
        role=role,
        image_uri=image_uri,
        instance_count=instance_count,
        instance_type=instance_type,
        environment=environment,
        sagemaker_session=sagemaker_session,
        subnets=subnets,
        security_group_ids=security_group_ids,
        keep_alive_period_in_seconds=keep_alive_period_in_seconds,
        max_run=max_run,
        input_mode=input_mode,
        job_name=job_name,
        checkpoint_s3_uri=checkpoint_s3_uri,
        checkpoint_local_path=checkpoint_local_path,
        code_location=code_location,
        distribution=distribution,
        tags=sagemaker_job_tags,
    )

    estimator.fit(inputs=training_inputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch_config', type=str)
    args = parser.parse_args()

    launch(args)
