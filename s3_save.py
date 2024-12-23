import os
import threading
from functools import partial
import shutil


# from pytorch_lightning.callbacks import Callback


def aws_s3_sync(source, destination):
    """aws s3 sync in quiet mode and time profile"""
    import time, subprocess

    cmd = ["aws", "s3", "sync", "--quiet", source, destination]
    print(f"Syncing files from {source} to {destination}")
    start_time = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.wait()
    end_time = time.time()
    print("Time Taken to Sync: ", (end_time - start_time))
    return


def sync_local_checkpoints_to_s3(
    local_path="/opt/ml/checkpoints",
    s3_uri=os.path.dirname(os.path.dirname(os.getenv("SM_MODULE_DIR", "")))
    + "/checkpoints",
):
    """ sample function to sync checkpoints from local path to s3 """

    import boto3

    # check if local path exists
    if not os.path.exists(local_path):
        raise RuntimeError(
            f"Provided local path {local_path} does not exist. Please check"
        )

    # check if s3 bucket exists
    s3 = boto3.resource("s3")
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Provided s3 uri {s3_uri} is not valid.")

    s3_bucket = s3_uri.replace("s3://", "").split("/")[0]
    print(f"S3 Bucket: {s3_bucket}")
    try:
        s3.meta.client.head_bucket(Bucket=s3_bucket)
    except Exception as e:
        raise e
    aws_s3_sync(local_path, s3_uri)
    return


def sync_s3_checkpoints_to_local(
    local_path="/opt/ml/checkpoints",
    s3_uri=os.path.dirname(os.path.dirname(os.getenv("SM_MODULE_DIR", "")))
    + "/checkpoints",
):
    """ sample function to sync checkpoints from s3 to local path """

    import boto3

    # try to create local path if it does not exist
    if not os.path.exists(local_path):
        print(f"Provided local path {local_path} does not exist. Creating...")
        try:
            os.makedirs(local_path)
        except Exception as e:
            raise RuntimeError(f"Failed to create {local_path}")

    # check if s3 bucket exists
    s3 = boto3.resource("s3")
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Provided s3 uri {s3_uri} is not valid.")

    s3_bucket = s3_uri.replace("s3://", "").split("/")[0]
    print(f"S3 Bucket: {s3_bucket}")
    try:
        s3.meta.client.head_bucket(Bucket=s3_bucket)
    except Exception as e:
        raise e
    aws_s3_sync(s3_uri, local_path)
    return



class S3SyncCallback:
    def __init__(self, local_path, s3_uri):#, sync_interval=5):
        """
        Args:
            local_path (str): Local directory path to sync to S3.
            s3_uri (str): S3 URI where data will be synced.
            sync_interval (int): Number of epochs between sync operations.
        """
        self.local_path = local_path
        self.s3_uri = s3_uri
        # self.sync_lock = threading.Lock()


    def on_train_epoch_end(self):
        self.sync_to_s3_background()

    def sync_to_s3_background(self):
        src_dir = self.local_path
        s3_dst_dir = self.s3_uri
        print(f"Syncing (\"{src_dir}\" to S3...")
        print(f"os.listdir(\"{src_dir}\"):", os.listdir(src_dir))
        sync_local_checkpoints_to_s3(src_dir, s3_dst_dir)
        print(f"Synced results from {src_dir} to {s3_dst_dir}.")
        # print(f"Removing {src_dir}...")
        # shutil.rmtree(src_dir)
        # print(f"Removed {src_dir}.")