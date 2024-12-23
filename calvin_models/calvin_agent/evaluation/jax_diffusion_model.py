import os
import time
import numpy as np
from PIL import Image

from susie.model import create_sample_fn, create_vae_encode_decode_fn
from susie.jax_utils import initialize_compilation_cache

class DiffusionModel:
    def __init__(self, num_denoising_steps=200, num_samples=1):
        initialize_compilation_cache()

        self.num_samples = num_samples

        self.sample_fn = create_sample_fn(
            os.getenv("DIFFUSION_MODEL_CHECKPOINT"),
            "kvablack/dlimp-diffusion/9n9ped8m",
            num_timesteps=num_denoising_steps,
            prompt_w=7.5,
            context_w=1.5,
            eta=0.0,
            pretrained_path="runwayml/stable-diffusion-v1-5:flax",
            num_samples=self.num_samples,
        )

        self.vae_encode_decode_fn = create_vae_encode_decode_fn(
            os.getenv("DIFFUSION_MODEL_CHECKPOINT"),
            "kvablack/dlimp-diffusion/9n9ped8m",
            eta=0.0,
            pretrained_path="runwayml/stable-diffusion-v1-5:flax",
            num_samples=self.num_samples,
        )




    def generate(self, language_command : str, image_obs : np.ndarray, return_inference_time=False, prompt_w=7.5, context_w=1.5):
        # Resize image to 256x256
        
        image_obs = np.array(Image.fromarray(image_obs).resize((256, 256))).astype(np.uint8)

        t0 = time.time()
        sample = self.sample_fn(image_obs, language_command, prompt_w=prompt_w, context_w=context_w)
        t1 = time.time()

        samples = np.array([np.array(Image.fromarray(s).resize((200, 200))).astype(np.uint8) for s in sample])

        if return_inference_time:
            return samples, t1 - t0
        else:
            return samples
    
        

    def vae_encode_decode(self, image_obs : np.ndarray, noise_scale=0, return_inference_time=False):
        # Resize image to 256x256
        
        image_obs = np.array(Image.fromarray(image_obs).resize((256, 256))).astype(np.uint8)
        t0 = time.time()
        sample = self.vae_encode_decode_fn(image_obs, noise_scale=noise_scale)
        t1 = time.time()
        samples = np.array([np.array(Image.fromarray(s).resize((200, 200))).astype(np.uint8) for s in sample])

        if return_inference_time:
            return samples, t1 - t0
        else:
            return samples
        

    

if __name__ == "__main__":
    model = DiffusionModel()