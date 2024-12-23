import os
import json
import torch
import sys

from einops import rearrange

from omegaconf import OmegaConf

from pvdm.exps.diffusion import diffusion
from pvdm.utils import set_random_seed
import tensorflow as tf

from pvdm.evals.eval import test_psnr, test_ifvd, test_fvd_ddpm
from transformers import T5Tokenizer, T5EncoderModel
from pvdm.models.autoencoder.autoencoder_vit import ViTAutoencoder
from pvdm.models.ddpm.unet import UNetModel, DiffusionWrapper
from pvdm.utils import file_name, Logger, download
from pvdm.tools.dataloader import get_loaders
import numpy as np
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
import imageio
from collections import OrderedDict

from pvdm.losses.ddpm import DDPM

from lvdm.models.samplers.ddim import DDIMSampler
from lvdm.models.samplers.ddim_multiplecond import DDIMSampler as DDIMSampler_multicond

from einops import rearrange, repeat
import torchvision.transforms as transforms

from copy import deepcopy

from utils.utils import instantiate_from_config

class VideoDiffusionModel:
    def __init__(self, config_path, ckpt_path):

        self.gpu_no = 1

        self.ckpt_path = ckpt_path
        self.ddim_eta = 1.0
        self.bs = 1
        self.height = 256
        self.width = 256
        self.frame_stride = 1 # 1 # 48 # 24 # 3
        self.unconditional_guidance_scale = 7.5
        self.seed = 123 
        self.video_length = 16 
        self.negative_prompt = False 
        self.text_input = True 
        self.multiple_cond_cfg = False 
        self.cfg_img = None 
        self.timestep_spacing = "uniform"
        self.guidance_rescale = 0.0 
        self.perframe_ae = False 
        self.loop = False 
        self.interp = False

        video_size = (self.height, self.width)
        self.transform = transforms.Compose([
            transforms.Resize(min(video_size)),
            transforms.CenterCrop(video_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])



        ## model config
        config = OmegaConf.load(config_path)
        model_config = config.pop("model", OmegaConf.create())
        
        ## set use_checkpoint as False as when using deepspeed, it encounters an error "deepspeed backend not set"
        model_config['params']['unet_config']['params']['use_checkpoint'] = False
        self.model = instantiate_from_config(model_config)
        self.model = self.model.cuda(self.gpu_no)
        self.model.perframe_ae = self.perframe_ae
        assert os.path.exists(self.ckpt_path), "Error: checkpoint Not Found!"
        self.model = load_model_checkpoint(self.model, self.ckpt_path)
        self.model.eval() 



        ## run over data
        assert (self.height % 16 == 0) and (self.width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
        assert self.bs == 1, "Current implementation only support [batch size = 1]!"
        ## latent noise shape
        h, w = self.height // 8, self.width // 8
        channels = self.model.model.diffusion_model.out_channels
        n_frames = self.video_length
        print(f'Inference with {n_frames} frames')
        self.noise_shape = [self.bs, channels, n_frames, h, w]

    def predict_video_sequence(self, language_command : str, image_obs : np.ndarray, n_samples=1, sampling_timesteps=50):
        image = Image.fromarray(image_obs).resize((self.width, self.height))


        image_tensor = self.transform(image).unsqueeze(1) # [c,1,h,w]
        videos = repeat(image_tensor, 'c t h w -> c (repeat t) h w', repeat=self.video_length).unsqueeze(0).to(f"cuda:{self.gpu_no}")
        noise_shape = deepcopy(self.noise_shape)

        if n_samples > 1:
            videos = torch.cat([videos for _ in range(n_samples)], dim=0)
            noise_shape[0] = n_samples
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            pred = image_guided_synthesis(self.model, [language_command] * n_samples, videos, noise_shape, 1, sampling_timesteps, self.ddim_eta, self.unconditional_guidance_scale, self.cfg_img, self.frame_stride, self.text_input, self.multiple_cond_cfg, self.loop, self.interp, self.timestep_spacing, self.guidance_rescale)
        

        assert pred.shape[1] == 1, f"pred.shape: {pred.shape}"
        overall_video_frames = []

        print(f"language_command: {language_command}, pred.shape: {pred.shape}")

        # b,c,t,h,w
        pred = pred.detach().cpu()
        pred = torch.clamp(pred.float(), -1., 1.)
        n = pred.shape[0]
        for i in range(n):
            grid = pred[i, 0,...]
            grid = (grid + 1.0) / 2.0
            grid = (grid * 255).to(torch.uint8).permute(1, 2, 3, 0) #thwc

            grid = grid.detach().cpu().numpy()
            grid = np.array([np.array(Image.fromarray(frame).resize((200, 200))) for frame in grid])

            overall_video_frames.append(grid)

        return np.array(overall_video_frames).astype(np.uint8)


def image_guided_synthesis(model, prompts, videos, noise_shape, n_samples=1, ddim_steps=50, ddim_eta=1., \
                        unconditional_guidance_scale=1.0, cfg_img=None, fs=None, text_input=False, multiple_cond_cfg=False, loop=False, interp=False, timestep_spacing='uniform', guidance_rescale=0.0, **kwargs):
    ddim_sampler = DDIMSampler(model) if not multiple_cond_cfg else DDIMSampler_multicond(model)
    batch_size = noise_shape[0]
    fs = torch.tensor([fs] * batch_size, dtype=torch.long, device=model.device)

    if not text_input:
        prompts = [""]*batch_size

    img = videos[:,:,0] #bchw
    img_emb = model.embedder(img) ## blc
    img_emb = model.image_proj_model(img_emb)
    cond_emb = model.get_learned_conditioning(prompts)
    cond = {"c_crossattn": [torch.cat([cond_emb,img_emb], dim=1)]}
    if model.model.conditioning_key == 'hybrid':
        z = get_latent_z(model, videos) # b c t h w
        if loop or interp:
            img_cat_cond = torch.zeros_like(z)
            img_cat_cond[:,:,0,:,:] = z[:,:,0,:,:]
            img_cat_cond[:,:,-1,:,:] = z[:,:,-1,:,:]
        else:
            img_cat_cond = z[:,:,:1,:,:]
            img_cat_cond = repeat(img_cat_cond, 'b c t h w -> b c (repeat t) h w', repeat=z.shape[2])
        cond["c_concat"] = [img_cat_cond] # b c 1 h w
    
    if unconditional_guidance_scale != 1.0:
        if model.uncond_type == "empty_seq":
            prompts = batch_size * [""]
            uc_emb = model.get_learned_conditioning(prompts)
        elif model.uncond_type == "zero_embed":
            uc_emb = torch.zeros_like(cond_emb)
        uc_img_emb = model.embedder(torch.zeros_like(img)) ## b l c
        uc_img_emb = model.image_proj_model(uc_img_emb)
        uc = {"c_crossattn": [torch.cat([uc_emb,uc_img_emb],dim=1)]}
        if model.model.conditioning_key == 'hybrid':
            uc["c_concat"] = [img_cat_cond]
    else:
        uc = None

    ## we need one more unconditioning image=yes, text=""
    if multiple_cond_cfg and cfg_img != 1.0:
        uc_2 = {"c_crossattn": [torch.cat([uc_emb,img_emb],dim=1)]}
        if model.model.conditioning_key == 'hybrid':
            uc_2["c_concat"] = [img_cat_cond]
        kwargs.update({"unconditional_conditioning_img_nonetext": uc_2})
    else:
        kwargs.update({"unconditional_conditioning_img_nonetext": None})

    z0 = None
    cond_mask = None

    batch_variants = []
    for _ in range(n_samples):

        if z0 is not None:
            cond_z0 = z0.clone()
            kwargs.update({"clean_cond": True})
        else:
            cond_z0 = None
        if ddim_sampler is not None:

            samples, _ = ddim_sampler.sample(S=ddim_steps,
                                            conditioning=cond,
                                            batch_size=batch_size,
                                            shape=noise_shape[1:],
                                            verbose=False,
                                            unconditional_guidance_scale=unconditional_guidance_scale,
                                            unconditional_conditioning=uc,
                                            eta=ddim_eta,
                                            cfg_img=cfg_img, 
                                            mask=cond_mask,
                                            x0=cond_z0,
                                            fs=fs,
                                            timestep_spacing=timestep_spacing,
                                            guidance_rescale=guidance_rescale,
                                            **kwargs
                                            )

        ## reconstruct from latent to pixel space
        batch_images = model.decode_first_stage(samples)
        batch_variants.append(batch_images)
    ## variants, batch, c, t, h, w
    batch_variants = torch.stack(batch_variants)
    return batch_variants.permute(1, 0, 2, 3, 4, 5)


def get_latent_z(model, videos):
    b, c, t, h, w = videos.shape
    x = rearrange(videos, 'b c t h w -> (b t) c h w')
    z = model.encode_first_stage(x)
    z = rearrange(z, '(b t) c h w -> b c t h w', b=b, t=t)
    return z

def save_video(output_video_file, frames):
    # import cv2
     # Extract frame dimensions
    height, width, _ = frames.shape[1:]

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also use other codecs such as 'XVID'
    fps = 30  # Adjust the frame rate as needed

    os.makedirs(os.path.dirname(output_video_file), exist_ok=True)
    video_writer = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height))

    # Write each frame to the video file
    for frame in frames:
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(bgr_frame)

    # Release the video writer object
    video_writer.release()


def load_model_checkpoint(model, ckpt):
    state_dict = torch.load(ckpt, map_location="cpu")
    if "state_dict" in list(state_dict.keys()):
        state_dict = state_dict["state_dict"]
        try:
            model.load_state_dict(state_dict, strict=True)
        except:
            ## rename the keys for 256x256 model
            new_pl_sd = OrderedDict()
            for k,v in state_dict.items():
                new_pl_sd[k] = v

            for k in list(new_pl_sd.keys()):
                if "framestride_embed" in k:
                    new_key = k.replace("framestride_embed", "fps_embedding")
                    new_pl_sd[new_key] = new_pl_sd[k]
                    del new_pl_sd[k]
            model.load_state_dict(new_pl_sd, strict=True)
    else:
        # deepspeed
        new_pl_sd = OrderedDict()
        for key in state_dict['module'].keys():
            new_pl_sd[key[16:]]=state_dict['module'][key]
        model.load_state_dict(new_pl_sd)
    print('>>> model checkpoint loaded.')
    return model

if __name__ == "__main__":
    import cv2
    video_diffusion_model = VideoDiffusionModel()


    image_obs = cv2.imread("example_rgb_obs.png")
    image_obs = image_obs[..., ::-1]

    language_command = "store the grasped block in the sliding cabinet"

    video_prediction = video_diffusion_model.predict_video_sequence(language_command, image_obs, n_samples=8, sampling_timesteps=100)


    vid = []
    for i in range(video_prediction.shape[0]):
        video = video_prediction[i]
        vid.append(video)

    assert len(video_prediction) % 4 == 0, f"len(video_prediction): {len(video_prediction)}"
    video_rows = []
    for row_idx in range(len(video_prediction) // 4):
        start = row_idx * 4
        end = start + 4
        video_row = np.concatenate([np.tile(image_obs, [16, 1, 1, 1])] + vid[start:end], axis=2)
        video_rows.append(video_row)

    query_video = np.concatenate(video_rows, axis=1)


    print(video_prediction.shape) # 16 frames
    print(query_video.shape) # 16 frames

    save_video("./query_video.mp4", query_video)