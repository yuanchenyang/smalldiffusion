import torch
from accelerate import Accelerator
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer
from torch import nn

from smalldiffusion import ModelMixin

def alpha_bar(sigma):
    return 1/(sigma**2+1)

class ModelLatentDiffusion(nn.Module, ModelMixin):
    def __init__(self, model_key, accelerator=None):
        super().__init__()
        self.accelerator = accelerator or Accelerator()
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae")
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder")
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet")
        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        self.input_dims = (self.unet.in_channels, self.unet.sample_size, self.unet.sample_size,)
        self.text_condition = None
        self.text_guidance_scale = None
        if is_xformers_available():
            self.unet.enable_xformers_memory_efficient_attention()
        self.to(self.accelerator.device)

    def tokenize(self, prompt):
        return self.tokenizer(
            prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors='pt'
        ).input_ids.to(self.accelerator.device)

    def set_text_condition(self, prompt, negative_prompt='', text_guidance_scale=7.5):
        with torch.no_grad():
            prompt_emb = self.text_encoder(self.tokenize(prompt))[0]
            uncond_emb = self.text_encoder(self.tokenize(negative_prompt))[0]
        self.text_condition = torch.cat([uncond_emb, prompt_emb])
        self.text_guidance_scale = text_guidance_scale

    @torch.no_grad()
    def decode_latents(self, latents):
        return self.vae.decode(latents / 0.18215).sample

    def sigma_to_t(self, sigma):
        idx = torch.searchsorted(reversed(self.scheduler.alphas_cumprod.to(sigma)), alpha_bar(sigma))
        return self.scheduler.num_train_timesteps - 1 - idx

    def forward(self, x, sigma, cond=None):
        z = alpha_bar(sigma).sqrt() * x
        z2 = torch.cat([z, z])
        eps = self.unet(z2, self.sigma_to_t(sigma), encoder_hidden_states=self.text_condition).sample
        eps_uncond, eps_prompt = eps.chunk(2)
        return eps_prompt + self.text_guidance_scale * (eps_prompt - eps_uncond)
