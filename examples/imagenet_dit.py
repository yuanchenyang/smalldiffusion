"""Self-contained DiT-B/2 training on ImageNet latents with flow matching.

Usage: uv run accelerate launch train_imagenet.py
"""

import random
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import load_dataset
from diffusers import AutoencoderKL
from smalldiffusion import (CondEmbedderLabel, DiT, PredFlow, ScheduleFlow,
                           ScheduleLogNormalFlow, TimestepEmbedder, samples, training_loop)
from torch.utils.data import DataLoader, Dataset
from torch_ema import ExponentialMovingAverage as EMA
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

# Latent dataset for ImageNet, with optional random horizontal flip augmentation
class LatentDataset(Dataset):
    def __init__(self, hf_dataset, latent_scale, latent_bias, use_flip=True):
        self.scale, self.bias, self.use_flip = latent_scale, latent_bias, use_flip
        self.ds = load_dataset(hf_dataset, split='train')

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        key = 'latent_mean_flip' if self.use_flip and random.random() < 0.5 else 'latent_mean'
        x = torch.tensor(row[key], dtype=torch.float32) * self.scale + self.bias
        return x, torch.tensor(row['label'], dtype=torch.int64)

def main(train_batch_size=256, epochs=80, steps=400000):
    a = Accelerator()
    set_seed(0, device_specific=True)
    device = a.device

    # VAE (only needed for decoding samples)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
    vae.eval()
    vae.requires_grad_(False)

    # Dataset
    dataset = LatentDataset(
        "yuanchenyang/imagenet-256-sd-vae-ft-mse-latents",
        latent_scale=vae.config.scaling_factor,
        latent_bias=vae.config.shift_factor or 0.0,
    )
    loader = DataLoader(
        dataset, batch_size=train_batch_size // a.num_processes,
        shuffle=True, num_workers=8, pin_memory=True, drop_last=True,
    )

    # Model
    model = PredFlow(DiT)(
        in_dim=32, channels=4, patch_size=2, depth=12,
        head_dim=64, num_heads=12, mlp_ratio=4.0,
        cond_embed=CondEmbedderLabel(12 * 64, num_classes=1000, dropout_prob=0.1),
        sig_embed=TimestepEmbedder(12 * 64),
    )
    if a.is_main_process:
        print(f"DiT parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    ema = EMA(model.parameters(), decay=0.9999)
    ema.to(device)
    step_pbar = tqdm(total=epochs * len(loader) // a.num_processes, disable=not a.is_main_process)
    for step, ns in enumerate(training_loop(loader, model, ScheduleLogNormalFlow(),
                              epochs=epochs, lr=1e-4, conditional=True, accelerator=a)):
        if step >= steps: break            
        ns.pbar.disable = True
        step_pbar.set_description(f'Loss={ns.loss.item():.5}')
        step_pbar.update(1)
        ema.update()

    # Save and sample
    with ema.average_parameters():
        if a.is_main_process:
            torch.save(model.state_dict(), 'checkpoint.pt')

        model.eval()
        for latents in samples(model, ScheduleFlow(t_min=0.04).sample_sigmas(10).to(device),
                               gam=1.6, mu=0.0, cfg_scale=2.0,
                               xt=torch.randn(16, 4, 32, 32, device=device, dtype=torch.float64),
                               cond=torch.arange(16, device=device)):
            pass
        scale, shift = vae.config.scaling_factor, vae.config.shift_factor or 0.0
        images = ((vae.decode((latents.float() - shift) / scale).sample + 1) / 2).clamp(0, 1)
        if a.is_main_process:
            save_image(make_grid(images), 'samples.png')

if __name__ == '__main__':
    main()
