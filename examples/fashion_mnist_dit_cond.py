import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchvision import transforms as tf
from torchvision.datasets import FashionMNIST
from torchvision.utils import make_grid, save_image
from torch_ema import ExponentialMovingAverage as EMA
from tqdm import tqdm

from smalldiffusion import (
    ScheduleDDPM, samples, training_loop, MappedDataset, DiT, CondEmbedderLabel,
    img_train_transform, img_normalize
)

def main(train_batch_size=1024, epochs=300, sample_batch_size=40):
    # Setup
    a = Accelerator()
    dataset = FashionMNIST('datasets', train=True, download=True,
                           transform=img_train_transform)
    loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True)
    schedule = ScheduleDDPM(beta_start=0.0001, beta_end=0.02, N=1000)
    model = DiT(in_dim=28, channels=1,
                patch_size=2, depth=6, head_dim=32, num_heads=6, mlp_ratio=4.0,
                cond_embed=CondEmbedderLabel(32*6, 10, 0.1))

    # Train
    ema = EMA(model.parameters(), decay=0.99)
    ema.to(a.device)
    for ns in training_loop(loader, model, schedule, epochs=epochs, lr=1e-3,
                            conditional=True, accelerator=a):
        ns.pbar.set_description(f'Loss={ns.loss.item():.5}')
        ema.update()

    # Sample
    with ema.average_parameters():
        *xt, x0 = samples(model, schedule.sample_sigmas(20), gam=1.6, batchsize=sample_batch_size,
                          cond=torch.tensor([i%10 for i in range(sample_batch_size)]),
                          accelerator=a)
        save_image(img_normalize(make_grid(x0)), 'samples.png')
        torch.save(model.state_dict(), 'checkpoint.pth')

if __name__=='__main__':
    main()
