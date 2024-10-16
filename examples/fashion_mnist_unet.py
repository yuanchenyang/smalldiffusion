import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchvision import transforms as tf
from torchvision.datasets import FashionMNIST
from torchvision.utils import make_grid, save_image
from torch_ema import ExponentialMovingAverage as EMA
from tqdm import tqdm

from smalldiffusion import (
    ScheduleLogLinear, samples, training_loop, MappedDataset, Unet, Scaled,
    img_train_transform, img_normalize
)

def main(train_batch_size=1024, epochs=300, sample_batch_size=64):
    # Setup
    a = Accelerator()
    dataset = MappedDataset(FashionMNIST('datasets', train=True, download=True,
                                         transform=img_train_transform),
                            lambda x: x[0])
    loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True)
    schedule = ScheduleLogLinear(sigma_min=0.01, sigma_max=20, N=800)
    model = Scaled(Unet)(28, 1, 1, ch=64, ch_mult=(1, 1, 2), attn_resolutions=(14,))

    # Train
    ema = EMA(model.parameters(), decay=0.999)
    ema.to(a.device)
    for ns in training_loop(loader, model, schedule, epochs=epochs, lr=7e-4, accelerator=a):
        ns.pbar.set_description(f'Loss={ns.loss.item():.5}')
        ema.update()

    # Sample
    with ema.average_parameters():
        *xt, x0 = samples(model, schedule.sample_sigmas(20), gam=1.6,
                          batchsize=sample_batch_size, accelerator=a)
        save_image(img_normalize(make_grid(x0)), 'samples.png')
        torch.save(model.state_dict(), 'checkpoint.pth')

if __name__=='__main__':
    main()
