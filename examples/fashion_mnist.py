import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchvision import transforms as tf
from torchvision.datasets import FashionMNIST
from torchvision.utils import make_grid, save_image
from torch_ema import ExponentialMovingAverage as EMA
from tqdm import tqdm

from smalldiffusion import ScheduleLogLinear, samples, training_loop, MappedDataset
from unet import Unet

# Setup
accelerator = Accelerator()
dataset = MappedDataset(FashionMNIST('datasets', train=True, download=True,
                                     transform=tf.Compose([
                                         tf.RandomHorizontalFlip(),
                                         tf.ToTensor(),
                                         tf.Lambda(lambda t: (t * 2) - 1)
                                     ])),
                        lambda x: x[0])
loader = DataLoader(dataset, batch_size=1024, shuffle=True)
schedule = ScheduleLogLinear(sigma_min=0.02, sigma_max=20, N=800)
model = Unet(dim=28, channels=1, dim_mults=(1,2,4,))

# Train
trainer = training_loop(loader, model, schedule, epochs=300, lr=7e-4, accelerator=accelerator)
ema = EMA(model.parameters(), decay=0.99)
ema.to(accelerator.device)
for ns in trainer:
    ns.pbar.set_description(f'Loss={ns.loss.item():.5}')
    ema.update()

# Sample
with ema.average_parameters():
    *xt, x0 = samples(model, schedule.sample_sigmas(20), gam=1.6, batchsize=64, accelerator=accelerator)
    save_image(((make_grid(x0) + 1)/2).clamp(0, 1), 'fashion_mnist_samples.png')
    torch.save(model.state_dict(), 'checkpoint.pth')
