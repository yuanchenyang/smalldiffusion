import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchvision import transforms as tf
from torchvision.datasets import FashionMNIST
from torchvision.utils import make_grid, save_image
from torch_ema import ExponentialMovingAverage as EMA
from tqdm import tqdm

from smalldiffusion import ScheduleDDPM, samples, training_loop, MappedDataset, DiT

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
schedule = ScheduleDDPM(beta_start=0.0001, beta_end=0.02, N=1000)
model = DiT(in_dim=28, channels=1,
            patch_size=2, depth=6, head_dim=32, num_heads=6, mlp_ratio=4.0)

# Train
trainer = training_loop(loader, model, schedule, epochs=300, lr=1e-3, accelerator=accelerator)
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
