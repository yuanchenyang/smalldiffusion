from torchvision import transforms
from tqdm import tqdm
from accelerate import Accelerator

from smalldiffusion import ScheduleLogLinear, samples, training_loop, get_hf_dataloader
from models.unet import Unet

from torch_ema import ExponentialMovingAverage as EMA

# Setup
accelerator = Accelerator()
loader = get_hf_dataloader(
    dataset_name='fashion_mnist',
    batch_size=1024,
    train_transforms=[transforms.RandomHorizontalFlip()]
)
schedule = ScheduleLogLinear(sigma_min=0.02, sigma_max=20, N=800)
model = Unet(dim=28, channels=1, dim_mults=(1,2,4,), sigma_to_t_scale=3, sigma_min=schedule[0])

# Train
trainer = training_loop(loader, model, schedule, epochs=250, lr=7e-4, accelerator=accelerator)
ema = EMA(model.parameters(), decay=0.99)
ema.to(accelerator.device)
for ns in trainer:
    ema.update()

# Sample
with ema.average_parameters():
    *xt, x0 = samples(model, schedule.sample_sigmas(steps), batchsize=100, device=accelerator.device)
    torch.save(model.state_dict(), 'checkpoint.pth')
