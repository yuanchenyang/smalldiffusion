import math
from itertools import pairwise

import torch
import numpy as np
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

class Schedule:
    '''Diffusion noise schedules parameterized by sigma'''
    def __init__(self, sigmas: torch.FloatTensor):
        self._sigmas = sigmas

    def __getitem__(self, i) -> torch.FloatTensor:
        return self._sigmas[i]

    def __len__(self) -> int:
        return len(self._sigmas)

    def sample_sigmas(self, steps: int) -> torch.FloatTensor:
        '''Called during sampling to get a decreasing sigma schedule with a
        specified number of sampling steps:
          - Spacing is "trailing" as in Table 2 of https://arxiv.org/abs/2305.08891
          - Includes initial and final sigmas
            i.e. len(schedule.sample_sigmas(steps)) == steps + 1
        '''
        indices = list((len(self) * (1 - np.arange(0, steps)/steps))
                       .round().astype(np.int64) - 1)
        return self[indices + [0]]

    def sample_batch(self, batchsize: int) -> torch.FloatTensor:
        '''Called during training to get a batch of randomly sampled sigma values
        '''
        return self[torch.randint(len(self), (batchsize,))]

def sigmas_from_betas(betas: torch.FloatTensor):
    return (1/torch.cumprod(1.0 - betas, dim=0) - 1).sqrt()

class ScheduleLogLinear(Schedule):
    def __init__(self, N: int, sigma_min: float=0.02, sigma_max: float=10):
        super().__init__(torch.logspace(math.log10(sigma_min), math.log10(sigma_max), N))

class ScheduleDDPM(Schedule):
    def __init__(self, N: int=1000, beta_start: float=0.0001, beta_end: float=0.02):
        '''Default parameters recover schedule used in most diffusion models'''
        super().__init__(sigmas_from_betas(torch.linspace(beta_start, beta_end, N)))

class ScheduleLDM(Schedule):
    def __init__(self, N: int=1000, beta_start: float=0.00085, beta_end: float=0.012):
        '''Default parameters recover schedule used in most latent diffusion
        models, e.g. Stable diffusion'''
        super().__init__(sigmas_from_betas(torch.linspace(beta_start**0.5, beta_end**0.5, N)**2))

def generate_train_sample(x0: torch.FloatTensor, schedule):
    # eps  : i.i.d. normal with same shape as x0
    # sigma: uniformly sampled from schedule, with shape Bx1x..x1 for broadcasting
    sigma = schedule.sample_batch(x0.shape[0]).to(x0)
    while len(sigma.shape) < len(x0.shape):
        sigma = sigma.unsqueeze(-1)
    eps = torch.randn(x0.shape).to(x0)
    return sigma, eps

# Model objects
# Always called with (x0, sigma):
#   If x0.shape == [B, D1, ..., Dk], sigma.shape == [] or [B, 1, ..., 1].
#   If sigma.shape == [], model will be called with the same sigma for each x0
#   Otherwise, x0[i] will be paired with sigma[i] when calling model
# Have a `rand_input` method for generating random xt during sampling

def training_loop(data, model, schedule, accelerator=None, epochs=10000, lr=1e-3):
    accelerator = accelerator or Accelerator()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model, optimizer, data = accelerator.prepare(model, optimizer, data)
    for _ in (pbar := tqdm(range(epochs))):
        for x0 in iter(data):
            optimizer.zero_grad()
            sigma, eps = generate_train_sample(x0, schedule)
            eps_hat = model(x0 + sigma * eps, sigma)
            loss = nn.MSELoss()(eps_hat, eps)
            yield locals() # For extracting training statistics
            accelerator.backward(loss)
            optimizer.step()

@torch.no_grad()
def samples(model, sigmas, xt=None, batchsize=1, gam=1., mu=0., device='cpu'):
    # sigmas: Iterable with N+1 values [sigma_N, ..., sigma_0] for N sampling steps
    # Need gam >= 1 and mu in [0, 1)
    # For DDPM   : gam=1, mu=0.5
    # For DDIM   : gam=1, mu=0
    # Accelerated: gam=2, mu=0
    if xt is None:
        xt = model.rand_input(batchsize, device) * sigmas[0]
    else:
        batchsize = xt.shape[0]
    eps = None
    for i, (sig, sig_prev) in enumerate(pairwise(sigmas)):
        eps, eps_prev = model(xt, sig.to(device)), eps
        eps_av = eps * gam + eps_prev * (1-gam)  if i > 0 else eps
        sig_p = (sig_prev/sig**mu)**(1/(1-mu)) # sig_prev == sig**mu sig_p**(1-mu)
        eta = (sig_prev**2 - sig_p**2).sqrt()
        xt = xt - (sig - sig_p) * eps_av + eta * model.rand_input(batchsize, device)
        yield xt
