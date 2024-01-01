import math
from itertools import pairwise

import torch
import numpy as np
from accelerate import Accelerator
from torch import nn
from tqdm import tqdm

class Schedule:
    '''Diffusion noise schedules parameterized by sigma'''
    def __init__(self, N: int):
        self._sigmas = torch.linspace(0, 1, N)

    def sigmas_from_betas(self, betas: torch.FloatTensor):
        alpha_prod = torch.cumprod(1.0 - betas, dim=0)
        self._sigmas = (1/alpha_prod - 1).sqrt()

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
        start_indices = list((len(self) * (1 - np.arange(0, steps)/steps))
                             .round().astype(np.int64) - 1)
        return self[start_indices + [0]]

    def sample_batch(self, batchsize: int) -> torch.FloatTensor:
        '''Called during training to get a batch of randomly sampled sigma values
        '''
        return self[torch.randint(len(self), (batchsize,))]

class ScheduleLogLinear(Schedule):
    def __init__(self, N, sigma_min=0.02, sigma_max=10):
        super().__init__(N)
        self._sigmas = torch.logspace(math.log10(sigma_min), math.log10(sigma_max), N)

class ScheduleDDPM(Schedule):
    def __init__(self, N, beta_start: float=0.0001, beta_end: float=0.02):
        '''Default parameters recover schedule used in most diffusion models'''
        super().__init__(N)
        self.sigmas_from_betas(torch.linspace(beta_start, beta_end, N))

class ScheduleLDM(Schedule):
    def __init__(self, N, beta_start: float=0.0001, beta_end: float=0.02):
        '''Default parameters recover schedule used in most latent diffusion
        models, e.g. Stable diffusion'''
        super().__init__(N)
        self.sigmas_from_betas(torch.linspace(beta_start**0.5, beta_end**0.5, N)**2)

def generate_train_sample(x0, schedule, device):
    sigma = schedule.sample_batch(x0.shape[0])
    while len(sigma.shape) < len(x0.shape):
        sigma = sigma.unsqueeze(-1)
    eps = torch.randn(x0.shape)
    return x0.to(device), sigma.to(device), eps.to(device)

def training_loop(data, model, schedule, epochs=10000, lr=1e-3, accelerator=None):
    accelerator = accelerator or Accelerator()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model, optimizer, data = accelerator.prepare(model, optimizer, data)
    for _ in tqdm(range(epochs)):
        for x0 in iter(data):
            optimizer.zero_grad()
            x0, sigma, eps = generate_train_sample(x0, schedule, accelerator.device)
            eps_hat = model(x0 + sigma * eps, sigma)
            loss = nn.MSELoss()(eps_hat, eps)
            yield locals() # For extracting training statistics
            accelerator.backward(loss)
            optimizer.step()

@torch.no_grad()
def samples(model, sigmas, batchsize=2048, gam=1, device='cpu', xt=None):
    if xt is None:
        xt = torch.randn((batchsize,) + model.input_dims).to(device) * sigmas[0]
    eps = None
    for i, (sig, sig_prev) in enumerate(pairwise(sigmas)):
        eps, eps_prev = model(xt, sig.to(device)), eps
        eps_av = eps_av = eps * gam + eps_prev * (1-gam)  if i > 0 else eps
        xt = xt - (sig - sig_prev) * eps_av
        yield xt
