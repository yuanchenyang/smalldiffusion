import unittest
import torch
import numpy as np
from torch.utils.data import DataLoader, RandomSampler
from diffusers import DDIMScheduler, DDPMScheduler
from accelerate import Accelerator

from smalldiffusion import *
from smalldiffusion.diffusion import generate_train_sample

def get_hf_sigmas(scheduler):
    return (1/scheduler.alphas_cumprod - 1).sqrt()

class DummyModel(torch.nn.Module, ModelMixin):
    def __init__(self, dims):
        super().__init__()
        self.input_dims = dims

    def __call__(self, x, sigma, cond=None):
        gen = torch.Generator().manual_seed(int(sigma * 100000))
        return torch.randn((x.shape[0],) + self.input_dims, generator=gen)

class TensorTest:
    def assertEqualTensors(self, t1, t2):
        error = sum(t1 - t2).item()
        self.assertEqual(error, 0.0, f'{error} != 0!')

    def assertAlmostEqualTensors(self, t1, t2, tol=1e-6):
        mse = (t1-t2).square().mean().item()
        self.assertTrue(mse < tol, f'{mse} > {tol}!')

class TestSchedule(unittest.TestCase, TensorTest):
    def setUp(self):
        self.params = [
            (1000, 0.0001, 0.02),
            (100, 0.01, 0.02),
            (659, 0.001, 0.02),
        ]

    def compare_scheduler_sigmas(self, sch_hf, sch_sd):
        self.assertEqualTensors(get_hf_sigmas(sch_hf), sch_sd.sigmas)

    def test_DDPMScheduler(self):
        for N, beta_start, beta_end in self.params:
            self.compare_scheduler_sigmas(
                DDIMScheduler(num_train_timesteps=N, beta_start=beta_start,
                              beta_end=beta_end),
                ScheduleDDPM(N, beta_start=beta_start, beta_end=beta_end))

    def test_CosineScheduler(self):
        for N, beta_start, beta_end in self.params:
            self.compare_scheduler_sigmas(
                DDIMScheduler(num_train_timesteps=N, beta_start=beta_start,
                              beta_end=beta_end, beta_schedule='squaredcos_cap_v2'),
                ScheduleCosine(N, beta_start=beta_start, beta_end=beta_end)
            )

    def test_SigmoidScheduler(self):
        for N, beta_start, beta_end in self.params:
            self.compare_scheduler_sigmas(
                DDPMScheduler(num_train_timesteps=N, beta_start=beta_start,
                              beta_end=beta_end, beta_schedule='sigmoid'),
                ScheduleSigmoid(N, beta_start=beta_start, beta_end=beta_end)
            )

    def test_LDMScheduler(self):
        for N, beta_start, beta_end in self.params:
            self.compare_scheduler_sigmas(
                DDIMScheduler(num_train_timesteps=N, beta_start=beta_start,
                              beta_end=beta_end, beta_schedule='scaled_linear'),
                ScheduleLDM(N, beta_start=beta_start, beta_end=beta_end)
            )

    def test_LDMScheduler_default(self):
        self.compare_scheduler_sigmas(
            DDIMScheduler.from_pretrained('stabilityai/stable-diffusion-2',
                                          subfolder='scheduler'),
            ScheduleLDM(),
        )

    def test_sample_sigma(self):
        for N, beta_start, beta_end in self.params:
            sc_hf = DDIMScheduler(num_train_timesteps=N, timestep_spacing='trailing',
                                  beta_start=beta_start, beta_end=beta_end)
            sc_sd = ScheduleDDPM(N, beta_start=beta_start, beta_end=beta_end)
            for n in range(1,100):
                sc_hf.set_timesteps(n)
                sig_hf = get_hf_sigmas(sc_hf)[sc_hf.timesteps]
                sig_sd = sc_sd.sample_sigmas(n)
                self.assertEqual(len(sig_sd), n+1)
                if N % n == 0:
                    # Numerical issues when rounding occues in diffusers
                    # implementation, only compare when steps are divisible
                    self.assertEqualTensors(sig_hf, sig_sd[:-1])

class TestSampler(unittest.TestCase, TensorTest):
    def setUp(self):
        self.dims = (2,10,40)
        self.N_train = 1000
        self.ntrials = 5
        self.nevals = (10,20,50)
        self.batches = 3
        self.sc_sd = ScheduleDDPM(self.N_train)
        self.model = DummyModel(self.dims)

    def test_DDIM_equivalence(self):
        sc_hf = DDIMScheduler(
            num_train_timesteps=self.N_train,
            timestep_spacing='trailing',
            clip_sample=False,
            set_alpha_to_one=False,
        )
        for _ in range(self.ntrials):
            for N in self.nevals:
                # Same initial noise
                xt = self.model.rand_input(self.batches)

                # Sample with smalldiffusion in DDIM mode
                sigmas = self.sc_sd.sample_sigmas(N)
                *_, x0_sd = samples(self.model, sigmas, gam=1, xt=xt*sigmas[0])

                # Sample with DDIM scheduler from diffusers library
                sc_hf.set_timesteps(N)
                x = xt
                for t in sc_hf.timesteps:
                    x = sc_hf.step(self.model(x, self.sc_sd[t]), t, x).prev_sample
                x0_hf = x

                self.assertAlmostEqualTensors(x0_sd, x0_hf, tol=1e-4)

    def test_DDPM_equivalence(self):
        sc_hf = DDPMScheduler(
            num_train_timesteps=self.N_train,
            timestep_spacing='trailing',
            clip_sample=False,
        )
        for seed in range(self.ntrials):
            for N in self.nevals:
                # Same initial noise
                xt = self.model.rand_input(self.batches)

                # Sample with smalldiffusion in DDPM mode
                sigmas = self.sc_sd.sample_sigmas(N)
                torch.manual_seed(seed)
                *_, x0_sd = samples(self.model, sigmas, gam=1, xt=xt*sigmas[0], mu=0.5)

                # Sample with DDPM scheduler from diffusers library
                sc_hf.set_timesteps(N)
                torch.manual_seed(seed)
                x = xt
                for t in sc_hf.timesteps:
                    x = sc_hf.step(self.model(x, self.sc_sd[t]), t, x).prev_sample
                x0_hf = x

                self.assertAlmostEqualTensors(x0_sd, x0_hf, tol=1e-3)

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.params = [
            (100, 2048, 3, 19),
            (10, 100, 42, 7),
        ]

    def test_swissroll(self):
        for npoints, B, epochs, sample_steps in self.params:
            accelerator = Accelerator()
            dataset = Swissroll(np.pi/2, 5*np.pi, npoints)
            loader = DataLoader(dataset, sampler=RandomSampler(dataset, num_samples=B), batch_size=B)

            # Test loader
            batch = next(iter(loader))
            self.assertEqual(batch.shape, (B, 2))
            self.assertEqual(len(set((x, y) for x, y in batch.numpy())), npoints)

            schedule = ScheduleLogLinear(N=200, sigma_min=0.01, sigma_max=10)
            model = TimeInputMLP(hidden_dims=(16,128,256,128,16))
            trainer = training_loop(loader, model, schedule, epochs=epochs, lr=1e-3,
                                    accelerator=accelerator)

            # Mainly to test that model trains without error
            losses = [ns.loss.item() for ns in trainer]
            self.assertEqual(len(losses), epochs)

            # Test sampling
            *_, sample = samples(model, schedule.sample_sigmas(sample_steps), gam=1, batchsize=B//2,
                                 accelerator=accelerator)
            self.assertEqual(sample.shape, (B//2, 2))

class TestIdeal(unittest.TestCase, TensorTest):
    # Test that ideal deoiser batching works
    def test_ideal(self):
        for N in [1, 10, 99]:
            loader = DataLoader(Swissroll(np.pi/2, 5*np.pi, 30), batch_size=2000)
            sigmas = torch.linspace(1, 2, N)
            idd = IdealDenoiser(loader.dataset)
            x0 = idd.rand_input(N)
            batched_output = idd(x0, sigmas.unsqueeze(1))
            singleton_output = torch.cat([idd(x0i.unsqueeze(0), s) for x0i, s in zip(x0, sigmas)])
            self.assertAlmostEqualTensors(batched_output, singleton_output, tol=1e-6)

# Just testing that model creation and forward pass works
class TestModels(unittest.TestCase):
    def setUp(self):
        self.modifiers = (
            Scaled, PredX0, PredV,
            lambda x: x,
            lambda x: Scaled(PredX0(x)),
            lambda x: Scaled(PredV(x))
        )
        self.batches = (1, 13)
        self.schedule = ScheduleDDPM(1000)

    def test_dit_uncond(self):
        for B in self.batches:
            for modifier in self.modifiers:
                model = modifier(DiT)(in_dim=16, channels=3, patch_size=2, depth=4, head_dim=32, num_heads=6)
                x = torch.randn(B, 3, 16, 16)
                for sigma in (torch.tensor(1), torch.randn(B, 1, 1, 1)):
                    y = model.predict_eps(x, sigma)
                    self.assertEqual(y.shape, x.shape)
                model.get_loss(*generate_train_sample(x, self.schedule, conditional=False))


    def test_dit_cond(self):
        for B in self.batches:
            for modifier in self.modifiers:
                model = modifier(DiT)(in_dim=16, channels=3, patch_size=2, depth=4, head_dim=32, num_heads=6,
                                      cond_embed=CondEmbedderLabel(32*6, 10))
                x = torch.randn(B, 3, 16, 16)
                labels = torch.tensor([i % 10 for i in range(B)])
                y = model.predict_eps_cfg(x, torch.tensor(1), cond=labels, cfg_scale=4.0)
                self.assertEqual(y.shape, x.shape)
                model.get_loss(*generate_train_sample((x, labels), self.schedule, conditional=True))

    def test_unet_uncond(self):
        for B in self.batches:
            for modifier in self.modifiers:
                model = modifier(Unet)(in_dim=16, in_ch=3, out_ch=3, ch=64, ch_mult=(1,1,2,),
                                       num_res_blocks= 2, attn_resolutions=(8,))
                x = torch.randn(B, 3, 16, 16)
                for sigma in (torch.tensor(1), torch.randn(B, 1, 1, 1)):
                    y = model.predict_eps(x, sigma)
                    self.assertEqual(y.shape, x.shape)
                model.get_loss(*generate_train_sample(x, self.schedule, conditional=False))
