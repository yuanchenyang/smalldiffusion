import unittest

from smalldiffusion import *
from diffusers import DDIMScheduler

def get_hf_sigmas(scheduler):
    return (1/scheduler.alphas_cumprod - 1).sqrt()

class TestSchedule(unittest.TestCase):
    def setUp(self):
        self.params = [
            (1000, 0.0001, 0.02),
            (100, 0.01, 0.02),
            (659, 0.001, 0.02),
        ]

    def assertEqualTensors(self, t1, t2):
        self.assertEqual(sum(t1 - t2).item(), 0.0)

    def compare_scheduler_sigmas(self, sch_hf, sch_sd):
        self.assertEqualTensors(get_hf_sigmas(sch_hf), sch_sd._sigmas)

    def test_DDPMScheduler(self):
        for N, beta_start, beta_end in self.params:
            self.compare_scheduler_sigmas(
                DDIMScheduler(num_train_timesteps=N, beta_start=beta_start, beta_end=beta_end),
                ScheduleDDPM(N, beta_start=beta_start, beta_end=beta_end)
            )

    def test_LDMScheduler(self):
        for N, beta_start, beta_end in self.params:
            self.compare_scheduler_sigmas(
                DDIMScheduler(num_train_timesteps=N, beta_start=beta_start,
                              beta_end=beta_end, beta_schedule='scaled_linear'),
                ScheduleLDM(N, beta_start=beta_start, beta_end=beta_end)
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

if __name__ == '__main__':
    unittest.main()
