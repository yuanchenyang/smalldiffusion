from diffusers_wrapper import ModelLatentDiffusion
from smalldiffusion import ScheduleLDM, samples
from torchvision.utils import save_image

schedule = ScheduleLDM(1000)
model    = ModelLatentDiffusion('CompVis/stable-diffusion-v1-4')
model.set_text_condition('An astronaut riding a horse')
*xts, x0 = samples(model, schedule.sample_sigmas(50))
decoded  = model.decode_latents(x0)
save_image(((decoded.squeeze()+1)/2).clamp(0,1), 'stablediffusion_output.png')
