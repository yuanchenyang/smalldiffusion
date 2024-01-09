# smalldiffusion

A lightweight diffusion library for training and sampling from diffusion
models. It is built for easy experimentation when training new models and
developing new samplers, supporting minimal toy models to state-of-the-art
pretrained models. The [core of this library](/src/smalldiffusion/diffusion.py)
for diffusion training and sampling is implemented in less than 100 lines of
very readable code.

### Toy models
To train and sample from the `Swissroll` toy dataset in 10 lines of code (see
[examples/toyexample.ipynb](/examples/toyexample.ipynb) for a detailed
guide):

```python
from torch.utils.data import DataLoader
from smalldiffusion import Swissroll, TimeInputMLP, ScheduleLogLinear, training_loop, samples

dataset  = Swissroll(np.pi/2, 5*np.pi, 100)
loader   = DataLoader(dataset, batch_size=2048)
model    = TimeInputMLP(hidden_dims=(16,128,128,128,128,16))
schedule = ScheduleLogLinear(N=200)
trainer  = training_loop(loader, model, schedule, epochs=10000)
losses   = [ns.loss.item() for ns in trainer]
*xt, x0  = samples(model, schedule.sample_sigmas(20))
```

### Unet models
The same code can be used to train Unet-based models. Multi-GPU training and
sampling is also supported via
[`accelerate`](https://github.com/huggingface/accelerate). To train a model on
the FashionMNIST dataset and generate a batch of samples:

```
$ accelerate config
$ accelerate launch examples/fashion_mnist.py
```

With the default parameters, the model can achieve a [FID
score](https://paperswithcode.com/sota/image-generation-on-fashion-mnist) of
around 12-13, producing the following generated outputs:
<p align="center">
<img src="/imgs/fashion-mnist-samples.png" width=50%>
</p>

### StableDiffusion
smalldiffusion's sampler works with any pretrained diffusion model, and supports
DDPM, DDIM as well as accelerated sampling algorithms. In
[examples/stablediffusion.ipynb](/examples/stablediffusion.ipynb), we provide a
simple wrapper for any pretrained [huggingface
diffusers](https://github.com/huggingface/diffusers) latent diffusion model,
with examples showing how to sample from pretrained models.

```python
from diffusers_wrapper import ModelLatentDiffusion
from smalldiffusion import ScheduleLDM, samples

schedule = ScheduleLDM(1000)
model    = ModelLatentDiffusion('stabilityai/stable-diffusion-2-1-base')
model.set_text_condition('An astronaut riding a horse')
*xts, x0 = samples(model, schedule.sample_sigmas(50))
decoded  = model.decode_latents(x0)
```

# How to use
The core of smalldiffusion depends on the interaction between `data`, `model`
and `schedule` objects. Here we give a specification of these objects. For a
detailed introduction to diffusion models and the notation used in the code, see
the [accompanying tutorial](https://www.chenyang.co/diffusion.html).

### Data
For training diffusion models, smalldiffusion supports [PyTorch `Datasets` and
`DataLoaders`](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html).
The training code expects the iterates from a `DataLoader` object to be batches
of data, without labels. To remove labels from existing datasets, extract the
data with the provided `MappedDataset` wrapper before constructing a
`DataLoader`.

### Model

### Schedule

### Training

### Sampling
