# smalldiffusion

[![Tutorial blog post][blog-img]][blog-url]
[![Paper link][arxiv-img]][arxiv-url]
[![Open in Colab][colab-img]][colab-url]
[![Pypi project][pypi-img]][pypi-url]
[![Build Status][build-img]][build-url]

A lightweight diffusion library for training and sampling from diffusion and
flow models. Features:
 - Designed for ease of experimentation when training new models or developing new samplers
 - Dataset support: 2D toy datasets, pixel and latent-space image datasets
 - Example training code (with close to SOTA FID): [FashionMNIST](/examples/fashion_mnist_dit.py), [CIFAR10](/examples/cifar_unet.py), [Imagenet](/examples/imagenet_dit.py)
 - Models: [MLP](/src/smalldiffusion/model.py), [U-Net](/src/smalldiffusion/model_unet.py) and [DiT](/src/smalldiffusion/model_dit.py)
 - Supports multiple parameterizations: score-, flow- or data-prediction
 - [Small but extensible core][diffusion-py]: less than 100 lines of code for training and sampling

To install from [pypi][pypi-url]:

```
pip install smalldiffusion
```

For local development with `uv`:

```
uv sync --extra dev --extra test --extra examples
uv run pytest
```

## Toy models
To train and sample from the `Swissroll` toy dataset in 10 lines of code (see
[examples/toyexample.ipynb](/examples/toyexample.ipynb) for a detailed
guide):

```python
from torch.utils.data import DataLoader
from smalldiffusion import Swissroll, TimeInputMLP, ScheduleLogLinear, training_loop, samples

dataset  = Swissroll(np.pi/2, 5*np.pi, 100)
loader   = DataLoader(dataset, batch_size=2048)
model    = TimeInputMLP(hidden_dims=(16,128,128,128,128,16))
schedule = ScheduleLogLinear(N=200, sigma_min=0.005, sigma_max=10)
trainer  = training_loop(loader, model, schedule, epochs=15000)
losses   = [ns.loss.item() for ns in trainer]
*xt, x0  = samples(model, schedule.sample_sigmas(20), gam=2)
```

Results on various toy datasets:

<p align="center">
  <img src="https://raw.githubusercontent.com/yuanchenyang/smalldiffusion/main/imgs/toy_models.png" width=100%>
</p>

### Conditional training and sampling with classifier-free guidance

We can also train conditional diffusion models and sample from them using
[classifier-free guidance][cfg-paper]. In
[examples/cond_tree_model.ipynb](/examples/cond_tree_model.ipynb), samples from
each class in the 2D tree dataset are represented with a different color.

<p align="center">
  <img src="https://raw.githubusercontent.com/yuanchenyang/smalldiffusion/main/imgs/cfg.png" width=100%>
</p>

## Diffusion transformer
We provide [a concise implementation][model-code] of the diffusion transformer introduced in
[[Peebles and Xie 2022]][dit-paper].

### DiT on ImageNet with flow matching
We provide [an example script](/examples/imagenet_dit.py) for training a DiT-B/2
model on ImageNet 256×256 using the flow matching formulation in the latent
space of [Stable Diffusion's
VAE](https://huggingface.co/stabilityai/sd-vae-ft-mse). The script trains on
precomputed VAE latents and supports multi-GPU training via `accelerate`:

```
uv run accelerate config
uv run accelerate launch examples/imagenet_dit.py
```

After training for 400k steps (~10 hours on 8 GPUs), the model achieves an
unconditional FID of around 27, compared to 33 for
[SiT](https://github.com/willisma/sit) and 43 for [DiT](https://github.com/facebookresearch/dit).

<p align="center">
  <img src="https://raw.githubusercontent.com/yuanchenyang/smalldiffusion/main/imgs/imagenet_samples.png" width=100%>
</p>

### FashionMNIST dataset
To train a diffusion transformer model on the FashionMNIST dataset and
generate a batch of samples (after first running `uv run accelerate config`):

```
uv run accelerate launch examples/fashion_mnist_dit.py
```

With the provided default parameters and training on a single GPU for around 2
hours, the model can achieve a [FID
score](https://paperswithcode.com/sota/image-generation-on-fashion-mnist) of
around 5-6, producing the following generated outputs:

<p align="center">
  <img src="https://raw.githubusercontent.com/yuanchenyang/smalldiffusion/main/imgs/fashion-mnist-samples.png" width=50%>
</p>

## U-Net models
The same code can be used to train [U-Net-based models][unet-py].

```
uv run accelerate launch examples/fashion_mnist_unet.py
```

We also provide example code to train a U-Net on the CIFAR-10 dataset, with an
unconditional generation FID of around 3-4:

```
uv run accelerate launch examples/cifar_unet.py
```

<p align="center">
  <img src="https://raw.githubusercontent.com/yuanchenyang/smalldiffusion/main/imgs/cifar-samples.png" width=50%>
</p>

## StableDiffusion
smalldiffusion's sampler works with any pretrained diffusion model, and supports
DDPM, DDIM as well as accelerated sampling algorithms. In
[examples/diffusers_wrapper.py][diffusers-wrapper], we provide a
simple wrapper for any pretrained [huggingface
diffusers](https://github.com/huggingface/diffusers) latent diffusion model,
enabling sampling from pretrained models with only a few lines of code:

```python
from diffusers_wrapper import ModelLatentDiffusion
from smalldiffusion import ScheduleLDM, samples

schedule = ScheduleLDM(1000)
model    = ModelLatentDiffusion('stabilityai/stable-diffusion-2-1-base')
model.set_text_condition('An astronaut riding a horse')
*xts, x0 = samples(model, schedule.sample_sigmas(50))
decoded  = model.decode_latents(x0)
```

It is easy to experiment with different sampler parameters and sampling
schedules, as demonstrated in [examples/stablediffusion.py][stablediffusion]. A
few examples on tweaking the parameter `gam`:

<p align="center">
  <img src="https://raw.githubusercontent.com/yuanchenyang/smalldiffusion/main/imgs/sd_examples.jpg" width=100%>
</p>

# How to use
The core of smalldiffusion depends on the interaction between `data`, `model`
and `schedule` objects. Here we give a specification of these objects. For a
detailed introduction to diffusion models and the notation used in the code, see
the [accompanying tutorial][blog-url].

The library targets **Python 3.10+** (see `requires-python` in
[`pyproject.toml`](pyproject.toml)). After a local checkout, install everything
needed for tests and examples with `uv sync --all-extras` (or
`make install-local`), or install subsets with
`uv sync --extra dev --extra test --extra examples`.

### Data
For training diffusion models, smalldiffusion uses PyTorch
[`Dataset`](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
and [`DataLoader`](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
batches.
By default, `training_loop` expects each batch to be a tensor of data only. To
drop labels (or otherwise map items), wrap the underlying dataset with
`MappedDataset(dataset, fn)` so each `__getitem__` returns what the loop should
see.

For **classifier-free guidance**–style conditional training, pass
`conditional=True` to `training_loop`. Each batch should then be a pair
`(x0, cond)` (e.g. images and integer class ids); see
[`generate_train_sample`](src/smalldiffusion/diffusion.py) for how batches are
split.

Three 2D toy datasets are provided: `Swissroll`,
[`DatasaurusDozen`](https://www.research.autodesk.com/publications/same-stats-different-graphs/),
and `TreeDataset`.

### Model
All model objects should subclass `torch.nn.Module` and integrate with the
training and sampling hooks used in [`diffusion.py`](src/smalldiffusion/diffusion.py).
They should define:

  - `input_dims`: a tuple of spatial/channel dimensions for one sample (no batch
    dimension).
  - `rand_input(batchsize)`: i.i.d. standard normal noise with shape
    `[batchsize, *input_dims]`. You can inherit this from `ModelMixin` once
    `input_dims` is set.

The model `forward` is called as `forward(x, sigma, cond=None)` where:

 - `x` has shape `[B, *input_dims]`.
 - `sigma` is either a scalar (`shape == ()`), broadcast to the whole batch, or
   per-sample with shape `[B, 1, …, 1]` matching `x`.
 - `cond` is optional conditioning (e.g. class indices for CFG).

By default, [`ModelMixin`](src/smalldiffusion/model.py) trains the network to
predict additive noise `eps` and implements `get_loss` accordingly. For other
targets, compose the model class with **`PredX0`**, **`PredV`**, or **`PredFlow`**
(score / v-prediction / flow-style objectives); those wrappers adjust the loss
and implement `predict_eps` so the shared `samples()` loop still applies.
Sampling calls `predict_eps` (and `predict_eps_cfg` when `cfg_scale > 0`).

### Schedule
A `Schedule` holds a 1D tensor of `sigma` values (noise level or, for flow
schedules, continuous time). Subclasses build that tensor; you can also
instantiate `Schedule(sigmas)` directly. Methods:

  - `sample_sigmas(steps)`: decreasing sequence for sampling (length `steps + 1`,
    trailing spacing as in Table 2 of [https://arxiv.org/abs/2305.08891](https://arxiv.org/abs/2305.08891)).
  - `sample_batch(x0)`: random `sigma` for each row of batch `x0`, with shape
    broadcastable to `x0` (uses batch size and device from `x0`).

Built-in schedules:

  1. `ScheduleLogLinear` — simple log-spaced sigmas; strong default for toys and
     small models.
  2. `ScheduleDDPM` — standard pixel-space diffusion.
  3. `ScheduleLDM` — latent diffusion (e.g. Stable Diffusion–style).
  4. `ScheduleSigmoid` — [GeoDiff][geodiff].
  5. `ScheduleCosine` — [iDDPM][iddpm].
  6. `ScheduleFlow` — flow matching with uniform time in `[t_min, t_max]`.
  7. `ScheduleLogNormalFlow` — extends `ScheduleFlow` with logit-normal training
     times (`sample_batch` overridden).

The figure below compares several of these with default parameters (diffusion
sigmas as a function of step index).
<p align="center">
  <img src="https://raw.githubusercontent.com/yuanchenyang/smalldiffusion/main/imgs/schedule.png" width=40%>
</p>

### Training
[`training_loop`](src/smalldiffusion/diffusion.py) is a generator that runs
`epochs` passes over `loader`, using `get_loss` from the model’s class (so
`PredX0` / `PredV` / `PredFlow` work without changing the loop). It accepts
optional `accelerator` ([Hugging Face Accelerate](https://github.com/huggingface/accelerate))
and `conditional` as above. Each step yields a namespace of locals (e.g. `loss`,
`optimizer`, `x0`, `sigma`).

```
for ns in training_loop(loader, model, schedule):
    print(ns.loss.item())
```

Pass a prepared `Accelerator` instance to use distributed or mixed-precision
training; the examples use `accelerate launch` for multi-GPU runs.

### Sampling
[`samples`](src/smalldiffusion/diffusion.py) is a generator that takes `model`, a
**decreasing** list/tensor of `sigmas` (typically `schedule.sample_sigmas(steps)`),
and sampler hyperparameters `gam` and `mu`. Optional arguments include
`batchsize`, initial latent `xt`, conditioning `cond`, `cfg_scale` for
classifier-free guidance, and `accelerator`. It yields successive `xt` states.

Common choices:

 - DDPM [[Ho et al.]](https://arxiv.org/abs/2006.11239): `gam=1`, `mu=0.5`.
 - DDIM [[Song et al.]](https://arxiv.org/abs/2010.02502): `gam=1`, `mu=0`.
 - Accelerated sampling [[Permenter and Yuan]][arxiv-url]: `gam=2` (with `mu=0`).

For more detail on unifying these samplers, see Appendix A of [[Permenter and
Yuan]][arxiv-url].


[diffusion-py]:https://github.com/yuanchenyang/smalldiffusion/blob/main/src/smalldiffusion/diffusion.py
[unet-py]:https://github.com/yuanchenyang/smalldiffusion/blob/main/src/smalldiffusion/model_unet.py
[diffusers-wrapper]:https://github.com/yuanchenyang/smalldiffusion/blob/main/examples/diffusers_wrapper.py
[stablediffusion]:https://github.com/yuanchenyang/smalldiffusion/blob/main/examples/stablediffusion.py
[build-img]:https://github.com/yuanchenyang/smalldiffusion/workflows/CI/badge.svg
[build-url]:https://github.com/yuanchenyang/smalldiffusion/actions?query=workflow%3ACI
[pypi-img]:https://img.shields.io/badge/pypi-blue
[pypi-url]:https://pypi.org/project/smalldiffusion/
[dit-paper]:https://arxiv.org/abs/2212.09748
[model-code]:https://github.com/yuanchenyang/smalldiffusion/blob/main/src/smalldiffusion/model.py
[blog-img]:https://img.shields.io/badge/Tutorial-blogpost-blue
[blog-url]:https://www.chenyang.co/diffusion.html
[arxiv-img]:https://img.shields.io/badge/Paper-arxiv-blue
[arxiv-url]:https://arxiv.org/abs/2306.04848
[colab-url]:https://colab.research.google.com/drive/1So1lb9fG-AnDeSXNbosCnDbxbzf5xbor?usp=sharing
[colab-img]:https://colab.research.google.com/assets/colab-badge.svg
[geodiff]:https://arxiv.org/abs/2203.02923
[iddpm]:https://arxiv.org/abs/2102.09672
[cfg-paper]:https://arxiv.org/abs/2207.12598
