# smalldiffusion

A lightweight diffusion library for training and sampling from diffusion
models. It is built for easy experimentation when training new models and
samplers, supporting minimal toy models to state-of-the-art pretrained
models. The [core of this library](/src/smalldiffusion/diffusion.py)
for diffusion training and sampling is implemented in less than 100 lines of
code.

For example, to train and sample from the `Swissroll` toy model (see
[examples/toyexample.ipynb](/examples/toyexample.ipynb) for a detailed
guide):

```python
from torch.utils.data import DataLoader
from smalldiffusion import Swissroll, ScheduleLogLinear, TimeInputMLP, training_loop, samples

dataset = Swissroll(np.pi/2, 5*np.pi, 100)
loader = DataLoader(dataset, batch_size=2048)
schedule = ScheduleLogLinear(N=200, sigma_min=0.01, sigma_max=10)
model = TimeInputMLP(hidden_dims=(16,128,128,128,128,16))
trainer = training_loop(loader, model, schedule, epochs=10000, lr=1e-3)
losses = [ns['loss'].item() for ns in trainer]
*xt, x0 = samples(model, schedule.sample_sigmas(20), batchsize=100)
```
