import math
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from einops import rearrange, repeat
from itertools import pairwise


## Basic functions used by all models

def sigma_log_scale(batches, sigma, scaling_factor):
    if sigma.shape == torch.Size([]):
        sigma = sigma.unsqueeze(0).repeat(batches)
    else:
        assert sigma.shape == (batches,), 'sigma.shape == [] or [batches]!'
    return torch.log(sigma)*scaling_factor

def get_sigma_embeds(batches, sigma, scaling_factor=0.5):
    s = sigma_log_scale(batches, sigma, scaling_factor).unsqueeze(1)
    return torch.cat([torch.sin(s), torch.cos(s)], dim=1)

class ModelMixin:
    def rand_input(self, batchsize):
        assert hasattr(self, 'input_dims'), 'Model must have "input_dims" attribute!'
        return torch.randn((batchsize,) + self.input_dims)

    # Currently predicts eps, override following methods to predict, for example, x0
    def get_loss(self, x0, sigma, eps):
        return nn.MSELoss()(eps, self(x0 + sigma * eps, sigma))

    def predict_eps(self, x, sigma):
        return self(x, sigma)

## Modifiers for models, such as including scaling or changing model predictions

def alpha(sigma):
    return 1/(1+sigma**2)

# Scale model input so that its norm stays constant for all sigma
def Scaled(cls: ModelMixin):
    def forward(self, x, sigma):
        return cls.forward(self, x * alpha(sigma).sqrt(), sigma)
    return type(cls.__name__ + 'Scaled', (cls,), dict(forward=forward))

# Train model to predict x0 instead of eps
def PredX0(cls: ModelMixin):
    def get_loss(self, x0, sigma, eps):
        return nn.MSELoss()(x0, self(x0 + sigma * eps, sigma))
    def predict_eps(self, x, sigma):
        x0_hat = self(x, sigma)
        return (x - x0_hat)/sigma
    return type(cls.__name__ + 'PredX0', (cls,),
                dict(get_loss=get_loss, predict_eps=predict_eps))

# Train model to predict v (https://arxiv.org/pdf/2202.00512.pdf) instead of eps
def PredV(cls: ModelMixin):
    def get_loss(self, x0, sigma, eps):
        xt = x0 + sigma * eps
        v = alpha(sigma).sqrt() * eps - (1-alpha(sigma)).sqrt() * x0
        return nn.MSELoss()(v, self(xt, sigma))
    def predict_eps(self, x, sigma):
        v_hat = self(x, sigma)
        return alpha(sigma).sqrt() * (v_hat + (1-alpha(sigma)).sqrt() * x)
    return type(cls.__name__ + 'PredV', (cls,),
                dict(get_loss=get_loss, predict_eps=predict_eps))


## Simple MLP for toy examples

class TimeInputMLP(nn.Module, ModelMixin):
    def __init__(self, dim=2, hidden_dims=(16,128,256,128,16)):
        super().__init__()
        layers = []
        for in_dim, out_dim in pairwise((dim + 2,) + hidden_dims):
            layers.extend([nn.Linear(in_dim, out_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dims[-1], dim))

        self.net = nn.Sequential(*layers)
        self.input_dims = (dim,)

    def forward(self, x, sigma):
        # x     shape: b x dim
        # sigma shape: b x 1 or scalar
        sigma_embeds = get_sigma_embeds(x.shape[0], sigma.squeeze()) # shape: b x 2
        nn_input = torch.cat([x, sigma_embeds], dim=1)               # shape: b x (dim + 2)
        return self.net(nn_input)


## Ideal denoiser defined by a dataset

def sq_norm(M, k):
    # M: b x n --(norm)--> b --(repeat)--> b x k
    return (torch.norm(M, dim=1)**2).unsqueeze(1).repeat(1,k)

class IdealDenoiser(ModelMixin):
    def __init__(self, dataset: torch.utils.data.Dataset):
        self.data = torch.stack([dataset[i] for i in range(len(dataset))])
        self.input_dims = self.data.shape[1:]

    def __call__(self, x, sigma):
        assert sigma.shape == tuple(), 'Only singleton sigma supported'
        data = self.data.to(x)
        x_flat = x.flatten(start_dim=1)
        d_flat = data.flatten(start_dim=1)
        xb, xr = x_flat.shape
        db, dr = d_flat.shape
        assert xr == dr, 'Input x must have same dimension as data!'
        # ||x - x0||^2 ,shape xb x db
        sq_diffs = sq_norm(x_flat, db) + sq_norm(d_flat, xb).T - 2 * x_flat @ d_flat.T
        weights = torch.nn.functional.softmax(-sq_diffs/2/sigma**2, dim=1)
        return (x - torch.einsum('ij,j...->i...', weights, data))/sigma


## Diffusion transformer

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=16, channels=3, embed_dim=768, bias=True):
        super().__init__()
        self.proj = nn.Conv2d(channels, embed_dim, stride=patch_size, kernel_size=patch_size, bias=bias)
        self.init()

    def init(self): # Init like nn.Linear
        w = self.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.proj.bias, 0)

    def forward(self, x):
        return rearrange(self.proj(x), 'b c h w -> b (h w) c')

class Attention(nn.Module):
    def __init__(self, head_dim, num_heads=8, qkv_bias=False, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        dim = head_dim * num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # (B, N, D) -> (B, N, D)
        # N = H * W / patch_size**2, D = num_heads * head_dim
        q, k, v = rearrange(self.qkv(x), 'b n (qkv h k) -> qkv b h n k',
                            h=self.num_heads, k=self.head_dim)
        x = rearrange(F.scaled_dot_product_attention(q, k, v),
                      'b h n k -> b n (h k)')
        return self.proj(x)

class Modulation(nn.Module):
    def __init__(self, dim, n):
        super().__init__()
        self.n = n
        self.proj = nn.Sequential(nn.SiLU(), nn.Linear(dim, n * dim, bias=True))
        nn.init.constant_(self.proj[-1].weight, 0)
        nn.init.constant_(self.proj[-1].bias, 0)

    def forward(self, y):
        return [m.unsqueeze(1) for m in self.proj(y).chunk(self.n, dim=1)]

class ModulatedLayerNorm(nn.LayerNorm):
    def __init__(self, dim, **kwargs):
        super().__init__(dim, **kwargs)
        self.modulation = Modulation(dim, 2)
    def forward(self, x, y):
        scale, shift = self.modulation(y)
        return super().forward(x) * (1 + scale) + shift

class DiTBlock(nn.Module):
    def __init__(self, head_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        dim = head_dim * num_heads
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm1 = ModulatedLayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(head_dim, num_heads=num_heads, qkv_bias=True)
        self.norm2 = ModulatedLayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
                nn.Linear(dim, mlp_hidden_dim, bias=True),
                nn.GELU(approximate="tanh"),
                nn.Linear(mlp_hidden_dim, dim, bias=True),
        )
        self.scale_modulation = Modulation(dim, 2)

    def forward(self, x, y):
        # (B, N, D), (B, D) -> (B, N, D)
        # N = H * W / patch_size**2, D = num_heads * head_dim
        gate_msa, gate_mlp = self.scale_modulation(y)
        x = x + gate_msa * self.attn(self.norm1(x, y))
        x = x + gate_mlp * self.mlp(self.norm2(x, y))
        return x

def get_pos_embed(in_dim, patch_size, dim, N=10000):
    n = in_dim // patch_size                                          # Number of patches per side
    assert dim % 4 == 0, 'Embedding dimension must be multiple of 4!'
    omega = 1/N**np.linspace(0, 1, dim // 4, endpoint=False)          # [dim/4]
    freqs = np.outer(np.arange(n), omega)                             # [n, dim/4]
    embeds = repeat(np.stack([np.sin(freqs), np.cos(freqs)]),
                       ' b n d -> b n k d', k=n)                      # [2, n, n, dim/4]
    embeds_2d = np.concatenate([
        rearrange(embeds, 'b n k d -> (k n) (b d)'),                  # [n*n, dim/2]
        rearrange(embeds, 'b n k d -> (n k) (b d)'),                  # [n*n, dim/2]
    ], axis=1)                                                        # [n*n, dim]
    return nn.Parameter(torch.tensor(embeds_2d).float().unsqueeze(0), # [1, n*n, dim]
                        requires_grad=False)

class DiT(nn.Module, ModelMixin):
    def __init__(self, in_dim=32, channels=3, patch_size=2, depth=12,
                 head_dim=64, num_heads=6, mlp_ratio=4.0, sig_embed_factor=0.5,
                 sig_embed_class=None):
        super().__init__()
        self.in_dim = in_dim
        self.channels = channels
        self.patch_size = patch_size
        self.input_dims = (channels, in_dim, in_dim)

        dim = head_dim * num_heads

        self.pos_embed = get_pos_embed(in_dim, patch_size, dim)
        self.x_embed = PatchEmbed(patch_size, channels, dim, bias=True)
        self.sig_embed = (sig_embed_class or SigmaEmbedderSinCos)(
            dim, scaling_factor=sig_embed_factor
        )

        self.blocks = nn.ModuleList([
            DiTBlock(head_dim, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])

        self.final_norm = ModulatedLayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.final_linear = nn.Linear(dim, patch_size**2 * channels)
        self.init()

    def init(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize sigma embedding MLP:
        nn.init.normal_(self.sig_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.sig_embed.mlp[2].weight, std=0.02)

        # Zero-out output layers:
        nn.init.constant_(self.final_linear.weight, 0)
        nn.init.constant_(self.final_linear.bias, 0)

    def unpatchify(self, x):
        # (B, N, patchsize**2 * channels) -> (B, channels, H, W)
        patches = self.in_dim // self.patch_size
        return rearrange(x, 'b (ph pw) (psh psw c) -> b c (ph psh) (pw psw)',
                         ph=patches, pw=patches,
                         psh=self.patch_size, psw=self.patch_size)

    def forward(self, x, sigma):
        # (B, C, H, W), Union[(B, 1, 1, 1), ()] -> (B, C, H, W)
        # N = num_patches, D = dim = head_dim * num_heads
        x = self.x_embed(x) + self.pos_embed                      # (B, N, D)
        y = self.sig_embed(x.shape[0], sigma.squeeze())           # (B, D)
        for block in self.blocks:
            x = block(x, y)                                       # (B, N, D)
        x = self.final_linear(self.final_norm(x, y))              # (B, N, patchsize**2 * channels)
        return self.unpatchify(x)

# A simple embedding that works just as well as usual sinusoidal embedding
class SigmaEmbedderSinCos(nn.Module):
    def __init__(self, hidden_size, scaling_factor=0.5):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, batches, sigma):
        sig_embed = get_sigma_embeds(batches, sigma, self.scaling_factor) # (B, 2)
        return self.mlp(sig_embed)                                        # (B, D)
