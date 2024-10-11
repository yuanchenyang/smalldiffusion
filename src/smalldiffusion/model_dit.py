import torch
import numpy as np
from torch import nn
from einops import rearrange, repeat
from .model import ModelMixin, Attention, SigmaEmbedderSinCos, CondSequential

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
                 head_dim=64, num_heads=6, mlp_ratio=4.0,
                 sig_embed=None, cond_embed=None):
        super().__init__()
        self.in_dim = in_dim
        self.channels = channels
        self.patch_size = patch_size
        self.input_dims = (channels, in_dim, in_dim)

        dim = head_dim * num_heads

        self.pos_embed = get_pos_embed(in_dim, patch_size, dim)
        self.x_embed = PatchEmbed(patch_size, channels, dim, bias=True)
        self.sig_embed = sig_embed or SigmaEmbedderSinCos(dim)
        self.cond_embed = cond_embed

        self.blocks = CondSequential(*[
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

    def forward(self, x, sigma, cond=None):
        # x: (B, C, H, W), sigma: Union[(B, 1, 1, 1), ()], cond: (B, *)
        # returns: (B, C, H, W)
        # N = num_patches, D = dim = head_dim * num_heads
        x = self.x_embed(x) + self.pos_embed            # (B, N, D)
        y = self.sig_embed(x.shape[0], sigma.squeeze()) # (B, D)
        if self.cond_embed is not None:
            assert cond is not None and x.shape[0] == cond.shape[0], \
                'Conditioning must have same batches as x!'
            y += self.cond_embed(cond)                  # (B, D)
        x = self.blocks(x, y)                           # (B, N, D)
        x = self.final_linear(self.final_norm(x, y))    # (B, N, patchsize**2 * channels)
        return self.unpatchify(x)
