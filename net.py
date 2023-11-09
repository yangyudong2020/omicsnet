import torch
import torch.nn as nn
import torchvision
import numpy as np
from collections import OrderedDict

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class FeatureNet(nn.Module):
    def __init__(self, transformer_width=512, transformer_heads=4, transformer_layers=36, embed_dim=512):
        super(FeatureNet, self).__init__()

        self.init_embed = 2884
        self.context_length = 110
        self.fc = nn.Linear(self.init_embed, 512)
        self.transformer_width = transformer_width
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, transformer_width))
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.ln_final = LayerNorm(transformer_width)
    
    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask    

    @property
    def dtype(self):
        return self.fc.weight.dtype
    
    def forward(self, x):
        # print('positional_embedding', self.positional_embedding.type(self.dtype).shape)
        x = x.reshape(-1, self.init_embed)
        x = self.fc(x)
        x = x.reshape(-1, self.context_length, self.transformer_width)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        # print('transformer shape', x.shape)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # print('ln_final shape', x.shape)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), 0] @ self.text_projection

        return x


# Testing the model
model = FeatureNet().cuda()
input_tensor = torch.randn(8, 110, 2884).cuda()
output = model(input_tensor)
print(output.shape)

# unet encoder shape torch.Size([6, 32, 640, 384])
# unet encoder shape torch.Size([6, 64, 320, 192])
# unet encoder shape torch.Size([6, 128, 160, 96])
# unet encoder shape torch.Size([6, 256, 80, 48])
# unet encoder shape torch.Size([6, 512, 40, 24])
# unet encoder shape torch.Size([6, 512, 20, 12])
# unet encoder shape torch.Size([6, 512, 10, 6])

# resnet shape torch.Size([8, 64, 40, 24])
# resnet shape torch.Size([8, 64, 40, 24])
# resnet shape torch.Size([8, 64, 40, 24])
# resnet shape torch.Size([8, 64, 20, 12])
# resnet shape torch.Size([8, 512, 20, 12])
# resnet shape torch.Size([8, 512, 10, 6])
