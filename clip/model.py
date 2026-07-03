"""
CLIP with class embeddings on CIFAR-100 @ 32x32.       ***TASK: implement ClassEncoder.***
Self-contained (no cross-imports).

CLIP jointly trains an image encoder and a "text" encoder into a shared embedding space,
matching real (image, text) pairs against the mismatched pairs in a batch. Here the "text"
is a class label, so the text encoder becomes a CLASS encoder:

    int label -> nn.Embedding -> small MLP -> projection      (CLIP's "hypernetwork":
                                                               a classifier weight per class)

The image encoders (PROVIDED) and the CLIP wrapper are given; you implement the class
encoder. Pick the image encoder with `vision=`: "resnet" (CIFAR ResNet-18) or "vit"
(CLIP-style Vision Transformer).
"""
import numpy as np
import torch
import torch.nn as nn


# =============================== ResNet image encoder (PROVIDED) ================ #
def conv3x3(cin, cout, stride=1):
    return nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = conv3x3(cin, cout, stride)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = conv3x3(cout, cout)
        self.bn2 = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or cin != cout:
            self.downsample = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        idt = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + idt)


class ResNetVision(nn.Module):
    """CIFAR ResNet-18 backbone (no classifier) + a linear projection to the shared space."""

    def __init__(self, embed_dim, depths=(2, 2, 2, 2), widths=(64, 128, 256, 512)):
        super().__init__()
        self.inplanes = widths[0]
        self.stem = nn.Sequential(conv3x3(3, widths[0]), nn.BatchNorm2d(widths[0]),
                                  nn.ReLU(inplace=True))
        self.layer1 = self._stage(widths[0], depths[0], 1)
        self.layer2 = self._stage(widths[1], depths[1], 2)
        self.layer3 = self._stage(widths[2], depths[2], 2)
        self.layer4 = self._stage(widths[3], depths[3], 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(widths[3], embed_dim)

    def _stage(self, planes, blocks, stride):
        layers = [BasicBlock(self.inplanes, planes, stride)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        return self.proj(torch.flatten(self.avgpool(x), 1))


# ================================ ViT image encoder (PROVIDED) ================= #
class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class Attention(nn.Module):
    """Multi-head self-attention, written out (softmax(QK^T / sqrt(d)) V)."""

    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.heads, C // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, L, C)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.ln_2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), QuickGELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class VisionTransformer(nn.Module):
    """CLIP-style ViT. forward(image [N,3,H,W]) -> [N, output_dim]."""

    def __init__(self, image_size, patch_size, width, layers, heads, output_dim, mlp_ratio=4):
        super().__init__()
        grid = image_size // patch_size
        n_tokens = grid * grid + 1
        self.patch_embed = nn.Conv2d(3, width, patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(n_tokens, width))
        self.ln_pre = nn.LayerNorm(width)
        self.blocks = nn.ModuleList([Block(width, heads, mlp_ratio) for _ in range(layers)])
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.class_embedding.view(1, 1, -1).expand(x.shape[0], 1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.positional_embedding
        x = self.ln_pre(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_post(x[:, 0])
        return x @ self.proj


# ============================== class encoder (TASK) =========================== #
class ClassEncoder(nn.Module):
    """int label -> embedding -> small MLP -> projection into the shared space."""

    def __init__(self, num_classes, embed_dim, width=256):
        super().__init__()
        # TODO: build
        #   embedding : nn.Embedding(num_classes, width)
        #   mlp       : Linear(width, width) -> GELU -> Linear(width, width)
        #   proj      : nn.Linear(width, embed_dim)
        self.embedding = nn.Embedding(num_classes, width)
        self.mlp = nn.Sequential(
            nn.Linear(width, width),
            QuickGELU(),
            nn.Linear(width, width)
        )
        self.proj = nn.Linear(width, embed_dim)

    def forward(self, labels):
        # labels: int [N].  TODO: e = embedding(labels); return proj(mlp(e))   [N, embed_dim]
        e = self.Embedding(labels)
        mlp = self.mlp(e)
        return self.proj(mlp)

# ================================= CLIP (PROVIDED) ============================= #
class CLIP(nn.Module):
    def __init__(self, vision="resnet", embed_dim=256, num_classes=100,
                 vit_width=384, vit_layers=6, vit_heads=6,
                 patch_size=4, image_resolution=32, class_width=256):
        super().__init__()
        if vision == "resnet":
            self.visual = ResNetVision(embed_dim)
        elif vision == "vit":
            self.visual = VisionTransformer(image_resolution, patch_size,
                                            vit_width, vit_layers, vit_heads, embed_dim)
        else:
            raise ValueError("vision must be 'resnet' or 'vit'")
        self.class_encoder = ClassEncoder(num_classes, embed_dim, width=class_width)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_image(self, image):
        return self.visual(image)

    def encode_class(self, labels):
        return self.class_encoder(labels)

    def forward(self, image, labels):
        """Return (image_embeddings, class_embeddings) -- L2-norm is applied in the loss."""
        return self.encode_image(image), self.encode_class(labels)


def build_model(vision="resnet", embed_dim=256, num_classes=100, **kwargs):
    return CLIP(vision=vision, embed_dim=embed_dim, num_classes=num_classes, **kwargs)
