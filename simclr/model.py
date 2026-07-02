    """
    SimCLR model = base encoder f(.) + projection head g(.).   ***TASK: fill in the TODOs.***
    Self-contained (no cross-imports).

    * `ResNetEncoder` (PROVIDED) -- CIFAR ResNet-18 backbone, outputs the 512-d h.
    * `ProjectionHead` -- the small MLP g(.) = W2 . ReLU(BN(W1 h)) -- YOU implement it.
    * `SimCLRModel` -- `forward` returns z (for the loss), `features` returns h (for kNN).
    """
    import torch
    import torch.nn as nn


    def conv3x3(cin, cout, stride=1):
        return nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)


    class BasicBlock(nn.Module):
        """Two 3x3 convs with a residual connection (the ResNet-18 block). PROVIDED."""

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


    DEPTHS = {"resnet18": (2, 2, 2, 2), "resnet34": (3, 4, 6, 3)}


    class ResNetEncoder(nn.Module):
        """CIFAR ResNet-18 backbone (3x3 s1 stem, no max-pool) -> global avg-pool. PROVIDED."""

        def __init__(self, depths=(2, 2, 2, 2), widths=(64, 128, 256, 512)):
            super().__init__()
            self.inplanes = widths[0]
            self.stem = nn.Sequential(conv3x3(3, widths[0]), nn.BatchNorm2d(widths[0]),
                                    nn.ReLU(inplace=True))
            self.layer1 = self._stage(widths[0], depths[0], 1)
            self.layer2 = self._stage(widths[1], depths[1], 2)
            self.layer3 = self._stage(widths[2], depths[2], 2)
            self.layer4 = self._stage(widths[3], depths[3], 2)
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.feature_dim = widths[3]

        def _stage(self, planes, blocks, stride):
            layers = [BasicBlock(self.inplanes, planes, stride)]
            self.inplanes = planes
            for _ in range(1, blocks):
                layers.append(BasicBlock(planes, planes, 1))
            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.stem(x)
            x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
            return torch.flatten(self.avgpool(x), 1)


    class ProjectionHead(nn.Module):
        """2-layer MLP g(.): Linear -> BN -> ReLU -> Linear. Output is NOT normalised here."""

        def __init__(self, in_dim, hidden_dim, out_dim):
            super().__init__()
            # TODO: build the MLP:
            #   Linear(in_dim, hidden_dim, bias=False) -> BatchNorm1d(hidden_dim)
            #     -> ReLU -> Linear(hidden_dim, out_dim)
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim)
            )
            
        def forward(self, h):
            # TODO: return z = g(h)   [N, out_dim]
            z = self.mlp(h)
            return z

    class SimCLRModel(nn.Module):
        def __init__(self, arch="resnet18", proj_dim=128, proj_hidden=None):
            super().__init__()
            self.encoder = ResNetEncoder(depths=DEPTHS[arch])             # provided
            self.feature_dim = self.encoder.feature_dim
            hidden = proj_hidden or self.feature_dim
            self.projector = ProjectionHead(self.feature_dim, hidden, proj_dim)

        def features(self, x):
            features = self.encoder(x) # Batch_size, Feature_dim
            return features

        def forward(self, x):
            z = self.projector(self.features(x))
            return z


    def build_model(arch="resnet18", proj_dim=128):
        return SimCLRModel(arch=arch, proj_dim=proj_dim)