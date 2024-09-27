import torch
from model.embedder import *
import numpy as np
import torch.nn as nn


class GeometryNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            d_in,
            d_out,
            dims,
            geometric_init=True,
            bias=1.0,
            skip_in=(),
            weight_norm=True,
            multires=0,
    ):
        super().__init__()
        dims = [d_in] + dims
        self.feature_vector_size = feature_vector_size
        self.embed_fn = None
        self.multires = multires
        self.bias = bias
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in

        self.softplus = nn.Softplus(beta=100)
        self.scaling_layer = nn.Sequential(nn.Linear(256, 256), self.softplus,
                                           nn.Linear(256, 3))
        self.rotations_layer = nn.Sequential(nn.Linear(256, 256), self.softplus,
                                             nn.Linear(256, 4))
        self.opacity_layer = nn.Sequential(nn.Linear(256, 256), self.softplus,
                                           nn.Linear(256, 1))
        self.scale_ac = nn.Softplus(beta=100)
        self.rotations_ac = nn.functional.normalize
        self.opacity_ac = nn.Sigmoid()

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.out_layer = nn.Sequential(nn.Linear(256, 256), self.softplus,
                                       nn.Linear(256, 256), nn.Linear(256, 3))

    def forward(self, input):
        if self.embed_fn is not None:
            input = self.embed_fn(input)
        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)
            x = lin(x)
            if l < self.num_layers - 1:
                x = self.softplus(x)

        color = self.out_layer(x)
        scales = self.scaling_layer(x)
        rotations = self.rotations_layer(x)
        opacity = self.opacity_layer(x)

        return color, scales, rotations, opacity
