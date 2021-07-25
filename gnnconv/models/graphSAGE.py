import torch
import torch_scatter
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from torch.nn import Parameter, Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax

class GraphSage(MessagePassing):

    def __init__(self, in_channels, out_channels, normalize=True,
                 bias=False, **kwargs):
        super(GraphSage, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        self.lin_l = None
        self.lin_r = None

        self.lin_l = nn.Linear(in_features=self.in_channels,
                               out_features=self.out_channels,
                               bias=bias)

        self.lin_r = nn.Linear(in_features=self.in_channels,
                               out_features=self.out_channels,
                               bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x, edge_index, size=None):
        out = self.propagate(edge_index=edge_index,
                             size=size,
                             x=(x, x))
        out = self.lin_r(out)
        out += self.lin_l(x)
        out = F.normalize(out, p=2., dim=-1)
        return out

    def message(self, x_j):
        out = x_j

        return out

    def aggregate(self, inputs, index, dim_size=None):
        node_dim = self.node_dim  # -2로 잡힘
        out = torch_scatter.scatter(src=inputs,
                                    index=index,
                                    dim=node_dim,
                                    reduce='mean')
        return out
