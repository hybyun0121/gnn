import torch
import torch_scatter
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from torch.nn import Parameter, Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax

class GAT(MessagePassing):

    def __init__(self, in_channels, out_channels, heads=2,
                 negative_slope=0.2, dropout=0., **kwargs):
        super(GAT, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin_l = None
        self.lin_r = None
        self.att_l = None
        self.att_r = None

        self.lin_l = nn.Linear(in_features=self.in_channels,
                               out_features=self.out_channels * self.heads,
                               bias=False)

        self.lin_r = self.lin_l

        self.att_l = nn.Parameter(Tensor(1, self.heads, self.out_channels))
        self.att_r = nn.Parameter(Tensor(1, self.heads, self.out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_l.weight)
        nn.init.xavier_uniform_(self.lin_r.weight)
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)

    def forward(self, x, edge_index, size=None):
        H, C = self.heads, self.out_channels

        out = None

        x_l = self.lin_l(x).view(-1, H, C)
        x_r = self.lin_r(x).view(-1, H, C)

        alpha_l = x_l * self.att_l
        alpha_r = x_r * self.att_r

        out = self.propagate(edge_index,
                             size=size,
                             x=(x_l, x_r),
                             alpha=(alpha_l, alpha_r))

        # concat
        out = out.view(-1, H * C)
        # mean
        # out = out.mean(dim=1)

        return out

    def message(self, x_j, alpha_j, alpha_i, index, ptr, size_i):
        alpha = alpha_i + alpha_j
        alpha = F.leaky_relu(alpha, negative_slope=self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout)
        out = x_j * alpha

        return out

    def aggregate(self, inputs, index, dim_size=None):
        node_dim = self.node_dim

        out = torch_scatter.scatter(src=inputs,
                                    index=index,
                                    dim=node_dim,
                                    dim_size=dim_size,
                                    reduce='sum')
        return out