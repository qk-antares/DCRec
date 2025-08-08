import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, List, Union

class GConv(nn.Module):
    """
    Graph Convolutional Layer which is inspired and developed based on Graph Convolutional Network (GCN).

    :param in_features: the dimension of input node features
    :param out_features: the dimension of output node features
    """

    def __init__(self, in_features: int, out_features: int):
        super(GConv, self).__init__()
        self.num_inputs = in_features
        self.num_outputs = out_features
        self.a_fc = nn.Linear(self.num_inputs, self.num_outputs)
        self.u_fc = nn.Linear(self.num_inputs, self.num_outputs)

    def forward(self, A: Tensor, x: Tensor, norm: bool=True) -> Tensor:
        """
        Forward computation of graph convolution network.

        :param A: [batch_size, n, n] {0,1} adjacency matrix. n is the max number of nodes
        :param x: [batch_size, n, d] input node embedding.
        :param norm: normalize connectivity matrix or not

        :return: [batch_size, n, d] new node embedding
        """
        if norm is True:
            A = F.normalize(A, p=1, dim=-1)
        ax = self.a_fc(x)
        ux = self.u_fc(x)
        x = torch.bmm(A, F.relu(ax)) + F.relu(ux) # has size (bs, N, num_outputs)
        return x

class Siamese_GConv(nn.Module):
    """
    Siamese Gconv neural network for processing arbitrary number of graphs.

    :param in_channels: the dimension of input node features
    :param out_channels: the dimension of output node features
    """

    def __init__(self, in_channels, out_channels):
        super(Siamese_GConv, self).__init__()
        self.gconv = GConv(in_channels, out_channels)

    def forward(self, g1: Tuple[Tensor, Tensor, bool], *args) -> Union[Tensor, List[Tensor]]:
        """
        Forward computation of Siamese GConv.

        :param g1: The first graph, which is a tuple of 
        [batch_size, n, n] {0,1} adjacency matrix,
        [batch_size, n, d] input node embedding, 
        normalize connectivity matrix or not)
        :param args: Other graphs
        
        :return: A list of tensors composed of new node embeddings [batch_size, n, d]
        """
        emb1 = self.gconv(*g1)
        if len(args) == 0:
            return emb1
        else:
            embs = [emb1]
            for g in args:
                embs.append(self.gconv(*g))
            return embs