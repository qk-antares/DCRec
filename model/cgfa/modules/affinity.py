import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch import Tensor
import math


class Affinity(nn.Module):
    """
    Affinity Layer to compute a learnable affinity (similarity) matrix between two sets of node features.

    This layer computes the affinity matrix M between feature tensors X and Y as:
        M = X * A * Y^T
    where A is a learnable weight matrix of shape (d, d), and X, Y are input feature tensors.

    Args:
        d (int): Feature dimension of input tensors X and Y.

    Input:
        X: Tensor of shape [batch_size, n, d] (node features for set 1)
        Y: Tensor of shape [batch_size, m, d] (node features for set 2)

    Output:
        M: Tensor of shape [batch_size, n, m] (affinity matrix between X and Y)
    """
    def __init__(self, d):
        super(Affinity, self).__init__()
        self.d = d
        # Learnable weight matrix for feature transformation
        self.A = Parameter(Tensor(self.d, self.d))
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize A with uniform distribution and add identity for stability
        stdv = 1. / math.sqrt(self.d)
        self.A.data.uniform_(-stdv, stdv)
        self.A.data += torch.eye(self.d)

    def forward(self, X, Y):
        """
        Args:
            X: [batch_size, n, d] input node features
            Y: [batch_size, m, d] input node features
        Returns:
            M: [batch_size, n, m] affinity matrix
        """
        assert X.shape[2] == Y.shape[2] == self.d
        # Transform X with learnable matrix A
        M = torch.matmul(X, self.A)  # [batch_size, n, d]
        # Compute affinity matrix: (X*A) * Y^T
        M = torch.matmul(M, Y.transpose(1, 2))  # [batch_size, n, m]
        return M

class AffinityInp(nn.Module):
    """
    Affinity Layer to compute the (non-learnable) inner product affinity matrix between two sets of node features.

    This layer computes the affinity matrix M as:
        M = X * Y^T
    (i.e., the standard inner product between feature vectors, without any learnable parameters)

    Args:
        d (int): Feature dimension of input tensors X and Y.

    Input:
        X: Tensor of shape [batch_size, n, d] (node features for set 1)
        Y: Tensor of shape [batch_size, m, d] (node features for set 2)

    Output:
        M: Tensor of shape [batch_size, n, m] (affinity matrix between X and Y)
    """
    def __init__(self, d):
        super(AffinityInp, self).__init__()
        self.d = d

    def forward(self, X, Y):
        """
        Args:
            X: [batch_size, n, d] input node features
            Y: [batch_size, m, d] input node features
        Returns:
            M: [batch_size, n, m] affinity matrix (inner product)
        """
        assert X.shape[2] == Y.shape[2] == self.d
        # Compute standard inner product affinity matrix
        M = torch.matmul(X, Y.transpose(1, 2))  # [batch_size, n, m]
        return M

# Difference between Affinity and AffinityInp:
# - Affinity: computes a learnable affinity matrix using a trainable weight matrix A (M = X * A * Y^T), allowing the model to learn a task-specific similarity metric between features.
# - AffinityInp: computes a fixed (non-learnable) inner product affinity matrix (M = X * Y^T), i.e., standard dot product similarity, with no learnable parameters.