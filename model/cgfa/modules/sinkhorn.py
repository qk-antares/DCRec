import torch.nn as nn
from torch import Tensor
import pygmtools as pygm


class Sinkhorn(nn.Module):
    """
    Sinkhorn algorithm turns the input matrix into a bi-stochastic matrix.

    Sinkhorn algorithm firstly applies an exp function with temperature tau:
    S[i][j] = exp(S[i][j] / tau)
    And then turns the matrix into doubly-stochastic matrix by iterative row- and column-wise normalization:

    :param max_iter: maximum iterations (default: 10)
    :param tau: the hyper parameter tau controlling the temperature (default: 1)
    :param epsilon: a small number for numerical stability (default: 1e-4)
    :param log_forward: apply log-scale computation for better numerical stability (default: True)
    :param batched_operation: apply batched_operation for better efficiency, but may cause issues for back-propagation (default: False)

    .. note::
        tau is an important hyper parameter to be set for Sinkhorn algorithm. tau controls the distance between
        the predicted doubly-stochastic matrix, and the discrete permutation matrix computed by Hungarian algorithm. 
        Given a small tau, Sinkhorn performs more closely to Hungarian, at the cost of slower convergence speed and reduced numerical stability.

    .. note::
        Setting batched_operation=True may be preferred when you are doing inference with this module and do not
        need the gradient.
    """
    def __init__(self, max_iter: int=10, tau: float=1., epsilon: float=1e-4, batched_operation: bool=False):
        super(Sinkhorn, self).__init__()
        self.max_iter = max_iter
        self.tau = tau
        self.epsilon = epsilon
        # batched operation may cause instability in backward computation, but will boost computation.
        self.batched_operation = batched_operation

    def forward(self, s: Tensor, nrows: Tensor=None, ncols: Tensor=None, dummy_row: bool=False) -> Tensor:
        """
        :param s: [batch_size, n1, n2] input 3d tensor
        :param nrows: [batch_size] number of objects in dim1
        :param ncols: [batch_size] number of objects in dim2
        :param dummy_row: whether to add dummy rows (rows whose elements are all 0) to pad the matrix to square matrix. (default: False)
        :return: [batch_size, n1, n2] the computed doubly-stochastic matrix

        .. note::
            We support batched instances with different number of nodes, therefore ``nrows`` and ``ncols`` are
            required to specify the exact number of objects of each dimension in the batch. If not specified, we assume
            the batched matrices are not padded.

        .. note::
            The original Sinkhorn algorithm only works for square matrices. To handle cases where the graphs to be
            matched have different number of nodes, it is a common practice to add dummy rows to construct a square
            matrix. After the row and column normalizations, the padded rows are discarded.

        .. note::
            We assume row number <= column number. If not, the input matrix will be transposed.
        """
        return pygm.sinkhorn(s, n1=nrows, n2=ncols, dummy_row=dummy_row, max_iter=self.max_iter, tau=self.tau, batched_operation=self.batched_operation, backend='pytorch')
