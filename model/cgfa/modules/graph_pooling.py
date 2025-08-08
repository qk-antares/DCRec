import torch

class DenseGraphPooling(torch.nn.Module):
    def __init__(self, node_dim):
        """
        :param node_dim: node emb dim
        """
        super(DenseGraphPooling, self).__init__()
        self.node_dim = node_dim
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.node_dim, self.node_dim))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, emb, mask=None):
        """
        Making a forward propagation pass to create a graph level representation.
        :param emb: [batch_size, n, d] Result of the GNN.
        :param mask: [batch_size, n] Mask matrix indicating the valid nodes for each graph.
        :return representation: [batch_size, d] A graph level representation matrix.
        """
        B, N, _ = emb.size()

        if mask is not None:
            # 统计每个图的有效节点数 [B, 1]
            num_nodes = mask.view(B, N).sum(dim=1).unsqueeze(-1)
            # 有效节点的特征均值 [B, F]
            mean = emb.sum(dim=1) / num_nodes.to(emb.dtype)
        else:
            # 所有节点的均值 [B, F]
            mean = emb.mean(dim=1)

        # 全局特征变换 [B, F]
        transformed_global = torch.tanh(torch.mm(mean, self.weight_matrix))

        # 计算每个节点的注意力系数 [B, N, 1]
        weight = torch.sigmoid(torch.matmul(emb, transformed_global.unsqueeze(-1)))
        # 加权节点特征 [B, N, F]
        global_emb = weight * emb

        if mask is not None:
            # 屏蔽无效节点
            global_emb = global_emb * mask.view(B, N, 1).to(emb.dtype)

        # 聚合得到图级别表示 [B, F]
        return global_emb.sum(dim=1)