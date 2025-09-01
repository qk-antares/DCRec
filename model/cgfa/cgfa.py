
import torch

from model.cgfa.modules.gconv import Siamese_GConv
from model.cgfa.modules.affinity import Affinity
from model.cgfa.modules.sinkhorn import Sinkhorn
from model.cgfa.modules.graph_pooling import DenseGraphPooling
from torch import Tensor, List

class CGFA(torch.nn.Module):
    """
    Cross Graph Feature Aggregation Module
    """

    def __init__(self, in_channels, out_channels, max_iter, tau):
        super(CGFA, self).__init__()

        # 图内聚合
        self.intra_gconv = Siamese_GConv(in_channels, out_channels)
        # 跨图聚合
        self.affinity = Affinity(d=out_channels)
        self.sinkhorn = Sinkhorn(max_iter=max_iter, tau=tau)
        self.cross_graph = torch.nn.Linear(out_channels * 2, out_channels)
        # 注意力图池化
        self.graph_pooling1 = DenseGraphPooling(node_dim=out_channels)
        self.graph_pooling2 = DenseGraphPooling(node_dim=out_channels)

    def forward(self, A_src: Tensor, emb_src: Tensor, n_nodes_src: Tensor, A_dst: Tensor, emb_dst: Tensor, n_nodes_dst: Tensor) -> List[Tensor]:
        """
        Forward pass for the CGFA module.

        :param A_src: [batch_size, n, n] {0,1} adjacency matrix of src graphs
        :param emb_src: [batch_size, n, d] node features of src graphs
        :param n_nodes_src: [batch_size] number of nodes in src graphs
        :param A_dst: [batch_size, n, n] {0,1} adjacency matrix of dst graphs
        :param emb_dst: [batch_size, n, d] node features of dst graphs
        :param n_nodes_dst: [batch_size] number of nodes in dst graphs

        :return: Output global emb of g1 and g2
        """
        emb1, emb2 = self.intra_gconv([A_src, emb_src, True], [A_dst, emb_dst, True])
        s = self.affinity(emb1, emb2)
        s = self.sinkhorn(s, n_nodes_src, n_nodes_dst, dummy_row=True)

        new_emb1 = self.cross_graph(torch.cat((emb1, torch.bmm(s, emb2)), dim=-1))
        new_emb2 = self.cross_graph(torch.cat((emb2, torch.bmm(s.transpose(1, 2), emb1)), dim=-1))

        global_emb1 = self.graph_pooling1(new_emb1)
        global_emb2 = self.graph_pooling2(new_emb2)

        return global_emb1, global_emb2
