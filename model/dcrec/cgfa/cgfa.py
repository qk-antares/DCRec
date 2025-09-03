
import logging
import torch

from model.dcrec.cgfa.modules.gconv import Siamese_GConv
from model.dcrec.cgfa.modules.affinity import Affinity
from model.dcrec.cgfa.modules.graph_pooling import DenseGraphPooling
from torch import Tensor, List

class CGFA(torch.nn.Module):
    """
    Cross Graph Feature Aggregation Module
    """

    def __init__(self, in_channels, out_channels):
        super(CGFA, self).__init__()

        self.logger = logging.getLogger(__name__)

        # 图内聚合
        self.intra_gconv = Siamese_GConv(in_channels, out_channels)
        # 跨图聚合
        self.affinity = Affinity(d=out_channels)
        self.cross_graph = torch.nn.Linear(out_channels * 2, out_channels)
        # 注意力图池化
        self.graph_pooling1 = DenseGraphPooling(node_dim=out_channels)
        self.graph_pooling2 = DenseGraphPooling(node_dim=out_channels)

    def compute_cgfa_embeddings(self, A_src: Tensor, emb_src: Tensor, n_nodes_src: Tensor, A_dst: Tensor, emb_dst: Tensor, n_nodes_dst: Tensor) -> List[Tensor]:
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
        n = A_src.shape[-1]

        # [batch_size, n, d]
        emb1, emb2 = self.intra_gconv([A_src, emb_src, True], [A_dst, emb_dst, True])

        # [batch_size, n, n]
        s = self.affinity(emb1, emb2)

        # 构造mask
        mask1 = torch.arange(n)[None, :] < n_nodes_src[:, None]  # [batch_size, n]
        mask2 = torch.arange(n)[None, :] < n_nodes_dst[:, None]  # [batch_size, n]

        # 对s做行softmax（G2的有效节点）
        mask2_expanded = mask2[:, None, :]  # [batch_size, 1, n]
        s_masked = s.masked_fill(~mask2_expanded, float('-1e9'))
        s_softmax = torch.softmax(s_masked, dim=-1)

        # 对s.transpose(1,2)做行softmax（G1的有效节点）
        mask1_expanded = mask1[:, None, :]  # [batch_size, 1, n]
        sT = s.transpose(1, 2)
        sT_masked = sT.masked_fill(~mask1_expanded, float('-1e9'))
        sT_softmax = torch.softmax(sT_masked, dim=-1)

        new_emb1 = self.cross_graph(torch.cat((emb1, torch.bmm(s_softmax, emb2)), dim=-1))
        new_emb2 = self.cross_graph(torch.cat((emb2, torch.bmm(sT_softmax, emb1)), dim=-1))

        global_emb1 = self.graph_pooling1(new_emb1, mask1)
        global_emb2 = self.graph_pooling2(new_emb2, mask2)

        return global_emb1, global_emb2
