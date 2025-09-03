import logging
import numpy as np
import torch

from model.dcrec.cgfa.cgfa import CGFA
from model.dcrec.tgn.tgn import TGN
from utils.utils import MergeLayer


class DCRec(torch.nn.Module):
    """
    Dual-Channel Recommendation System (DCRec)
    融合TGN的时序协同信息和CGFA的侧信息交互
    """
    def __init__(self, neighbor_finder, node_features, edge_features, device, 
                 n_layers=2, n_heads=2, dropout=0.1, use_memory=False,
                 memory_update_at_start=True, message_dimension=100,
                 memory_dimension=500, embedding_module_type="graph_attention",
                 message_function="mlp",
                 mean_time_shift_src=0, std_time_shift_src=1, mean_time_shift_dst=0,
                 std_time_shift_dst=1, n_neighbors=None, aggregator_type="last",
                 memory_updater_type="gru",
                 use_destination_embedding_in_message=False,
                 use_source_embedding_in_message=False,
                 dyrep=False,
                 # CGFA相关参数
                 cgfa_in_channels=None, cgfa_out_channels=128, 
                 # 融合网络参数
                 fusion_hidden_dim=256, final_hidden_dim=128,
                 # TODO: 数据集，后面要去掉
                 dataset=None):
        super(DCRec, self).__init__()
        
        self.logger = logging.getLogger(__name__)

        self.dataset = dataset

        self.device = device
        
        # TGN分支 - 处理时序协同信息
        self.tgn = TGN(
            neighbor_finder=neighbor_finder, 
            node_features=node_features, 
            edge_features=edge_features, 
            device=device,
            n_layers=n_layers,
            n_heads=n_heads, 
            dropout=dropout, 
            use_memory=use_memory,
            message_dimension=message_dimension, 
            memory_dimension=memory_dimension,
            memory_update_at_start=memory_update_at_start,
            embedding_module_type=embedding_module_type,
            message_function=message_function,
            aggregator_type=aggregator_type,
            memory_updater_type=memory_updater_type,
            n_neighbors=n_neighbors,
            mean_time_shift_src=mean_time_shift_src, 
            std_time_shift_src=std_time_shift_src,
            mean_time_shift_dst=mean_time_shift_dst, 
            std_time_shift_dst=std_time_shift_dst,
            use_destination_embedding_in_message=use_destination_embedding_in_message,
            use_source_embedding_in_message=use_source_embedding_in_message,
            dyrep=dyrep
        )
        
        # 获取TGN输出维度
        self.tgn_output_dim = node_features.shape[1]
        
        # CGFA分支 - 处理侧信息交互
        self.cgfa = CGFA(
            in_channels=cgfa_in_channels,
            out_channels=cgfa_out_channels,
        )
        self.cgfa_output_dim = cgfa_out_channels
        
        # 融合网络
        self.user_fusion_mlp = MergeLayer(self.tgn_output_dim, self.cgfa_output_dim,
                                          fusion_hidden_dim,
                                          final_hidden_dim)

        self.item_fusion_mlp = MergeLayer(self.tgn_output_dim, self.cgfa_output_dim,
                                          fusion_hidden_dim,
                                          final_hidden_dim)

        # 最终预测网络
        self.affinity_score = MergeLayer(final_hidden_dim, final_hidden_dim,
                                         final_hidden_dim,
                                         1)
        
    def __init_memory__(self):
        self.tgn.memory.__init_memory__()
    
    def set_neighbor_finder(self, neighbor_finder):
        self.tgn.neighbor_finder = neighbor_finder
        self.tgn.embedding_module.neighbor_finder = neighbor_finder

    def clear_all_messages(self):
        self.tgn.memory.clear_all_messages()

    def detach_memory(self):
        self.tgn.memory.detach_memory()

    def compute_edge_scores(self, source_nodes, destination_nodes, negative_nodes, 
                            source_counts, source_embs, source_adjs,
                            dest_counts, dest_embs, dest_adjs,
                            neg_counts, neg_embs, neg_adjs,
                            edge_times, edge_idxs, n_neighbors=20):
        """
        Compute raw scores (logits) for edges between sources and destination and between sources and
        negatives. This is used for BPR loss computation.
        :param destination_nodes [batch_size]: destination ids
        :param negative_nodes [n_neg, batch_size]: ids of negative sampled destinations
        :param edge_times [batch_size]: timestamp of interaction
        :param edge_idxs [batch_size]: index of interaction
        :param n_neighbors [scalar]: number of temporal neighbor to consider in each convolutional
        layer
        :return: Raw scores for both the positive and negative edges
        """
        n_samples = len(source_nodes)
        n_neg = negative_nodes.shape[0]

        # 1. get source_embedding1 and target_embeddings1 by tgn
        source_node_embedding1, destination_node_embedding1, negative_node_embedding1 = self.tgn.compute_temporal_embeddings(
        source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, n_neighbors)

        # Repeat source embeddings (1 + n_neg) times: once for positive, n_neg times for negatives
        source_embedding1 = source_node_embedding1.repeat(1 + n_neg, 1)
        # Flatten negative embeddings from [n_neg, batch_size, feat_dim] to [n_neg * batch_size, feat_dim]
        negative_node_embedding_flat = negative_node_embedding1.view(-1, negative_node_embedding1.shape[-1])
        # Concatenate destination and negative embeddings
        target_embeddings1 = torch.cat([destination_node_embedding1, negative_node_embedding_flat], dim=0)

        # 2. get source_embedding2 and target_embeddings2 by cgfa
        source_adjs = torch.FloatTensor(source_adjs).to(self.device)
        source_embs = torch.FloatTensor(source_embs).to(self.device)
        source_counts = torch.LongTensor(source_counts).to(self.device)


        dest_embs = torch.FloatTensor(dest_embs).to(self.device)
        dest_adjs = torch.FloatTensor(dest_adjs).to(self.device)
        
        dest_counts = torch.LongTensor(dest_counts).to(self.device)
        
        # 通过CGFA获取图级别表示
        source_graph_emb, dest_graph_emb = self.cgfa(

        )
        


        source_embedding2, target_embeddings2 = self.cgfa.compute_cgfa_embeddings(
            A_src=source_adjs,
            emb_src=source_embs,
            n_nodes_src=source_counts,
            A_dst=dest_adjs,
            emb_dst=dest_embs,
            n_nodes_dst=dest_counts

            np.repeat(source_nodes, repeats=(1 + n_neg), axis=0), np.concatenate([destination_nodes, negative_nodes.reshape(-1)], axis=0)
        )

        # 3. get score
        # 通过融合MLP
        source_fused = self.user_fusion_mlp(source_embedding1, source_embedding2)
        target_fused = self.item_fusion_mlp(target_embeddings1, target_embeddings2)

        # 最终预测
        score = self.affinity_score(source_fused, target_fused).squeeze(dim=-1)
        pos_score = score[:n_samples]
        neg_score = score[n_samples:]
        
        neg_score = neg_score.view(n_neg, n_samples)

        return pos_score, neg_score
