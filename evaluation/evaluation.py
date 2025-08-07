import math

import numpy as np
import torch
from tqdm import tqdm

from utils.dataset import NegativeSampler


def eval_edge_prediction(model, data, n_neighbors, negative_sampler: NegativeSampler, n_test_neg, batch_size=256):
  mrr_list, recall_10_list, recall_20_list = [], [], []
  with torch.no_grad():
    model = model.eval()
    # While usually the test batch size is as big as it fits in memory, here we keep it the same
    # size as the training batch size, since it allows the memory to be updated more frequently,
    # and later test batches to access information from interactions in previous test batches
    # through the memory
    TEST_BATCH_SIZE = batch_size
    num_test_instance = len(data.sources)
    num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

    # 添加进度条
    progress_bar = tqdm(range(num_test_batch), desc='Evaluating', unit='batch')
    
    for k in progress_bar:
      s_idx = k * TEST_BATCH_SIZE
      e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
      sources_batch = data.sources[s_idx:e_idx]
      destinations_batch = data.destinations[s_idx:e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      edge_idxs_batch = data.edge_idxs[s_idx: e_idx]

      # 生成负样本，现在返回shape为[n_test_neg, batch_size]
      negatives_batch = negative_sampler.generate_negative_samples(sources_batch, timestamps_batch, n_test_neg)

      # 计算正样本和负样本的原始分数（用于计算MRR,Recall@10，@Recall@20）
      pos_scores, neg_scores = model.compute_edge_scores(sources_batch, destinations_batch, negatives_batch, 
                                                         timestamps_batch, edge_idxs_batch, n_neighbors)

      # 计算每个用户的MRR和Recall
      size = len(sources_batch)
      for i in range(size):
        # 对于每个用户，获取正样本分数和所有负样本分数
        pos_score = pos_scores[i].item()  # 正样本分数：标量
        neg_scores_user = neg_scores[:, i].cpu().numpy()  # 该用户的所有负样本分数：[n_test_neg]
        
        # 构建候选列表：正样本 + 负样本
        all_scores = np.concatenate([[pos_score], neg_scores_user])  # [1 + n_test_neg]
        
        # 按分数从高到低排序，获取排名（rank从1开始）
        sorted_indices = np.argsort(all_scores)[::-1]  # 降序排列的索引
        pos_rank = np.where(sorted_indices == 0)[0][0] + 1  # 正样本的排名（索引0对应正样本）
        
        # 计算MRR
        mrr = 1.0 / pos_rank
        mrr_list.append(mrr)
        
        # 计算Recall@10和Recall@20
        recall_10 = 1.0 if pos_rank <= 10 else 0.0
        recall_20 = 1.0 if pos_rank <= 20 else 0.0
        recall_10_list.append(recall_10)
        recall_20_list.append(recall_20)
      
      # 更新进度条显示当前的评估指标
      if len(mrr_list) > 0:
        current_mrr = np.mean(mrr_list)
        current_recall_10 = np.mean(recall_10_list)
        current_recall_20 = np.mean(recall_20_list)
        progress_bar.set_postfix({
          'MRR': f'{current_mrr:.4f}',
          'R@10': f'{current_recall_10:.4f}',
          'R@20': f'{current_recall_20:.4f}'
        })

  return np.mean(mrr_list), np.mean(recall_10_list), np.mean(recall_20_list)

