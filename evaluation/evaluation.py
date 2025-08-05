import math

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from utils.dataset import NegativeSampler


def eval_edge_prediction(model, data, n_neighbors, negative_sampler: NegativeSampler, num_neg, batch_size=256):
  val_ap, val_auc = [], []
  with torch.no_grad():
    model = model.eval()
    # While usually the test batch size is as big as it fits in memory, here we keep it the same
    # size as the training batch size, since it allows the memory to be updated more frequently,
    # and later test batches to access information from interactions in previous test batches
    # through the memory
    TEST_BATCH_SIZE = batch_size
    num_test_instance = len(data.sources)
    num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

    for k in range(num_test_batch):
      s_idx = k * TEST_BATCH_SIZE
      e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
      sources_batch = data.sources[s_idx:e_idx]
      destinations_batch = data.destinations[s_idx:e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      edge_idxs_batch = data.edge_idxs[s_idx: e_idx]

      # 生成负样本，现在返回shape为[num_neg, batch_size]
      negatives_batch = negative_sampler.generate_negative_samples(sources_batch, timestamps_batch, num_neg)

      pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch, destinations_batch,
                                                            negatives_batch, timestamps_batch,
                                                            edge_idxs_batch, n_neighbors)

      size = len(sources_batch)
      # pos_prob: [batch_size], neg_prob: [num_neg, batch_size]
      # 将neg_prob展平为[num_neg * batch_size]
      neg_prob_flat = neg_prob.view(-1) if neg_prob.dim() > 1 else neg_prob
      
      pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob_flat).cpu().numpy()])
      true_label = np.concatenate([np.ones(size), np.zeros(size * num_neg)])

      val_ap.append(average_precision_score(true_label, pred_score))
      val_auc.append(roc_auc_score(true_label, pred_score))

  return np.mean(val_ap), np.mean(val_auc)

