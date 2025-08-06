#!/usr/bin/env python3

import numpy as np
import torch
import sys
import os

# Add the project root to Python path
sys.path.append('/root/workplace/python/DCRec')

from utils.dataset import Dataset
from trainer import TGNTrainer
from evaluation.evaluation import eval_edge_prediction
import argparse

def test_evaluation_metrics():
    """Test the refactored evaluation function with MRR, Recall@10, and Recall@20"""
    print("Testing refactored evaluation function with MRR, Recall@10, and Recall@20...")
    
    # Create mock arguments
    args = argparse.Namespace()
    args.data = 'ml-1m'
    args.batch_size = 16
    args.n_epoch = 1
    args.n_layers = 1
    args.n_heads = 2
    args.drop_out = 0.1
    args.use_memory = False
    args.memory_update_at_end = False
    args.message_dim = 100
    args.memory_dim = 500
    args.embedding_module = "graph_attention"
    args.message_function = "mlp"
    args.aggregator = "last"
    args.memory_updater = "gru"
    args.n_neighbors = 10
    args.use_destination_embedding_in_message = False
    args.use_source_embedding_in_message = False
    args.dyrep = False
    args.learning_rate = 0.0001
    args.n_neg = 3
    args.n_test_neg = 99  # Test with 99 negative samples for evaluation
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    dataset = Dataset('ml-1m', randomize_features=False)
    
    # Create trainer and model
    import logging
    logger = logging.getLogger(__name__)
    trainer = TGNTrainer(args, dataset, device, logger)
    model = trainer.create_model()
    
    print(f"Model created successfully!")
    print(f"Testing evaluation with n_test_neg={args.n_test_neg}")
    
    # Test evaluation on a small subset of validation data
    print("\n--- Testing on validation data subset ---")
    
    # Take first 32 samples from validation data for quick testing
    num_samples = 32
    
    val_data_subset = type('', (), {})()
    val_data_subset.sources = dataset.val_data.sources[:num_samples]
    val_data_subset.destinations = dataset.val_data.destinations[:num_samples]
    val_data_subset.timestamps = dataset.val_data.timestamps[:num_samples]
    val_data_subset.edge_idxs = dataset.val_data.edge_idxs[:num_samples]
    
    model.eval()
    # Set the model to use the full neighbor finder for evaluation
    model.set_neighbor_finder(dataset.full_ngh_finder)
    
    with torch.no_grad():
        # Run evaluation
        mrr, recall_10, recall_20 = eval_edge_prediction(
            model, val_data_subset, args.n_neighbors,
            dataset.full_negative_sampler, args.n_test_neg, batch_size=16)
    
    print(f"Evaluation Results:")
    print(f"  MRR: {mrr:.4f}")
    print(f"  Recall@10: {recall_10:.4f}")
    print(f"  Recall@20: {recall_20:.4f}")
    
    # Verify the results are reasonable
    assert 0 <= mrr <= 1, f"MRR should be between 0 and 1, got {mrr}"
    assert 0 <= recall_10 <= 1, f"Recall@10 should be between 0 and 1, got {recall_10}"
    assert 0 <= recall_20 <= 1, f"Recall@20 should be between 0 and 1, got {recall_20}"
    assert recall_10 <= recall_20, f"Recall@10 should be <= Recall@20, got {recall_10} > {recall_20}"
    
    print(f"\n✅ All validation checks passed!")
    
    # Test with different n_test_neg values
    print(f"\n--- Testing with different n_test_neg values ---")
    
    for n_test_neg in [10, 50, 100]:
        print(f"Testing with n_test_neg={n_test_neg}")
        with torch.no_grad():
            mrr, recall_10, recall_20 = eval_edge_prediction(
                model, val_data_subset, args.n_neighbors,
                dataset.full_negative_sampler, n_test_neg, batch_size=16)
        
        print(f"  MRR: {mrr:.4f}, Recall@10: {recall_10:.4f}, Recall@20: {recall_20:.4f}")
        
        # With more negative samples, MRR and Recall should generally be lower
        assert 0 <= mrr <= 1 and 0 <= recall_10 <= 1 and 0 <= recall_20 <= 1
        assert recall_10 <= recall_20
    
    print(f"\n✅ All tests passed! Evaluation function works correctly with MRR, Recall@10, and Recall@20.")
    return True

if __name__ == "__main__":
    test_evaluation_metrics()
