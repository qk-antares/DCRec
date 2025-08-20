import argparse
import sys

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser('TGN self-supervised training')
    
    # 数据相关
    parser.add_argument('-d', '--data', type=str, help='Dataset name (ml-1m or taobao)', default='ml-1m')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch_size')
    parser.add_argument('--n_epoch', type=int, default=50, help='Number of epochs')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Training set ratio')
    parser.add_argument('--valid_ratio', type=float, default=0.1, help='Validation set ratio')
    parser.add_argument('--randomize_features', action='store_true', help='Whether to randomize node features')
    parser.add_argument('--uniform', action='store_true', help='take uniform sampling from temporal neighbors')
    parser.add_argument('--inductive', action='store_true', help='Whether to use inductive learning setting')

    # 训练相关
    parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
    parser.add_argument('--n_neighbors', type=int, default=10, help='Number of neighbors to sample')
    parser.add_argument('--n_neg', type=int, default=10, help='Number of negative samples to generate')
    parser.add_argument('--n_test_neg', type=int, default=1000, help='Number of negative samples to generate when evaluating')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--n_skip_val', type=int, default=30, help='Number of epochs to skip validation (model warm-up period)')
    parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
    parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
    parser.add_argument('--node_dim', type=int, default=128, help='Dimensions of the node embedding')
    parser.add_argument('--time_dim', type=int, default=128, help='Dimensions of the time embedding')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    
    # 噪声剪枝相关参数
    parser.add_argument('--noise_pruning_ratio', type=float, default=0.0, help='Noise pruning ratio (w), 0.0 means no pruning')
    parser.add_argument('--l2_regularization', type=float, default=0.0, help='L2 regularization coefficient (alpha)')

    # 模型相关
    parser.add_argument('--n_heads', type=int, default=2, help='Number of heads used in attention layer')
    parser.add_argument('--n_layers', type=int, default=1, help='Number of network layers')
    parser.add_argument('--use_memory', action='store_true', help='Whether to augment the model with a node memory')
    parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=["graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
    parser.add_argument('--message_function', type=str, default="identity", choices=["mlp", "identity"], help='Type of message function')
    parser.add_argument('--memory_updater', type=str, default="gru", choices=["gru", "rnn"], help='Type of memory updater')
    parser.add_argument('--aggregator', type=str, default="last", help='Type of message aggregator')
    parser.add_argument('--memory_update_at_end', action='store_true', help='Whether to update memory at the end or at the start of the batch')
    parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
    parser.add_argument('--memory_dim', type=int, default=129, help='Dimensions of the memory for each node, 129 for ml-1m')
    parser.add_argument('--use_destination_embedding_in_message', action='store_true', help='Whether to use the embedding of the destination node as part of the message')
    parser.add_argument('--use_source_embedding_in_message', action='store_true', help='Whether to use the embedding of the source node as part of the message')
    parser.add_argument('--dyrep', action='store_true', help='Whether to run the dyrep model')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    
    return args