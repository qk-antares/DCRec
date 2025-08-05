import math
import logging
import time
import sys
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm

from evaluation.evaluation import eval_edge_prediction
from model.tgn import TGN
from utils.dataset import Dataset
from utils.utils import EarlyStopMonitor

torch.manual_seed(0)
np.random.seed(0)

### Argument and global variables
parser = argparse.ArgumentParser('TGN self-supervised training')
# 数据相关
parser.add_argument('-d', '--data', type=str, help='Dataset name (ml-1m or taobao)', default='ml-1m')
parser.add_argument('--bs', type=int, default=256, help='Batch_size')
parser.add_argument('--n_epoch', type=int, default=50, help='Number of epochs')
parser.add_argument('--train_ratio', type=float, default=0.8, help='Training set ratio')
parser.add_argument('--valid_ratio', type=float, default=0.1, help='Validation set ratio')
parser.add_argument('--randomize_features', action='store_true', help='Whether to randomize node features')
parser.add_argument('--uniform', action='store_true', help='take uniform sampling from temporal neighbors')

# 训练相关
parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
parser.add_argument('--n_neighbors', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_neg', type=int, default=1, help='Number of negative samples to generate')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=128, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=128, help='Dimensions of the time embedding')

# 模型相关
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--use_memory', action='store_true', help='Whether to augment the model with a node memory')
parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=["graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
parser.add_argument('--message_function', type=str, default="identity", choices=[ "mlp", "identity"], help='Type of message function')
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

DATA = args.data
BATCH_SIZE = args.bs
NUM_EPOCH = args.n_epoch
TRAIN_RATIO = args.train_ratio
VALID_RATIO = args.valid_ratio
RANDOMIZE_FEATURES = args.randomize_features
UNIFORM = args.uniform

NUM_NEIGHBORS = args.n_neighbors
NUM_NEG = args.n_neg
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
USE_MEMORY = args.use_memory
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim

Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}.pth'
get_checkpoint_path = lambda epoch: f'./saved_checkpoints/{args.prefix}-{args.data}-{epoch}.pth'

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
Path("log/").mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

### Extract data for training, validation and testing
dataset = Dataset(DATA, RANDOMIZE_FEATURES, TRAIN_RATIO, VALID_RATIO, UNIFORM)

# Set device
device = torch.device('cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu')

for i in range(args.n_runs):
  results_path = "results/{}_{}.pkl".format(args.prefix, i) if i > 0 else "results/{}.pkl".format(args.prefix)
  Path("results/").mkdir(parents=True, exist_ok=True)

  # Initialize Model
  tgn = TGN(neighbor_finder=dataset.train_ngh_finder, 
            node_features=dataset.node_features, edge_features=dataset.edge_features, device=device,
            n_layers=NUM_LAYER,
            n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
            message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
            memory_update_at_start=not args.memory_update_at_end,
            embedding_module_type=args.embedding_module,
            message_function=args.message_function,
            aggregator_type=args.aggregator,
            memory_updater_type=args.memory_updater,
            n_neighbors=NUM_NEIGHBORS,
            mean_time_shift_src=dataset.mean_time_shift_src, std_time_shift_src=dataset.std_time_shift_src,
            mean_time_shift_dst=dataset.mean_time_shift_dst, std_time_shift_dst=dataset.std_time_shift_dst,
            use_destination_embedding_in_message=args.use_destination_embedding_in_message,
            use_source_embedding_in_message=args.use_source_embedding_in_message,
            dyrep=args.dyrep)
  criterion = torch.nn.BCELoss()
  optimizer = torch.optim.Adam(tgn.parameters(), lr=LEARNING_RATE)
  tgn = tgn.to(device)

  num_instance = len(dataset.train_data.sources)
  num_batch = math.ceil(num_instance / BATCH_SIZE)

  logger.info('num of training instances: {}'.format(num_instance))
  logger.info('num of batches per epoch: {}'.format(num_batch))
  idx_list = np.arange(num_instance)

  new_nodes_val_aps = []
  val_aps = []
  epoch_times = []
  total_epoch_times = []
  train_losses = []

  early_stopper = EarlyStopMonitor(max_round=args.patience)
  for epoch in range(NUM_EPOCH):
    start_epoch = time.time()
    ### Training

    # Reinitialize memory of the model at the start of each epoch
    if USE_MEMORY:
      tgn.memory.__init_memory__()

    # Train using only training graph
    tgn.set_neighbor_finder(dataset.train_ngh_finder)
    m_loss = []

    logger.info('start {} epoch'.format(epoch))
    pbar = tqdm(range(num_batch), desc=f'Epoch {epoch}/{NUM_EPOCH-1}')
    for batch_idx in pbar:
      optimizer.zero_grad()

      start_idx = batch_idx * BATCH_SIZE
      end_idx = min(num_instance, start_idx + BATCH_SIZE)
      
      sources_batch = dataset.train_data.sources[start_idx:end_idx]
      destinations_batch = dataset.train_data.destinations[start_idx:end_idx]
      edge_idxs_batch = dataset.train_data.edge_idxs[start_idx:end_idx]
      timestamps_batch = dataset.train_data.timestamps[start_idx:end_idx]

      negatives_batch = dataset.train_negative_sampler.generate_negative_samples(sources_batch, timestamps_batch, NUM_NEG)
      # TODO: 支持多个负样本
      negatives_batch = np.array(negatives_batch).flatten()

      size = len(sources_batch)
      pos_label = torch.ones(size, dtype=torch.float, device=device)
      neg_label = torch.zeros(size, dtype=torch.float, device=device)

      tgn = tgn.train()
      pos_prob, neg_prob = tgn.compute_edge_probabilities(sources_batch, destinations_batch, negatives_batch, timestamps_batch, edge_idxs_batch, NUM_NEIGHBORS)

      loss = criterion(pos_prob.squeeze(), pos_label) + criterion(neg_prob.squeeze(), neg_label)

      loss.backward()
      optimizer.step()
      m_loss.append(loss.item())
      
      # Update progress bar with current loss
      pbar.set_postfix({'loss': f'{loss.item():.4f}'})
      
      if USE_MEMORY:
        tgn.memory.detach_memory()

    epoch_time = time.time() - start_epoch
    epoch_times.append(epoch_time)

    ### Validation
    # Validation uses the full graph
    tgn.set_neighbor_finder(dataset.full_ngh_finder)

    if USE_MEMORY:
      # Backup memory at the end of training, so later we can restore it and use it for the
      # validation on unseen nodes
      train_memory_backup = tgn.memory.backup_memory()

    val_ap, val_auc = eval_edge_prediction(tgn, dataset.val_data, NUM_NEIGHBORS, dataset.full_negative_sampler, NUM_NEG, BATCH_SIZE)
    if USE_MEMORY:
      val_memory_backup = tgn.memory.backup_memory()
      # Restore memory we had at the end of training to be used when validating on new nodes.
      # Also backup memory after validation so it can be used for testing (since test edges are
      # strictly later in time than validation edges)
      tgn.memory.restore_memory(train_memory_backup)

    # Validate on unseen nodes
    nn_val_ap, nn_val_auc = eval_edge_prediction(tgn, dataset.new_node_val_data, NUM_NEIGHBORS, dataset.full_negative_sampler, NUM_NEG, BATCH_SIZE)

    if USE_MEMORY:
      # Restore memory we had at the end of validation
      tgn.memory.restore_memory(val_memory_backup)

    new_nodes_val_aps.append(nn_val_ap)
    val_aps.append(val_ap)
    train_losses.append(np.mean(m_loss))

    # Save temporary results to disk
    pickle.dump({
      "val_aps": val_aps,
      "new_nodes_val_aps": new_nodes_val_aps,
      "train_losses": train_losses,
      "epoch_times": epoch_times,
      "total_epoch_times": total_epoch_times
    }, open(results_path, "wb"))

    total_epoch_time = time.time() - start_epoch
    total_epoch_times.append(total_epoch_time)

    logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
    logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
    logger.info(
      'val auc: {}, new node val auc: {}'.format(val_auc, nn_val_auc))
    logger.info(
      'val ap: {}, new node val ap: {}'.format(val_ap, nn_val_ap))

    # Early stopping
    if early_stopper.early_stop_check(val_ap):
      logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
      logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
      best_model_path = get_checkpoint_path(early_stopper.best_epoch)
      tgn.load_state_dict(torch.load(best_model_path))
      logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
      tgn.eval()
      break
    else:
      torch.save(tgn.state_dict(), get_checkpoint_path(epoch))

  # Training has finished, we have loaded the best model, and we want to backup its current
  # memory (which has seen validation edges) so that it can also be used when testing on unseen
  # nodes
  if USE_MEMORY:
    val_memory_backup = tgn.memory.backup_memory()

  ### Test
  tgn.set_neighbor_finder(dataset.full_ngh_finder)
  test_ap, test_auc = eval_edge_prediction(tgn, dataset.test_data, NUM_NEIGHBORS, dataset.full_negative_sampler, NUM_NEG, BATCH_SIZE)

  if USE_MEMORY:
    tgn.memory.restore_memory(val_memory_backup)

  # Test on unseen nodes
  nn_test_ap, nn_test_auc = eval_edge_prediction(tgn, dataset.new_node_test_data, NUM_NEIGHBORS, dataset.full_negative_sampler, NUM_NEG, BATCH_SIZE)

  logger.info(
    'Test statistics: Old nodes -- auc: {}, ap: {}'.format(test_auc, test_ap))
  logger.info(
    'Test statistics: New nodes -- auc: {}, ap: {}'.format(nn_test_auc, nn_test_ap))
  # Save results for this run
  pickle.dump({
    "val_aps": val_aps,
    "new_nodes_val_aps": new_nodes_val_aps,
    "test_ap": test_ap,
    "new_node_test_ap": nn_test_ap,
    "epoch_times": epoch_times,
    "train_losses": train_losses,
    "total_epoch_times": total_epoch_times
  }, open(results_path, "wb"))

  logger.info('Saving TGN model')
  if USE_MEMORY:
    # Restore memory at the end of validation (save a model which is ready for testing)
    tgn.memory.restore_memory(val_memory_backup)
  torch.save(tgn.state_dict(), MODEL_SAVE_PATH)
  logger.info('TGN model saved')
