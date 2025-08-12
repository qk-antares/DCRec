import numpy as np
import pandas as pd

# 训练/验证/测试数据集类
class Data:
  def __init__(self, sources, destinations, timestamps, edge_idxs, labels):
    self.sources = sources
    self.destinations = destinations
    self.timestamps = timestamps
    self.edge_idxs = edge_idxs
    self.labels = labels

    self.n_interactions = len(sources)
    self.unique_nodes = set(sources) | set(destinations)
    self.n_unique_nodes = len(self.unique_nodes)
    self.n_unique_sources = len(set(sources))
    self.n_unique_destinations = len(set(destinations))

# 整个数据集，包含多个Data对象，提供访问和处理数据的功能
class Dataset:
    def __init__(self, dataset_name, randomize_features=False, train_ratio=0.8, val_ratio=0.1, inductive=False, uniform=False, logger=None):
        """
        初始化数据集
        :param dataset_name: 数据集名称
        :param randomize_features: 是否随机化节点特征
        :param train_ratio: 训练集比例
        :param val_ratio: 验证集比例
        :param uniform: 是否进行均匀采样
        :param logger: 日志记录器
        """
        self.logger = logger

        assert train_ratio + val_ratio < 1, "Train and validation ratios must sum to less than 1."

        self.graph_df = None
        self.node_features = None
        self.edge_features = None
        self.read_data(dataset_name, randomize_features)

        self.full_data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.split_data(train_ratio, val_ratio, inductive)

        # Initialize training neighbor finder to retrieve temporal graph
        self.train_ngh_finder = NeighborFinder(self.train_data, uniform)
        # Initialize validation and test neighbor finder to retrieve temporal graph
        self.full_ngh_finder = NeighborFinder(self.full_data, uniform)

        self.train_negative_sampler = NegativeSampler(self.train_data, self.train_ngh_finder)
        self.full_negative_sampler = NegativeSampler(self.full_data, self.full_ngh_finder)

        self.mean_time_shift_src = None
        self.std_time_shift_src = None
        self.mean_time_shift_dst = None
        self.std_time_shift_dst = None
        self.compute_time_statistics(self.full_data.sources, self.full_data.destinations, self.full_data.timestamps)

    def read_data(self, dataset_name, randomize_features):
        self.logger.info(f"Reading dataset {dataset_name}...")
        # 读取图数据
        self.graph_df = pd.read_csv(f'data/{dataset_name}/processed/graph.csv')
        self.edge_features = np.load(f'data/{dataset_name}/processed/edges.npy')
        self.node_features = np.load(f'data/{dataset_name}/processed/nodes.npy')

        if randomize_features:
            self.node_features = np.random.rand(self.node_features.shape[0], self.node_features.shape[1])

    def split_data(self, train_ratio, val_ratio, inductive=False):
        self.logger.info(f"Splitting data into train, val, and test sets with ratios {train_ratio}, {val_ratio}...")

        # 1. 构造全体数据
        sources = self.graph_df.u.values
        destinations = self.graph_df.i.values
        edge_idxs = self.graph_df.idx.values
        labels = self.graph_df.label.values
        timestamps = self.graph_df.ts.values

        self.full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

        val_time, test_time = list(np.quantile(self.graph_df.ts, [train_ratio, train_ratio + val_ratio]))

        if not inductive:
          # 2. 构造完整的训练集、验证集和测试集
          train_mask = timestamps <= val_time
          val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)
          test_mask = timestamps > test_time

          # train validation and test with all edges
          self.train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                          edge_idxs[train_mask], labels[train_mask])
          self.val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                          edge_idxs[val_mask], labels[val_mask])
          self.test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                          edge_idxs[test_mask], labels[test_mask])
        else:
          # 3. 构造训练集（屏蔽了验证机和测试集的节点）
          node_set = set(sources) | set(destinations)
          n_total_unique_nodes = len(node_set)

          # 3.1 验证集和测试集的节点
          test_node_set = set(sources[timestamps > val_time]).union(set(destinations[timestamps > val_time]))
          # 3.2 从中抽取10%的节点作为new_test_node_set，并从训练集中删除，来测试模型的归纳能力
          new_test_node_set = set(np.random.choice(list(test_node_set), int(0.1 * n_total_unique_nodes), replace=False))

          # 3.3 训练集的src和dst不包含new_test_node_set中的节点
          # Mask saying for each source and destination whether they are new test nodes
          new_test_source_mask = self.graph_df.u.map(lambda x: x in new_test_node_set).values
          new_test_destination_mask = self.graph_df.i.map(lambda x: x in new_test_node_set).values
          # Mask which is true for edges with both destination and source not being new test nodes (because we want to remove all edges involving any new test node)
          observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)

          # For train we keep edges happening before the validation time which do not involve any new node used for inductiveness
          train_mask = np.logical_and(timestamps <= val_time, observed_edges_mask)

          self.train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                              edge_idxs[train_mask], labels[train_mask])

          # 4. 构造用于inductive测试的新节点验证集和测试集
          # 4.1 构造new_node_set（所有节点 - 训练集的节点）
          train_node_set = set(self.train_data.sources).union(self.train_data.destinations)
          assert len(train_node_set & new_test_node_set) == 0
          new_node_set = node_set - train_node_set

          edge_contains_new_node_mask = np.array([(a in new_node_set or b in new_node_set) for a, b in zip(sources, destinations)])
          new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
          new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

          # validation and test with edges that at least has one new node (not in training set)
          self.val_data = Data(sources[new_node_val_mask], destinations[new_node_val_mask],
                                  timestamps[new_node_val_mask],
                                  edge_idxs[new_node_val_mask], labels[new_node_val_mask])

          self.test_data = Data(sources[new_node_test_mask], destinations[new_node_test_mask],
                                      timestamps[new_node_test_mask], edge_idxs[new_node_test_mask],
                                      labels[new_node_test_mask])

        self.logger.info(f"The dataset has {self.full_data.n_interactions} interactions, involving {self.full_data.n_unique_nodes} different nodes ({self.full_data.n_unique_sources} unique sources and {self.full_data.n_unique_destinations} unique destinations)")
        if inductive:
          self.logger.info("Using inductive setting, all nodes in val and test set are unseen during training")
        self.logger.info(f"The training dataset has {self.train_data.n_interactions} interactions, involving {self.train_data.n_unique_nodes} different nodes ({self.train_data.n_unique_sources} unique sources and {self.train_data.n_unique_destinations} unique destinations)")
        self.logger.info(f"The validation dataset has {self.val_data.n_interactions} interactions, involving {self.val_data.n_unique_nodes} different nodes ({self.val_data.n_unique_sources} unique sources and {self.val_data.n_unique_destinations} unique destinations)")
        self.logger.info(f"The test dataset has {self.test_data.n_interactions} interactions, involving {self.test_data.n_unique_nodes} different nodes ({self.test_data.n_unique_sources} unique sources and {self.test_data.n_unique_destinations} unique destinations)")

    def compute_time_statistics(self, sources, destinations, timestamps):
      last_timestamp_sources = dict()
      last_timestamp_dst = dict()
      all_timediffs_src = []
      all_timediffs_dst = []
      for k in range(len(sources)):
        source_id = sources[k]
        dest_id = destinations[k]
        c_timestamp = timestamps[k]
        if source_id not in last_timestamp_sources.keys():
          last_timestamp_sources[source_id] = 0
        if dest_id not in last_timestamp_dst.keys():
          last_timestamp_dst[dest_id] = 0
        all_timediffs_src.append(c_timestamp - last_timestamp_sources[source_id])
        all_timediffs_dst.append(c_timestamp - last_timestamp_dst[dest_id])
        last_timestamp_sources[source_id] = c_timestamp
        last_timestamp_dst[dest_id] = c_timestamp
      assert len(all_timediffs_src) == len(sources)
      assert len(all_timediffs_dst) == len(sources)
      
      self.mean_time_shift_src = np.mean(all_timediffs_src)
      self.std_time_shift_src = np.std(all_timediffs_src)
      self.mean_time_shift_dst = np.mean(all_timediffs_dst)
      self.std_time_shift_dst = np.std(all_timediffs_dst)

# 根据训练/验证/测试数据集创建邻居查找器
class NeighborFinder:
  def __init__(self, data: Data, uniform=False):
    """
    Initializes the NeighborFinder with the given data.
    :param data: Data object containing sources, destinations, timestamps, edge_idxs, and labels.
    :param uniform: If True, samples neighbors uniformly; otherwise, takes the most recent neighbors
    """
    max_node_idx = max(data.sources.max(), data.destinations.max())
    self.adj_list = [[] for _ in range(max_node_idx + 1)]
    for source, destination, edge_idx, timestamp in zip(data.sources, data.destinations,
                                                        data.edge_idxs,
                                                        data.timestamps):
        self.adj_list[source].append((destination, edge_idx, timestamp))
        self.adj_list[destination].append((source, edge_idx, timestamp))

    self.node_to_neighbors = []
    self.node_to_edge_idxs = []
    self.node_to_edge_timestamps = []

    for neighbors in self.adj_list:
      # Neighbors is a list of tuples (neighbor, edge_idx, timestamp)
      # We sort the list based on timestamp
      sorted_neighbors = sorted(neighbors, key=lambda x: x[2])
      self.node_to_neighbors.append(np.array([x[0] for x in sorted_neighbors]))
      self.node_to_edge_idxs.append(np.array([x[1] for x in sorted_neighbors]))
      self.node_to_edge_timestamps.append(np.array([x[2] for x in sorted_neighbors]))

    self.uniform = uniform

  def find_before(self, src_idx, cut_time):
    """
    Extracts all the interactions happening [before cut_time (timestamp < cut_time)] for user src_idx in the overall interaction graph. The returned interactions are sorted by time.
    
    Returns 3 lists: neighbors, edge_idxs, timestamps
    """
    i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)

    return self.node_to_neighbors[src_idx][:i], self.node_to_edge_idxs[src_idx][:i], self.node_to_edge_timestamps[src_idx][:i]

  def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=20):
    """
    Given a list of users ids and relative cut times, extracts a sampled temporal neighborhood of each user in the list.

    Params
    ------
    src_idx_l: List[int]
    cut_time_l: List[float],
    num_neighbors: int
    """
    assert (len(source_nodes) == len(timestamps))

    tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
    # NB! All interactions described in these matrices are sorted in each row by time
    neighbors = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.int32)  # each entry in position (i,j) represent the id of the item targeted by user src_idx_l[i] with an interaction happening before cut_time_l[i]
    edge_times = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.float32)  # each entry in position (i,j) represent the timestamp of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]
    edge_idxs = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.int32)  # each entry in position (i,j) represent the interaction index of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]

    for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):
      source_neighbors, source_edge_idxs, source_edge_times = self.find_before(source_node,
                                                   timestamp)  # extracts all neighbors, interactions indexes and timestamps of all interactions of user source_node happening before cut_time

      if len(source_neighbors) > 0 and n_neighbors > 0:
        if self.uniform:  # if we are applying uniform sampling, shuffles the data above before sampling
          sampled_idx = np.random.randint(0, len(source_neighbors), n_neighbors)

          neighbors[i, :] = source_neighbors[sampled_idx]
          edge_times[i, :] = source_edge_times[sampled_idx]
          edge_idxs[i, :] = source_edge_idxs[sampled_idx]

          # re-sort based on time
          pos = edge_times[i, :].argsort()
          neighbors[i, :] = neighbors[i, :][pos]
          edge_times[i, :] = edge_times[i, :][pos]
          edge_idxs[i, :] = edge_idxs[i, :][pos]
        else:
          # Take most recent interactions
          source_edge_times = source_edge_times[-n_neighbors:]
          source_neighbors = source_neighbors[-n_neighbors:]
          source_edge_idxs = source_edge_idxs[-n_neighbors:]

          assert (len(source_neighbors) <= n_neighbors)
          assert (len(source_edge_times) <= n_neighbors)
          assert (len(source_edge_idxs) <= n_neighbors)

          neighbors[i, n_neighbors - len(source_neighbors):] = source_neighbors
          edge_times[i, n_neighbors - len(source_edge_times):] = source_edge_times
          edge_idxs[i, n_neighbors - len(source_edge_idxs):] = source_edge_idxs

    return neighbors, edge_idxs, edge_times

# 负采样器，用于生成负样本，负样本的定义是在timestamps之前用户未交互的物品
class NegativeSampler:
    def __init__(self, data: Data, neighbor_finder: NeighborFinder):
        self.all_destinations = set(data.destinations)
        self.neighbor_finder = neighbor_finder

    def generate_negative_samples(self, source_nodes, timestamps, n_neg: int = 1):
        """
        对给定的source_nodes和timestamps进行负采样
        Args:
            source_nodes: 用户节点ID数组 [batch_size]
            timestamps: 时间戳数组 [batch_size]
            n_neg: 每个用户生成的负样本数量
        Returns:
            neg_destinations: 负样本数组，shape为[n_neg, batch_size]
                             neg_destinations[i]代表对batch中所有source_nodes节点所采样的第i个负样本
        """
        batch_size = len(source_nodes)
        # 初始化结果矩阵 [n_neg, batch_size]
        neg_destinations = np.zeros((n_neg, batch_size), dtype=np.int32)
        
        for batch_idx, (user_id, timestamp) in enumerate(zip(source_nodes, timestamps)):
            # 获取该用户在timestamp之前交互过的物品
            interacted_items, _, _ = self.neighbor_finder.find_before(user_id, timestamp)
            interacted_set = set(interacted_items)
            candidate_destinations = list(self.all_destinations - interacted_set)
            if not candidate_destinations:
                # 无法采样负样本，这表明用户在该时间点与所有物品都有过交互，通常是不常见的情况
                raise ValueError(f"User {user_id} at timestamp {timestamp} has no candidate negative items.")
            # 如果候选物品不足n_neg，允许有放回采样
            replace = len(candidate_destinations) < n_neg
            sampled_items = np.random.choice(candidate_destinations, size=n_neg, replace=replace)
            
            # 将采样结果填入对应位置：第i个负样本放在neg_destinations[i, batch_idx]
            for neg_idx in range(n_neg):
                neg_destinations[neg_idx, batch_idx] = sampled_items[neg_idx]

        return neg_destinations
