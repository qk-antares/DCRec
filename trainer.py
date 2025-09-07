import math
import time
import logging
import torch
import numpy as np
from tqdm import tqdm

from model.dcrec.cgfa.cgfa import CGFA
from model.dcrec.dcrec import DCRec
from model.dcrec.tgn.tgn import TGN
from utils.utils import EarlyStopMonitor
from utils.dataset import NegativeSampler

class Trainer:
    def __init__(self, args, dataset, device):
        self.args = args
        self.dataset = dataset
        self.device = device
        self.logger = logging.getLogger(__name__)

    def train_model(self):
        pass
    
    def compute_bpr_loss(self, pos_scores, neg_scores):
        """
        计算基于噪声剪枝的BPR损失
        Args:
            pos_scores: [batch_size] - 正样本分数
            neg_scores: [n_neg, batch_size] - 负样本分数
        Returns:
            总损失（BPR损失 + L2正则化）
        """
        # 将pos_scores扩展为[n_neg, batch_size]以便与neg_scores进行比较
        pos_scores_expanded = pos_scores.unsqueeze(0).expand_as(neg_scores)  # [n_neg, batch_size]
        
        # BPR损失: -log(sigmoid(pos_score - neg_score))
        # 等价于: log(1 + exp(neg_score - pos_score))
        score_diff = neg_scores - pos_scores_expanded  # [n_neg, batch_size]
        
        # 使用logsigmoid来数值稳定地计算log(sigmoid(x))
        # log(sigmoid(x)) = -log(1 + exp(-x)) = -softplus(-x)
        bpr_losses = -torch.nn.functional.logsigmoid(-score_diff)  # [n_neg, batch_size]
        
        # 噪声剪枝：只保留损失最小的(1-w)*n_neg个样本对
        # 对每个batch样本，按损失升序排序
        sorted_losses, _ = torch.sort(bpr_losses, dim=0)  # [n_neg, batch_size]
        
        # 计算保留的负样本数量
        n_neg = bpr_losses.shape[0]
        n_keep = max(1, int((1 - self.args.noise_pruning_ratio) * n_neg))
        
        # 只保留损失最小的n_keep个样本对
        pruned_losses = sorted_losses[:n_keep, :]  # [n_keep, batch_size]
        bpr_loss = pruned_losses.mean()
        
        return bpr_loss

class DCRecBaseTrainer(Trainer):
    """TGN模型训练器"""
    def __init__(self, args, dataset, device):
        super().__init__(args, dataset, device)

    def create_model(self):
        pass
    
    def train_model(self):
        """完整的模型训练流程"""
        model = self.create_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)
        
        # 训练统计
        val_mrrs = []
        val_recall_10s = []
        val_recall_20s = []
        train_total_losses = []
        train_bpr_losses = []
        train_l2_losses = []
        epoch_times = []
        total_epoch_times = []
        
        # 早停监控器只在开始验证后启用
        early_stopper = None
        best_val_mrr = 0.0
        best_epoch = -1
        
        num_instance = len(self.dataset.train_data.sources)
        num_batch = math.ceil(num_instance / self.args.batch_size)
        
        self.logger.info('num of training instances: {}'.format(num_instance))
        self.logger.info('num of batches per epoch: {}'.format(num_batch))
        self.logger.info(f'Skipping validation for first {self.args.n_skip_val} epochs (warm-up period)')
        
        # 记录噪声剪枝和正则化配置
        self.logger.info(f'Using noise pruning with ratio: {self.args.noise_pruning_ratio}')
        self.logger.info(f'Using L2 regularization with coefficient: {self.args.l2_regularization}')
        
        # 模型保存路径
        model_save_path = f'saved_models/{self.args.prefix}-{self.args.data}.pth'

        for epoch in range(self.args.n_epoch):
            start_epoch = time.time()
            
            # 训练
            train_loss_dict = self.train_epoch(model, optimizer, epoch)
            train_total_losses.append(train_loss_dict['total_loss'])
            train_bpr_losses.append(train_loss_dict['bpr_loss'])
            train_l2_losses.append(train_loss_dict['l2_loss'])
            
            epoch_time = time.time() - start_epoch
            epoch_times.append(epoch_time)
            
            total_epoch_time = time.time() - start_epoch
            total_epoch_times.append(total_epoch_time)
            
            # 记录训练日志
            self.logger.info(f'epoch: {epoch+1} took {total_epoch_time:.2f}s')
            self.logger.info(f'Epoch losses - Total: {train_loss_dict["total_loss"]:.4f}, BPR: {train_loss_dict["bpr_loss"]:.4f}, L2: {train_loss_dict["l2_loss"]:.6f}')
            
            # 验证阶段：从第n_skip_val+1个epoch开始
            if epoch >= self.args.n_skip_val:
                if epoch == self.args.n_skip_val:
                    # 初次开始验证时初始化早停监控器
                    early_stopper = EarlyStopMonitor(max_round=self.args.patience)
                    self.logger.info(f'Starting validation from epoch {epoch+1}')
                
                # 验证
                val_mrr, val_recall_10, val_recall_20 = self.validate(model)
                val_mrrs.append(val_mrr)
                val_recall_10s.append(val_recall_10)
                val_recall_20s.append(val_recall_20)
                
                self.logger.info(f'val MRR: {val_mrr:.4f}, Recall@10: {val_recall_10:.4f}, Recall@20: {val_recall_20:.4f}')
                
                # 如果是新的最佳性能，保存模型
                if val_mrr > best_val_mrr:
                    best_val_mrr = val_mrr
                    best_epoch = epoch
                    
                    # 保存最佳模型
                    torch.save(model.state_dict(), model_save_path)
                    self.logger.info(f'New best validation MRR: {val_mrr:.4f} at epoch {epoch+1}, model saved')
                
                # 早停检查
                if early_stopper.early_stop_check(val_mrr):
                    self.logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
                    self.logger.info(f'Best model was at epoch {best_epoch+1} with MRR: {best_val_mrr:.4f}')
                    break
            else:
                # 填充空值以保持数组长度一致
                val_mrrs.append(0.0)
                val_recall_10s.append(0.0)
                val_recall_20s.append(0.0)
                self.logger.info('Validation skipped (warm-up period)')

        # 如果没有进行过验证（所有epoch都被跳过），保存最后的模型
        if best_epoch == -1:
            self.logger.info('No validation performed, saving final model')
            torch.save(model.state_dict(), model_save_path)
            best_epoch = self.args.n_epoch - 1
        
        # 加载最佳模型进行测试（包括模型的记忆）
        if(self.args.use_memory):
            model.clear_all_messages()
        model.load_state_dict(torch.load(model_save_path, weights_only=True))
        model.eval()
        
        # 测试
        test_mrr, test_recall_10, test_recall_20 = self.test(model)
        
        self.logger.info('Test statistics: MRR: {:.4f}, Recall@10: {:.4f}, Recall@20: {:.4f}'.format(test_mrr, test_recall_10, test_recall_20))
        self.logger.info('Model training completed')
        
        return {
            "val_mrrs": val_mrrs,
            "val_recall_10s": val_recall_10s,
            "val_recall_20s": val_recall_20s,
            "test_mrr": test_mrr,
            "test_recall_10": test_recall_10,
            "test_recall_20": test_recall_20,
            "epoch_times": epoch_times,
            "train_total_losses": train_total_losses,
            "train_bpr_losses": train_bpr_losses,
            "train_l2_losses": train_l2_losses,
            "total_epoch_times": total_epoch_times,
            "best_epoch": best_epoch,
            "best_val_mrr": best_val_mrr
        }

    def train_epoch(self, model, optimizer, epoch):
        """训练一个epoch"""
        if self.args.use_memory:
            model.__init_memory__()

        model.set_neighbor_finder(self.dataset.train_ngh_finder)
        model.train()
        
        num_instance = len(self.dataset.train_data.sources)
        num_batch = math.ceil(num_instance / self.args.batch_size)
        
        epoch_total_losses = []
        epoch_bpr_losses = []
        epoch_l2_losses = []
        progress_bar = tqdm(range(num_batch), desc=f'Epoch {epoch+1}/{self.args.n_epoch}', unit='batch')
        
        for batch_idx in progress_bar:
            total_loss, bpr_loss, l2_loss = self.train_batch(model, optimizer, batch_idx, num_instance)
            epoch_total_losses.append(total_loss)
            epoch_bpr_losses.append(bpr_loss)
            epoch_l2_losses.append(l2_loss)
            
            # 更新进度条显示
            postfix = {'total_loss': f'{total_loss:.4f}', 'bpr_loss': f'{bpr_loss:.4f}', 'l2_loss': f'{l2_loss:.6f}'}
            progress_bar.set_postfix(postfix)
            
            if self.args.use_memory:
                model.detach_memory()
        
        return {
            'total_loss': np.mean(epoch_total_losses),
            'bpr_loss': np.mean(epoch_bpr_losses),
            'l2_loss': np.mean(epoch_l2_losses),
        }

    def train_batch(self, model, optimizer, batch_idx, num_instance):
        pass
    
    def validate(self, model):
        """验证模型"""
        pass
    
    def test(self, model):
        """测试模型"""
        pass

    def eval_edge_prediction(self, model, data, n_neighbors, negative_sampler: NegativeSampler, n_test_neg, batch_size=256):
        pass

class TGNTrainer(DCRecBaseTrainer):
    def __init__(self, args, dataset, device):
        super().__init__(args, dataset, device)
        self.logger.info("Initializing TGN trainer")

    def create_model(self):
        """创建TGN模型"""
        tgn = TGN(
            neighbor_finder=self.dataset.train_ngh_finder, 
            node_features=self.dataset.node_features, 
            edge_features=self.dataset.edge_features, 
            device=self.device,
            n_layers=self.args.n_layers,
            n_heads=self.args.n_heads, 
            dropout=self.args.drop_out, 
            use_memory=self.args.use_memory,
            message_dimension=self.args.message_dim, 
            memory_dimension=self.args.memory_dim,
            memory_update_at_start=not self.args.memory_update_at_end,
            embedding_module_type=self.args.embedding_module,
            message_function=self.args.message_function,
            aggregator_type=self.args.aggregator,
            memory_updater_type=self.args.memory_updater,
            n_neighbors=self.args.n_neighbors,
            mean_time_shift_src=self.dataset.mean_time_shift_src, 
            std_time_shift_src=self.dataset.std_time_shift_src,
            mean_time_shift_dst=self.dataset.mean_time_shift_dst, 
            std_time_shift_dst=self.dataset.std_time_shift_dst,
            use_destination_embedding_in_message=self.args.use_destination_embedding_in_message,
            use_source_embedding_in_message=self.args.use_source_embedding_in_message,
            dyrep=self.args.dyrep
        )
        return tgn.to(self.device)

    def train_batch(self, model, optimizer, batch_idx, num_instance):
        """训练一个batch"""
        optimizer.zero_grad()
        
        start_idx = batch_idx * self.args.batch_size
        end_idx = min(num_instance, start_idx + self.args.batch_size)
        
        # 获取batch数据
        sources_batch = self.dataset.train_data.sources[start_idx:end_idx]
        destinations_batch = self.dataset.train_data.destinations[start_idx:end_idx]
        edge_idxs_batch = self.dataset.train_data.edge_idxs[start_idx:end_idx]
        timestamps_batch = self.dataset.train_data.timestamps[start_idx:end_idx]
        
        # 生成负样本，现在返回shape为[n_neg, batch_size]
        negatives_batch = self.dataset.train_negative_sampler.generate_negative_samples(
            sources_batch, timestamps_batch, self.args.n_neg)
        
        # 计算正样本和负样本的原始分数（用于BPR损失）
        pos_scores, neg_scores = model.compute_edge_scores(
            sources_batch, destinations_batch, negatives_batch, 
            timestamps_batch, edge_idxs_batch, self.args.n_neighbors)
        
        # 计算基于噪声剪枝的BPR损失
        bpr_loss = self.compute_bpr_loss(pos_scores, neg_scores)
        
        # 添加L2正则化项
        l2_loss = 0.0
        if self.args.l2_regularization > 0:
            for param in model.parameters():
                l2_loss += torch.norm(param, 2) ** 2
            l2_loss *= self.args.l2_regularization
        
        # 总损失 = BPR损失 + L2正则化
        total_loss = bpr_loss + l2_loss
        
        total_loss.backward()
        
        optimizer.step()

        return total_loss.item(), bpr_loss.item(), l2_loss if isinstance(l2_loss, float) else l2_loss.item()

    def validate(self, model):
        """验证模型"""
        model.set_neighbor_finder(self.dataset.full_ngh_finder)
        
        val_mrr, val_recall_10, val_recall_20 = self.eval_edge_prediction(
            model, self.dataset.val_data, self.args.n_neighbors, 
            self.dataset.full_negative_sampler, self.args.n_test_neg, self.args.batch_size)
        
        return val_mrr, val_recall_10, val_recall_20
    
    def test(self, model):
        """测试模型"""
        model.set_neighbor_finder(self.dataset.full_ngh_finder)
        
        test_mrr, test_recall_10, test_recall_20 = self.eval_edge_prediction(
            model, self.dataset.test_data, self.args.n_neighbors,
            self.dataset.full_negative_sampler, self.args.n_test_neg, self.args.batch_size)
        
        return test_mrr, test_recall_10, test_recall_20

    def eval_edge_prediction(self, model, data, n_neighbors, negative_sampler: NegativeSampler, n_test_neg, batch_size=256):
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

class DCRecTrainer(DCRecBaseTrainer):
    def __init__(self, args, dataset, device):
        super().__init__(args, dataset, device)
        self.logger.info("Initializing DCRec trainer")
    
    def create_model(self):
        """创建DCRec模型"""
        dcrec = DCRec(
            neighbor_finder=self.dataset.train_ngh_finder, 
            node_features=self.dataset.node_features, 
            edge_features=self.dataset.edge_features, 
            device=self.device,
            n_layers=self.args.n_layers,
            n_heads=self.args.n_heads, 
            dropout=self.args.drop_out, 
            use_memory=self.args.use_memory,
            message_dimension=self.args.message_dim, 
            memory_dimension=self.args.memory_dim,
            memory_update_at_start=not self.args.memory_update_at_end,
            embedding_module_type=self.args.embedding_module,
            message_function=self.args.message_function,
            aggregator_type=self.args.aggregator,
            memory_updater_type=self.args.memory_updater,
            n_neighbors=self.args.n_neighbors,
            mean_time_shift_src=self.dataset.mean_time_shift_src, 
            std_time_shift_src=self.dataset.std_time_shift_src,
            mean_time_shift_dst=self.dataset.mean_time_shift_dst, 
            std_time_shift_dst=self.dataset.std_time_shift_dst,
            use_destination_embedding_in_message=self.args.use_destination_embedding_in_message,
            use_source_embedding_in_message=self.args.use_source_embedding_in_message,
            dyrep=self.args.dyrep,
            # CGFA参数
            cgfa_in_channels=self.dataset.node_cg_emb.shape[-1],
            cgfa_out_channels=self.args.cgfa_out_channels,
            # 融合网络参数
            fusion_hidden_dim=self.args.fusion_hidden_dim,
            final_hidden_dim=self.args.final_hidden_dim
        )
        return dcrec.to(self.device)

    def train_batch(self, model, optimizer, batch_idx, num_instance):
        """训练一个batch"""
        optimizer.zero_grad()
        
        start_idx = batch_idx * self.args.batch_size
        end_idx = min(num_instance, start_idx + self.args.batch_size)
        
        # 获取batch数据
        sources_batch = self.dataset.train_data.sources[start_idx:end_idx]
        destinations_batch = self.dataset.train_data.destinations[start_idx:end_idx]
        edge_idxs_batch = self.dataset.train_data.edge_idxs[start_idx:end_idx]
        timestamps_batch = self.dataset.train_data.timestamps[start_idx:end_idx]

        # 生成负样本，现在返回shape为[n_neg, batch_size]
        negatives_batch = self.dataset.train_negative_sampler.generate_negative_samples(
            sources_batch, timestamps_batch, self.args.n_neg)

        source_masks = self.dataset.node_cg_mask[sources_batch]
        source_embs = self.dataset.node_cg_emb[sources_batch]
        source_adjs = self.dataset.node_cg_A[sources_batch]

        dest_masks = self.dataset.node_cg_mask[destinations_batch]
        dest_embs = self.dataset.node_cg_emb[destinations_batch]
        dest_adjs = self.dataset.node_cg_A[destinations_batch]

        neg_masks = self.dataset.node_cg_mask[negatives_batch]
        neg_embs = self.dataset.node_cg_emb[negatives_batch]
        neg_adjs = self.dataset.node_cg_A[negatives_batch]

        # 计算正样本和负样本的原始分数（用于BPR损失）
        pos_scores, neg_scores = model.compute_edge_scores(
            sources_batch, destinations_batch, negatives_batch, 
            source_masks, source_embs, source_adjs,
            dest_masks, dest_embs, dest_adjs,
            neg_masks, neg_embs, neg_adjs,
            timestamps_batch, edge_idxs_batch, self.args.n_neighbors)
        
        # 计算基于噪声剪枝的BPR损失
        bpr_loss = self.compute_bpr_loss(pos_scores, neg_scores)
        
        # 添加L2正则化项
        l2_loss = 0.0
        if self.args.l2_regularization > 0:
            for param in model.parameters():
                l2_loss += torch.norm(param, 2) ** 2
            l2_loss *= self.args.l2_regularization
        
        # 总损失 = BPR损失 + L2正则化
        total_loss = bpr_loss + l2_loss
        
        total_loss.backward()
        
        optimizer.step()

        return total_loss.item(), bpr_loss.item(), l2_loss if isinstance(l2_loss, float) else l2_loss.item()

    def validate(self, model):
        """验证模型"""
        model.set_neighbor_finder(self.dataset.full_ngh_finder)
        
        val_mrr, val_recall_10, val_recall_20 = self.eval_edge_prediction(
            model, self.dataset.val_data, self.args.n_neighbors, 
            self.dataset.full_negative_sampler, self.args.n_test_neg, self.args.batch_size)
        
        return val_mrr, val_recall_10, val_recall_20
    
    def test(self, model):
        """测试模型"""
        model.set_neighbor_finder(self.dataset.full_ngh_finder)
        
        test_mrr, test_recall_10, test_recall_20 = self.eval_edge_prediction(
            model, self.dataset.test_data, self.args.n_neighbors,
            self.dataset.full_negative_sampler, self.args.n_test_neg, self.args.batch_size)
        
        return test_mrr, test_recall_10, test_recall_20

    def eval_edge_prediction(self, model, data, n_neighbors, negative_sampler: NegativeSampler, n_test_neg, batch_size=256):
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

                source_masks = self.dataset.node_cg_mask[sources_batch]
                source_embs = self.dataset.node_cg_emb[sources_batch]
                source_adjs = self.dataset.node_cg_A[sources_batch]

                dest_masks = self.dataset.node_cg_mask[destinations_batch]
                dest_embs = self.dataset.node_cg_emb[destinations_batch]
                dest_adjs = self.dataset.node_cg_A[destinations_batch]

                neg_masks = self.dataset.node_cg_mask[negatives_batch]
                neg_embs = self.dataset.node_cg_emb[negatives_batch]
                neg_adjs = self.dataset.node_cg_A[negatives_batch]

                # 计算正样本和负样本的原始分数（用于BPR损失）
                pos_scores, neg_scores = model.compute_edge_scores(
                    sources_batch, destinations_batch, negatives_batch, 
                    source_masks, source_embs, source_adjs,
                    dest_masks, dest_embs, dest_adjs,
                    neg_masks, neg_embs, neg_adjs,
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


class CGFATrainer(DCRecBaseTrainer):
    def __init__(self, args, dataset, device):
        super().__init__(args, dataset, device)
        self.logger.info("Initializing CGFA trainer")

    def create_model(self):
        """创建CGFA模型"""
        cgfa = CGFA(
            in_channels=self.dataset.node_cg_emb.shape[-1],
            out_channels=self.args.cgfa_out_channels,
        )
        return cgfa.to(self.device)

    def train_batch(self, model, optimizer, batch_idx, num_instance):
        """训练一个batch"""
        optimizer.zero_grad()
        
        start_idx = batch_idx * self.args.batch_size
        end_idx = min(num_instance, start_idx + self.args.batch_size)
        
        # 获取batch数据
        sources_batch = self.dataset.train_data.sources[start_idx:end_idx]
        destinations_batch = self.dataset.train_data.destinations[start_idx:end_idx]
        edge_idxs_batch = self.dataset.train_data.edge_idxs[start_idx:end_idx]
        timestamps_batch = self.dataset.train_data.timestamps[start_idx:end_idx]

        # 生成负样本，现在返回shape为[n_neg, batch_size]
        negatives_batch = self.dataset.train_negative_sampler.generate_negative_samples(
            sources_batch, timestamps_batch, self.args.n_neg)

        source_masks = self.dataset.node_cg_mask[sources_batch]
        source_embs = self.dataset.node_cg_emb[sources_batch]
        source_adjs = self.dataset.node_cg_A[sources_batch]

        dest_masks = self.dataset.node_cg_mask[destinations_batch]
        dest_embs = self.dataset.node_cg_emb[destinations_batch]
        dest_adjs = self.dataset.node_cg_A[destinations_batch]

        neg_masks = self.dataset.node_cg_mask[negatives_batch]
        neg_embs = self.dataset.node_cg_emb[negatives_batch]
        neg_adjs = self.dataset.node_cg_A[negatives_batch]

        # 计算正样本和负样本的原始分数（用于BPR损失）
        pos_scores, neg_scores = model.compute_edge_scores(
            source_masks, source_embs, source_adjs,
            dest_masks, dest_embs, dest_adjs,
            neg_masks, neg_embs, neg_adjs)
        
        # 计算基于噪声剪枝的BPR损失
        bpr_loss = self.compute_bpr_loss(pos_scores, neg_scores)
        
        # 添加L2正则化项
        l2_loss = 0.0
        if self.args.l2_regularization > 0:
            for param in model.parameters():
                l2_loss += torch.norm(param, 2) ** 2
            l2_loss *= self.args.l2_regularization
        
        # 总损失 = BPR损失 + L2正则化
        total_loss = bpr_loss + l2_loss
        
        total_loss.backward()
        
        optimizer.step()

        return total_loss.item(), bpr_loss.item(), l2_loss if isinstance(l2_loss, float) else l2_loss.item()

    
    def validate(self, model):
        """验证模型"""
        model.set_neighbor_finder(self.dataset.full_ngh_finder)
        
        val_mrr, val_recall_10, val_recall_20 = self.eval_edge_prediction(
            model, self.dataset.val_data, self.args.n_neighbors, 
            self.dataset.full_negative_sampler, self.args.n_test_neg, self.args.batch_size)
        
        return val_mrr, val_recall_10, val_recall_20
    
    def test(self, model):
        """测试模型"""
        model.set_neighbor_finder(self.dataset.full_ngh_finder)
        
        test_mrr, test_recall_10, test_recall_20 = self.eval_edge_prediction(
            model, self.dataset.test_data, self.args.n_neighbors,
            self.dataset.full_negative_sampler, self.args.n_test_neg, self.args.batch_size)
        
        return test_mrr, test_recall_10, test_recall_20

    def eval_edge_prediction(self, model, data, n_neighbors, negative_sampler: NegativeSampler, n_test_neg, batch_size=256):
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

                source_masks = self.dataset.node_cg_mask[sources_batch]
                source_embs = self.dataset.node_cg_emb[sources_batch]
                source_adjs = self.dataset.node_cg_A[sources_batch]

                dest_masks = self.dataset.node_cg_mask[destinations_batch]
                dest_embs = self.dataset.node_cg_emb[destinations_batch]
                dest_adjs = self.dataset.node_cg_A[destinations_batch]

                neg_masks = self.dataset.node_cg_mask[negatives_batch]
                neg_embs = self.dataset.node_cg_emb[negatives_batch]
                neg_adjs = self.dataset.node_cg_A[negatives_batch]

                # 计算正样本和负样本的原始分数（用于BPR损失）
                pos_scores, neg_scores = model.compute_edge_scores(
                    source_masks, source_embs, source_adjs,
                    dest_masks, dest_embs, dest_adjs,
                    neg_masks, neg_embs, neg_adjs)

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