import gzip
import math
import pickle
import shutil
import time
import logging
import torch
import numpy as np
from tqdm import tqdm

from evaluation.evaluation import eval_edge_prediction
from model.tgn.tgn import TGN
from utils.utils import EarlyStopMonitor

class TGNTrainer:
    """TGN模型训练器"""
    def __init__(self, args, dataset, device, logger):
        self.args = args
        self.dataset = dataset
        self.device = device
        self.logger = logger
        
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
    
    def train_epoch(self, model, optimizer, epoch):
        """训练一个epoch"""
        if self.args.use_memory:
            model.memory.__init_memory__()
        
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
            postfix = {'total_loss': f'{total_loss:.4f}', 'bpr_loss': f'{bpr_loss:.4f}'}
            if self.args.l2_regularization > 0:
                postfix['l2_loss'] = f'{l2_loss:.6f}'
            progress_bar.set_postfix(postfix)
            
            if self.args.use_memory:
                model.memory.detach_memory()
        
        return {
            'total_loss': np.mean(epoch_total_losses),
            'bpr_loss': np.mean(epoch_bpr_losses),
            'l2_loss': np.mean(epoch_l2_losses),
        }
    
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
    
    def validate(self, model):
        """验证模型"""
        model.set_neighbor_finder(self.dataset.full_ngh_finder)
        
        val_mrr, val_recall_10, val_recall_20 = eval_edge_prediction(
            model, self.dataset.val_data, self.args.n_neighbors, 
            self.dataset.full_negative_sampler, self.args.n_test_neg, self.args.batch_size)
        
        return val_mrr, val_recall_10, val_recall_20
    
    def test(self, model):
        """测试模型"""
        model.set_neighbor_finder(self.dataset.full_ngh_finder)
        
        test_mrr, test_recall_10, test_recall_20 = eval_edge_prediction(
            model, self.dataset.test_data, self.args.n_neighbors,
            self.dataset.full_negative_sampler, self.args.n_test_neg, self.args.batch_size)
        
        return test_mrr, test_recall_10, test_recall_20
    
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
        if self.args.noise_pruning_ratio > 0:
            self.logger.info(f'Using noise pruning with ratio: {self.args.noise_pruning_ratio}')
        if self.args.l2_regularization > 0:
            self.logger.info(f'Using L2 regularization with coefficient: {self.args.l2_regularization}')
        
        # 模型保存路径
        model_save_path = f'saved_models/{self.args.prefix}-{self.args.data}.pth'
        memory_save_path = f'saved_models/{self.args.prefix}-{self.args.data}-memory.pkl'

        # for epoch in range(self.args.n_epoch):
        #     start_epoch = time.time()
            
        #     # 训练
        #     train_loss_dict = self.train_epoch(model, optimizer, epoch)
        #     train_total_losses.append(train_loss_dict['total_loss'])
        #     train_bpr_losses.append(train_loss_dict['bpr_loss'])
        #     train_l2_losses.append(train_loss_dict['l2_loss'])
            
        #     epoch_time = time.time() - start_epoch
        #     epoch_times.append(epoch_time)
            
        #     total_epoch_time = time.time() - start_epoch
        #     total_epoch_times.append(total_epoch_time)
            
        #     # 记录训练日志
        #     self.logger.info(f'epoch: {epoch+1} took {total_epoch_time:.2f}s')
        #     log_msg = f'Epoch losses - Total: {train_loss_dict["total_loss"]:.4f}, BPR: {train_loss_dict["bpr_loss"]:.4f}'
        #     if self.args.l2_regularization > 0:
        #         log_msg += f', L2: {train_loss_dict["l2_loss"]:.6f}'
        #     self.logger.info(log_msg)
            
        #     # 验证阶段：从第n_skip_val+1个epoch开始
        #     if epoch >= self.args.n_skip_val:
        #         if epoch == self.args.n_skip_val:
        #             # 初次开始验证时初始化早停监控器
        #             early_stopper = EarlyStopMonitor(max_round=self.args.patience)
        #             self.logger.info(f'Starting validation from epoch {epoch+1}')
                
        #         # 验证
        #         val_mrr, val_recall_10, val_recall_20 = self.validate(model)
        #         val_mrrs.append(val_mrr)
        #         val_recall_10s.append(val_recall_10)
        #         val_recall_20s.append(val_recall_20)
                
        #         self.logger.info(f'val MRR: {val_mrr:.4f}, Recall@10: {val_recall_10:.4f}, Recall@20: {val_recall_20:.4f}')
                
        #         # 如果是新的最佳性能，保存模型
        #         if val_mrr > best_val_mrr:
        #             best_val_mrr = val_mrr
        #             best_epoch = epoch
                    
        #             # 保存最佳模型
        #             torch.save(model.state_dict(), model_save_path)
        #             with gzip.open(memory_save_path, 'wb') as f:
        #                 pickle.dump(model.memory.messages, f)
                    
        #             self.logger.info(f'New best validation MRR: {val_mrr:.4f} at epoch {epoch+1}, model saved')
                
        #         # 早停检查
        #         if early_stopper.early_stop_check(val_mrr):
        #             self.logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
        #             self.logger.info(f'Best model was at epoch {best_epoch+1} with MRR: {best_val_mrr:.4f}')
        #             break
        #     else:
        #         # 填充空值以保持数组长度一致
        #         val_mrrs.append(0.0)
        #         val_recall_10s.append(0.0)
        #         val_recall_20s.append(0.0)
        #         self.logger.info('Validation skipped (warm-up period)')

        # # 如果没有进行过验证（所有epoch都被跳过），保存最后的模型
        # if best_epoch == -1:
        #     self.logger.info('No validation performed, saving final model')
        #     torch.save(model.state_dict(), model_save_path)
        #     best_epoch = self.args.n_epoch - 1
        
        # 加载最佳模型进行测试（包括模型的记忆）
        model.memory.clear_all_messages()
        model.load_state_dict(torch.load(model_save_path, weights_only=True))
        model.eval()
        
        # 测试
        test_mrr, test_recall_10, test_recall_20 = self.test(model)
        
        self.logger.info('Test statistics: MRR: {:.4f}, Recall@10: {:.4f}, Recall@20: {:.4f}'.format(test_mrr, test_recall_10, test_recall_20))
        self.logger.info('TGN model training completed')
        
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
