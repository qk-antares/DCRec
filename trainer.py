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
        
        epoch_losses = []
        progress_bar = tqdm(range(num_batch), desc=f'Epoch {epoch+1}/{self.args.n_epoch}', unit='batch')
        
        for batch_idx in progress_bar:
            loss = self.train_batch(model, optimizer, batch_idx, num_instance)
            epoch_losses.append(loss)
            progress_bar.set_postfix({'loss': f'{loss:.4f}'})
            
            if self.args.use_memory:
                model.memory.detach_memory()
        
        return np.mean(epoch_losses)
    
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
        
        # 计算BPR损失: -log(sigmoid(pos_score - neg_score))
        bpr_loss = self.compute_bpr_loss(pos_scores, neg_scores)
        
        bpr_loss.backward()
        optimizer.step()
        
        return bpr_loss.item()
    
    def compute_bpr_loss(self, pos_scores, neg_scores):
        """
        计算BPR损失
        Args:
            pos_scores: [batch_size] - 正样本分数
            neg_scores: [n_neg, batch_size] - 负样本分数
        """
        # 将pos_scores扩展为[n_neg, batch_size]以便与neg_scores进行比较
        pos_scores_expanded = pos_scores.unsqueeze(0).expand_as(neg_scores)  # [n_neg, batch_size]
        
        # BPR损失: -log(sigmoid(pos_score - neg_score))
        # 等价于: log(1 + exp(neg_score - pos_score))
        score_diff = neg_scores - pos_scores_expanded  # [n_neg, batch_size]
        
        # 使用logsigmoid来数值稳定地计算log(sigmoid(x))
        # log(sigmoid(x)) = -log(1 + exp(-x)) = -softplus(-x)
        loss = -torch.nn.functional.logsigmoid(-score_diff)  # [n_neg, batch_size]
        
        # 对所有负样本和batch样本取平均
        return loss.mean()
    
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
        train_losses = []
        epoch_times = []
        total_epoch_times = []
        
        early_stopper = EarlyStopMonitor(max_round=self.args.patience)
        
        num_instance = len(self.dataset.train_data.sources)
        num_batch = math.ceil(num_instance / self.args.batch_size)
        
        self.logger.info('num of training instances: {}'.format(num_instance))
        self.logger.info('num of batches per epoch: {}'.format(num_batch))
        
        for epoch in range(self.args.n_epoch):
            start_epoch = time.time()
            
            # 训练
            train_loss = self.train_epoch(model, optimizer, epoch)
            train_losses.append(train_loss)
            
            epoch_time = time.time() - start_epoch
            epoch_times.append(epoch_time)
            
            # 验证
            val_mrr, val_recall_10, val_recall_20 = self.validate(model)
            val_mrrs.append(val_mrr)
            val_recall_10s.append(val_recall_10)
            val_recall_20s.append(val_recall_20)
            
            total_epoch_time = time.time() - start_epoch
            total_epoch_times.append(total_epoch_time)
            
            # 记录日志
            self.logger.info(f'epoch: {epoch+1} took {total_epoch_time:.2f}s')
            self.logger.info(f'Epoch mean loss: {train_loss}')
            self.logger.info(f'val MRR: {val_mrr:.4f}, Recall@10: {val_recall_10:.4f}, Recall@20: {val_recall_20:.4f}')
            
            get_checkpoint_path = lambda epoch: f'saved_checkpoints/{self.args.prefix}-{self.args.data}-{epoch+1}.pth'
            get_memory_path = lambda epoch: f'saved_memory/{self.args.prefix}-{self.args.data}-{epoch+1}.gz'

            # 保存模型
            torch.save(model.state_dict(), get_checkpoint_path(epoch))
            # with gzip.open(get_memory_path(epoch), 'wb') as f:
            #     pickle.dump(model.memory.messages, f)

            # 早停检查（使用MRR作为主要指标）
            if early_stopper.early_stop_check(val_mrr):
                self.logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
                self.logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
                best_model_path = get_checkpoint_path(early_stopper.best_epoch)
                model.load_state_dict(torch.load(best_model_path))
                # with gzip.open(get_memory_path(epoch), 'rb') as f:
                #     model.messages = pickle.load(f)
                self.logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
                model.eval()
                break

        # 获取【最佳的那个epoch】验证结束时的内存状态
        # if self.args.use_memory:
        #     val_memory_backup = model.memory.backup_memory()
        
        # 测试
        test_mrr, test_recall_10, test_recall_20 = self.test(model)
        
        self.logger.info('Test statistics: MRR: {:.4f}, Recall@10: {:.4f}, Recall@20: {:.4f}'.format(test_mrr, test_recall_10, test_recall_20))
        
        # 保存模型
        self.logger.info('Saving TGN model')
        shutil.copy(get_checkpoint_path(early_stopper.best_epoch), f'saved_models/{self.args.prefix}-{self.args.data}.pth')
        shutil.copy(get_memory_path(early_stopper.best_epoch), f'saved_models/{self.args.prefix}-{self.args.data}-memory.pkl')
        # if self.args.use_memory:
        #     model.memory.restore_memory(val_memory_backup)
        # model_save_path = 
        # torch.save(model.state_dict(), model_save_path)
        self.logger.info('TGN model saved')
        
        return {
            "val_mrrs": val_mrrs,
            "val_recall_10s": val_recall_10s,
            "val_recall_20s": val_recall_20s,
            "test_mrr": test_mrr,
            "test_recall_10": test_recall_10,
            "test_recall_20": test_recall_20,
            "epoch_times": epoch_times,
            "train_losses": train_losses,
            "total_epoch_times": total_epoch_times
        }
