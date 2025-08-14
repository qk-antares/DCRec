import torch
import numpy as np
import pickle
import random
from pathlib import Path

from utils.config import parse_arguments
from utils.logger_utils import setup_logger
from utils.dataset import Dataset
from trainer import TGNTrainer

def set_all_seeds(seed=0):
    """设置所有随机种子以确保可重现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 如果使用GPU，也设置GPU的随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 对于多GPU
    
    # 设置PyTorch的随机数生成器状态
    torch.use_deterministic_algorithms(True, warn_only=True)

def create_directories():
    """创建必要的目录"""
    Path("saved_models/").mkdir(parents=True, exist_ok=True)
    Path("saved_checkpoints/").mkdir(parents=True, exist_ok=True)
    Path("saved_memory/").mkdir(parents=True, exist_ok=True)
    Path("results/").mkdir(parents=True, exist_ok=True)
    Path("log/").mkdir(parents=True, exist_ok=True)


def main():
    """主函数"""
    # 解析参数和设置配置
    args = parse_arguments()
    
    # 创建目录
    create_directories()

    # 设置日志
    logger = setup_logger(args)
    
    # 加载数据集
    dataset = Dataset(args.data, args.randomize_features, args.train_ratio, args.valid_ratio, args.inductive, args.uniform, logger)

    # 设置设备
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # 运行实验
    set_all_seeds(args.seed)
    
    # 创建训练器并训练模型
    trainer = TGNTrainer(args, dataset, device, logger)
    results = trainer.train_model()
    
    # 保存结果
    results_path = f"results/{args.prefix}.pkl"
    pickle.dump(results, open(results_path, "wb"))
    logger.info(f'Results saved to {results_path}')

if __name__ == '__main__':
    main()
