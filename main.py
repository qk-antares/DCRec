import torch
import numpy as np
import pickle
from pathlib import Path

from utils.config import parse_arguments
from utils.logger_utils import setup_logger
from utils.dataset import Dataset
from trainer import TGNTrainer

torch.manual_seed(0)
np.random.seed(0)

def create_directories():
    """创建必要的目录"""
    Path("saved_models/").mkdir(parents=True, exist_ok=True)
    Path("saved_checkpoints/").mkdir(parents=True, exist_ok=True)
    Path("results/").mkdir(parents=True, exist_ok=True)
    Path("log/").mkdir(parents=True, exist_ok=True)


def run_single_experiment(args, dataset, device, logger, run_id):
    """运行单次实验"""
    results_path = f"results/{args.prefix}_{run_id}.pkl"
    
    # 创建训练器并训练模型
    trainer = TGNTrainer(args, dataset, device, logger)
    results = trainer.train_model()
    
    # 保存结果
    pickle.dump(results, open(results_path, "wb"))
    logger.info(f'Results saved to {results_path}')
    
    return results

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
    
    # 运行多次实验
    all_results = []
    for run_id in range(args.n_runs):
        logger.info(f'Starting run {run_id+1}/{args.n_runs}')
        results = run_single_experiment(args, dataset, device, logger, run_id)
        all_results.append(results)
        logger.info(f'Run {run_id} completed')

if __name__ == '__main__':
    main()
