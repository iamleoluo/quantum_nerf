"""
NeRF 主腳本

用於運行 NeRF 的訓練和渲染流程
"""

import os
import torch
import argparse
from src.models.nerf import NeRFNetwork
from src.models.encoding import PositionalEncoder
from src.rendering.volume_renderer import VolumeRenderer
from src.rendering.ray_sampling import RaySampler
from src.training.trainer import NeRFTrainer
from src.utils.data_loader import create_data_loaders


def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='NeRF Training')
    
    # 數據相關參數
    parser.add_argument('--data_dir', type=str, required=True,
                      help='數據集目錄路徑')
    parser.add_argument('--batch_size', type=int, default=1024,
                      help='批次大小')
    
    # 模型相關參數
    parser.add_argument('--pos_encode_dim', type=int, default=63,
                      help='位置編碼維度')
    parser.add_argument('--dir_encode_dim', type=int, default=27,
                      help='方向編碼維度')
    parser.add_argument('--hidden_dim', type=int, default=256,
                      help='隱藏層維度')
    parser.add_argument('--num_layers', type=int, default=8,
                      help='網絡層數')
    
    # 渲染相關參數
    parser.add_argument('--n_samples', type=int, default=64,
                      help='每條射線的採樣點數量')
    parser.add_argument('--n_fine', type=int, default=128,
                      help='細採樣的點數量')
    parser.add_argument('--white_bkgd', action='store_true',
                      help='是否使用白色背景')
    
    # 訓練相關參數
    parser.add_argument('--lr', type=float, default=5e-4,
                      help='學習率')
    parser.add_argument('--num_epochs', type=int, default=100,
                      help='訓練輪數')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                      help='檢查點保存目錄')
    parser.add_argument('--log_dir', type=str, default='logs',
                      help='日誌保存目錄')
    
    return parser.parse_args()


def main():
    """主函數"""
    # 解析參數
    args = parse_args()
    
    # 設置設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用設備: {device}')
    
    # 創建數據加載器
    train_loader, val_loader = create_data_loaders(
        args.data_dir,
        batch_size=args.batch_size
    )
    
    # 創建模型組件
    model = NeRFNetwork(
        pos_encode_dim=args.pos_encode_dim,
        dir_encode_dim=args.dir_encode_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    ).to(device)
    
    encoder = PositionalEncoder(
        input_dim=3,
        num_freqs=10,
        include_input=True
    ).to(device)
    
    renderer = VolumeRenderer({
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': 0.0
    })
    
    sampler = RaySampler({
        'n_samples': args.n_samples,
        'perturb': True,
        'n_fine': args.n_fine
    })
    
    # 創建訓練器
    trainer = NeRFTrainer(
        model=model,
        encoder=encoder,
        renderer=renderer,
        sampler=sampler,
        train_loader=train_loader,
        val_loader=val_loader,
        config={
            'lr': args.lr,
            'checkpoint_dir': args.checkpoint_dir,
            'log_dir': args.log_dir,
            'rgb_weight': 1.0,
            'depth_weight': 0.1,
            'l1_weight': 0.0,
            'l2_weight': 0.0,
            'tv_weight': 0.0,
            'optimizer_type': 'adam',
            'scheduler_type': 'warmup_cosine',
            'warmup_steps': 1000,
            'max_steps': args.num_epochs * len(train_loader)
        }
    )
    
    # 開始訓練
    print('開始訓練...')
    trainer.train(args.num_epochs)
    print('訓練完成！')


if __name__ == '__main__':
    main() 