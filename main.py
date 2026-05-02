import os
import torch
import random
import warnings
import numpy as np
from exp.calmpro_trainer import calmproTrainer
from data_provider.processors import AliPreprocessor

warnings.filterwarnings('ignore')

model_trainer_dict = {
    'calmpro': calmproTrainer
}

data_process_dict = {
    'ali': AliPreprocessor
}

def setup_reproducibility(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    # ==========================
    # 核心环境与模式配置 (Core)
    # ==========================
    parser.add_argument('--device', type=str, default='cuda', help='运算设备 (cuda/cpu)')
    parser.add_argument('--phase', type=str, default='pretrain', choices=['pretrain', 'train', 'test', 'all'], help='运行阶段')
    parser.add_argument('--train_ssn', type=int, default=0, help='是否训练SSN (1: 训练, 0: 不训练)')
    parser.add_argument('--train_cls', type=int, default=0, help='是否训练patch分类头 (1: 训练, 0: 不训练)')
    parser.add_argument('--test_ssn', type=int, default=0, help='是否测试SSN (1: 测试, 0: 不测试)')
    parser.add_argument('--test_cls', type=int, default=0, help='是否测试patch分类头 (1: 测试, 0: 不测试)')
    parser.add_argument('--model', type=str, default='calmpro', help='选择训练哪个模型')
    parser.add_argument('--seed', type=int, default=42, help='随机种子 (虽然代码里写死在setup func, 但保留参数是个好习惯)')
    parser.add_argument('--do_stasca', action='store_true', help='是否对数据做标准化，与ReVin配套使用，如果该项为false，则ReVin要为true，反之该项为true，ReVin要为false')

    # ==========================
    # 数据集配置 (Data)
    # ==========================
    parser.add_argument('--data', type=str, default='d1', help='数据集名称 (对应 processors key)')
    parser.add_argument('--raw_root_path', type=str, default='data/raw_data/d1', help='原始数据根目录')
    parser.add_argument('--sta_root_path', type=str, default='data/sta_data/d1', help='处理后数据保存目录')
    parser.add_argument('--data_path', type=str, nargs='+', default=[], help='(可选) 指定读取的具体文件名列表')
    
    # 数据预处理参数
    parser.add_argument('--win_size', type=int, default=6, help='时间窗口大小 (T)')
    parser.add_argument('--test_size', type=float, default=0.3, help='测试集划分比例')
    parser.add_argument('--num_channels', type=int, default=15, help='输入特征维度')
    parser.add_argument('--num_class', type=int, default=2, help='分类数')
    parser.add_argument('--gene_des', type=int, default=0, help='是否生成描述性统计特征 (1: 生成, 0: 不生成)')
    parser.add_argument('--llm_model', type=str, default='gpt-3.5-turbo', help='用于生成描述性统计特征的LLM模型名称')
    parser.add_argument('--llm_max_tokens', type=int, default=100, help='用于生成描述性统计特征的LLM模型最大输出token数')
    parser.add_argument('--llm_temperature', type=float, default=0.7, help='用于生成描述性统计特征的LLM模型温度参数')
    parser.add_argument('--llm_api_key', type=str, default='', help='用于调用LLM的API Key')
    parser.add_argument('--llm_base_url', type=str, default='', help='用于调用LLM的Base URL')
    parser.add_argument('--gene_splits', nargs='+', default=['train', 'test'], choices=['train', 'test'], help='指定为哪些数据集生成描述性统计特征 (如 --gene_splits train test)')


    # ==========================
    # 模型训练超参数 (Training)
    # ==========================
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='通用批次大小')
    parser.add_argument('--backbone_batch_size', type=int, default=32, help='骨干批次大小')
    parser.add_argument('--ssn_batch_size', type=int, default=32, help='SSN批次大小')
    parser.add_argument('--cls_batch_size', type=int, default=32, help='分类头批次大小')
    parser.add_argument('--lr', type=float, default=5e-2, help='学习率')
    parser.add_argument('--lr_ssn', type=float, default=1e-4, help='学习率')
    parser.add_argument('--lr_cls', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout 概率')
    parser.add_argument('--early_stopping', action='store_true', help='启用早停策略')
    parser.add_argument('--patience', type=int, default=20, help='早停容忍的无提升轮数')
    parser.add_argument('--warmup_start_ratio', type=float, default=0.1, help='学习率预热起始比例')
    parser.add_argument('--plateau_patience', type=int, default=10, help='学习率调整中plateau策略的patience参数')
    parser.add_argument('--lr_decay_factor', type=float, default=0.5, help='学习率衰减因子 (用于plateau策略)')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='学习率下限 (用于plateau策略)')
    parser.add_argument('--T_0', type=int, default=10, help='Cosine Annealing周期长度 (用于cosanneal策略)')
    parser.add_argument('--T_mult', type=int, default=2, help='Cosine Annealing周期倍增因子 (用于cosanneal策略)')

    # ==========================
    # 检查点路径 (Checkpoints)
    # ==========================
    parser.add_argument('--ckpt_path', type=str, default='checkpoints/TranAD.pth', help='通用模型保存路径')
    parser.add_argument('--SSN_path', type=str, default='checkpoints/SSN.pth', help='SoftShapeNet 模型保存路径')
    parser.add_argument('--cls_path', type=str, default='checkpoints/cls_head.pth', help='分类模型保存路径')
    
    # causal相关配置
    parser.add_argument('--lam', type=float, default=2e-3)
    parser.add_argument('--lam_ridge', type=float, default=1e-2)
    parser.add_argument('--penalty', type=str, default='H', choices=['H', 'GL', 'GSGL'])

    # tokenizer相关配置
    parser.add_argument('--emb_dim', type=int, default=256, help='SoftShapeNet embedding dimension')
    parser.add_argument('--depth', type=int, default=2, help='SoftShapeNet depth')
    parser.add_argument('--sparse_rate', type=float, default=0.60, help='SoftShapeNet sparse rate')
    parser.add_argument('--shape_size', type=int, default=5, help='SoftShapeNet shape size')
    parser.add_argument('--shape_use_ratio', type=int, default=0, help='SoftShapeNet shape use ratio')
    parser.add_argument('--shape_ratio', type=float, default=0.0, help='SoftShapeNet shape ratio')
    parser.add_argument('--shape_stride', type=int, default=2, help='SoftShapeNet shape stride')
    parser.add_argument('--moe_num_experts', type=int, default=3, help='SoftShapeNet Mixture of Experts数量')
    parser.add_argument('--warm_up_epoch', type=int, default=100, help='SoftShapeNet 预热轮数')
    parser.add_argument('--moeloss_rate', type=float, default=0.001, help='SoftShapeNet MoE损失权重')
    parser.add_argument('--RevIN', type=int, default=1, help='是否使用RevIN (1: 使用, 0: 不使用)')
    parser.add_argument('--process', type=str, default='normal', help='数据处理方式 (normal/shape)')
    parser.add_argument('--alpha', type=float, default=0.7, help='SoftShapeNet alpha参数')
    parser.add_argument('--attention_head_dim', type=int, default=8, help='SoftShapeNet attention head dimension')
    parser.add_argument('--raw', type=int, default=1, help='SoftShapeNet raw parameter')
    parser.add_argument('--verbose', type=int, default=1, help='SoftShapeNet verbose parameter')
    parser.add_argument('--affine', type=int, default=0)
    parser.add_argument('--subtract_last', type=int, default=0)

    args = parser.parse_args()

    args.ckpt_path = os.path.join('checkpoints', f'{args.model}_win_{args.win_size}_lr{args.lr}_batchsize{args.backbone_batch_size}.pth')
    args.SSN_path = os.path.join('checkpoints', f'softshape_embdim{args.emb_dim}_affine{args.affine}_a{args.alpha}_shsi{args.shape_size}_shst{args.shape_stride}_spra{args.sparse_rate}_attehdim{args.attention_head_dim}_lr{args.lr_ssn}_batchsize{args.ssn_batch_size}.pth')
    args.cls_path = os.path.join('checkpoints', f'{args.model}_cls_head_ssnattehdim{args.attention_head_dim}_ssnlr{args.lr_ssn}_ssnbatchsize{args.ssn_batch_size}_lr{args.lr_cls}_batchsize{args.cls_batch_size}.pth')
    
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('dataset', exist_ok=True)

    setup_reproducibility(seed=args.seed)

    if not os.path.exists(os.path.join(args.sta_root_path, 'X_train.npy')):
        print('数据还未处理, 正在处理...')
        # 获取类并实例化
        ProcessorClass = data_process_dict[args.data]
        processor = ProcessorClass(args)
        
        # 运行
        processor.run()
    else:
        print('数据已存在, 无须处理直接读取训练')

    model_trainer = model_trainer_dict[args.model](args)
    model_trainer.run()
