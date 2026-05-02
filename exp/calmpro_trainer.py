import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from data_provider.unified_dataset import UnifiedDataset
from sklearn.metrics import classification_report, confusion_matrix
from utils.utils import ridge_regularize, prox_update, regularize, convert_to_list
from utils.visual_utils import ssn_vis
from utils.llm_utils import load_baseline_gc_lag1, _query_patch_description_only
from exp.base_trainer import BaseTrainer
from models.tc_mlp import TCMLP
from models.polymorphic_patch_tokenizer import PolymorphicPatchTokenizer
from torch.utils.data import DataLoader, Subset
import json
from models.OpenClipTextEncoder import OpenClipTextEncoder
from models.Multi_CaLMPro import MultiPatchTextFusionModelDirect_6way
from openai import OpenAI


class calmproTrainer(BaseTrainer):
    def pretrain(self, args):
        full_dataset = UnifiedDataset(root_path=args.sta_root_path, flag='train')
        N, T, F = full_dataset.X.shape
        health_idx = np.where(full_dataset.y == 0)[0]
        train_dataset = Subset(full_dataset, health_idx)

        print(f'筛选完成: 总训练集 {len(full_dataset)} -> 预训练健康样本 {len(train_dataset)}')

        train_dl = DataLoader(train_dataset, batch_size=args.backbone_batch_size, shuffle=True)

        cmlp = TCMLP(num_series=F, lag=args.win_size, affine=0, subtract_last=0).to(device=args.device)

        cmlp.train()
        batch_lam_ridge = args.lam_ridge / len(train_dl)
        batch_lam_prox = args.lam / len(train_dl)
        loss_fn = nn.MSELoss(reduction='mean')

        resume_path = args.ckpt_path.replace('.pth', '_pretrain_resume.pth')
        start_epoch = 0
        best_loss = float('inf')
        early_stop_counter = 0

        ckpt = self._load_ckpt(resume_path, device=args.device)
        if ckpt is not None:
            cmlp.load_state_dict(ckpt['model_state'])
            start_epoch = ckpt['epoch'] + 1
            best_loss = ckpt['best_loss']
            early_stop_counter = ckpt['early_stop_counter']
            print(f'[Resume] 从Epoch {start_epoch} 继续, 历史最优 Loss {best_loss:.6f}, 早停计数 {early_stop_counter}')

        if start_epoch >= args.epochs:
            print(f'预训练已完成: 已达到指定的最大 Epoch {args.epochs}')
            return

        for epoch in range(start_epoch, args.epochs):
            epoch_mse_loss = []
            loop = tqdm(train_dl, desc=f'Epoch {epoch+1}/{args.epochs}')

            for batch_X, _, _ in loop:
                batch_X = batch_X.to(args.device)

                for param in cmlp.parameters():
                    param.grad = None
                
                pred_X, _, _, _, true_X = cmlp(batch_X)
                
                mse_loss = loss_fn(pred_X, true_X) * true_X.shape[-1]
                ridge_loss = sum(ridge_regularize(net, batch_lam_ridge) for net in cmlp.networks)
                smooth_loss = mse_loss + ridge_loss

                smooth_loss.backward()
                nn.utils.clip_grad_norm_(cmlp.parameters(), max_norm=1.0)
                with torch.no_grad():
                    for param in cmlp.parameters():
                        param -= args.lr * param.grad
                
                if args.lam > 0:
                    with torch.no_grad():
                        for net in cmlp.networks:
                            prox_update(net, batch_lam_prox, args.lr, args.penalty)

                epoch_mse_loss.append(mse_loss.item())

            with torch.no_grad():
                avg_mse_loss = sum(epoch_mse_loss) / len(epoch_mse_loss)
                all_ridge_loss = sum(ridge_regularize(net, args.lam_ridge) for net in cmlp.networks)
                all_nonsmooth_loss = sum([regularize(net, args.lam, args.penalty) for net in cmlp.networks])
                mean_loss = (avg_mse_loss + all_ridge_loss.item() + all_nonsmooth_loss.item()) / F

                if mean_loss < best_loss:
                    best_loss = mean_loss
                    early_stop_counter = 0
                    torch.save(cmlp.state_dict(), args.ckpt_path)
                    print(f'---> 最佳模型已更新, 当前最优 Loss: {best_loss:.6f}')
                else:
                    early_stop_counter += 1
                    if args.early_stopping:
                        print(f'早停计数: {early_stop_counter}/{args.patience}, 还差 {args.patience - early_stop_counter} 轮触发早停')

                if args.verbose > 0:
                    print(('-' * 10 + 'Iter = %d' + '-' * 10) % (epoch + 1))
                    print('Loss = %f' % mean_loss)
                    print('Variable usage = %.2f%%'
                        % (100 * torch.mean(cmlp.GC().float())))

            self._save_ckpt(resume_path, {
                'epoch': epoch,
                'model_state': cmlp.state_dict(),
                'best_loss': best_loss,
                'early_stop_counter': early_stop_counter
            })

            if args.early_stopping and early_stop_counter >= args.patience:
                print(f'早停触发: 连续 {args.patience} 轮无提升, 于 Epoch {epoch+1} 停止训练')
                break
        
        if os.path.exists(resume_path):
            os.remove(resume_path)
            print(f'预训练完成, 已删除断点文件: {resume_path}')
        print(f'Stage 1 完成, 最佳模型已保存至: {args.ckpt_path}')

    def train(self, args):
        if args.train_ssn:
            self._train_ssn(args)
        
        gen_splits = getattr(args, 'gen_splits', ['train', 'test'])

        if args.gene_des:
            for split in gen_splits:
                feat_path = os.path.join(args.raw_root_path, f"text_features_{split}.npy")

                print(f'[{split}] 生成描述中...')
                self._gene_des(args, split=split)

                if not os.path.exists(feat_path):
                    print(f'[{split}] 文本特征不存在，正在生成...')
                    self._gene_text_features(args, split=split)
                else:
                    print(f'[{split}] 文本特征已存在，跳过生成。')
        
        if args.train_cls:
            self._train_cls(args)
        
    def _train_ssn(self, args):
        train_dataset = UnifiedDataset(root_path=args.sta_root_path, flag='train')
        _, T, F = train_dataset.X.shape
        train_dl = DataLoader(train_dataset, batch_size=args.ssn_batch_size, shuffle=True)

        max_feature_path = os.path.join(args.sta_root_path, 'max_feature.npy')
        if os.path.exists(max_feature_path):
            max_feature = torch.tensor(np.load(max_feature_path), device=args.device, dtype=torch.float32)
        else:
            raise FileNotFoundError(f"Error: {max_feature_path} not found. Run preprocessor first!")

        cmlp = TCMLP(num_series=F, lag=args.win_size, affine=0, subtract_last=0).to(device=args.device)
        cmlp.load_state_dict(torch.load(args.ckpt_path, map_location=args.device))
        cmlp.eval()
        for param in cmlp.parameters():
            param.requires_grad = False

        ss_net = PolymorphicPatchTokenizer(
            seq_len=T - args.win_size, 
            shape_size=args.shape_size, 
            num_channels=args.num_channels, 
            emb_dim=args.emb_dim, 
            sparse_rate=args.sparse_rate, 
            depth=args.depth, 
            num_experts=args.moe_num_experts, 
            num_classes=args.num_class,
            raw=args.raw, 
            stride=args.shape_stride, 
            RevIN=args.RevIN,
            affine=args.affine,
            subtract_last=args.subtract_last,
            alpha=args.alpha,
            attention_head_dim=args.attention_head_dim
        ).to(args.device)
        
        optimizer = torch.optim.Adam(ss_net.parameters(), lr=args.lr_ssn, weight_decay=args.weight_decay)

        warmup_start_ratio = getattr(args, 'warmup_start_ratio', 0.1)

        # 核心修正 1：预见性 Lambda 逻辑，提前计算下一轮的倍率
        def warmup_lambda(current_step):
            next_epoch = current_step + 1
            if next_epoch < args.warm_up_epoch:
                return warmup_start_ratio + (1.0 - warmup_start_ratio) * (next_epoch / max(1, args.warm_up_epoch - 1))
            return 1.0
        
        # 必须在修改 optimizer 现行 lr 之前定义 Scheduler，以正确捕获 args.lr_ssn 作为 base_lr
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)

        # 核心修正 2：移除弃用的 verbose 参数
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=args.lr_decay_factor, 
            patience=args.plateau_patience,
            min_lr=args.min_lr
        )

        loss_fn = nn.CrossEntropyLoss()

        resume_path = args.SSN_path.replace('.pth', '_resume.pth')
        start_epoch = 0
        min_train_loss = float('inf')
        best_epoch = -1
        early_stop_counter = 0

        ckpt = self._load_ckpt(resume_path, device=args.device)
        if ckpt is not None:
            ss_net.load_state_dict(ckpt['model_state'])
            optimizer.load_state_dict(ckpt['optimizer_state'])
            if 'warmup_scheduler_state' in ckpt and 'plateau_scheduler_state' in ckpt:
                warmup_scheduler.load_state_dict(ckpt['warmup_scheduler_state'])
                plateau_scheduler.load_state_dict(ckpt['plateau_scheduler_state'])
            start_epoch = ckpt['epoch'] + 1
            min_train_loss = ckpt['min_train_loss']
            best_epoch = ckpt['best_epoch']
            early_stop_counter = ckpt['early_stop_counter']

            # 核心修正 3：展示层采用 "+1" 逻辑，统一使用人类直觉的 1-indexed 显示
            if start_epoch < args.warm_up_epoch:
                min_train_loss = float('inf')
                best_epoch = -1
                tqdm.write(f'[Resume] 从 Epoch {start_epoch + 1} 继续 (Warm-up阶段), 历史最优 Loss 已重置, 早停计数 {early_stop_counter}')
            else:
                tqdm.write(f'[Resume] 从 Epoch {start_epoch + 1} 继续, 历史最优 Loss {min_train_loss:.6f} at Epoch {best_epoch + 1}, 早停计数 {early_stop_counter}')
        else:
            # 核心修正 4：新训练起步，手动将 Epoch 0 的初始学习率压降至安全阈值
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr_ssn * warmup_start_ratio
        
        if start_epoch >= args.epochs:
            print(f'[Skip] 已达到指定的最大 Epoch {args.epochs}, 跳过')
            return

        # 核心修正 5：原代码此处为 range(args.epochs)，会导致恢复训练时重复从 0 遍历，现改为 start_epoch
        for epoch in range(start_epoch, args.epochs):
            in_warmup = epoch < args.warm_up_epoch
            epoch_display = epoch + 1  # 专门用于对外展示的 Epoch 变量

            ss_net.train()
            epoch_train_loss = 0
            num_iterations = 0
            correct = total = 0
            current_lr = optimizer.param_groups[0]['lr']
            
            # 使用 epoch_display
            loop = tqdm(train_dl, desc=f"Epoch {epoch_display}/{args.epochs} | LR: {current_lr:.2e} | {'Warm-up' if in_warmup else 'Main'}")
            
            for batch_X, batch_y, _ in loop:
                batch_X, batch_y = batch_X.to(args.device), batch_y.to(args.device)
                optimizer.zero_grad()

                with torch.no_grad():
                    _, _, _, res, _ = cmlp(batch_X)

                target_slice = batch_X[:, args.win_size:, :]
                current_stats = target_slice / (max_feature)
                
                logits, moe_loss, _, _, _ = ss_net(res, current_stats, num_epoch_i=epoch, warm_up_epoch=args.warm_up_epoch)
                
                cls_loss = loss_fn(logits, batch_y)
                total_loss = cls_loss + args.moeloss_rate * moe_loss
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(ss_net.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_train_loss += total_loss.item()
                num_iterations += 1
                pred = torch.argmax(logits.detach(), dim=1)
                total += batch_y.size(0)
                correct += (pred == batch_y).sum().item()
                
                loop.set_postfix(loss=total_loss.item(), lr=f'{optimizer.param_groups[0]["lr"]:.2e}')
            
            avg_train_loss = epoch_train_loss / num_iterations
            train_acc = 100 * correct / total
            
            if in_warmup:
                warmup_scheduler.step()
                tqdm.write(f"[Warm-up] Epoch {epoch_display}: Loss {avg_train_loss:.6f} | Acc: {train_acc:.6f}% (LR: {optimizer.param_groups[0]['lr']:.2e})")
            else:
                # 建议：如果后续引入了验证集，请将 avg_train_loss 替换为 val_loss 以更准确地监测平台期
                plateau_scheduler.step(avg_train_loss)

            tqdm.write(f"Epoch {epoch_display}: Avg Loss {avg_train_loss:.6f} | Acc: {train_acc:.6f}% | Phase: {'Warmup (' + str(epoch_display) + '/' + str(args.warm_up_epoch) + ')' if in_warmup else 'Main'}")

            if in_warmup:
                tqdm.write('---> 处于 Warm-up 阶段，当前模型性能不参与最优模型判定与早停计数')
            else:
                if avg_train_loss < min_train_loss:
                    min_train_loss = avg_train_loss
                    best_epoch = epoch  # 底层记录保持 0-indexed
                    early_stop_counter = 0
                    torch.save(ss_net.state_dict(), args.SSN_path)
                    tqdm.write(f'[Saved] Epoch {epoch_display}: Train Loss -> {avg_train_loss:.6f} | Acc: {train_acc:.6f}%')
                else:
                    early_stop_counter += 1
                    if epoch % 10 == 0 or args.early_stopping:
                        tqdm.write(f'Epoch {epoch_display}: Train Loss {avg_train_loss:.6f} (Best: {min_train_loss:.6f} at Epoch {best_epoch + 1}), 早停计数 {early_stop_counter}/{args.patience}')

            # 模型检查点保存，严格存储底层的 0-indexed 数据
            self._save_ckpt(resume_path, {
                'epoch': epoch,
                'model_state': ss_net.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'warmup_scheduler_state': warmup_scheduler.state_dict(),
                'plateau_scheduler_state': plateau_scheduler.state_dict(),
                'min_train_loss': min_train_loss,
                'best_epoch': best_epoch,
                'early_stop_counter': early_stop_counter
            })
            
            if not in_warmup and args.early_stopping and early_stop_counter >= args.patience:
                tqdm.write(f'早停触发: 连续 {args.patience} 轮无提升, 于 Epoch {epoch_display} 停止训练')
                break
        
        if os.path.exists(resume_path):
            os.remove(resume_path)
            
        print(f'SSN Training 完成, Best @ Epoch {best_epoch + 1}, Loss {min_train_loss:.6f}, 模型已保存至: {args.SSN_path}')

    
    def _gene_des(self, args, split='train'):
        feature_names = [
            'baddllperrors',                # 0
            'badtlperrors',                 # 1
            'portreceivererrors',           # 2
            'recoverydiagnosticserrors',    # 3
            'pcie_fatal_error',             # 4
            'retire_page_dbe',              # 5
            'ecc_v6_sram_uce',              # 6
            'pcie_l0_recovery_count',       # 7
            'temperature',                  # 8
            'retire_page_total',            # 9
            'current_sm_utilization',       # 10
            'pcie_ce_count',                # 11
            'ecc_v6_dram_uce',              # 12
            'retire_page_sbe',              # 13
            'pcie_non_fatal_error'          # 14
        ]
        
        client = OpenAI(api_key=args.llm_api_key, base_url=args.llm_base_url)
        llm_model = args.llm_model
        llm_temp =args.llm_temp
        llm_max_tokens = args.llm_max_tokens

        health_lib = load_baseline_gc_lag1(
            'dataset/raw_data/ali/gpt52_thinking_lag1_e3_style_from_motifs_en.json'
        )
        save_path = os.path.join(args.sta_root_path, f"desc_{split}.json")

        # 如果存在就加载（断点续跑）
        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                results = json.load(f)
            print(f"Loaded existing {len(results)} descriptions")
        else:
            results = {}
        
        save_counter = 0

        # 动态 split
        dataset = UnifiedDataset(root_path=args.sta_root_path, flag=split)
        _, T, F = dataset.X.shape
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        max_fea_path = os.path.join(args.sta_root_path, 'max_feature.npy')
        if not os.path.exists(max_fea_path):
            raise FileNotFoundError(max_fea_path)

        max_fea = torch.tensor(
            np.load(max_fea_path),
            device=args.device,
            dtype=torch.float32
        )

        # load model
        cmlp = TCMLP(num_series=F, lag=args.win_size).to(args.device)
        cmlp.load_state_dict(torch.load(args.ckpt_path, map_location=args.device))
        cmlp.eval()

        ss_net = PolymorphicPatchTokenizer(
            seq_len=T - args.win_size, 
            shape_size=args.shape_size, 
            num_channels=args.num_channels, 
            emb_dim=args.emb_dim, 
            sparse_rate=args.sparse_rate, 
            depth=args.depth, 
            num_experts=args.moe_num_experts, 
            num_classes=args.num_class,
            raw=args.raw, 
            stride=args.shape_stride, 
            RevIN=args.RevIN,
            affine=args.affine,
            subtract_last=args.subtract_last,
            alpha=args.alpha,
            attention_head_dim=args.attention_head_dim
        ).to(args.device)
        ss_net.load_state_dict(torch.load(args.SSN_path, map_location=args.device))
        ss_net.eval()

        def _is_valid_desc(desc) -> bool:
            if desc is None:
                return False
            if isinstance(desc, dict) and len(desc) == 0:
                return False
            if isinstance(desc, str) and len(desc.strip()) == 0:
                return False
            return True

        with torch.no_grad():
            for batch_X, batch_y, batch_id in tqdm(dataloader):
                sample_id = str(int(batch_id.item()))

                # 已生成就跳过
                if sample_id in results and _is_valid_desc(results[sample_id]['description']):
                    continue

                batch_X = batch_X.to(args.device)
                batch_y = batch_y.to(args.device)

                _, pred_raw, _, res, _ = cmlp(batch_X)

                target_slice = batch_X[:, args.win_size:, :]
                current_stats = target_slice / max_fea

                _, _, idx, _, _ = ss_net(res, current_stats)

                raw_np = batch_X.squeeze(0).cpu().numpy()
                pred_np = pred_raw.squeeze(0).cpu().numpy()
                segment_idx = convert_to_list(idx)

                img_b64 = ssn_vis(
                    raw_np.T, pred_np.T,
                    feature_names,
                    highlight_idx=segment_idx
                )

                parsed, _ = _query_patch_description_only(
                    client=client,
                    model=llm_model,
                    img_b64=img_b64,
                    feature_names=feature_names,
                    segment_idx=segment_idx,
                    health_lib=health_lib,
                    temperature=llm_temp,
                    max_tokens=llm_max_tokens
                )

                desc = str(parsed['description']).strip()

                results[sample_id] = {
                    "description": desc,
                    "label": int(batch_y.item())
                }

                save_counter += 1
                if save_counter % 50 == 0:
                    tmp_path = save_path + ".tmp"
                    with open(tmp_path, "w", encoding='utf-8') as f:
                        json.dump(results, f, indent=2)
                    os.replace(tmp_path, save_path)

        # 按 split 保存
        tmp_path = save_path + ".tmp"
        with open(tmp_path, "w", encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        os.replace(tmp_path, save_path)

        print(f"{split} 描述已保存: {save_path}")

    def _gene_text_features(self, args, split='train'):
        desc_path = os.path.join(args.sta_root_path, f"desc_{split}.json")
        if not os.path.exists(desc_path):
            raise FileNotFoundError(f"Missing description file: {desc_path}")

        with open(desc_path, "r", encoding='utf-8') as f:
            desc_dict = json.load(f)

        text_encoder = OpenClipTextEncoder('ViT-B-32', 'laion2b_s34b_b79k', args.device)

        all_txt_feats = []
        all_ids = []

        for sample_id, content in tqdm(desc_dict.items(), desc=f"Encoding {split} descriptions"):
            desc = content["description"]
            txt_feat = text_encoder.encode_tokens(desc).cpu().numpy()
            all_txt_feats.append(txt_feat)
            all_ids.append(sample_id)

        all_txt_feats = np.concatenate(all_txt_feats, axis=0)
        np.save(os.path.join(args.sta_root_path, f"text_features_{split}.npy"), all_txt_feats)

        with open(os.path.join(args.sta_root_path, f"text_ids_{split}.json"), "w", encoding='utf-8') as f:
            json.dump(all_ids, f)

        print(f"✅ {split} 文本特征已保存: {os.path.join(args.sta_root_path, f'text_features_{split}.npy')}")

    def _train_cls(self, args):

        train_dataset = UnifiedDataset(root_path=args.sta_root_path, flag='train')
        _, T, F = train_dataset.X.shape
        # ✅ shuffle=True，用 batch_id 查表保证对应关系，顺序无关
        train_dl = DataLoader(train_dataset, batch_size=args.cls_batch_size, shuffle=True)

        max_fea_path = os.path.join(args.sta_root_path, 'max_feature.npy')
        if os.path.exists(max_fea_path):
            max_fea = torch.tensor(np.load(max_fea_path), device=args.device, dtype=torch.float32)
            print(f"Loaded MAX feature from {max_fea_path}")
        else:
            raise FileNotFoundError(f"Error: {max_fea_path} not found. Run preprocessor first!")

        cmlp = TCMLP(num_series=F, lag=args.win_size, affine=0, subtract_last=0).to(args.device)
        cmlp.load_state_dict(torch.load(args.ckpt_path, map_location=args.device))
        cmlp.eval()

        ss_net = PolymorphicPatchTokenizer(
            seq_len=T - args.win_size,
            shape_size=args.shape_size,
            num_channels=args.num_channels,
            emb_dim=args.emb_dim,
            sparse_rate=args.sparse_rate,
            depth=args.depth,
            num_experts=args.moe_num_experts,
            num_classes=args.num_class,
            raw=args.raw,
            stride=args.shape_stride,
            RevIN=args.RevIN,
            affine=args.affine,
            subtract_last=args.subtract_last,
            alpha=args.alpha,
            attention_head_dim=args.attention_head_dim
        ).to(args.device)
        ss_net.load_state_dict(torch.load(args.SSN_path, map_location=args.device))
        ss_net.eval()

        # ✅ 预加载文本特征与查找表
        txt_feats = np.load(os.path.join(args.sta_root_path, "text_features_train.npy"))  # [N, 77, 512]
        with open(os.path.join(args.sta_root_path, "text_ids_train.json"), "r", encoding='utf-8') as f:
            txt_ids = json.load(f)
        id_to_idx = {k: i for i, k in enumerate(txt_ids)}

        patch_cls = MultiPatchTextFusionModelDirect_6way(embed_dim=args.emb_dim, num_classes=2).to(args.device)
        patch_cls.load_state_dict(torch.load(args.cls_path, map_location=args.device))

        optimizer = torch.optim.AdamW(patch_cls.parameters(), lr=args.lr_cls, weight_decay=args.weight_decay)

        T_0 = args.T_0
        T_mult = args.T_mult
        min_lr = args.min_lr

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_0=T_0,
            T_mult=T_mult,
            eta_min=min_lr
        )
        loss_fn = nn.CrossEntropyLoss()

        resume_path = args.cls_path.replace('.pth', '_resume.pth')
        start_epoch = 0
        min_train_loss = float('inf')
        best_epoch = -1
        early_stop_counter = 0

        ckpt = self._load_ckpt(resume_path, device=args.device)
        if ckpt is not None:
            patch_cls.load_state_dict(ckpt['model_state'])
            optimizer.load_state_dict(ckpt['optimizer_state'])
            scheduler.load_state_dict(ckpt['scheduler_state'])
            start_epoch = ckpt['epoch'] + 1
            min_train_loss = ckpt['min_train_loss']
            best_epoch = ckpt['best_epoch']
            early_stop_counter = ckpt['early_stop_counter']
            print(f'[Resume] 从 Epoch {start_epoch + 1} 继续, 历史最优 Loss {min_train_loss:.6f} at Epoch {best_epoch + 1}, 当前 LR {optimizer.param_groups[0]["lr"]:.2e}, 早停计数 {early_stop_counter}')

        if start_epoch >= args.epochs:
            print(f'[Skip] 已达到指定的最大 Epoch {args.epochs}, 跳过')
            return


        for epoch in range(start_epoch, args.epochs):
            current_lr = optimizer.param_groups[0]['lr']
            patch_cls.train()  # 每个 epoch 开头重置训练模式
            epoch_train_loss = 0
            correct = 0
            total = 0
            num_iterations = 0

            loop = tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs}")

            for batch_X, batch_y, batch_id in loop:
                batch_X, batch_y = batch_X.to(args.device), batch_y.to(args.device)
                batch_ids = [str(int(i)) for i in batch_id]

                # ✅ np.stack 保证 shape [B, 77, 512]，避免 torch.stack 引入多余维度
                txt_feat_batch = torch.tensor(
                    np.stack([txt_feats[id_to_idx[sid]] for sid in batch_ids], axis=0),
                    dtype=torch.float32,
                    device=args.device
                )

                optimizer.zero_grad()

                with torch.no_grad():
                    _, _, _, res, _ = cmlp(batch_X)
                    target_slice = batch_X[:, args.win_size:, :]
                    current_stats = target_slice / max_fea
                    _, _, _, _, patch_tokens = ss_net(res, current_stats)

                logits = patch_cls(patch_tokens, txt_feat_batch)
                loss = loss_fn(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(patch_cls.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_train_loss += loss.item()
                pred = torch.argmax(logits.detach(), dim=1)
                total += batch_y.size(0)
                correct += (pred == batch_y).sum().item()
                num_iterations += 1
                loop.set_postfix(loss=loss.item())

            scheduler.step()
            new_lr = optimizer.param_groups[0]['lr']
            tqdm.write(f"Epoch {epoch}: Avg Train Loss {epoch_train_loss / num_iterations:.6f}, Train Acc {100 * correct / total:.2f}%, LR {new_lr:.2e}")

            avg_train_loss = epoch_train_loss / num_iterations
            train_acc = 100 * correct / total

            # 补全保存与早停逻辑
            if avg_train_loss < min_train_loss:
                min_train_loss = avg_train_loss
                best_epoch = epoch
                early_stop_counter = 0
                torch.save(patch_cls.state_dict(), args.cls_path)
                tqdm.write(f"[Saved] Epoch {epoch}: Loss -> {avg_train_loss:.6f} | Acc: {train_acc:.6f}%")
            else:
                if epoch % 10 == 0:
                    tqdm.write(f"Epoch {epoch}: Loss {avg_train_loss:.6f} (Best: {min_train_loss:.6f} @ Ep {best_epoch})")
                early_stop_counter += 1
                if args.early_stopping:
                    tqdm.write(f'早停计数: {early_stop_counter}/{args.patience}')
    
            self._save_ckpt(resume_path, {
                'epoch': epoch,
                'model_state': patch_cls.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'min_train_loss': min_train_loss,
                'best_epoch': best_epoch,
                'early_stop_counter': early_stop_counter
            })

            if args.early_stopping and early_stop_counter >= args.patience:
                print(f'早停触发: 连续 {args.patience} 轮无提升, 于 Epoch {epoch} 停止训练')
                break
        
        if os.path.exists(resume_path):
            os.remove(resume_path)
        print(f"CLS Training Done. Best @ Epoch {best_epoch}, Loss {min_train_loss:.4f}")

    def test(self, args):
        if not args.test_ssn and not args.test_cls:
            print('两个测试选项都未启用, 直接退出')
            return
        
        # 公共资源加载
        max_feature_path = os.path.join(args.sta_root_path, 'max_feature.npy')
        if os.path.exists(max_feature_path):
            max_feature = torch.tensor(np.load(max_feature_path), device=args.device, dtype=torch.float32)
        else:
            raise FileNotFoundError(f"Missing max_feature.npy at {max_feature_path}")
        
        train_dataset = UnifiedDataset(root_path=args.sta_root_path, flag='train')
        _, T_full, F = train_dataset.X.shape

        cmlp = TCMLP(num_series=F, lag=args.win_size, affine=0, subtract_last=0).to(device=args.device)
        cmlp.load_state_dict(torch.load(args.ckpt_path, map_location=args.device))
        cmlp.eval()

        ss_net = PolymorphicPatchTokenizer(
            seq_len=T_full - args.win_size, 
            shape_size=args.shape_size, 
            num_channels=args.num_channels, 
            emb_dim=args.emb_dim, 
            sparse_rate=args.sparse_rate, 
            depth=args.depth, 
            num_experts=args.moe_num_experts, 
            num_classes=args.num_class,
            raw=args.raw, 
            stride=args.shape_stride, 
            RevIN=args.RevIN, 
            affine=args.affine,
            subtract_last=args.subtract_last, 
            alpha=args.alpha,
            attention_head_dim=args.attention_head_dim
        ).to(args.device)
        ss_net.load_state_dict(torch.load(args.SSN_path, map_location=args.device))
        ss_net.eval()

        patch_cls = MultiPatchTextFusionModelDirect_6way(embed_dim=args.emb_dim, num_classes=2).to(args.device)
        patch_cls.load_state_dict(torch.load(args.cls_path, map_location=args.device))
        patch_cls.eval()

        print('ALL models loaded successfully')

        for split in ('train', 'test'):
            print(f'\n{"="*20} Evaluating on {split} set {"="*20}')
            if args.test_ssn:
                self._test_ssn(args, split, cmlp, ss_net, max_feature)
            if args.test_patch_cls:
                self._test_patch_cls(args, split, cmlp, ss_net, patch_cls, max_feature)

    def _test_ssn(self, args, split, cmlp, ss_net, max_feature):
        dataset = UnifiedDataset(root_path=args.sta_root_path, flag=split)
        dataloader = DataLoader(dataset, batch_size=args.ssn_batch_size, shuffle=False)

        ssn_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_X, batch_y, _ in tqdm(dataloader, desc=f"Testing SSN on {split} set"):
                batch_X, batch_y = batch_X.to(args.device), batch_y.to(args.device)

                _, _, _, res, _ = cmlp(batch_X)

                target_slice = batch_X[:, args.win_size:, :]
                current_stats = target_slice / max_feature

                logits, _, _, _, _ = ss_net(res, current_stats)
                preds = torch.argmax(logits, dim=1)

                ssn_preds.append(preds.cpu())
                all_labels.append(batch_y.cpu())

        ssn_preds = torch.cat(ssn_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        print(f"\nClassification Report for {split} set:")
        print(classification_report(all_labels, ssn_preds, digits=6))
        print(f"Confusion Matrix for {split} set:")
        print(confusion_matrix(all_labels, ssn_preds))

    def _test_patch_cls(self, args, split, cmlp, ss_net, patch_cls, max_feature):
        dataset = UnifiedDataset(root_path=args.sta_root_path, flag=split)
        dataloader = DataLoader(dataset, batch_size=args.cls_batch_size, shuffle=False)

        txt_feats = np.load(os.path.join(args.sta_root_path, f"text_features_{split}.npy"))
        with open(os.path.join(args.sta_root_path, f"text_ids_{split}.json"), "r", encoding='utf-8') as f:
            txt_ids = json.load(f)
        id_to_idx = {k: i for i, k in enumerate(txt_ids)}

        patch_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_X, batch_y, batch_id in tqdm(dataloader, desc=f"Testing Patch CLS on {split} set"):
                batch_X, batch_y = batch_X.to(args.device), batch_y.to(args.device)
                batch_ids = [str(int(i)) for i in batch_id]

                txt_feat_batch = torch.tensor(
                    np.stack([txt_feats[id_to_idx[sid]] for sid in batch_ids], axis=0),
                    dtype=torch.float32,
                    device=args.device
                )

                _, _, _, res, _ = cmlp(batch_X)
                target_slice = batch_X[:, args.win_size:, :]
                current_stats = target_slice / max_feature
                _, _, _, _, patch_tokens = ss_net(res, current_stats)

                logits = patch_cls(patch_tokens, txt_feat_batch)
                preds = torch.argmax(logits, dim=1)

                patch_preds.append(preds.cpu())
                all_labels.append(batch_y.cpu())

        patch_preds = torch.cat(patch_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        print(f"\nClassification Report for {split} set:")
        print(classification_report(all_labels, patch_preds, digits=6))
        print(f"Confusion Matrix for {split} set:")
        print(confusion_matrix(all_labels, patch_preds))
    
    def _save_ckpt(self, path: str, payload: dict):
        tmp_path = path + ".tmp"
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    
    def _load_ckpt(self, path: str, device: str = 'cpu'):
        if os.path.exists(path):
            print(f'[Resume] 发现断点文件 {path}，正在加载...')
            return torch.load(path, map_location=device)
        else:
            print(f'未找到断点文件 {path}，将从头开始训练')
            return None