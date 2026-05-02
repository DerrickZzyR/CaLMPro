import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_ # pip install timm==1.0.17
from fastai.basics import * # pip install fastai==2.8.6 pip install ipython
from layers.ppt_layers import *
from layers.RevIN_em import RevIN_em
    
# 下标映射
def restore_and_merge_intervals(idx, shape_size, stride, seq_len=None):
    """
    将 patch 索引还原为原始时间区间，并合并为无重叠区间。

    参数:
        idx: list[int]
            patch 下标列表，例如 [1,2,3,4,6,7,9,10,11,12,13]
        shape_size: int
            每个 patch 覆盖的原始时间长度，例如 8
        stride: int
            patch 滑动步长，例如 2
        seq_len: int or None
            原始序列长度。若提供，则区间右端会被裁剪到 seq_len-1，避免越界

    返回:
        merged_intervals: list[tuple[int, int]]
            合并后的无重叠区间，例如 [(2, 33)]
    """
    if idx is None or len(idx) == 0:
        return []

    # Normalize idx to a flat list[int] to tolerate tensor/ndarray/nested inputs.
    flat_idx = []
    if torch.is_tensor(idx):
        flat_idx = idx.detach().cpu().reshape(-1).tolist()
    elif isinstance(idx, np.ndarray):
        flat_idx = idx.reshape(-1).tolist()
    elif isinstance(idx, (list, tuple)):
        stack = list(idx)
        while stack:
            cur = stack.pop()
            if torch.is_tensor(cur):
                flat_idx.extend(cur.detach().cpu().reshape(-1).tolist())
            elif isinstance(cur, np.ndarray):
                flat_idx.extend(cur.reshape(-1).tolist())
            elif isinstance(cur, (list, tuple)):
                stack.extend(cur)
            else:
                flat_idx.append(cur)
    else:
        flat_idx = [idx]

    # Deduplicate + sort after int casting.
    idx = sorted(
        set(
            int(v.item()) if torch.is_tensor(v) and v.numel() == 1 else int(v)
            for v in flat_idx
        )
    )

    # 1. patch idx -> 原始时间区间
    intervals = []
    for i in idx:
        start = i * stride
        end = start + shape_size - 1

        if seq_len is not None:
            start = max(0, start)
            end = min(seq_len - 1, end)
            if start > end:
                continue

        intervals.append((start, end))

    if not intervals:
        return []

    # 2. 合并重叠/相邻区间
    merged = [intervals[0]]
    for cur_start, cur_end in intervals[1:]:
        last_start, last_end = merged[-1]

        # 重叠或相邻就合并
        if cur_start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, cur_end))
        else:
            merged.append((cur_start, cur_end))

    return merged

class ChangeAwareAttentionHead(nn.Module):
    """融合 patch 自身特征 + 相邻 patch 差异来打分，增强突变感知"""
    def __init__(self, emb_dim, head_dim=32):
        super().__init__()
        self.self_score = nn.Sequential(
            nn.Linear(emb_dim, head_dim),
            nn.GELU(),
            nn.Linear(head_dim, 1),
        )
        self.diff_score = nn.Sequential(
            nn.Linear(emb_dim, head_dim),
            nn.GELU(),
            nn.Linear(head_dim, 1),
        )
        self.gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        # x: [B, N_patches, D]
        s1 = self.self_score(x)

        diff = torch.zeros_like(x)
        diff[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
        diff[:, 0, :] = diff[:, 1, :]
        s2 = self.diff_score(diff)

        g = torch.sigmoid(self.gate)
        score = g * s1 + (1 - g) * s2
        return torch.sigmoid(score)
# PolymorphicPatchTokenizer

class PolymorphicPatchTokenizer(nn.Module):
    def __init__(self, seq_len, shape_size, num_channels, emb_dim, sparse_rate,
                 depth, num_classes, raw, affine, subtract_last, RevIN,
                 alpha, attention_head_dim, num_experts=8, stride=2):
        super().__init__()

        self.seq_len = seq_len
        self.shape_size = shape_size
        self.num_channels = num_channels
        self.emb_dim = emb_dim
        self.sparse_rate = sparse_rate
        self.depth = depth
        self.num_classes = num_classes
        self.RevIN = RevIN
        self.raw = raw
        self.alpha = alpha
        self.shape_stride = stride

        # 归一化
        self.revin_d_layer = RevIN_em(num_features=num_channels, affine=affine, subtract_last=subtract_last)
        self.revin_r_layer = RevIN_em(num_features=num_channels, affine=affine, subtract_last=subtract_last)

        
        self.main_projection = nn.Linear(num_channels, emb_dim)
        self.stats_projection = nn.Linear(num_channels, emb_dim)

        if raw == 1:
            self.shape_embed = ShapeEmbedLayer(
                seq_len=self.seq_len, shape_size=self.shape_size,
                in_chans=self.emb_dim, embed_dim=self.emb_dim, stride=stride
            )
        else:
            self.shape_embed = ShapeEmbedLayer(
                seq_len=self.seq_len, shape_size=self.shape_size,
                in_chans=self.num_channels, embed_dim=self.emb_dim, stride=stride
            )

        self.attention_head = nn.Sequential(
            nn.Linear(self.emb_dim, attention_head_dim),
            nn.Tanh(),
            nn.Linear(attention_head_dim, 1),
            nn.Sigmoid(),
        )

        # self.attention_head = ChangeAwareAttentionHead(self.emb_dim, head_dim=attention_head_dim)

        self.moe = MoE_Block(input_size=self.emb_dim, output_size=self.emb_dim, num_experts=num_experts, hidden_size=self.emb_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.shape_embed.num_patches, self.emb_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=0.15)
        self.sparse_ratio_d = [x.item() for x in torch.linspace(0, self.sparse_rate, self.depth)]  # stochastic depth decay rule

        self.shape_blocks = nn.ModuleList([
            SoftShapeNet_layer(dim=self.emb_dim, moe_nets=self.moe, atten_head=self.attention_head)
            for i in range(self.depth)]
        )

        # Classifier head
        self.head = nn.Linear(self.emb_dim, self.num_classes)

        # init weights
        self.apply(self._init_weights)
                
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                trunc_normal_(m.weight, std=.02)
            if m.bias is not None: 
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            with torch.no_grad():
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        x,
        stats,
        num_epoch_i=100,
        warm_up_epoch=50,
    ):
        x_norm, _ = self.revin_r_layer(x, 'norm')
        x_emb = self.main_projection(x_norm)
        stats_emb = self.stats_projection(stats)
        x = self.alpha*x_emb + (1 - self.alpha)*stats_emb
        x = x.permute(0, 2, 1)

        x = self.shape_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # 创建不断更新的全局索引表，返回最终保留的token的索引
        B, num_patches, _ = x.shape 
        global_idx = torch.arange(num_patches, device=x.device).unsqueeze(0).expand(B, -1)


        moe_loss = None
        d = 0
        end_attn_x_score = None
        
        for shape_blk in self.shape_blocks:
            depth_remain_ratio = 1.0 - self.sparse_ratio_d[d]
            if num_epoch_i < warm_up_epoch:
                depth_remain_ratio = 1.0

            judge_end = False
            if (d + 1) == self.depth:
                judge_end = True

            x, _temp_mloss, end_attn_x_score, idx = shape_blk(x, end_depth=judge_end, remain_ratio=depth_remain_ratio)


            # 在global_idx筛选保留的token的索引
            if idx is not None:
                local_idx_squeezed = idx.squeeze(-1) # local_idx_squeezed [B, K]
                global_idx = torch.gather(global_idx, dim=1, index=local_idx_squeezed) # global_idx [B, K]
                dummy_idx = torch.full((x.shape[0], 1), -1, dtype=global_idx.dtype, device=global_idx.device) # 创建虚拟索引-1以占位extra_token，防止越界
                global_idx = torch.cat([global_idx, dummy_idx], dim=1) # 将虚拟索引与全局索引拼接


            d = d + 1

            if moe_loss == None:
                moe_loss = _temp_mloss
            else:
                moe_loss = moe_loss + _temp_mloss

        instance_logits = self.head(x)
        weighted_instance_logits = instance_logits * end_attn_x_score
        cls_logits = torch.mean(weighted_instance_logits, dim=1)

        valid_patch_mask = global_idx != -1
        valid_patch_counts = valid_patch_mask.sum(dim=1)
        max_valid_patches = int(valid_patch_counts.max().item()) if valid_patch_counts.numel() > 0 else 0

        patch_tokens = x.new_zeros((x.shape[0], max_valid_patches, x.shape[2]))
        patch_mask = torch.zeros((x.shape[0], max_valid_patches), dtype=torch.bool, device=x.device)
        for b in range(x.shape[0]):
            n_valid = int(valid_patch_counts[b].item())
            if n_valid > 0:
                patch_tokens[b, :n_valid] = x[b, valid_patch_mask[b]]
                patch_mask[b, :n_valid] = True
        global_idx = [gi[gi != -1] for gi in global_idx] # 去除全局索引中的虚拟索引-1，得到最终保留的token的全局索引列表

        return cls_logits, moe_loss, global_idx, x_norm, patch_tokens, patch_mask
