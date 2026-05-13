"""
第七史诗选秀辅助 - Transformer 模型定义
用于预测最佳英雄选择和阵容胜率
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model: int, max_len: int = 50):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)  # [max_len, 1, d_model]
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [seq_len, batch, d_model]
        """
        return x + self.pe[:x.size(0), :, :]


class DraftTransformer(nn.Module):
    """
    选秀 Transformer 模型
    
    输入：已选英雄序列（包含 Ban 位、我方英雄、敌方英雄）
    输出：下一个最佳选择的概率分布
    """
    def __init__(
        self,
        num_heroes: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 20
    ):
        super().__init__()
        
        self.num_heroes = num_heroes
        self.d_model = d_model
        
        # 英雄 Embedding
        self.hero_embedding = nn.Embedding(num_heroes + 1, d_model)  # +1 为 padding
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        
        # 阵营 Embedding (0=padding, 1=我方，2=敌方，3=ban 位)
        self.side_embedding = nn.Embedding(4, d_model)
        
        # 阶段 Embedding (preban, pick1-5, finalban)
        self.phase_embedding = nn.Embedding(8, d_model)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层
        self.hero_classifier = nn.Linear(d_model, num_heroes)
        
        # 胜率预测头（可选）
        self.win_rate_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_mask(self, src):
        """生成注意力掩码，防止看到未来"""
        seq_len = src.size(0)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=src.device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, hero_ids, side_ids, phase_ids, src_mask=None):
        """
        前向传播

        Args:
            hero_ids: [batch, seq_len] - 英雄 ID 序列
            side_ids: [batch, seq_len] - 阵营标记 (0=padding, 1=我方，2=敌方，3=ban 位)
            phase_ids: [batch] - 当前阶段 (0-7)
            src_mask: [batch, seq_len] - 源序列 padding 掩码 (1=真实数据，0=padding)

        Returns:
            next_pick_logits: [batch, num_heroes] - 下一个选择的 logits
            win_rate: [batch, 1] - 当前阵容胜率预测
        """
        batch_size = hero_ids.size(0)
        seq_len = hero_ids.size(1)

        # Embedding
        hero_emb = self.hero_embedding(hero_ids) * math.sqrt(self.d_model)  # [batch, seq, d_model]
        side_emb = self.side_embedding(side_ids)  # [batch, seq, d_model]
        phase_emb = self.phase_embedding(phase_ids).unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq, d_model]

        # 组合 Embedding
        x = hero_emb + side_emb + phase_emb

        # 添加位置编码
        pos_enc = self.pos_encoder.pe[:seq_len, :, :].transpose(0, 1)  # [1, seq, d_model]
        x = x + pos_enc

        # Transformer 编码
        # src_key_padding_mask: True 表示忽略该位置（padding）
        # 我们的 mask 是 1=真实，0=padding，需要取反
        if src_mask is not None:
            padding_mask = (src_mask == 0)  # [batch, seq_len], True=padding
            memory = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        else:
            memory = self.transformer_encoder(x)  # [batch, seq, d_model]

        # 取最后一个位置作为当前状态表示
        last_output = memory[:, -1, :]  # [batch, d_model]

        # 预测下一个英雄
        next_pick_logits = self.hero_classifier(last_output)  # [batch, num_heroes]

        # 预测胜率
        win_rate = self.win_rate_head(last_output)  # [batch, 1]

        return next_pick_logits, win_rate
    
    def predict_next_pick(self, hero_sequence, side_sequence, phase_id, available_mask, top_k=10):
        """
        推理：预测下一个最佳选择
        
        Args:
            hero_sequence: List[int] - 已选英雄 ID 列表
            side_sequence: List[int] - 对应的阵营列表
            phase_id: int - 当前阶段
            available_mask: torch.Tensor - 可选英雄掩码
            top_k: int - 返回前 K 个推荐
        
        Returns:
            recommendations: List[Dict] - 推荐列表
        """
        self.eval()
        device = next(self.parameters()).device

        if len(hero_sequence) == 0:
            # 空上文（如第一步preban）：无法推理，返回空
            return []

        # 不追加哑元 token，与训练时保持一致（训练取 last real token 的表示）
        hero_ids  = torch.tensor([hero_sequence], dtype=torch.long, device=device)   # [1, seq]
        side_ids  = torch.tensor([side_sequence], dtype=torch.long, device=device)   # [1, seq]
        phase_ids = torch.tensor([phase_id],      dtype=torch.long, device=device)   # [1]
        src_mask  = torch.ones(1, len(hero_sequence), dtype=torch.float, device=device)

        with torch.no_grad():
            logits, win_rate = self.forward(hero_ids, side_ids, phase_ids, src_mask=src_mask)
            
            # 应用可选掩码
            if available_mask is not None:
                available_mask = available_mask.to(device)
                logits = logits.masked_fill(available_mask == 0, float('-inf'))
            
            # 获取 top_k
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            top_probs, top_indices = torch.topk(probs, min(top_k, probs.size(-1)))
            
            recommendations = []
            for prob, idx in zip(top_probs, top_indices):
                if prob > 0:
                    recommendations.append({
                        'hero_idx': idx.item(),
                        'probability': prob.item(),
                        'win_rate': win_rate.item()
                    })
            
            return recommendations


class WinRatePredictor(nn.Module):
    """
    阵容胜率预测器
    
    输入：完整阵容（双方各 5 英雄 + Ban 位）
    输出：我方胜率
    """
    def __init__(
        self,
        num_heroes: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3
    ):
        super().__init__()
        
        self.hero_embedding = nn.Embedding(num_heroes + 1, d_model)
        self.side_embedding = nn.Embedding(3, d_model)  # 0=我方，1=敌方，2=ban
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, hero_ids, side_ids):
        """
        Args:
            hero_ids: [batch, seq_len] - 英雄 ID 序列
            side_ids: [batch, seq_len] - 阵营标记
        """
        hero_emb = self.hero_embedding(hero_ids)
        side_emb = self.side_embedding(side_ids)
        x = hero_emb + side_emb
        
        memory = self.encoder(x)
        
        # 全局池化
        pooled = memory.mean(dim=1)
        
        win_rate = self.classifier(pooled)
        return win_rate


if __name__ == '__main__':
    # 测试模型
    num_heroes = 200
    batch_size = 4
    seq_len = 10
    
    model = DraftTransformer(num_heroes=num_heroes)
    
    hero_ids = torch.randint(0, num_heroes, (seq_len, batch_size))
    side_ids = torch.randint(0, 4, (seq_len, batch_size))
    phase_ids = torch.randint(0, 8, (batch_size,))
    
    logits, win_rate = model(hero_ids, side_ids, phase_ids)
    
    print(f"Next pick logits shape: {logits.shape}")
    print(f"Win rate shape: {win_rate.shape}")
    print("模型测试通过!")
