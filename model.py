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

        # 每个 token 被选时所处的阶段 (0=preban, 1-6=各pick轮次)
        self.token_phase_embedding = nn.Embedding(7, d_model)

        # 当前预测的是哪方 (0=padding, 1=我方, 2=敌方)
        self.prediction_side_embedding = nn.Embedding(3, d_model)

        # 当前预测方是先手还是后手 (0=padding, 1=先手方, 2=后手方)
        self.first_pick_embedding = nn.Embedding(3, d_model)

        # 开局规则 (0=未知/preban阶段, 1=category_1, 2=category_2, 3=category_4, 4=category_5)
        self.opening_rule_embedding = nn.Embedding(5, d_model)
        
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
    
    def forward(self, hero_ids, side_ids, phase_ids, src_mask=None, token_phase_ids=None, prediction_side_ids=None, first_pick_ids=None, opening_rule_ids=None):
        """
        前向传播

        Args:
            hero_ids: [batch, seq_len] - 英雄 ID 序列
            side_ids: [batch, seq_len] - 阵营标记 (0=padding, 1=我方，2=敌方，3=ban 位)
            phase_ids: [batch] - 当前阶段 (0-7)
            src_mask: [batch, seq_len] - 源序列 padding 掩码 (1=真实数据，0=padding)
            token_phase_ids: [batch, seq_len] - 每个 token 被选时的阶段 (0=preban, 1-6=pick轮次)
            prediction_side_ids: [batch] - 当前预测的是哪方 (1=我方, 2=敌方)
            first_pick_ids: [batch] - 当前预测方是先手还是后手 (1=先手, 2=后手)

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
        if token_phase_ids is not None:
            x = x + self.token_phase_embedding(token_phase_ids)
        if first_pick_ids is not None:
            x = x + self.first_pick_embedding(first_pick_ids).unsqueeze(1).expand(-1, seq_len, -1)
        if opening_rule_ids is not None:
            x = x + self.opening_rule_embedding(opening_rule_ids).unsqueeze(1).expand(-1, seq_len, -1)
        if prediction_side_ids is not None:
            x = x + self.prediction_side_embedding(prediction_side_ids).unsqueeze(1).expand(-1, seq_len, -1)

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

        # mean pooling over real tokens（避免 padding 位置影响结果）
        if src_mask is not None:
            mask_f = src_mask.unsqueeze(-1)  # [batch, seq, 1]
            pooled = (memory * mask_f).sum(1) / mask_f.sum(1)
        else:
            pooled = memory.mean(dim=1)

        next_pick_logits = self.hero_classifier(pooled)  # [batch, num_heroes]
        win_rate = self.win_rate_head(pooled)  # [batch, 1]

        return next_pick_logits, win_rate

    def predict_next_pick(self, hero_sequence, side_sequence, phase_id, available_mask, top_k=10, token_phase_sequence=None, prediction_side_id=1, is_first_pick=True, opening_rule_id=0):
        self.eval()
        device = next(self.parameters()).device

        if hero_sequence:
            hero_ids = torch.tensor([hero_sequence], dtype=torch.long, device=device)
            side_ids = torch.tensor([side_sequence], dtype=torch.long, device=device)
            src_mask = torch.ones(1, len(hero_sequence), dtype=torch.float, device=device)
            token_phase_ids = torch.tensor([token_phase_sequence], dtype=torch.long, device=device) if token_phase_sequence is not None else None
        else:
            # 空序列（首次preban）用单个dummy token
            hero_ids = torch.tensor([[0]], dtype=torch.long, device=device)
            side_ids = torch.tensor([[0]], dtype=torch.long, device=device)
            src_mask = torch.ones(1, 1, dtype=torch.float, device=device)
            token_phase_ids = None

        phase_ids = torch.tensor([phase_id], dtype=torch.long, device=device)
        pred_side_ids = torch.tensor([prediction_side_id], dtype=torch.long, device=device)
        first_pick_ids = torch.tensor([1 if is_first_pick else 2], dtype=torch.long, device=device)
        opening_rule_ids = torch.tensor([opening_rule_id], dtype=torch.long, device=device)

        with torch.no_grad():
            logits, win_rate = self.forward(hero_ids, side_ids, phase_ids, src_mask=src_mask, token_phase_ids=token_phase_ids, prediction_side_ids=pred_side_ids, first_pick_ids=first_pick_ids, opening_rule_ids=opening_rule_ids)

            if available_mask is not None:
                available_mask = available_mask.to(device)
                logits = logits.masked_fill(available_mask == 0, float('-inf'))

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


if __name__ == '__main__':
    num_heroes = 200
    batch_size = 4
    seq_len = 10

    model = DraftTransformer(num_heroes=num_heroes)

    hero_ids = torch.randint(0, num_heroes, (batch_size, seq_len))
    side_ids = torch.randint(0, 4, (batch_size, seq_len))
    phase_ids = torch.randint(0, 8, (batch_size,))
    src_mask = torch.ones(batch_size, seq_len)

    logits, win_rate = model(hero_ids, side_ids, phase_ids, src_mask=src_mask)

    print(f"logits shape: {logits.shape}")
    print(f"win_rate shape: {win_rate.shape}")
    print("模型测试通过!")
