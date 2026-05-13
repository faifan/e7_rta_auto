"""
第七史诗选秀辅助 - Transformer 模型推理
用于实时推荐英雄选择
"""

import torch
import json
import os
from model import DraftTransformer

class DraftRecommender:
    """选秀推荐器"""
    
    def __init__(self, model_path='draft_transformer.pth', hero_list_path='hero_list_146.json'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.hero_list = []
        self.hero_to_idx = {}
        self.idx_to_hero = {}
        self.num_heroes = 0
        
        # 加载英雄列表
        if os.path.exists(hero_list_path):
            with open(hero_list_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.hero_list = data['hero_list']
                self.hero_to_idx = data['hero_to_idx']
                self.idx_to_hero = {i: h for i, h in enumerate(self.hero_list)}
                self.num_heroes = len(self.hero_list)
            print(f"OK 英雄列表加载成功：{self.num_heroes} 个英雄")
        else:
            print(f"W 找不到英雄列表文件：{hero_list_path}")

        # 加载模型
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model = DraftTransformer(
                num_heroes=checkpoint['config']['num_heroes'],
                d_model=checkpoint['config']['d_model'],
                nhead=checkpoint['config']['nhead'],
                num_layers=checkpoint['config']['num_layers'],
                dropout=checkpoint['config']['dropout'],
                dim_feedforward=checkpoint['config']['dim_feedforward']
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            print(f"OK 模型加载成功 (验证准确率：{checkpoint.get('val_acc', 0):.4f})")
        else:
            print(f"W 找不到模型文件：{model_path}")
    
    def get_available_mask(self, banned, used):
        """生成可选英雄掩码"""
        mask = torch.ones(self.num_heroes, dtype=torch.float)
        for hero in banned + used:
            if hero in self.hero_to_idx:
                mask[self.hero_to_idx[hero]] = 0
        return mask
    
    def recommend(self, my_picks, enemy_picks, banned, phase='pick1', my_first=True, top_k=10):
        """
        推荐下一个英雄选择

        Preban 阶段：
        - 双方各 Ban 2 个，可以重复 Ban 同一个英雄
        - 样本：[], [my_ban1], [enemy_ban1], [my_ban1, enemy_ban1, my_ban2]

        Pick 阶段：
        - 10 个英雄不能重复
        - ABBA 顺序

        Args:
            my_picks: List[str] - 我方已选英雄代码
            enemy_picks: List[str] - 敌方已选英雄代码
            banned: List[str] - 已禁用英雄代码 (preban)
            phase: str - 当前阶段 (preban, pick1, pick2, pick3, pick4, pick5)
            my_first: bool - 我方是否先手
            top_k: int - 返回前 K 个推荐

        Returns:
            List[Dict] - 推荐列表
        """
        if self.model is None or self.num_heroes == 0:
            return []

        # ========== Preban 阶段 ==========
        if phase == 'preban':
            return self.recommend_preban(
                my_banned=my_picks,  # 在 preban 阶段，my_picks 存储的是我方已 Ban 的英雄
                enemy_banned=enemy_picks,  # enemy_picks 存储的是敌方已 Ban 的英雄
                all_banned=banned,
                top_k=top_k
            )

        # ========== Pick 阶段 ==========
        # 确定当前该谁选择
        my_count = len(my_picks)
        enemy_count = len(enemy_picks)

        # 构建选秀序列
        hero_seq = []
        side_seq = []

        # 1. 添加 ban 位
        for hero in banned:
            if hero in self.hero_to_idx:
                hero_seq.append(self.hero_to_idx[hero])
                side_seq.append(3)  # 3 = ban 位

        # 2. 按实际选秀顺序添加已选英雄
        # 确定先手方和后手方
        if my_first:
            first_side_picks = my_picks  # 先手方（我方）
            second_side_picks = enemy_picks  # 后手方（敌方）
            first_side_id = 1  # 我方
            second_side_id = 2  # 敌方
        else:
            first_side_picks = enemy_picks  # 先手方（敌方）
            second_side_picks = my_picks  # 后手方（我方）
            first_side_id = 2  # 敌方
            second_side_id = 1  # 我方

        # 按照 ABBA 顺序重建序列
        picks_order = [
            (first_side_picks, 1, first_side_id),      # pick1: 先手方选 1
            (second_side_picks, 2, second_side_id),    # pick2: 后手方选 2
            (first_side_picks, 2, first_side_id),      # pick3: 先手方选 2
            (second_side_picks, 2, second_side_id),    # pick4: 后手方选 2
            (first_side_picks, 2, first_side_id),      # pick5: 先手方选 2
            (second_side_picks, 1, second_side_id),    # pick5: 后手方选 1
        ]

        for picks, count, side_id in picks_order:
            for i in range(min(count, len(picks))):
                hero = picks[i]
                if hero in self.hero_to_idx:
                    hero_seq.append(self.hero_to_idx[hero])
                    side_seq.append(side_id)

        # 3. 阶段映射
        phase_map = {'preban': 0, 'pick1': 1, 'pick2': 2, 'pick3': 3, 'pick4': 4, 'pick5': 5, 'finalban': 6}
        phase_id = phase_map.get(phase, 1)

        # 4. 生成可选掩码
        used = set(my_picks + enemy_picks)
        available_mask = self.get_available_mask(banned, list(used))

        # 5. 预测
        recommendations = self.model.predict_next_pick(
            hero_seq, side_seq, phase_id, available_mask, top_k
        )

        # 6. 转换结果
        result = []
        for rec in recommendations:
            hero_code = self.idx_to_hero.get(rec['hero_idx'], 'unknown')
            result.append({
                'hero_code': hero_code,
                'probability': rec['probability'],
                'win_rate': rec['win_rate']
            })

        return result

    def recommend_preban(self, my_banned, enemy_banned, all_banned, top_k=10):
        """
        Preban 阶段推荐

        Preban 特点：
        - 双方独立 Ban，不知道对面 Ban 了什么
        - 可以重复 Ban 同一个英雄

        Args:
            my_banned: List[str] - 我方已 Ban 的英雄
            enemy_banned: List[str] - 敌方已 Ban 的英雄
            all_banned: List[str] - 所有已 Ban 的英雄（用于掩码）
            top_k: int - 返回前 K 个推荐

        Returns:
            List[Dict] - 推荐列表
        """
        # 确定该谁 Ban
        my_count = len(my_banned)
        enemy_count = len(enemy_banned)
        total_banned = my_count + enemy_count

        if total_banned >= 4:
            return []  # Preban 完成

        # 判断该哪方 Ban
        if my_count == enemy_count:
            # 该我方 Ban（0-0 或 1-1）
            current_side = 'my'
            side_id = 1
            # 只看我方已 Ban 的
            ban_seq = [self.hero_to_idx[h] for h in my_banned if h in self.hero_to_idx]
        else:
            # 该敌方 Ban（1-0 或 0-1）
            current_side = 'enemy'
            side_id = 2
            # 只看敌方已 Ban 的
            ban_seq = [self.hero_to_idx[h] for h in enemy_banned if h in self.hero_to_idx]

        # 构建 side_seq（都是 Ban）
        side_seq = [3] * len(ban_seq)

        # 阶段 ID（Preban）
        phase_id = 0  # Preban 对应 phase=0

        # 生成可选掩码（Preban 可以重复 Ban，所以不用掩码）
        available_mask = torch.ones(self.num_heroes, dtype=torch.float)

        # 预测
        recommendations = self.model.predict_next_pick(
            ban_seq, side_seq, phase_id, available_mask, top_k
        )

        # 转换结果
        result = []
        for rec in recommendations:
            hero_code = self.idx_to_hero.get(rec['hero_idx'], 'unknown')
            result.append({
                'hero_code': hero_code,
                'probability': rec['probability'],
                'win_rate': rec.get('win_rate', 0)
            })

        return result

    def recommend_preban_simple(self, my_banned, enemy_banned, top_k=5):
        """
        Preban 阶段推荐：永远返回 5 个常 Ban 英雄（不过滤已 Ban 的）
        因为 Preban 双方独立 Ban，对方可能也要 Ban 同一个

        Args:
            my_banned: List[str] - 我方已 Ban 的英雄
            enemy_banned: List[str] - 敌方已 Ban 的英雄
            top_k: int - 返回前 K 个推荐

        Returns:
            List[Dict] - 推荐列表（永远 5 个）
        """
        # 常 Ban 英雄列表（按 Ban 率排序）
        ban_priority = [
            'c1153',  # 光呆
            'c1133',  # 暗呆
            'c1117',  # 水弓
            'c6005',  # 新英雄
            'c2076',  # 火奶
        ]
        
        # 返回全部 5 个，不过滤（已 Ban 的在前端显示为灰色）
        result = []
        for hero_code in ban_priority[:top_k]:
            result.append({
                'hero_code': hero_code,
                'probability': 1.0 / (len(result) + 1),  # 简单优先级
                'win_rate': 0.5
            })
        
        return result

    def recommend_finalban(self, my_picks, enemy_picks, banned, my_first=True, top_k=5):
        """选秀后禁用推荐：序列上下文完整，available_mask 只开放 enemy_picks。"""
        if self.model is None or not enemy_picks:
            return []

        hero_seq, side_seq = [], []
        for hero in banned:
            if hero in self.hero_to_idx:
                hero_seq.append(self.hero_to_idx[hero])
                side_seq.append(3)

        if my_first:
            first_picks, second_picks = my_picks, enemy_picks
            first_id, second_id = 1, 2
        else:
            first_picks, second_picks = enemy_picks, my_picks
            first_id, second_id = 2, 1

        for picks, count, side_id in [
            (first_picks,  1, first_id),
            (second_picks, 2, second_id),
            (first_picks,  2, first_id),
            (second_picks, 2, second_id),
            (first_picks,  2, first_id),
            (second_picks, 1, second_id),
        ]:
            for hero in picks[:count]:
                if hero in self.hero_to_idx:
                    hero_seq.append(self.hero_to_idx[hero])
                    side_seq.append(side_id)

        import torch as _torch
        available_mask = _torch.zeros(self.num_heroes, dtype=_torch.float)
        for hero in enemy_picks:
            if hero in self.hero_to_idx:
                available_mask[self.hero_to_idx[hero]] = 1.0

        recs = self.model.predict_next_pick(hero_seq, side_seq, 6, available_mask, top_k)
        return [{'hero_code': self.idx_to_hero.get(r['hero_idx'], 'unknown'),
                 'probability': r['probability']} for r in recs]

    def predict_win_rate(self, my_picks, enemy_picks, banned):
        """
        预测阵容胜率
        
        Args:
            my_picks: List[str] - 我方 5 英雄
            enemy_picks: List[str] - 敌方 5 英雄
            banned: List[str] - Ban 位
        
        Returns:
            float - 预测胜率
        """
        # TODO: 实现完整的阵容胜率预测
        return 0.5


# 测试
if __name__ == '__main__':
    recommender = DraftRecommender()
    
    # 测试推荐
    recs = recommender.recommend(
        my_picks=[],
        enemy_picks=[],
        banned=[],
        phase='pick1',
        my_first=True
    )
    
    print("\n推荐测试:")
    for i, rec in enumerate(recs[:5], 1):
        print(f"{i}. {rec['hero_code']} - 概率：{rec['probability']:.4f}, 胜率：{rec['win_rate']:.4f}")
