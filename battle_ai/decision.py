import random
from battle_ai.recognition import identify
from battle_ai.perception import is_skill_on_cooldown

_PRIORITY = ['S3', 'S2', 'S1']

def get_candidates(img=None):
    """
    返回经过OCR冷却过滤后的候选技能列表 [(skill, stype), ...]，按优先级排序。
    stype: '单体' | '群体' | '自动'
    """
    char_name, skill_types = identify(img) if img is not None else (None, None)
    candidates = []
    for skill in _PRIORITY:
        stype = (skill_types or {}).get(skill, '自动')
        if stype == '被动':
            continue
        # Gate 1: OCR冷却检测（S2/S3，主动/被动均会显示回合数）
        if skill != 'S1' and img is not None and is_skill_on_cooldown(img, skill):
            continue
        candidates.append((skill, stype))
    return candidates

def decide(img=None):
    """向后兼容接口，返回 (skill, target_idx, stype)。"""
    candidates = get_candidates(img)
    if not candidates:
        return 'S1', random.randint(0, 3), '单体'
    skill, stype = candidates[0]
    target = None if stype == '群体' else random.randint(0, 3)
    return skill, target, stype
