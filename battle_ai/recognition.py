"""
根据战斗界面三个技能图标综合匹配当前出手角色。
"""
import os
import json
import cv2
import numpy as np
from PIL import Image

_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_FILE = os.path.join(_ROOT, 'skill_db', 'skills.json')
MATCH_SIZE = (64, 64)
MIN_SCORE  = 0.05   # 第一名最低分（过滤完全错误）
MIN_GAP    = 0.05   # 第一名比第二名至少高出这么多才认为识别成功

_DEFAULT_BATTLE_CROPS = {
    'S1': (1455, 945,  1611, 1110),
    'S2': (1611, 945,  1755, 1110),
    'S3': (1755, 945,  1911, 1110),
}

def _get_battle_crops() -> dict:
    try:
        from config_loader import cfg
        if cfg.is_loaded():
            raw = cfg.section('recognition').get('battle_crops', {})
            if raw:
                return {k: tuple(v) for k, v in raw.items()}
    except ImportError:
        pass
    return _DEFAULT_BATTLE_CROPS

BATTLE_CROPS = _DEFAULT_BATTLE_CROPS  # 向后兼容
SKILLS = ['S1', 'S2', 'S3']


def _preprocess(gray: np.ndarray) -> np.ndarray:
    resized = cv2.resize(gray, MATCH_SIZE)
    return resized.astype(np.float32)


def _load():
    if not os.path.exists(DB_FILE):
        return {}, {}
    db = json.load(open(DB_FILE, encoding='utf-8'))
    templates   = {}
    skill_types = {}
    for name, entry in db.items():
        tmpl = {}
        for skill in SKILLS:
            path = entry['skills'][skill]['icon']
            if os.path.exists(path):
                gray = np.array(Image.open(path).convert('L'))
                tmpl[skill] = _preprocess(gray)
        if tmpl:
            templates[name] = tmpl
        skill_types[name] = {s: v['type'] for s, v in entry['skills'].items()}
    return templates, skill_types


_templates, _skill_types = _load()


def _similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32); b = b.astype(np.float32)
    a -= a.mean();             b -= b.mean()
    denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float(np.sum(a * b) / denom) if denom > 1e-6 else 0.0


def _crop_query(img: np.ndarray, skill: str) -> np.ndarray:
    x1, y1, x2, y2 = _get_battle_crops()[skill]
    crop = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    return _preprocess(gray)


def identify(img: np.ndarray):
    """
    传入游戏截图，综合三个技能图标打分，返回 (角色名, skill_types字典)。
    识别失败返回 (None, None)。
    """
    queries = {s: _crop_query(img, s) for s in SKILLS}

    all_scores = []
    for name, tmpl_dict in _templates.items():
        s_list = [_similarity(queries[sk], tmpl_dict[sk]) for sk in SKILLS if sk in tmpl_dict]
        if s_list:
            all_scores.append((sum(s_list) / len(s_list), name))

    all_scores.sort(reverse=True)
    if not all_scores:
        return None, None

    best_score, best_name = all_scores[0]
    second_score = all_scores[1][0] if len(all_scores) > 1 else 0.0

    if best_score >= MIN_SCORE and (best_score - second_score) >= MIN_GAP:
        return best_name, _skill_types[best_name]
    return None, None
