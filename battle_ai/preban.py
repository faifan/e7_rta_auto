"""预禁用阶段：检测界面 → 点击2个英雄禁用 → 确认"""
import os
import time
import io
import numpy as np
import cv2
from PIL import Image
from battle_ai.executor import click_at
from battle_ai.perception import capture

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PREBAN_NCC_SIZE  = (80, 16)
_HERO_TMPL_SIZE   = (80, 80)
_FIRST_PICK_SIZE  = (80, 20)
_NCC_THRESHOLD    = 0.50

_DEFAULT_REGION   = (181, 137, 505, 203)
_DEFAULT_HERO_1   = (1695, 609)
_DEFAULT_HERO_2   = (1695, 759)
_DEFAULT_CONFIRM  = (1624, 945)

_DEFAULT_CANDIDATE_SLOTS  = [(1614,208,1767,340),(1610,362,1778,495),(1604,515,1777,639),(1607,658,1784,792)]
_DEFAULT_CANDIDATE_CLICKS = [(1690,274),(1694,428),(1690,577),(1695,725)]
_DEFAULT_FIRST_PICK_REGION = (1508,802,1736,861)


def _pcfg() -> dict:
    try:
        from config_loader import cfg
        if cfg.is_loaded():
            return cfg.section('preban')
    except ImportError:
        pass
    return {}

def _lang(key, default=''):
    try:
        from config_loader import cfg
        if cfg.is_loaded():
            return cfg.lang(key, default)
    except ImportError:
        pass
    return default

def _resolution():
    try:
        from config_loader import cfg
        if cfg.is_loaded():
            return cfg._profile.get('resolution', [1920, 1080])
    except Exception:
        pass
    return [1920, 1080]

def _scale():
    r = _resolution()
    return r[0] / 1920, r[1] / 1080

def _region():
    p = _pcfg()
    return tuple(p['region']) if 'region' in p else _DEFAULT_REGION

def _hero1():
    p = _pcfg()
    return tuple(p['ban_hero_1']) if 'ban_hero_1' in p else _DEFAULT_HERO_1

def _hero2():
    p = _pcfg()
    return tuple(p['ban_hero_2']) if 'ban_hero_2' in p else _DEFAULT_HERO_2

def _confirm():
    p = _pcfg()
    return tuple(p['ban_confirm']) if 'ban_confirm' in p else _DEFAULT_CONFIRM

def _candidate_slots():
    p = _pcfg()
    if 'candidate_slots' in p:
        return [tuple(s) for s in p['candidate_slots']]
    sx, sy = _scale()
    return [(round(x1*sx),round(y1*sy),round(x2*sx),round(y2*sy)) for x1,y1,x2,y2 in _DEFAULT_CANDIDATE_SLOTS]

def _candidate_clicks():
    p = _pcfg()
    if 'candidate_clicks' in p:
        return [tuple(c) for c in p['candidate_clicks']]
    sx, sy = _scale()
    return [(round(x*sx),round(y*sy)) for x,y in _DEFAULT_CANDIDATE_CLICKS]

def _first_pick_region():
    p = _pcfg()
    if 'first_pick_region' in p:
        return tuple(p['first_pick_region'])
    sx, sy = _scale()
    x1,y1,x2,y2 = _DEFAULT_FIRST_PICK_REGION
    return (round(x1*sx),round(y1*sy),round(x2*sx),round(y2*sy))


# ── NCC 工具 ──────────────────────────────────────────────────

def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a**2).sum() * (b**2).sum())
    return float(np.sum(a * b) / denom) if denom > 1e-6 else 0.0


# ── preban界面检测 ────────────────────────────────────────────

_ocr_preban      = None
_preban_ncc_tmpl = None


def _load_preban_ncc():
    global _preban_ncc_tmpl
    tmpl_path = os.path.join(_ROOT, 'templates', 'phase', 'preban.png')
    if not os.path.exists(tmpl_path):
        return
    img = np.array(Image.open(tmpl_path).convert('L'))
    tmpl_h, tmpl_w = img.shape[:2]
    x1, y1, x2, y2 = _region()
    r = _resolution()
    sx, sy = tmpl_w / r[0], tmpl_h / r[1]
    crop = img[int(y1*sy):int(y2*sy), int(x1*sx):int(x2*sx)]
    _preban_ncc_tmpl = cv2.resize(crop, _PREBAN_NCC_SIZE).astype(np.float32)


def _preban_ncc_score(img) -> float:
    if _preban_ncc_tmpl is None:
        return 0.0
    x1, y1, x2, y2 = _region()
    crop = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    query = cv2.resize(gray, _PREBAN_NCC_SIZE).astype(np.float32)
    return _ncc(query, _preban_ncc_tmpl)


def _preban_ocr(img) -> str:
    global _ocr_preban
    if _ocr_preban is None:
        import ddddocr
        _ocr_preban = ddddocr.DdddOcr(show_ad=False)
    x1, y1, x2, y2 = _region()
    crop = img[y1:y2, x1:x2]
    buf  = io.BytesIO()
    Image.fromarray(crop).save(buf, format='PNG')
    try:
        return _ocr_preban.classification(buf.getvalue())
    except Exception:
        return ''


def is_in_preban(img=None) -> bool:
    if img is None:
        img = capture()
    text     = _preban_ocr(img)
    keywords = _lang('preban_keywords', ['禁用', '预先'])
    if any(k in text for k in keywords):
        return True
    if _preban_ncc_tmpl is None:
        _load_preban_ncc()
    return _preban_ncc_score(img) >= 0.60


# ── 先/后手检测 ───────────────────────────────────────────────

_first_pick_tmpl = None


def _load_first_pick_tmpl():
    global _first_pick_tmpl
    path = os.path.join(_ROOT, 'templates', 'phase', 'preban_first_pick.png')
    if not os.path.exists(path):
        return
    img = np.array(Image.open(path).convert('L'))
    _first_pick_tmpl = cv2.resize(img, _FIRST_PICK_SIZE).astype(np.float32)


def is_first_pick(img=None) -> bool:
    if img is None:
        img = capture()
    x1, y1, x2, y2 = _first_pick_region()
    crop = img[y1:y2, x1:x2]

    # OCR 主检测
    try:
        global _ocr_preban
        if _ocr_preban is None:
            import ddddocr
            _ocr_preban = ddddocr.DdddOcr(show_ad=False)
        buf = io.BytesIO()
        Image.fromarray(crop).save(buf, format='PNG')
        text = _ocr_preban.classification(buf.getvalue()).upper()
        if 'FIRST' in text:
            return True
        if 'SECOND' in text:
            return False
    except Exception:
        pass

    # NCC 兜底
    if _first_pick_tmpl is None:
        _load_first_pick_tmpl()
    if _first_pick_tmpl is not None:
        gray  = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        query = cv2.resize(gray, _FIRST_PICK_SIZE).astype(np.float32)
        return _ncc(query, _first_pick_tmpl) >= 0.60

    return True


# ── 候选格识别 ────────────────────────────────────────────────

_preban_hero_tmpls: dict = {}   # code -> list[np.ndarray]


def _load_preban_hero_tmpls():
    global _preban_hero_tmpls
    tmpl_dir = os.path.join(_ROOT, 'templates', 'preban_heroes')
    if not os.path.exists(tmpl_dir):
        return
    tmpls: dict = {}
    for fname in os.listdir(tmpl_dir):
        if not fname.endswith('.png'):
            continue
        code = fname[:-4].split('_')[0]
        img  = cv2.imdecode(np.fromfile(os.path.join(tmpl_dir, fname), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        tmpls.setdefault(code, []).append(
            cv2.resize(img, _HERO_TMPL_SIZE).astype(np.float32)
        )
    _preban_hero_tmpls = tmpls


def identify_preban_candidates(img) -> list:
    """返回4个格子各自识别到的英雄code，未命中返回None。"""
    if not _preban_hero_tmpls:
        _load_preban_hero_tmpls()
    results = []
    for x1, y1, x2, y2 in _candidate_slots():
        crop  = img[y1:y2, x1:x2]
        gray  = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        query = cv2.resize(gray, _HERO_TMPL_SIZE).astype(np.float32)
        best_code, best_score = None, 0.0
        for code, tmpl_list in _preban_hero_tmpls.items():
            for tmpl in tmpl_list:
                s = _ncc(query, tmpl)
                if s > best_score:
                    best_score, best_code = s, code
        results.append(best_code if best_score >= _NCC_THRESHOLD else None)
    return results


# ── 禁用执行 ──────────────────────────────────────────────────

def do_preban():
    """点击2个英雄禁用，然后点击确认（无智能识别版）。"""
    click_at(*_hero1(), delay=0.6)
    click_at(*_hero2(), delay=0.6)
    click_at(*_confirm(), delay=0.8)


def do_smart_preban(first_targets: list, second_targets: list, log_fn=None):
    """
    智能预禁用：识别先/后手 → 扫描4个候选格 → 优先点目标英雄 → 兜底点第3、4格。
    first_targets / second_targets: 英雄code列表，如 ['c1117', 'c2181']
    """
    if not _preban_hero_tmpls:
        _load_preban_hero_tmpls()
    if _first_pick_tmpl is None:
        _load_first_pick_tmpl()

    img    = capture()
    first  = is_first_pick(img)
    role   = '先手' if first else '后手'
    targets = first_targets if first else second_targets
    if log_fn:
        log_fn(f'  预禁用：{role}，目标={targets}')

    candidates = identify_preban_candidates(img)
    if log_fn:
        log_fn(f'  候选格识别：{candidates}')

    clicks  = _candidate_clicks()
    chosen  = []

    for target in targets:
        for i, code in enumerate(candidates):
            if code == target and i not in chosen:
                chosen.append(i)
                break

    # 兜底：不足2个时用第3、4格（index 2, 3）
    for fb in (2, 3):
        if len(chosen) >= 2:
            break
        if fb not in chosen:
            chosen.append(fb)

    chosen = chosen[:2]
    if log_fn:
        log_fn(f'  点击槽位：{[i+1 for i in chosen]}')

    for idx in chosen:
        click_at(*clicks[idx], delay=0.6)
    click_at(*_confirm(), delay=0.8)
