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
# _HERO_TMPL_SIZE   = (80, 80)   # 旧：NCC resize 尺寸
# _NCC_THRESHOLD    = 0.50       # 旧：NCC 阈值
_FIRST_PICK_SIZE  = (80, 20)
_HERO_IMAGES_DIR_PREBAN = os.path.join(_ROOT, 'templates', 'hero_images')
# _PREBAN_MS_SCALES = [0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.03, 1.05]  # 旧：固定比例，720p 只有 0.75 有效
# 新：fill_pcts 按搜索区比例，自适应分辨率（1080p 128px / 720p 86px 均有 6 个有效 scale）
_PREBAN_FILL_PCTS = [0.65, 0.72, 0.80, 0.88, 0.95, 0.98]
_PREBAN_THRESHOLD = 0.40   # 多尺度 RGB matchTemplate，最低约 0.40

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

def _region():          return tuple(_pcfg()['region'])
def _hero1():           return tuple(_pcfg()['ban_hero_1'])
def _hero2():           return tuple(_pcfg()['ban_hero_2'])
def _confirm():         return tuple(_pcfg()['ban_confirm'])
def _candidate_slots():  return [tuple(s) for s in _pcfg()['candidate_slots']]
def _candidate_clicks(): return [tuple(c) for c in _pcfg()['candidate_clicks']]
def _first_pick_region(): return tuple(_pcfg()['first_pick_region'])


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
# 多尺度 RGB matchTemplate + hero_images，覆盖 380+ 英雄

_preban_hero_tmpls: dict = {}   # code -> uint8 RGB (112×112)


def _ms_preban_score(search_rgb: np.ndarray, tmpl_rgb: np.ndarray) -> float:
    """多尺度 matchTemplate，preban 候选槽专用（RGB）。模板尺寸按搜索区比例计算，自适应分辨率。"""
    best = -1.0
    sh, sw = search_rgb.shape[:2]
    for pct in _PREBAN_FILL_PCTS:
        tw = int(sw * pct)
        th = int(sh * pct)
        if tw >= sw or th >= sh or tw < 8 or th < 8:
            continue
        t = cv2.resize(tmpl_rgb, (tw, th))
        r = cv2.matchTemplate(search_rgb, t, cv2.TM_CCOEFF_NORMED)
        best = max(best, float(r.max()))
    return best


def _load_preban_hero_tmpls():
    global _preban_hero_tmpls
    # 旧：preban_heroes/ 实拍模板（仅约20英雄，NCC 80×80 灰度）
    # tmpl_dir = os.path.join(_ROOT, 'templates', 'preban_heroes')
    # for fname in os.listdir(tmpl_dir):
    #     code = fname[:-4].split('_')[0]
    #     img = cv2.imdecode(np.fromfile(...), cv2.IMREAD_GRAYSCALE)
    #     tmpls.setdefault(code, []).append(cv2.resize(img, (80,80)).astype(np.float32))
    if not os.path.exists(_HERO_IMAGES_DIR_PREBAN):
        return
    tmpls: dict = {}
    for fname in os.listdir(_HERO_IMAGES_DIR_PREBAN):
        if not fname.endswith('.png'):
            continue
        code = fname[:-4]
        path = os.path.join(_HERO_IMAGES_DIR_PREBAN, fname)
        try:
            h = Image.open(path).convert('RGBA')
            bg = Image.new('RGB', h.size, (0, 0, 0))
            bg.paste(h.convert('RGB'), mask=h.split()[3])
            tmpls[code] = np.array(bg)
        except Exception:
            pass
    _preban_hero_tmpls = tmpls


def identify_preban_candidates(img) -> list:
    """返回4个格子各自识别到的英雄code，未命中返回None。"""
    if not _preban_hero_tmpls:
        _load_preban_hero_tmpls()
    results = []
    for x1, y1, x2, y2 in _candidate_slots():
        crop = img[y1:y2, x1:x2]   # RGB
        # 旧：gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        # 旧：query = cv2.resize(gray, (80,80)).astype(np.float32)
        # 旧：for code, tmpl_list in _preban_hero_tmpls.items():
        # 旧：    for tmpl in tmpl_list: s = _ncc(query, tmpl)
        best_code, best_score = None, 0.0
        for code, tmpl in _preban_hero_tmpls.items():
            s = _ms_preban_score(crop, tmpl)
            if s > best_score:
                best_score, best_code = s, code
        results.append(best_code if best_score >= _PREBAN_THRESHOLD else None)
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
