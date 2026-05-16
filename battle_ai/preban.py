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
_PREBAN_NCC_SIZE = (80, 16)

_DEFAULT_REGION   = (181, 137, 505, 203)
_DEFAULT_HERO_1   = (1695, 609)
_DEFAULT_HERO_2   = (1695, 759)
_DEFAULT_CONFIRM  = (1624, 945)


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


_ocr_preban = None
_preban_ncc_tmpl = None


def _load_preban_ncc():
    global _preban_ncc_tmpl
    tmpl_path = os.path.join(_ROOT, 'templates', 'phase', 'preban.png')
    if not os.path.exists(tmpl_path):
        return
    img = np.array(Image.open(tmpl_path).convert('L'))
    tmpl_h, tmpl_w = img.shape[:2]
    x1, y1, x2, y2 = _region()
    try:
        from config_loader import cfg
        r = cfg._profile.get('resolution', [1922, 1115]) if cfg.is_loaded() else [1922, 1115]
        ew, eh = int(r[0]), int(r[1])
    except Exception:
        ew, eh = 1922, 1115
    sx, sy = tmpl_w / ew, tmpl_h / eh
    crop = img[int(y1*sy):int(y2*sy), int(x1*sx):int(x2*sx)]
    _preban_ncc_tmpl = cv2.resize(crop, _PREBAN_NCC_SIZE).astype(np.float32)


def _preban_ncc_score(img) -> float:
    if _preban_ncc_tmpl is None:
        return 0.0
    x1, y1, x2, y2 = _region()
    crop = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    query = cv2.resize(gray, _PREBAN_NCC_SIZE).astype(np.float32)
    a = query - query.mean()
    b = _preban_ncc_tmpl - _preban_ncc_tmpl.mean()
    denom = np.sqrt((a**2).sum() * (b**2).sum())
    return float(np.sum(a * b) / denom) if denom > 1e-6 else 0.0


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
    # OCR未命中时用NCC模板匹配兜底
    if _preban_ncc_tmpl is None:
        _load_preban_ncc()
    return _preban_ncc_score(img) >= 0.60


def do_preban():
    """点击2个英雄禁用，然后点击确认。"""
    click_at(*_hero1(), delay=0.6)
    click_at(*_hero2(), delay=0.6)
    click_at(*_confirm(), delay=0.8)
