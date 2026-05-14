"""预禁用阶段：检测界面 → 点击2个英雄禁用 → 确认"""
import time
import io
from PIL import Image
from battle_ai.executor import click_at
from battle_ai.perception import capture

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
    return any(k in text for k in keywords)


def do_preban():
    """点击2个英雄禁用，然后点击确认。"""
    click_at(*_hero1(), delay=0.6)
    click_at(*_hero2(), delay=0.6)
    click_at(*_confirm(), delay=0.8)
