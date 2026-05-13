"""预禁用阶段：检测界面 → 点击2个英雄禁用 → 确认"""
import time
from PIL import Image
from battle_ai.executor import click_at
from battle_ai.perception import capture

# 预禁用界面特征区域（"选择预先禁用英雄"文字所在矩形）
_PREBAN_REGION = (181, 137, 505, 203)

# 预禁用操作坐标（来自 说明_utf8.txt 2.png）
_BAN_HERO_1  = (1695, 609)
_BAN_HERO_2  = (1695, 759)
_BAN_CONFIRM = (1624, 945)


_ocr_preban = None

def _preban_ocr(img: object) -> str:
    global _ocr_preban
    import io
    if _ocr_preban is None:
        import ddddocr
        _ocr_preban = ddddocr.DdddOcr(show_ad=False)
    x1, y1, x2, y2 = _PREBAN_REGION
    crop = img[y1:y2, x1:x2]
    pil = Image.fromarray(crop)
    buf = io.BytesIO()
    pil.save(buf, format='PNG')
    try:
        return _ocr_preban.classification(buf.getvalue())
    except Exception:
        return ''


def is_in_preban(img: object = None) -> bool:
    if img is None:
        img = capture()
    text = _preban_ocr(img)
    return '禁用' in text or '预先' in text


def do_preban():
    """点击2个英雄禁用，然后点击确认"""
    click_at(*_BAN_HERO_1, delay=0.6)
    click_at(*_BAN_HERO_2, delay=0.6)
    click_at(*_BAN_CONFIRM, delay=0.8)
