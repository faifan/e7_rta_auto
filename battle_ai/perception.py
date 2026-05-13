import ctypes
import ctypes.wintypes as wt
import io
import os
import numpy as np
import pyautogui
import cv2
from PIL import Image

from battle_ai.executor import WINDOW_TITLE, SKILL_POS

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    pass

_u32 = ctypes.windll.user32

class _RECT(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

class _POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

def _get_window_region():
    """返回客户区屏幕坐标 (left, top, width, height)，不干扰窗口状态。"""
    hwnd = _u32.FindWindowW(None, WINDOW_TITLE)
    if not hwnd:
        raise RuntimeError(f"找不到窗口：{WINDOW_TITLE!r}")
    pt = _POINT(0, 0)
    _u32.ClientToScreen(hwnd, ctypes.byref(pt))
    rc = _RECT()
    _u32.GetClientRect(hwnd, ctypes.byref(rc))
    return pt.x, pt.y, rc.right, rc.bottom   # left, top, w, h

def capture() -> np.ndarray:
    """截取游戏客户区，返回 numpy RGB 数组。"""
    left, top, w, h = _get_window_region()
    pil = pyautogui.screenshot(region=(left, top, w, h))
    return np.array(pil)

# ── 我的回合检测 ──────────────────────────────────────────
# S1 图标激活时该区域是亮金色；敌方回合时为黑色背景
_S1_X, _S1_Y = SKILL_POS['S1']
_PATCH = 20
MY_TURN_THRESHOLD = 60   # 我的回合~86，敌方回合~26

def is_my_turn(img: np.ndarray = None) -> bool:
    if img is None:
        img = capture()
    patch = img[_S1_Y - _PATCH:_S1_Y + _PATCH,
                _S1_X - _PATCH:_S1_X + _PATCH]
    return float(patch.mean()) > MY_TURN_THRESHOLD


def is_in_battle(img: np.ndarray = None) -> bool:
    """S1技能按钮区域有内容（战斗中无论我方/对手回合均>15，非战斗界面接近0）"""
    if img is None:
        img = capture()
    patch = img[_S1_Y - _PATCH:_S1_Y + _PATCH,
                _S1_X - _PATCH:_S1_X + _PATCH]
    return float(patch.mean()) > 15

# ── 结算检测（模板匹配）──────────────────────────────────
_VIC_REGION  = (660, 292, 1246, 500)   # x1,y1,x2,y2
_VIC_THRESHOLD = 0.7
_RESULT_SIZE   = (256, 90)

def _load_tmpl(path):
    return cv2.resize(np.array(Image.open(path).convert('L')), _RESULT_SIZE).astype(np.float32)

_vic_tmpl    = _load_tmpl(os.path.join(_ROOT, 'templates', 'result', 'victory.png'))
_defeat_tmpl = _load_tmpl(os.path.join(_ROOT, 'templates', 'result', 'defeat.png'))

def _match(query: np.ndarray, tmpl: np.ndarray) -> float:
    a, b = query - query.mean(), tmpl - tmpl.mean()
    denom = np.sqrt((a**2).sum() * (b**2).sum())
    return float(np.sum(a * b) / denom) if denom > 1e-6 else 0.0

def is_battle_over(img: np.ndarray = None) -> bool:
    if img is None:
        img = capture()
    x1, y1, x2, y2 = _VIC_REGION
    crop = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    query = cv2.resize(gray, _RESULT_SIZE).astype(np.float32)
    return _match(query, _vic_tmpl) >= _VIC_THRESHOLD or \
           _match(query, _defeat_tmpl) >= _VIC_THRESHOLD

# 技能图标中心区域（用于亮度检测冷却/被动）
_SKILL_ICON_CROPS = {
    'S1': (1490, 960, 1595, 1095),
    'S2': (1635, 960, 1740, 1095),
    'S3': (1780, 960, 1885, 1095),
}
_COOLDOWN_RATIO = 0.70  # 亮度低于最亮技能的70%认为不可用

def skill_brightness(img: np.ndarray, skill: str) -> float:
    """返回技能图标区域平均亮度。"""
    x1, y1, x2, y2 = _SKILL_ICON_CROPS[skill]
    return float(img[y1:y2, x1:x2].mean())

def is_skill_ready(img: np.ndarray, skill: str) -> bool:
    """
    用亮度判断技能是否可用（冷却中或被动都会变暗）。
    与三个技能里最亮的比，低于70%认为不可用。
    """
    brightest = max(skill_brightness(img, s) for s in ('S1', 'S2', 'S3'))
    brightest = max(brightest, 1.0)
    return (skill_brightness(img, skill) / brightest) >= _COOLDOWN_RATIO

# ── OCR冷却检测 ──────────────────────────────────────────────
_COOLDOWN_TEXT_CROPS = {
    'S2': (1644, 1002, 1731, 1051),
    'S3': (1790, 1003, 1882, 1059),
}

_ocr_inst = None
def _get_ocr():
    global _ocr_inst
    if _ocr_inst is None:
        import ddddocr
        _ocr_inst = ddddocr.DdddOcr(show_ad=False)
    return _ocr_inst

def is_skill_on_cooldown(img: np.ndarray, skill: str) -> bool:
    """OCR检测技能冷却：识别到'回合'则冷却中。S1永不冷却返回False。"""
    if skill not in _COOLDOWN_TEXT_CROPS:
        return False
    x1, y1, x2, y2 = _COOLDOWN_TEXT_CROPS[skill]
    crop = img[y1:y2, x1:x2]
    buf = io.BytesIO()
    Image.fromarray(crop).save(buf, format='PNG')
    text = _get_ocr().classification(buf.getvalue())
    return '回合' in text

def skill_area_unchanged(img_before: np.ndarray, img_after: np.ndarray, skill: str,
                         threshold: float = 0.98) -> bool:
    """判断技能按钮区域是否基本未变（用于识别被动技能）。"""
    x1, y1, x2, y2 = _SKILL_ICON_CROPS[skill]
    return img_similarity(img_before[y1:y2, x1:x2], img_after[y1:y2, x1:x2]) >= threshold

def img_similarity(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """两帧截图的归一化相关系数，1.0=完全相同，越低越不同。"""
    a = img_a.mean(axis=2).astype(np.float32)
    b = img_b.mean(axis=2).astype(np.float32)
    a -= a.mean(); b -= b.mean()
    denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float(np.sum(a * b) / denom) if denom > 1e-6 else 0.0

# ── 调试工具 ──────────────────────────────────────────────
def debug_thresholds():
    """打印当前帧各检测区域的实际数值，用于校准阈值。"""
    img = capture()
    print(f"截图尺寸: {img.shape}")

    patch_s1 = img[_S1_Y - _PATCH:_S1_Y + _PATCH,
                   _S1_X - _PATCH:_S1_X + _PATCH]
    s1_mean = patch_s1.mean()

    print(f"S1区域亮度:    {s1_mean:.1f}  (阈值 {MY_TURN_THRESHOLD} → {'我的回合' if s1_mean > MY_TURN_THRESHOLD else '敌方回合/等待'})")
    print(f"结算检测: {'结算' if is_battle_over(img) else '战斗中'}")

def save_debug(path: str = None):
    """保存当前截图，方便检查区域是否对准。"""
    from PIL import Image
    if path is None:
        path = os.path.join(_ROOT, 'debug', 'debug_capture.png')
    img = capture()
    Image.fromarray(img).save(path)
    print(f"已保存: {path}  尺寸: {img.shape}")
