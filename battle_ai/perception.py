import ctypes
import io
import os
import random
import time
import numpy as np
import pyautogui
import cv2
from PIL import Image

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_u32 = ctypes.windll.user32

class _RECT(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

class _POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


def _pcfg() -> dict:
    """返回 perception section，cfg 未加载时返回空字典。"""
    try:
        from config_loader import cfg
        if cfg.is_loaded():
            return cfg.section('perception')
    except ImportError:
        pass
    return {}


def _lang(key: str, default: str = '') -> str:
    try:
        from config_loader import cfg
        if cfg.is_loaded():
            return cfg.lang(key, default)
    except ImportError:
        pass
    return default


def _get_window_region():
    """返回游戏画面屏幕坐标 (left, top, width, height)。
    用 GetWindowRect 定位窗口（参考 get_coords.py 思路），
    再用 ClientToScreen + drawn_title_h 算出游戏区域偏移。
    """
    from battle_ai.executor import (get_window_title, _find_main_hwnd,
                                     _drawn_title_h, _get_expected_resolution)
    hwnd = _find_main_hwnd(get_window_title())
    if not hwnd:
        raise RuntimeError("找不到窗口，请确认游戏已打开")

    ew, eh = _get_expected_resolution()
    pt = _POINT(0, 0)
    _u32.ClientToScreen(hwnd, ctypes.byref(pt))
    rc = _RECT()
    _u32.GetClientRect(hwnd, ctypes.byref(rc))
    tab_bar = _drawn_title_h(hwnd, rc.bottom)
    return pt.x, pt.y + tab_bar, ew, eh


def capture() -> np.ndarray:
    """截取游戏画面，返回 numpy RGB 数组（profile 分辨率 ew×eh）。
    Win32 返回物理坐标，PIL grab 用逻辑坐标，用 dpi_scale 换算后再裁剪。
    """
    from PIL import ImageGrab, Image as PILImage
    from battle_ai.executor import get_window_title, _find_main_hwnd, _get_expected_resolution

    hwnd = _find_main_hwnd(get_window_title())
    ew, eh = _get_expected_resolution()

    wr = _RECT(); _u32.GetWindowRect(hwnd, ctypes.byref(wr))

    # 找最大子窗口（游戏渲染区）
    best_area = [0]; best_child = [0]; best_cw = [0]; best_ch = [0]
    _PROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def _cb(child, _):
        cr = _RECT(); _u32.GetClientRect(child, ctypes.byref(cr))
        area = cr.right * cr.bottom
        if area > best_area[0]:
            best_area[0] = area; best_child[0] = child
            best_cw[0] = cr.right; best_ch[0] = cr.bottom
        return True
    _u32.EnumChildWindows(hwnd, _PROC(_cb), 0)

    if best_child[0]:
        cpt = _POINT(0, 0)
        _u32.ClientToScreen(best_child[0], ctypes.byref(cpt))
        gx_phys = cpt.x - wr.left
        gy_phys = cpt.y - wr.top
        gw_phys = best_cw[0]
        gh_phys = best_ch[0]
        # dpi_scale = 物理像素 / 逻辑像素（100% DPI → 1.0，150% DPI → 1.5）
        dpi_scale = best_cw[0] / ew if best_cw[0] >= ew else 1.0
    else:
        from battle_ai.executor import _drawn_title_h
        pt = _POINT(0, 0); _u32.ClientToScreen(hwnd, ctypes.byref(pt))
        rc = _RECT(); _u32.GetClientRect(hwnd, ctypes.byref(rc))
        drawn = _drawn_title_h(hwnd, rc.bottom)
        gx_phys = pt.x - wr.left
        gy_phys = (pt.y - wr.top) + drawn
        gw_phys, gh_phys = ew, eh
        dpi_scale = 1.0

    # PIL grab 永远用逻辑坐标 → 物理坐标 ÷ dpi_scale
    s = dpi_scale
    bbox = (int(wr.left/s), int(wr.top/s), int(wr.right/s), int(wr.bottom/s))
    gx = int(gx_phys / s)
    gy = int(gy_phys / s)
    gw = int(gw_phys / s)   # = ew（逻辑游戏宽度）
    gh = int(gh_phys / s)   # = eh（逻辑游戏高度）

    full = ImageGrab.grab(bbox=bbox)
    arr  = np.array(full)
    game = arr[gy : gy + gh, gx : gx + gw]

    if game.shape[1] != ew or game.shape[0] != eh:
        pil = PILImage.fromarray(game)
        pil = pil.resize((ew, eh), PILImage.LANCZOS)
        return np.array(pil)
    return game


# ── 我的回合检测（亮度，向后兼容保留）────────────────────────
_DEFAULT_S1_POS  = (1543, 1029)
_DEFAULT_PATCH   = 20
_DEFAULT_BRIGHTNESS = 60

def _s1_pos():
    p = _pcfg()
    return tuple(p['s1_check_pos']) if 's1_check_pos' in p else _DEFAULT_S1_POS

def _s1_patch():
    return _pcfg().get('s1_patch', _DEFAULT_PATCH)

def _brightness_threshold():
    return _pcfg().get('my_turn_brightness', _DEFAULT_BRIGHTNESS)

# 向后兼容常量
MY_TURN_THRESHOLD = _DEFAULT_BRIGHTNESS
_PATCH            = _DEFAULT_PATCH

def is_my_turn(img: np.ndarray = None) -> bool:
    if img is None:
        img = capture()
    x, y   = _s1_pos()
    patch  = _s1_patch()
    return float(img[y - patch:y + patch, x - patch:x + patch].mean()) > _brightness_threshold()


def is_in_battle(img: np.ndarray = None) -> bool:
    """S1 区域亮度 > 15 说明战斗界面可见。"""
    if img is None:
        img = capture()
    x, y  = _s1_pos()
    patch = _s1_patch()
    return float(img[y - patch:y + patch, x - patch:x + patch].mean()) > 15


# ── 结算检测（模板匹配）──────────────────────────────────────
_DEFAULT_VIC_REGION    = (660, 292, 1246, 500)
_DEFAULT_VIC_THRESHOLD = 0.7
_RESULT_SIZE           = (256, 90)

def _vic_region():
    p = _pcfg()
    return tuple(p['vic_region']) if 'vic_region' in p else _DEFAULT_VIC_REGION

def _vic_threshold():
    return _pcfg().get('vic_threshold', _DEFAULT_VIC_THRESHOLD)


def _load_tmpl(path):
    return cv2.resize(np.array(Image.open(path).convert('L')), _RESULT_SIZE).astype(np.float32)

_vic_tmpl    = _load_tmpl(os.path.join(_ROOT, 'templates', 'result', 'victory.png'))
_defeat_tmpl = _load_tmpl(os.path.join(_ROOT, 'templates', 'result', 'defeat.png'))


def _match(query: np.ndarray, tmpl: np.ndarray) -> float:
    a, b = query - query.mean(), tmpl - tmpl.mean()
    denom = np.sqrt((a**2).sum() * (b**2).sum())
    return float(np.sum(a * b) / denom) if denom > 1e-6 else 0.0


def is_battle_victory(img: np.ndarray = None) -> bool:
    """单独检测胜利画面，用于战绩记录。"""
    if img is None:
        img = capture()
    x1, y1, x2, y2 = _vic_region()
    crop  = img[y1:y2, x1:x2]
    gray  = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    query = cv2.resize(gray, _RESULT_SIZE).astype(np.float32)
    return _match(query, _vic_tmpl) >= _vic_threshold()


def is_battle_over(img: np.ndarray = None) -> bool:
    if img is None:
        img = capture()
    x1, y1, x2, y2 = _vic_region()
    crop  = img[y1:y2, x1:x2]
    gray  = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    query = cv2.resize(gray, _RESULT_SIZE).astype(np.float32)
    thr   = _vic_threshold()
    if _match(query, _vic_tmpl) >= thr or _match(query, _defeat_tmpl) >= thr:
        return True
    if is_in_battle(img):
        return False
    return False


# ── 技能图标亮度 ──────────────────────────────────────────────
_DEFAULT_ICON_CROPS = {
    'S1': (1490, 960, 1595, 1095),
    'S2': (1635, 960, 1740, 1095),
    'S3': (1780, 960, 1885, 1095),
}
_DEFAULT_COOLDOWN_RATIO = 0.70

def _icon_crops() -> dict:
    p = _pcfg()
    if 'skill_icon_crops' in p:
        return {k: tuple(v) for k, v in p['skill_icon_crops'].items()}
    return _DEFAULT_ICON_CROPS

_SKILL_ICON_CROPS = _DEFAULT_ICON_CROPS  # 向后兼容

def skill_brightness(img: np.ndarray, skill: str) -> float:
    crops = _icon_crops()
    x1, y1, x2, y2 = crops[skill]
    return float(img[y1:y2, x1:x2].mean())


def is_skill_ready(img: np.ndarray, skill: str) -> bool:
    ratio    = _pcfg().get('cooldown_ratio', _DEFAULT_COOLDOWN_RATIO)
    brightest = max(skill_brightness(img, s) for s in ('S1', 'S2', 'S3'))
    brightest = max(brightest, 1.0)
    return (skill_brightness(img, skill) / brightest) >= ratio


# ── 亲密度等级上升弹窗检测 ────────────────────────────────────
# 弹窗叠在胜利界面上方，会遮挡 vic_region 和结算确认按钮
_DEFAULT_INTIMACY_REGION = (918, 205, 1361, 292)   # 标题"英雄亲密度等级上升"区域

def _intimacy_region():
    p = _pcfg()
    return tuple(p['intimacy_region']) if 'intimacy_region' in p else _DEFAULT_INTIMACY_REGION

def is_intimacy_levelup(img: np.ndarray = None) -> bool:
    if img is None:
        img = capture()
    x1, y1, x2, y2 = _intimacy_region()
    crop = img[y1:y2, x1:x2]
    buf  = io.BytesIO()
    Image.fromarray(crop).save(buf, format='PNG')
    text = _get_ocr().classification(buf.getvalue())
    return '亲密' in text


# ── 段位结算检测 ──────────────────────────────────────────────
_DEFAULT_LEVELUP_BTN = (785, 978, 1138, 1069)

def _levelup_btn_region():
    p = _pcfg()
    return tuple(p['levelup_btn_region']) if 'levelup_btn_region' in p else _DEFAULT_LEVELUP_BTN


def is_levelup_screen(img: np.ndarray = None) -> bool:
    if img is None:
        img = capture()
    x1, y1, x2, y2 = _levelup_btn_region()
    crop = img[y1:y2, x1:x2]
    buf  = io.BytesIO()
    Image.fromarray(crop).save(buf, format='PNG')
    text    = _get_ocr().classification(buf.getvalue())
    keyword = _lang('levelup_confirm', '确认')
    return keyword in text


# ── OCR 冷却检测 ──────────────────────────────────────────────
_DEFAULT_COOLDOWN_CROPS = {
    'S2': (1644, 1002, 1731, 1051),
    'S3': (1790, 1003, 1882, 1059),
}

def _cooldown_crops() -> dict:
    p = _pcfg()
    if 'cooldown_text_crops' in p:
        return {k: tuple(v) for k, v in p['cooldown_text_crops'].items()}
    return _DEFAULT_COOLDOWN_CROPS

_COOLDOWN_TEXT_CROPS = _DEFAULT_COOLDOWN_CROPS  # 向后兼容

_ocr_inst = None
def _get_ocr():
    global _ocr_inst
    if _ocr_inst is None:
        import ddddocr
        _ocr_inst = ddddocr.DdddOcr(show_ad=False)
    return _ocr_inst


def is_skill_on_cooldown(img: np.ndarray, skill: str) -> bool:
    crops = _cooldown_crops()
    if skill not in crops:
        return False
    x1, y1, x2, y2 = crops[skill]
    crop = img[y1:y2, x1:x2]
    buf  = io.BytesIO()
    Image.fromarray(crop).save(buf, format='PNG')
    text    = _get_ocr().classification(buf.getvalue())
    keyword = _lang('cooldown', '回合')
    return keyword in text


# ── 回合徽章模板匹配 + 角色名 OCR ────────────────────────────
_DEFAULT_BADGE_REGION = (889, 233, 1026, 293)
_DEFAULT_NAME_REGION  = (376, 971, 665, 1033)
_DEFAULT_BADGE_SIZE   = (137, 60)
_DEFAULT_BADGE_THR    = 0.7
_BATTLE_TMPL_DIR      = os.path.join(_ROOT, 'templates', 'battle')

def _badge_region():
    p = _pcfg()
    return tuple(p['badge_region']) if 'badge_region' in p else _DEFAULT_BADGE_REGION

def _badge_size():
    p = _pcfg()
    return tuple(p['badge_size']) if 'badge_size' in p else _DEFAULT_BADGE_SIZE

def _badge_threshold():
    return _pcfg().get('badge_threshold', _DEFAULT_BADGE_THR)

def _name_region():
    p = _pcfg()
    return tuple(p['name_region']) if 'name_region' in p else _DEFAULT_NAME_REGION


def _load_badge_tmpl(name: str):
    path = os.path.join(_BATTLE_TMPL_DIR, name)
    if not os.path.exists(path):
        return None
    return cv2.resize(
        np.array(Image.open(path).convert('L')), _DEFAULT_BADGE_SIZE
    ).astype(np.float32)

_my_turn_tmpl    = _load_badge_tmpl('my_turn.png')
_enemy_turn_tmpl = _load_badge_tmpl('enemy_turn.png')


def read_turn_badge(img: np.ndarray = None) -> str:
    """
    判断中央回合徽章状态：'my_turn' / 'enemy_turn' / 'none'。
    优先 NCC 模板匹配，模板不存在时降级 OCR。
    """
    if img is None:
        img = capture()
    x1, y1, x2, y2 = _badge_region()
    crop  = img[y1:y2, x1:x2]
    bsize = _badge_size()
    thr   = _badge_threshold()

    if _my_turn_tmpl is not None and _enemy_turn_tmpl is not None:
        gray  = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        query = cv2.resize(gray, bsize).astype(np.float32)
        if _match(query, _my_turn_tmpl) >= thr:
            return 'my_turn'
        if _match(query, _enemy_turn_tmpl) >= thr:
            return 'enemy_turn'
        # 模板未匹配，OCR兜底

    buf  = io.BytesIO()
    Image.fromarray(crop).save(buf, format='PNG')
    text = _get_ocr().classification(buf.getvalue())
    if _lang('turn_enemy', '对手') in text:
        return 'enemy_turn'
    if _lang('turn_my', '回合') in text:
        return 'my_turn'
    return 'none'


def save_badge_template(state: str):
    """截取当前屏幕徽章区域保存为模板（state='my_turn' 或 'enemy_turn'）。"""
    assert state in ('my_turn', 'enemy_turn')
    img = capture()
    x1, y1, x2, y2 = _badge_region()
    crop = img[y1:y2, x1:x2]
    os.makedirs(_BATTLE_TMPL_DIR, exist_ok=True)
    path = os.path.join(_BATTLE_TMPL_DIR, f'{state}.png')
    Image.fromarray(crop).save(path)
    global _my_turn_tmpl, _enemy_turn_tmpl
    _my_turn_tmpl    = _load_badge_tmpl('my_turn.png')
    _enemy_turn_tmpl = _load_badge_tmpl('enemy_turn.png')
    print(f"模板已保存并加载: {path}")


def read_char_name(img: np.ndarray = None) -> str:
    """OCR 左下角角色名。"""
    if img is None:
        img = capture()
    x1, y1, x2, y2 = _name_region()
    crop = img[y1:y2, x1:x2]
    buf  = io.BytesIO()
    Image.fromarray(crop).save(buf, format='PNG')
    return _get_ocr().classification(buf.getvalue()).strip()


def skill_area_unchanged(img_before: np.ndarray, img_after: np.ndarray,
                         skill: str, threshold: float = 0.98) -> bool:
    crops = _icon_crops()
    x1, y1, x2, y2 = crops[skill]
    return img_similarity(img_before[y1:y2, x1:x2],
                          img_after[y1:y2, x1:x2]) >= threshold


def img_similarity(img_a: np.ndarray, img_b: np.ndarray) -> float:
    a = img_a.mean(axis=2).astype(np.float32)
    b = img_b.mean(axis=2).astype(np.float32)
    a -= a.mean(); b -= b.mean()
    denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float(np.sum(a * b) / denom) if denom > 1e-6 else 0.0


# ── 烧魂检测 ──────────────────────────────────────────────────
_DEFAULT_BURN_BTN_REGION    = (878, 961, 1212, 1092)
_DEFAULT_CANCEL_BTN_REGION  = (899, 981, 1171, 1070)   # 1922×1115 默认值
_BURN_AVAILABLE_RATIO       = 0.05   # 蓝色像素占比 ≥ 5% → Burn可用

def _burn_btn_region() -> tuple:
    p = _pcfg()
    return tuple(p['burn_btn_region']) if 'burn_btn_region' in p else _DEFAULT_BURN_BTN_REGION

def _cancel_btn_region() -> tuple:
    p = _pcfg()
    return tuple(p['cancel_btn_region']) if 'cancel_btn_region' in p else _DEFAULT_CANCEL_BTN_REGION

def _check_blue_ratio(img: np.ndarray, region: tuple) -> float:
    """返回region内HSV蓝色像素占比。"""
    x1, y1, x2, y2 = region
    crop = img[y1:y2, x1:x2]
    hsv  = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    mask = (
        (hsv[:, :, 0] >= 95) & (hsv[:, :, 0] <= 135) &
        (hsv[:, :, 1] >= 60) &
        (hsv[:, :, 2] >= 60)
    )
    return float(mask.sum()) / max(mask.size, 1)


# ── 烧魂按钮模板NCC检测 ───────────────────────────────────────
_BURN_TMPL_SIZE = (157, 62)
_burn_templates: list = []

def _load_burn_templates():
    global _burn_templates
    if _burn_templates:
        return
    tmpl_dir = os.path.join(_ROOT, 'templates')
    for i in range(1, 4):
        path = os.path.join(tmpl_dir, f'burn_btn_{i}.png')
        if os.path.exists(path):
            img = np.array(Image.open(path).convert('L'))
            tmpl = cv2.resize(img, _BURN_TMPL_SIZE).astype(np.float32)
            _burn_templates.append(tmpl)

def _ncc_burn(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.flatten() - a.mean(), b.flatten() - b.mean()
    denom = np.sqrt((a**2).sum() * (b**2).sum())
    return float(np.dot(a, b) / denom) if denom > 1e-6 else 0.0

_BURN_NCC_THRESHOLD = 0.65

def _check_burn_ncc(img: np.ndarray) -> bool:
    """用模板NCC检测烧魂按钮是否亮起。"""
    _load_burn_templates()
    if not _burn_templates:
        return False
    x1, y1, x2, y2 = _burn_btn_region()
    crop = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    query = cv2.resize(gray, _BURN_TMPL_SIZE).astype(np.float32)
    return max(_ncc_burn(query, t) for t in _burn_templates) >= _BURN_NCC_THRESHOLD

_BURN_WHITE_RATIO = 0.02  # 白色像素占比 >= 2% 认为有文字

def _burn_has_text(img: np.ndarray, region: tuple) -> bool:
    """检测 burn 区域是否有白色文字（阈值后白像素占比，无需额外依赖）。"""
    try:
        x1, y1, x2, y2 = region
        crop = img[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        ratio = thresh.sum() / (255 * thresh.size)
        return ratio >= _BURN_WHITE_RATIO
    except Exception:
        return True  # 出错时不拦截


def is_soul_burn_available(img: np.ndarray = None) -> bool:
    """
    检测"Burn!"按钮是否可用。
    有模板时用NCC匹配；无模板降级到HSV蓝色检测。
    传入img时单次检测；不传时3秒内轮询，命中即返回True。
    """
    _load_burn_templates()
    use_ncc = bool(_burn_templates)
    region = _burn_btn_region()

    if img is not None:
        return _check_burn_ncc(img) if use_ncc else _check_blue_ratio(img, region) >= _BURN_AVAILABLE_RATIO

    deadline = time.time() + 3.0
    while time.time() < deadline:
        try:
            frame = capture()
            hit = _check_burn_ncc(frame) if use_ncc else _check_blue_ratio(frame, region) >= _BURN_AVAILABLE_RATIO
            if hit:
                return True
        except Exception:
            pass
        time.sleep(random.uniform(0.05, 0.35))
    return False

def sample_soul_burn_available(duration: float = 1.5, interval: float = 0.12) -> bool:
    """多帧OR采样检测烧魂按钮（专为闪烁场景设计）。
    duration秒内每隔interval截一帧，任意帧命中即返回True。
    有模板用NCC，无模板用HSV+白字双校验。
    """
    _load_burn_templates()
    use_ncc = bool(_burn_templates)
    region  = _burn_btn_region()
    deadline = time.time() + duration
    while time.time() < deadline:
        try:
            frame = capture()
            if use_ncc:
                hit = _check_burn_ncc(frame)
            else:
                hit = (_check_blue_ratio(frame, region) >= _BURN_AVAILABLE_RATIO
                       and _burn_has_text(frame, region))
            if hit:
                return True
        except Exception:
            pass
        remaining = deadline - time.time()
        if remaining > 0:
            time.sleep(min(interval, remaining))
    return False


def is_soul_burn_activated(img: np.ndarray = None) -> bool:
    """检测烧魂已激活（Cancel按钮出现）。
    无参数时：OCR裁Cancel区域识别文字，每0.15s一帧，最多3秒。
    """
    if img is not None:
        # 兼容旧单帧调用，保留原逻辑
        region = _burn_btn_region()
        _BURN_ACTIVATED_RATIO = 0.015
        return _check_blue_ratio(img, region) < _BURN_ACTIVATED_RATIO

    x1, y1, x2, y2 = _cancel_btn_region()
    start = time.time()
    while time.time() - start < 3.0:
        frame = capture()
        crop  = frame[y1:y2, x1:x2]
        buf   = io.BytesIO()
        Image.fromarray(crop).save(buf, format='PNG')
        text  = _get_ocr().classification(buf.getvalue())
        if 'cancel' in text.lower():
            return True
        time.sleep(0.15)
    return False


# ── 敌方血量检测 ──────────────────────────────────────────────
_DEAD_RATIO = 0.02   # 绿色占比低于此视为已死亡/空槽
_SCAN_HALF  = 100    # 血条Y扫描范围 ±100px

def _enemy_hp_regions() -> list:
    try:
        from config_loader import cfg
        if cfg.is_loaded():
            p = cfg.section('executor')
            return [tuple(r) for r in p['enemy_hp_regions']] if 'enemy_hp_regions' in p else []
    except Exception:
        pass
    return []

def _enemy_pos_list() -> list:
    try:
        from config_loader import cfg
        if cfg.is_loaded():
            p = cfg.section('executor')
            return [tuple(r) for r in p.get('enemy_pos', [])]
    except Exception:
        pass
    return []

# ── 集火：回合开始扫一次血条，全场复用缓存 ────────────────────
_detected_hp_regions: list = []   # [(x1,y1,x2,y2)] 实际血条框
_detected_click_pos:  list = []   # [(x,y)] 动态攻击点位

def reset_hp_regions():
    """每场战斗开始调用，清空缓存，让下一回合重新扫。"""
    global _detected_hp_regions, _detected_click_pos
    _detected_hp_regions = []
    _detected_click_pos  = []

def _find_bar_y_sat(img: np.ndarray, x1: int, x2: int, y_min: int, y_max: int):
    """在扫描区域内按行统计绿色像素数，最多的行即血条所在行。
    只统计 H 35~90（绿色），排除蓝色buff等干扰。
    返回 (bar_y, green_ratio)；绿色像素不足返回 (None, 0.0)。
    """
    y_min = max(0, y_min);  y_max = min(img.shape[0], y_max)
    x1    = max(0, x1);     x2    = min(img.shape[1], x2)
    if y_min >= y_max or x1 >= x2:
        return None, 0.0
    hsv   = cv2.cvtColor(img[y_min:y_max, x1:x2], cv2.COLOR_RGB2HSV)
    green = (
        (hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 90) &
        (hsv[:, :, 1] >= 80) & (hsv[:, :, 2] >= 60)
    ).astype(np.float32)
    row_counts = green.sum(axis=1)
    best_row   = int(row_counts.argmax())
    best_count = float(row_counts[best_row])
    width      = x2 - x1
    if best_count < max(5, width * 0.05):
        return None, best_count / max(width, 1)
    return y_min + best_row, best_count / width

def _detect_and_cache_hp(img: np.ndarray):
    """扫描所有槽位血条并缓存坐标和动态攻击点位。"""
    global _detected_hp_regions, _detected_click_pos
    hp_regs  = _enemy_hp_regions()
    pos_list = _enemy_pos_list()
    bars, clicks = [], []
    for i, (x1, y1, x2, y2) in enumerate(hp_regs):
        yc    = (y1 + y2) // 2
        bar_y, _ = _find_bar_y_sat(img, x1, x2, yc - _SCAN_HALF, yc + _SCAN_HALF)
        if bar_y is not None:
            bars.append((x1, bar_y - 4, x2, bar_y + 4))
            if i < len(pos_list):
                ex, ey = pos_list[i]
                # 攻击点Y = 检测到的血条Y + (profile攻击Y - profile血条中心Y)
                clicks.append((ex, int(bar_y + (ey - yc))))
            else:
                clicks.append(((x1 + x2) // 2, bar_y + 180))
        else:
            # 扫描失败，fallback到profile坐标
            bars.append((x1, y1, x2, y2))
            clicks.append(tuple(pos_list[i]) if i < len(pos_list) else ((x1+x2)//2, yc+180))
    _detected_hp_regions = bars
    _detected_click_pos  = clicks

def get_dynamic_click_pos(idx: int) -> 'tuple | None':
    """返回动态攻击点位；未缓存时返回None（调用方fallback到profile）。"""
    return _detected_click_pos[idx] if 0 <= idx < len(_detected_click_pos) else None

def get_enemy_hp_ratios(img: np.ndarray) -> list[float]:
    """返回4个敌方目标的血量比例（0.0~1.0，越高血越多）。
    每回合重新扫描血条坐标（约2~5ms），同时更新动态攻击点位。
    """
    _detect_and_cache_hp(img)
    ratios = []
    for x1, y1, x2, y2 in _detected_hp_regions:
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            ratios.append(0.0)
            continue
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        green = (
            (hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 90) &
            (hsv[:, :, 1] >= 80) &
            (hsv[:, :, 2] >= 60)
        )
        width = x2 - x1
        row_max = float(green.sum(axis=1).max()) if green.shape[0] > 0 else 0.0
        ratios.append(row_max / max(width, 1))
    return ratios


# ── 调试工具 ──────────────────────────────────────────────────
def debug_thresholds():
    img    = capture()
    x, y   = _s1_pos()
    patch  = _s1_patch()
    mean   = img[y - patch:y + patch, x - patch:x + patch].mean()
    print(f"截图尺寸: {img.shape}")
    print(f"S1亮度: {mean:.1f}  (阈值 {_brightness_threshold()} → {'我的回合' if mean > _brightness_threshold() else '等待'})")
    print(f"结算: {'结算' if is_battle_over(img) else '战斗中'}")


def save_debug(path: str = None):
    if path is None:
        path = os.path.join(_ROOT, 'debug', 'debug_capture.png')
    img = capture()
    Image.fromarray(img).save(path)
    print(f"已保存: {path}  尺寸: {img.shape}")
