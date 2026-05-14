"""选秀阶段：检测回合 → 模型推荐 → 搜索选人 → 识别对手"""
import os
import re as _re
import random
import time
import json
import numpy as np
import cv2
from PIL import Image
from battle_ai.executor import click_at, type_text_chinese
from battle_ai.perception import capture

_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEBUG_DIR = os.path.join(_ROOT, 'debug')
os.makedirs(_DEBUG_DIR, exist_ok=True)


# ── 配置加载 ──────────────────────────────────────────────────
def _dcfg() -> dict:
    try:
        from config_loader import cfg
        if cfg.is_loaded():
            return cfg.section('draft')
    except ImportError:
        pass
    return {}

def _dlang(key: str, default=None):
    try:
        from config_loader import cfg
        if cfg.is_loaded():
            return cfg.lang(key, default)
    except ImportError:
        pass
    return default


# ── 坐标默认值 ────────────────────────────────────────────────
_DEFAULT_SEARCH_OPEN          = (1820, 173)
_DEFAULT_SEARCH_INPUT         = (1073, 269)
_DEFAULT_SEARCH_BTN           = (1437, 264)
_DEFAULT_FIRST_RESULT         = (1010, 443)
_DEFAULT_CONFIRM_BTN          = (1274, 1029)
_DEFAULT_POST_BAN_BTN         = (1282, 1027)
_DEFAULT_OPP_TURN_REGION      = (679, 65, 823, 115)
_DEFAULT_BANNER_REGION        = (550, 3, 950, 55)
_DEFAULT_LABEL_REGION         = (632, 53, 862, 127)
_DEFAULT_SEARCH_RESULT_REGION = (855, 402, 1264, 469)
_DEFAULT_SELECTED_HERO_REGION = (1541, 389, 1699, 458)
_DEFAULT_SEARCH_STATE_REGION  = (1747, 142, 1897, 197)
_DEFAULT_SEARCH_EXEC_REGION   = (1341, 227, 1538, 300)
_DEFAULT_SEARCH_CLEAR_BTN     = (1284, 264)
_DEFAULT_PREBAN_REGION        = (181, 137, 505, 203)

_DEFAULT_MY_SLOTS = [
    ( 37, 216, 461, 343),
    ( 34, 367, 361, 491),
    ( 37, 520, 318, 638),
    ( 35, 668, 314, 788),
    ( 41, 819, 359, 937),
]
_DEFAULT_OPP_SLOTS = [
    (1033, 219, 1456, 337),
    (1133, 367, 1456, 487),
    (1175, 518, 1453, 638),
    (1177, 668, 1463, 790),
    (1136, 818, 1462, 937),
]
_DEFAULT_MY_BAN_SLOTS  = [(565, 1010, 646, 1096), (646, 1010, 741, 1096)]
_DEFAULT_OPP_BAN_SLOTS = [(753, 1006, 842, 1095), (842, 1006, 937, 1095)]

# 向后兼容模块级常量
MY_SLOTS  = _DEFAULT_MY_SLOTS
OPP_SLOTS = _DEFAULT_OPP_SLOTS


# ── 坐标访问（懒加载，cfg 未加载时用默认值）──────────────────
def _search_open():           return tuple(_dcfg().get('search_open',          _DEFAULT_SEARCH_OPEN))
def _search_input():          return tuple(_dcfg().get('search_input',         _DEFAULT_SEARCH_INPUT))
def _search_btn():            return tuple(_dcfg().get('search_btn',           _DEFAULT_SEARCH_BTN))
def _first_result():          return tuple(_dcfg().get('first_result',         _DEFAULT_FIRST_RESULT))
def _confirm_btn():           return tuple(_dcfg().get('confirm_btn',          _DEFAULT_CONFIRM_BTN))
def _post_ban_btn():          return tuple(_dcfg().get('post_ban_btn',         _DEFAULT_POST_BAN_BTN))
def _banner_region():         return tuple(_dcfg().get('banner_region',        _DEFAULT_BANNER_REGION))
def _label_region():          return tuple(_dcfg().get('label_region',         _DEFAULT_LABEL_REGION))
def _search_result_region():  return tuple(_dcfg().get('search_result_region', _DEFAULT_SEARCH_RESULT_REGION))
def _selected_hero_region():  return tuple(_dcfg().get('selected_hero_region', _DEFAULT_SELECTED_HERO_REGION))
def _search_exec_region():    return tuple(_dcfg().get('search_exec_region',   _DEFAULT_SEARCH_EXEC_REGION))
def _search_clear_btn():      return tuple(_dcfg().get('search_clear_btn',     _DEFAULT_SEARCH_CLEAR_BTN))
def _preban_region_draft():   return tuple(_dcfg().get('preban_region',        _DEFAULT_PREBAN_REGION))
def _my_slots():   return [tuple(v) for v in _dcfg().get('my_slots',   _DEFAULT_MY_SLOTS)]
def _opp_slots():  return [tuple(v) for v in _dcfg().get('opp_slots',  _DEFAULT_OPP_SLOTS)]
def _my_ban_slots():  return [tuple(v) for v in _dcfg().get('my_ban_slots',  _DEFAULT_MY_BAN_SLOTS)]
def _opp_ban_slots(): return [tuple(v) for v in _dcfg().get('opp_ban_slots', _DEFAULT_OPP_BAN_SLOTS)]
def _opp_slot_centers():
    return [((x1 + x2) // 2, (y1 + y2) // 2) for x1, y1, x2, y2 in _opp_slots()]


# ── 文字匹配辅助（通过 lang 文件支持多语言）──────────────────
def _is_my_turn_text(text: str) -> bool:
    kw = _dlang('draft_my_turn_kw', '你的')
    return bool(kw) and kw in text

def _is_opp_turn_text(text: str) -> bool:
    kws = _dlang('draft_opp_turn_kw', ['对手', '对方'])
    if isinstance(kws, list):
        return any(k in text for k in kws)
    return bool(kws) and kws in text

def _is_ban_kw(text: str) -> bool:
    kw = _dlang('draft_ban_kw', '禁用')
    return bool(kw) and kw in text

def _is_ready_kw(text: str) -> bool:
    kw = _dlang('draft_ready_kw', '准备')
    return bool(kw) and kw in text

def _check_preban_text(text: str) -> bool:
    kws = _dlang('preban_keywords', ['禁用', '预先'])
    if isinstance(kws, list):
        return any(k in text for k in kws)
    return bool(kws) and kws in text


# ── 英雄名称映射 ──────────────────────────────────────────────
_E7_JSON = os.path.join(_ROOT, 'e7.json')
_code_to_name: dict = {}


def _load_hero_names():
    if not _code_to_name and os.path.exists(_E7_JSON):
        with open(_E7_JSON, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for hero in data:
            _code_to_name[hero['code']] = hero['name']


# ── 选秀卡识别 ────────────────────────────────────────────────
_DRAFT_CARDS_DIR = os.path.join(_ROOT, 'templates', 'draft_cards')
_TMPL_BASE       = os.path.join(_ROOT, 'templates')
_TMPL_SIZE       = (96, 56)
_NCC_THRESHOLD   = 0.5
_NCC_GAP         = 0.02

_draft_templates: dict = {}


def _load_draft_templates():
    if not os.path.exists(_DRAFT_CARDS_DIR):
        return
    for fname in os.listdir(_DRAFT_CARDS_DIR):
        if not fname.endswith('.png'):
            continue
        base = _re.sub(r'_\d+$', '', fname[:-4])
        path = os.path.join(_DRAFT_CARDS_DIR, fname)
        try:
            img = np.array(Image.open(path).convert('L'))
            tmpl = cv2.resize(img, _TMPL_SIZE).astype(np.float32)
            _draft_templates.setdefault(base, []).append(tmpl)
        except Exception:
            pass


def _ncc_flat(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.flatten(), b.flatten()
    a = a - a.mean(); b = b - b.mean()
    denom = np.sqrt((a**2).sum() * (b**2).sum())
    return float(np.dot(a, b) / denom) if denom > 1e-6 else 0.0


def identify_slot(img: np.ndarray, region: tuple, exclude: set = None) -> str:
    code, _, _ = identify_slot_debug(img, region, exclude=exclude)
    return code


def identify_slot_debug(img: np.ndarray, region: tuple, exclude: set = None) -> tuple:
    """返回 (code, best_score, gap)，方便日志排查。exclude 中的 code 跳过，自动取次高分。"""
    if not _draft_templates:
        _load_draft_templates()
    if not _draft_templates:
        return 'unknown', 0.0, 0.0
    x1, y1, x2, y2 = region
    crop = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    query = cv2.resize(gray, _TMPL_SIZE).astype(np.float32)
    best_code, best_score, second_score = 'unknown', -1.0, -1.0
    for code, tmpls in _draft_templates.items():
        if exclude and code in exclude:
            continue
        s = max(_ncc_flat(query, t) for t in tmpls)
        if s > best_score:
            second_score = best_score
            best_score, best_code = s, code
        elif s > second_score:
            second_score = s
    gap = best_score - second_score
    if best_score >= _NCC_THRESHOLD and gap >= _NCC_GAP:
        return best_code, best_score, gap
    return 'unknown', best_score, gap


# ── 禁用槽识别（CDN头像匹配）────────────────────────────────
_HERO_IMAGES_DIR = os.path.join(_ROOT, 'templates', 'hero_images')
_BAN_TMPL_SIZE   = (64, 64)
_BAN_THRESHOLD   = 0.35

_ban_templates: dict = {}


def _load_ban_templates():
    if not os.path.exists(_HERO_IMAGES_DIR):
        return
    for fname in os.listdir(_HERO_IMAGES_DIR):
        if not fname.endswith('.png'):
            continue
        code = fname[:-4]
        path = os.path.join(_HERO_IMAGES_DIR, fname)
        try:
            img = np.array(Image.open(path).convert('L'))
            _ban_templates[code] = cv2.resize(img, _BAN_TMPL_SIZE).astype(np.float32)
        except Exception:
            pass


def _identify_ban_slot(crop_gray: np.ndarray) -> tuple:
    query = cv2.resize(crop_gray, _BAN_TMPL_SIZE).astype(np.float32)
    best_code, best_score = 'empty', -1.0
    for code, tmpl in _ban_templates.items():
        s = _ncc_flat(query, tmpl)
        if s > best_score:
            best_score, best_code = s, code
    if best_score >= _BAN_THRESHOLD:
        return best_code, best_score
    return 'empty', best_score


def identify_ban_slots(img: np.ndarray = None) -> tuple:
    """识别底部4个禁用头像，返回 ([my1, my2, opp1, opp2], [scores])，空槽code='empty'"""
    if not _ban_templates:
        _load_ban_templates()
    if img is None:
        img = capture()
    try:
        cv2.imwrite(os.path.join(_DEBUG_DIR, 'dbg_ban_full.png'),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    except Exception:
        pass
    result = []
    scores = []
    debug_crops = []
    labels = ['我方ban1', '我方ban2', '对手ban1', '对手ban2']
    all_slots = _my_ban_slots() + _opp_ban_slots()
    for i, (x1, y1, x2, y2) in enumerate(all_slots):
        crop = img[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        code, ban_score = _identify_ban_slot(gray)
        result.append(code)
        scores.append(ban_score)
        enlarged = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
                              (crop.shape[1]*3, crop.shape[0]*3),
                              interpolation=cv2.INTER_NEAREST)
        debug_crops.append((labels[i], code, enlarged))
    try:
        for i, (label, code, c) in enumerate(debug_crops):
            cv2.imwrite(os.path.join(_DEBUG_DIR, f'dbg_ban_{i+1}.png'), c)
        h = max(c.shape[0] for _, _, c in debug_crops)
        row = np.zeros((h + 20, sum(c.shape[1] + 4 for _, _, c in debug_crops), 3), np.uint8)
        x = 0
        for label, code, c in debug_crops:
            row[20:20+c.shape[0], x:x+c.shape[1]] = c
            cv2.putText(row, f'{label}:{code}', (x, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 0), 1)
            x += c.shape[1] + 4
        cv2.imwrite(os.path.join(_DEBUG_DIR, 'dbg_ban_slots.png'), row)
    except Exception:
        pass
    return result, scores


# ── 顶部文字区域检测（你的回合 / 对手回合 / 选择禁用英雄）────────
_LABEL_SIZE   = (115, 37)

_MY_TURN_TMPL  = None
_OPP_TURN_TMPL = None
_POSTBAN_TMPL  = None


def _load_label_tmpls():
    global _MY_TURN_TMPL, _OPP_TURN_TMPL, _POSTBAN_TMPL
    x1, y1, x2, y2 = _label_region()
    for var, fname in [('_MY_TURN_TMPL',  'my_turn.png'),
                       ('_OPP_TURN_TMPL', 'opp_turn.png'),
                       ('_POSTBAN_TMPL',  'postban.png')]:
        path = os.path.join(_TMPL_BASE, 'phase', fname)
        img  = np.array(Image.open(path).convert('L'))
        crop = img[y1:y2, x1:x2]
        globals()[var] = cv2.resize(crop, _LABEL_SIZE).astype(np.float32)


def _ncc(a, b):
    a, b = a - a.mean(), b - b.mean()
    denom = np.sqrt((a**2).sum() * (b**2).sum())
    return float(np.sum(a * b) / denom) if denom > 1e-6 else 0.0


def _label_ncc(img: np.ndarray, tmpl: np.ndarray) -> float:
    x1, y1, x2, y2 = _label_region()
    crop = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    query = cv2.resize(gray, _LABEL_SIZE).astype(np.float32)
    return _ncc(query, tmpl)


def _ensure_label_tmpls():
    if _MY_TURN_TMPL is None:
        _load_label_tmpls()


def _opp_turn_score(img: np.ndarray = None) -> float:
    _ensure_label_tmpls()
    if img is None:
        img = capture()
    return _label_ncc(img, _OPP_TURN_TMPL)


def _my_turn_score(img: np.ndarray = None) -> float:
    _ensure_label_tmpls()
    if img is None:
        img = capture()
    return _label_ncc(img, _MY_TURN_TMPL)


# ── OCR 回合检测 ──────────────────────────────────────────────
_ocr_instance = None

def _get_ocr():
    global _ocr_instance
    if _ocr_instance is None:
        import ddddocr
        _ocr_instance = ddddocr.DdddOcr(show_ad=False)
    return _ocr_instance


def _read_turn_text(img: np.ndarray = None) -> str:
    """OCR 识别顶部回合文字。"""
    import io
    if img is None:
        img = capture()
    x1, y1, x2, y2 = _label_region()
    crop = img[y1:y2, x1:x2]
    pil = Image.fromarray(crop)
    buf = io.BytesIO()
    pil.save(buf, format='PNG')
    try:
        text = _get_ocr().classification(buf.getvalue())
    except Exception:
        text = ''
    return text


def is_my_turn_ocr(img: np.ndarray = None) -> bool:
    return _is_my_turn_text(_read_turn_text(img))


def is_opponent_turn_ocr(img: np.ndarray = None) -> bool:
    return _is_opp_turn_text(_read_turn_text(img))


def is_opponent_turn(img: np.ndarray = None) -> bool:
    _ensure_label_tmpls()
    if img is None:
        img = capture()
    return _opp_turn_score(img) >= 0.55


# ── 选秀后禁用 & 战斗开始检测 ────────────────────────────────
_POST_BAN_TMPL   = None
_BATTLE_RDY_TMPL = None
_BANNER_SIZE     = (100, 13)


def _load_banner_tmpls():
    global _POST_BAN_TMPL, _BATTLE_RDY_TMPL
    x1, y1, x2, y2 = _banner_region()
    for attr, path in [('_POST_BAN_TMPL',   os.path.join(_TMPL_BASE, 'phase', 'postban.png')),
                       ('_BATTLE_RDY_TMPL', os.path.join(_TMPL_BASE, 'phase', 'battle_ready.png'))]:
        img = np.array(Image.open(path).convert('L'))
        crop = img[y1:y2, x1:x2]
        tmpl = cv2.resize(crop, _BANNER_SIZE).astype(np.float32)
        globals()[attr] = tmpl


def _banner_ncc(img: np.ndarray, tmpl: np.ndarray) -> float:
    x1, y1, x2, y2 = _banner_region()
    crop = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    query = cv2.resize(gray, _BANNER_SIZE).astype(np.float32)
    return _ncc(query, tmpl)


def is_in_draft(img: np.ndarray = None) -> bool:
    """检测是否在选秀阶段：OCR 读到我的/对手的回合关键词。"""
    if img is None:
        img = capture()
    text = _read_turn_text(img)
    return _is_my_turn_text(text) or _is_opp_turn_text(text)


def is_in_post_draft_ban(img: np.ndarray = None) -> bool:
    """检测顶部文字是否为"选择禁用英雄"（选秀后禁用阶段）"""
    if img is None:
        img = capture()
    text = _read_turn_text(img)
    return _is_ban_kw(text) and not _is_my_turn_text(text) and not _is_opp_turn_text(text)


def is_battle_ready(img: np.ndarray = None) -> bool:
    """检测是否在准备战斗阶段。"""
    if img is None:
        img = capture()
    if not _is_ready_kw(_read_turn_text(img)):
        return False
    preban_text = _ocr_region(img, _preban_region_draft())
    return not _check_preban_text(preban_text)


def do_post_draft_ban(enemy_picks: list, recommender=None, my_picks=None,
                      banned=None, my_first=True, log_fn=None):
    """选秀后禁用：模型推荐应禁的对手英雄，点击对应槽位后确认。"""
    _log = log_fn or (lambda m: None)
    _load_hero_names()

    ban_idx = None

    if recommender is not None and enemy_picks:
        try:
            recs = recommender.recommend_finalban(
                my_picks=my_picks or [],
                enemy_picks=enemy_picks,
                banned=banned or [],
                my_first=my_first,
                top_k=5,
            )
            for rec in recs:
                code = rec['hero_code']
                if code in enemy_picks:
                    ban_idx = enemy_picks.index(code)
                    _log(f'  [禁用] 模型推荐禁: {_code_to_name.get(code, code)}（对手槽{ban_idx+1}）')
                    break
        except Exception as e:
            _log(f'  [禁用] 模型推荐异常: {e}')

    if ban_idx is None:
        ban_idx = random.randrange(len(enemy_picks)) if enemy_picks else 0
        code = enemy_picks[ban_idx] if enemy_picks else '?'
        _log(f'  [禁用] 随机禁: {_code_to_name.get(code, code)}（对手槽{ban_idx+1}）')

    cx, cy = _opp_slot_centers()[ban_idx]
    click_at(cx, cy, delay=1.5)
    click_at(*_post_ban_btn(), delay=1.5)


def click_battle_start():
    """点击战斗开始按钮（与postban完成选择同坐标）"""
    click_at(*_post_ban_btn(), delay=1.5)


# ── 选秀顺序（仅保留供参考，不再驱动逻辑） ────────────────────
_TURN_SEQ_FIRST  = [1, 2, 2, 1, 1, 2, 2, 1, 1, 2]
_TURN_SEQ_SECOND = [2, 1, 1, 2, 2, 1, 1, 2, 2, 1]

_PHASE_BY_POS = [
    'pick1',
    'pick2', 'pick2',
    'pick3', 'pick3',
    'pick4', 'pick4',
    'pick5', 'pick5', 'pick5',
]


def _get_phase(total_picks: int) -> str:
    if total_picks < len(_PHASE_BY_POS):
        return _PHASE_BY_POS[total_picks]
    return 'pick5'


def _read_clipboard() -> str:
    """读取当前剪贴板文字，用于调试。"""
    import ctypes as c
    u, k = c.windll.user32, c.windll.kernel32
    u.GetClipboardData.restype  = c.c_void_p
    k.GlobalLock.restype        = c.c_void_p
    k.GlobalLock.argtypes       = [c.c_void_p]
    if not u.OpenClipboard(0):
        return '<无法打开剪贴板>'
    try:
        h = u.GetClipboardData(13)
        if not h:
            return '<空>'
        p = k.GlobalLock(h)
        if not p:
            return '<锁定失败>'
        text = c.wstring_at(p)
        k.GlobalUnlock(h)
        return text
    finally:
        u.CloseClipboard()


def _type_unicode(text: str):
    """用 SendInput KEYEVENTF_UNICODE 逐字发送Unicode，不依赖剪贴板。"""
    import ctypes

    KEYEVENTF_UNICODE = 0x0004
    KEYEVENTF_KEYUP   = 0x0002

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [
            ('wVk',         ctypes.c_ushort),
            ('wScan',       ctypes.c_ushort),
            ('dwFlags',     ctypes.c_ulong),
            ('time',        ctypes.c_ulong),
            ('dwExtraInfo', ctypes.c_size_t),
        ]

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ('dx',          ctypes.c_long),
            ('dy',          ctypes.c_long),
            ('mouseData',   ctypes.c_ulong),
            ('dwFlags',     ctypes.c_ulong),
            ('time',        ctypes.c_ulong),
            ('dwExtraInfo', ctypes.c_size_t),
        ]

    class _INPUT_UNION(ctypes.Union):
        _fields_ = [('ki', KEYBDINPUT), ('mi', MOUSEINPUT)]

    class INPUT(ctypes.Structure):
        _fields_ = [('type', ctypes.c_ulong), ('_u', _INPUT_UNION)]

    user32 = ctypes.windll.user32
    sz = ctypes.sizeof(INPUT)

    for ch in text:
        for flag in (KEYEVENTF_UNICODE, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP):
            inp = INPUT()
            inp.type       = 1
            inp._u.ki.wVk   = 0
            inp._u.ki.wScan = ord(ch)
            inp._u.ki.dwFlags = flag
            user32.SendInput(1, ctypes.byref(inp), sz)
        time.sleep(0.05)


def _paste_via_keybd_event():
    """用 keybd_event 模拟物理按键 Ctrl+V，比 SendInput 更接近真实键盘。"""
    import ctypes
    u = ctypes.windll.user32
    VK_CONTROL, VK_V = 0x11, 0x56
    KEYDOWN, KEYUP = 0, 2
    u.keybd_event(VK_CONTROL, 0x1d, KEYDOWN, 0)
    u.keybd_event(VK_V,       0x2f, KEYDOWN, 0)
    time.sleep(0.05)
    u.keybd_event(VK_V,       0x2f, KEYUP,   0)
    u.keybd_event(VK_CONTROL, 0x1d, KEYUP,   0)


def _ocr_region(img, region) -> str:
    """截取区域并 OCR 识别文字。"""
    import io
    x1, y1, x2, y2 = region
    crop = img[y1:y2, x1:x2]
    buf = io.BytesIO()
    Image.fromarray(crop).save(buf, format='PNG')
    try:
        return _get_ocr().classification(buf.getvalue())
    except Exception:
        return ''


def _ocr_region_robust(img, region) -> str:
    """截取区域，尝试原图/正向OTSU/反向OTSU三种预处理，返回第一个非空结果。"""
    import io
    x1, y1, x2, y2 = region
    crop = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    candidates_img = [
        crop,
        cv2.cvtColor(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY     + cv2.THRESH_OTSU)[1], cv2.COLOR_GRAY2RGB),
        cv2.cvtColor(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1], cv2.COLOR_GRAY2RGB),
    ]
    ocr = _get_ocr()
    for ci in candidates_img:
        buf = io.BytesIO()
        Image.fromarray(ci).save(buf, format='PNG')
        try:
            text = ocr.classification(buf.getvalue())
            if text.strip():
                return text
        except Exception:
            pass
    return ''


def _name_matches(hero_name: str, ocr_text: str) -> bool:
    """名字字符匹配：≤4字需2+，5字需3+，6字需4+，>6字需4+（上限4）。"""
    if not ocr_text:
        return False
    min_match = min(4, max(2, len(hero_name) - 2))
    matched = sum(1 for ch in hero_name if ch in ocr_text)
    return matched >= min_match


def _is_search_popup_open(img=None) -> bool:
    """检测搜索弹窗是否打开：绿色搜索按钮像素占比足够高则认为弹窗打开。"""
    if img is None:
        img = capture()
    x1, y1, x2, y2 = _search_exec_region()
    patch = img[y1:y2, x1:x2].astype(float)
    r, g, b = patch[:, :, 0], patch[:, :, 1], patch[:, :, 2]
    green_mask = (g > 80) & (g > r * 1.3) & (g > b * 1.3)
    ratio = green_mask.mean()
    return ratio > 0.05


def search_and_pick_candidates(candidates: list, log_fn=None, unavailable: set = None) -> str:
    """
    搜索并选人。candidates = [(code, name, prob), ...]
    unavailable: 可选集合，搜索到禁用/已选的英雄会写入，下次调用可提前过滤。
    """
    import pyautogui
    _log = log_fn or (lambda m: None)

    if not candidates:
        return '', [], []

    _log('  [搜索] 点击搜索按钮')
    click_at(*_search_open(), delay=1.5)

    deadline = time.time() + 10
    while time.time() < deadline:
        img = capture()
        if _is_search_popup_open(img):
            _log('  [搜索] 弹窗已打开')
            break
        time.sleep(0.5)
    else:
        _log('  [搜索] 超时：弹窗未打开，放弃本轮')
        return '', [], []

    click_at(*_search_input(), delay=1.0)

    picked_code = ''
    seen_selected = []
    seen_banned   = []
    banned_kw   = _dlang('draft_hero_banned',   '禁用')
    selected_kw = _dlang('draft_hero_selected',  '已选')

    for code, name, prob in candidates:
        if not _is_search_popup_open():
            _log('  [搜索] 弹窗未打开，反复尝试重开...')
            opened = False
            for _attempt in range(30):
                click_at(*_search_open(), delay=1.5)
                if _is_search_popup_open():
                    _log(f'  [搜索] 弹窗已重新打开（第{_attempt+1}次）')
                    click_at(*_search_input(), delay=0.5)
                    opened = True
                    break
            if not opened:
                _log('  [搜索] 多次尝试仍无法打开弹窗，放弃本次选人')
                break

        _log(f'  [搜索] 搜索: {name!r}（{code}）')

        click_at(*_search_clear_btn(), delay=0.5)
        _type_unicode(name)
        time.sleep(3.0)

        img = capture()
        try:
            x1, y1, x2, y2 = _search_result_region()
            cv2.imwrite(os.path.join(_DEBUG_DIR, 'dbg_result.png'),
                        cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_RGB2BGR))
        except Exception:
            pass
        result_text = _ocr_region_robust(img, _search_result_region())
        _log(f'  [搜索] 结果OCR="{result_text}"')

        if _name_matches(name, result_text):
            click_at(*_first_result(), delay=2.0)

            img2 = capture()
            right_text = _ocr_region_robust(img2, _selected_hero_region())
            _log(f'  [搜索] 右侧状态OCR="{right_text}"')

            if banned_kw in right_text or selected_kw in right_text:
                _log(f'  [搜索] 英雄不可选（{right_text.strip()}），换下一个')
                if unavailable is not None:
                    unavailable.add(code)
                if selected_kw in right_text:
                    seen_selected.append(code)
                elif banned_kw in right_text:
                    seen_banned.append(code)
            else:
                click_at(*_confirm_btn(), delay=2.0)
                click_at(*_search_open(), delay=1.5)
                _log(f'  [搜索] 选人成功: {name}')
                picked_code = code
                break
        else:
            _log(f'  [搜索] 无结果，换下一个')
            if unavailable is not None:
                unavailable.add(code)

    if not picked_code:
        import pyautogui as _pag
        _pag.press('escape')
        time.sleep(0.5)

    return picked_code, seen_selected, seen_banned


def scan_existing_picks(img=None, log_fn=None) -> tuple:
    """扫描当前已有picks，返回(my_picks, enemy_picks)，过滤掉unknown"""
    _load_hero_names()
    if img is None:
        img = capture()
    my_picks, enemy_picks = [], []
    for i, region in enumerate(_my_slots()):
        code, score, gap = identify_slot_debug(img, region)
        if log_fn:
            log_fn(f"  我方槽{i+1}: {_code_to_name.get(code, code)}（score={score:.3f}）")
        if code != 'unknown' and code not in my_picks:
            my_picks.append(code)
    for i, region in enumerate(_opp_slots()):
        code, score, gap = identify_slot_debug(img, region)
        if log_fn:
            log_fn(f"  对手槽{i+1}: {_code_to_name.get(code, code)}（score={score:.3f}）")
        if code != 'unknown' and code not in enemy_picks and code not in my_picks:
            enemy_picks.append(code)
    return my_picks, enemy_picks


def run_draft(recommender, my_first: bool = True, banned: list = None,
              log_fn=print, stop_event=None,
              init_my_picks=None, init_enemy_picks=None,
              prepend_candidates=None) -> dict:
    """
    运行完整选秀流程（10轮）。
    每步直接 OCR 判断是我方/对手回合，不依赖预设序列。
    my_first 从第一步实际 OCR 结果推断，传入值仅作初始参考。
    """
    _load_hero_names()
    if banned is None:
        banned = []

    def _stopped():
        return stop_event is not None and stop_event.is_set()

    my_picks          = list(init_my_picks    or [])
    enemy_picks       = list(init_enemy_picks or [])
    enemy_pick_scores = [0.5] * len(enemy_picks)
    banned_scores     = [0.5] * len(banned)
    start_pos         = len(my_picks) + len(enemy_picks)

    unavailable_codes: set = set(banned)

    my_first_locked = (start_pos > 0)

    label = f"（先后手待OCR确认）" if not my_first_locked else f"（{'先手' if my_first else '后手'}，中途接入）"
    if start_pos > 0:
        label += f" 从第{start_pos + 1}步继续"
    log_fn(f"选秀开始{label}")

    cur_opp_slots = _opp_slots()

    for pos in range(start_pos, 10):
        if _stopped():
            break
        total_before = len(my_picks) + len(enemy_picks)

        whose, turn_img = _detect_current_turn(pos, stop_event=stop_event, log_fn=log_fn)
        if whose == 0:
            break

        if not my_first_locked:
            my_first = (whose == 1)
            my_first_locked = True
            log_fn(f'  {"先手" if my_first else "后手"}（OCR锁定，全程固定）')

            time.sleep(2.0)
            ban_frame = capture()
            raw_bans, raw_ban_scores = identify_ban_slots(ban_frame)
            detected = [(c, s) for c, s in zip(raw_bans, raw_ban_scores) if c != 'empty']
            if detected:
                banned       = [c for c, s in detected]
                banned_scores= [s for c, s in detected]
                unavailable_codes.update(banned)
            ban_names = [_code_to_name.get(c, c) for c in banned]
            log_fn(f'  禁用: {ban_names or "无"}（{len(banned)}/4）')

        if whose == 1:
            if len(my_picks) >= 5:
                log_fn(f'  我方已满5人，跳过选人')
                time.sleep(1.0)
                continue
            time.sleep(1.5)

            img_scan = capture()
            _my_set = set(my_picks)
            for slot_i in range(len(enemy_picks), 5):
                code_s, score_s, gap_s = identify_slot_debug(img_scan, cur_opp_slots[slot_i], exclude=_my_set)
                if code_s == 'unknown':
                    continue
                name_s = _code_to_name.get(code_s, code_s)
                if code_s in banned:
                    if code_s not in enemy_picks:
                        enemy_picks.append(code_s)
                        enemy_pick_scores.append(score_s)
                    ban_j = banned.index(code_s)
                    all_used_now = set(my_picks + enemy_picks + banned)
                    try:
                        repl_recs = recommender.recommend_preban(
                            my_banned=[], enemy_banned=[], all_banned=[], top_k=20)
                        repl_code = next(
                            (r['hero_code'] for r in repl_recs if r['hero_code'] not in all_used_now),
                            None)
                    except Exception:
                        repl_code = None
                    if repl_code is None:
                        repl_code = next(
                            (h for h in recommender.hero_list if h not in all_used_now), 'unknown')
                    old_ban = banned[ban_j]
                    log_fn(f"  扫描对手槽{slot_i+1}: {name_s}（score={score_s:.3f}）[在ban列表，判定ban识别有误]")
                    log_fn(f"  [修正] ban[{ban_j}] {_code_to_name.get(old_ban, old_ban)}"
                           f" → {_code_to_name.get(repl_code, repl_code)}（score {banned_scores[ban_j]:.3f}→0.0）")
                    banned[ban_j] = repl_code
                    banned_scores[ban_j] = 0.0
                    unavailable_codes.add(repl_code)
                elif code_s not in enemy_picks:
                    log_fn(f"  扫描对手槽{slot_i+1}: {name_s}（score={score_s:.3f}）")
                    enemy_picks.append(code_s)
                    enemy_pick_scores.append(score_s)

            phase = _get_phase(total_before)
            recs = recommender.recommend(
                my_picks=my_picks,
                enemy_picks=enemy_picks,
                banned=banned,
                phase=phase,
                my_first=my_first,
                top_k=5,
            )

            if recs:
                candidates = list(prepend_candidates or [])
                for rec in recs[:5]:
                    code = rec['hero_code']
                    if code in unavailable_codes:
                        name = _code_to_name.get(code, code)
                        log_fn(f"  → 跳过不可用: {name}（{code}）")
                        continue
                    name = _code_to_name.get(code, '')
                    if not name:
                        log_fn(f"  → 跳过无中文名: {code}")
                        continue
                    candidates.append((code, name, rec['probability']))
                    log_fn(f"  → 候选: {name}（{code}，概率 {rec['probability']:.3f}）")

                picked_code, seen_selected, seen_banned = search_and_pick_candidates(
                    candidates, log_fn=log_fn, unavailable=unavailable_codes)

                for conf_code in seen_selected:
                    if (conf_code not in my_picks and conf_code not in banned
                            and conf_code not in enemy_picks):
                        unconf = [(i, s) for i, s in enumerate(enemy_pick_scores) if s < 1.0]
                        if unconf:
                            min_idx = min(unconf, key=lambda x: x[1])[0]
                            old = enemy_picks[min_idx]
                            log_fn(f'  [修正] 对手持有 {_code_to_name.get(conf_code, conf_code)}'
                                   f'（已选择确认），替换低置信度 {_code_to_name.get(old, old)}'
                                   f'（{enemy_pick_scores[min_idx]:.3f}→1.0）')
                            enemy_picks[min_idx] = conf_code
                            enemy_pick_scores[min_idx] = 1.0
                            unavailable_codes.add(conf_code)

                for conf_code in seen_banned:
                    if conf_code not in banned:
                        unconf = [(i, s) for i, s in enumerate(banned_scores) if s < 1.0]
                        if unconf:
                            min_idx = min(unconf, key=lambda x: x[1])[0]
                            old = banned[min_idx]
                            log_fn(f'  [修正] {_code_to_name.get(conf_code, conf_code)}'
                                   f' 已被禁用（确认），替换低置信度禁用 {_code_to_name.get(old, old)}'
                                   f'（{banned_scores[min_idx]:.3f}→1.0）')
                            banned[min_idx] = conf_code
                            banned_scores[min_idx] = 1.0
                            unavailable_codes.add(conf_code)

                if picked_code:
                    time.sleep(1.0)
                    my_picks.append(picked_code)
                else:
                    placeholder = candidates[0][0] if candidates else (recs[0]['hero_code'] if recs else 'unknown')
                    log_fn(f"  ⚠ 搜索全部失败，以 {_code_to_name.get(placeholder, placeholder)} 占位")
                    my_picks.append(placeholder)
            else:
                all_used = set(my_picks + enemy_picks + banned)
                fallback = next((h for h in recommender.hero_list if h not in all_used), 'unknown')
                log_fn(f"  ⚠ 无推荐可用，以 {_code_to_name.get(fallback, fallback)} 兜底占位")
                my_picks.append(fallback)

            time.sleep(1.5)

        else:
            _wait_after_opponent_pick(stop_event=stop_event, log_fn=log_fn)
            if _stopped():
                break
            opp_slot_idx = len(enemy_picks)
            if opp_slot_idx >= len(cur_opp_slots):
                log_fn(f"  对手槽已满，跳过识别")
                continue
            code = 'unknown'
            pick_score = 0.0
            for attempt in range(1, 6):
                time.sleep(0.4)
                img_after = capture()
                code, score, gap = identify_slot_debug(img_after, cur_opp_slots[opp_slot_idx], exclude=set(my_picks))
                name = _code_to_name.get(code, code)
                log_fn(f"  识别第{attempt}次: {name}（{code}）分数={score:.3f} gap={gap:.3f}")
                if code != 'unknown':
                    pick_score = score
                    break

            all_used = set(my_picks + enemy_picks + banned)
            if code != 'unknown' and code in all_used:
                log_fn(f"  ⚠ {_code_to_name.get(code, code)} 已被占用，视为识别失败")
                code = 'unknown'
                pick_score = 0.0

            if code == 'unknown':
                pick_score = 0.0
                phase = _get_phase(total_before)
                fb_recs = recommender.recommend(
                    my_picks=my_picks, enemy_picks=enemy_picks,
                    banned=banned, phase=phase, my_first=my_first, top_k=1,
                )
                if fb_recs:
                    code = fb_recs[0]['hero_code']
                    log_fn(f"  ⚠ 识别失败，以模型推荐最高 {_code_to_name.get(code, code)} 占位")
                else:
                    all_used = set(my_picks + enemy_picks + banned)
                    fallback = next((h for h in recommender.hero_list if h not in all_used), 'unknown')
                    code = fallback
                    log_fn(f"  ⚠ 识别失败且无推荐，以 {_code_to_name.get(code, code)} 兜底占位")

            log_fn(f"  → 对手选了: {_code_to_name.get(code, code)}（{code}）")
            enemy_picks.append(code)
            enemy_pick_scores.append(pick_score)

        log_fn(f"  进度 我方:{[_code_to_name.get(c,c) for c in my_picks]} "
               f"对方:{[_code_to_name.get(c,c) for c in enemy_picks]}")

    img_final = capture()
    cur_my_slots  = _my_slots()
    cur_opp_slots = _opp_slots()
    while len(enemy_picks) < 5:
        slot_i = len(enemy_picks)
        code_new, score, _ = identify_slot_debug(img_final, cur_opp_slots[slot_i], exclude=set(my_picks))
        all_used = set(my_picks + enemy_picks + banned)
        if code_new != 'unknown' and code_new in all_used:
            log_fn(f'  [过渡] {_code_to_name.get(code_new, code_new)} 已被占用，视为识别失败')
            code_new = 'unknown'
            score = 0.0
        if code_new != 'unknown':
            log_fn(f'  [过渡] 对手槽{slot_i+1}: {_code_to_name.get(code_new, code_new)}（score={score:.3f}）')
        else:
            score = 0.0
            fb = recommender.recommend(my_picks=my_picks, enemy_picks=enemy_picks,
                                       banned=banned, phase='pick5', my_first=my_first, top_k=1)
            code_new = fb[0]['hero_code'] if fb else next(
                (h for h in recommender.hero_list if h not in all_used), 'unknown')
            log_fn(f'  [过渡] 对手槽{slot_i+1} 识别失败，占位: {_code_to_name.get(code_new, code_new)}')
        enemy_picks.append(code_new)
        enemy_pick_scores.append(score)
    while len(my_picks) < 5:
        slot_i = len(my_picks)
        code_new, score, _ = identify_slot_debug(img_final, cur_my_slots[slot_i])
        all_used = set(my_picks + enemy_picks + banned)
        if code_new != 'unknown' and code_new in all_used:
            log_fn(f'  [过渡] {_code_to_name.get(code_new, code_new)} 已被占用，视为识别失败')
            code_new = 'unknown'
            score = 0.0
        if code_new != 'unknown':
            log_fn(f'  [过渡] 我方槽{slot_i+1}: {_code_to_name.get(code_new, code_new)}（score={score:.3f}）')
        else:
            score = 0.0
            fb = recommender.recommend(my_picks=my_picks, enemy_picks=enemy_picks,
                                       banned=banned, phase='pick5', my_first=my_first, top_k=1)
            all_used = set(my_picks + enemy_picks + banned)
            code_new = fb[0]['hero_code'] if fb else next(
                (h for h in recommender.hero_list if h not in all_used), 'unknown')
            log_fn(f'  [过渡] 我方槽{slot_i+1} 识别失败，占位: {_code_to_name.get(code_new, code_new)}')
        my_picks.append(code_new)

    log_fn(f"选秀完成！我方: {[_code_to_name.get(c,c) for c in my_picks]} "
           f"对方: {[_code_to_name.get(c,c) for c in enemy_picks]}")
    _save_debug_slots()
    return {'my_picks': my_picks, 'enemy_picks': enemy_picks,
            'my_first': my_first, 'banned': banned}


def _save_debug_slots():
    """截图并用红框标出所有我方/对手槽位，存 dbg_slots.png 供人工核验坐标。"""
    try:
        img = capture()
        out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for i, (x1, y1, x2, y2) in enumerate(_my_slots()):
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(out, f'MY{i+1}', (x1+2, y1+14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        for i, (x1, y1, x2, y2) in enumerate(_opp_slots()):
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(out, f'OPP{i+1}', (x1+2, y1+14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        cv2.imwrite(os.path.join(_DEBUG_DIR, 'dbg_slots.png'), out)
    except Exception:
        pass


def _detect_current_turn(pos: int, timeout: int = 90, stop_event=None, log_fn=None) -> tuple:
    """OCR轮询直到检测到明确的回合标志。
    返回 (whose, img)：whose=1我的回合/2对手回合/0结束，img为检测到时的原始帧。"""
    _log = log_fn or (lambda m: None)
    deadline = time.time() + timeout
    last_log = 0
    while time.time() < deadline:
        if stop_event and stop_event.is_set():
            return 0, None
        img = capture()
        text = _read_turn_text(img)
        if time.time() - last_log >= 3:
            _log(f'  [第{pos+1}步] 等待回合... OCR="{text}"')
            last_log = time.time()
        if _is_my_turn_text(text):
            _log(f'  [第{pos+1}步] 我的回合（OCR="{text}"）')
            return 1, img
        if _is_opp_turn_text(text):
            _log(f'  [第{pos+1}步] 对手回合（OCR="{text}"）')
            return 2, img
        if _is_ban_kw(text) or _is_ready_kw(text):
            _log(f'  [第{pos+1}步] 检测到下一阶段（OCR="{text}"），退出')
            return 0, None
        time.sleep(0.5)
    _log(f'  [第{pos+1}步] 回合检测超时')
    return 0, None


def _wait_after_opponent_pick(timeout: int = 90, stop_event=None, log_fn=None):
    """已确认是对手回合，等对手选完（等到我的回合出现）。"""
    _log = log_fn or (lambda m: None)
    deadline = time.time() + timeout
    last_log = 0
    while time.time() < deadline:
        if stop_event and stop_event.is_set():
            return
        text = _read_turn_text()
        if time.time() - last_log >= 3:
            _log(f'  等待对手选完... OCR="{text}"')
            last_log = time.time()
        if _is_my_turn_text(text):
            _log(f'  对手选完，我方回合（OCR="{text}"）')
            return
        if _is_ban_kw(text) or _is_ready_kw(text):
            _log(f'  检测到下一阶段（OCR="{text}"），退出等待')
            return
        time.sleep(0.5)
    _log('  等待对手超时')


def _wait_my_turn(timeout: int = 60, stop_event=None, log_fn=None):
    _log = log_fn or (lambda m: None)
    deadline = time.time() + timeout
    last_log = 0
    while time.time() < deadline:
        if stop_event and stop_event.is_set():
            return
        img = capture()
        text = _read_turn_text(img)
        if time.time() - last_log >= 3:
            _log(f'  等待我方回合... OCR="{text}"')
            last_log = time.time()
        if _is_my_turn_text(text):
            _log(f'  我方回合开始（OCR="{text}"）')
            time.sleep(1.5)
            return
        preban_text = _ocr_region(img, _preban_region_draft())
        if _check_preban_text(preban_text):
            _log(f'  检测到preban阶段，退出等待')
            return
        time.sleep(0.5)
    _log('  等待我方回合超时')


def _wait_opponent_pick(timeout: int = 90, stop_event=None, log_fn=None):
    _log = log_fn or (lambda m: None)
    deadline = time.time() + timeout
    start = time.time()
    while time.time() - start < 8:
        if stop_event and stop_event.is_set():
            return
        text = _read_turn_text()
        if _is_opp_turn_text(text):
            _log(f'  检测到对手回合（OCR="{text}"）')
            break
        time.sleep(0.5)
    else:
        _log('  未检测到对手回合标志，继续等待')
    last_log = 0
    while time.time() < deadline:
        if stop_event and stop_event.is_set():
            return
        img = capture()
        text = _read_turn_text(img)
        if time.time() - last_log >= 3:
            _log(f'  等待对手选完... OCR="{text}"')
            last_log = time.time()
        if _is_my_turn_text(text):
            _log(f'  对手选完，我方回合（OCR="{text}"）')
            return
        if _is_ban_kw(text) or _is_ready_kw(text):
            _log(f'  检测到下一阶段（OCR="{text}"），退出等待')
            return
        time.sleep(0.5)
    _log('  等待对手超时')


def detect_my_first(img: np.ndarray = None) -> bool:
    if img is None:
        img = capture()
    text = _read_turn_text(img)
    if _is_opp_turn_text(text):
        return False
    _, score, _ = identify_slot_debug(img, _opp_slots()[0])
    if score > 0.35:
        return False
    return True
