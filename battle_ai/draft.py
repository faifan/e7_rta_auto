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

# ── 坐标 ─────────────────────────────────────────────────────
_SEARCH_OPEN   = (1820, 173)
_SEARCH_INPUT  = (1073, 269)
_SEARCH_BTN    = (1437, 264)
_FIRST_RESULT  = (1010, 443)
_CONFIRM_BTN   = (1274, 1029)

# 选秀后禁用 & 战斗开始按钮（来自 说明_utf8.txt 12.png，取矩形中心）
_POST_BAN_BTN  = (1282, 1027)

# ── 对手回合检测区域 ──────────────────────────────────────────
_OPP_TURN_REGION = (679, 65, 823, 115)

# ── 选秀后禁对手英雄 & 准备战斗：顶部 banner 检测区域 ────────
_BANNER_REGION = (550, 3, 950, 55)

# ── 英雄名称映射 ──────────────────────────────────────────────
_E7_JSON = os.path.join(_ROOT, 'e7.json')
_code_to_name: dict = {}


def _load_hero_names():
    if not _code_to_name and os.path.exists(_E7_JSON):
        with open(_E7_JSON, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for hero in data:
            _code_to_name[hero['code']] = hero['name']


# ── pick 槽坐标 ───────────────────────────────────────────────
MY_SLOTS = [
    ( 37, 216, 461, 343),
    ( 34, 367, 361, 491),
    ( 37, 520, 318, 638),
    ( 35, 668, 314, 788),
    ( 41, 819, 359, 937),
]
OPP_SLOTS = [
    (1033, 219, 1456, 337),
    (1133, 367, 1456, 487),
    (1175, 518, 1453, 638),
    (1177, 668, 1463, 790),
    (1136, 818, 1462, 937),
]

# 对手 pick 槽中心坐标（用于选秀后禁用点击）
_OPP_SLOT_CENTERS = [
    ((x1 + x2) // 2, (y1 + y2) // 2)
    for x1, y1, x2, y2 in OPP_SLOTS
]

_TMPL_BASE = os.path.join(_ROOT, 'templates')

# ── 选秀卡识别 ────────────────────────────────────────────────
_DRAFT_CARDS_DIR = os.path.join(_ROOT, 'templates', 'draft_cards')
_TMPL_SIZE       = (96, 56)
_NCC_THRESHOLD   = 0.5
_NCC_GAP         = 0.02

_draft_templates: dict = {}   # code -> list[ndarray]


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

_MY_BAN_SLOTS = [
    (565, 1010, 646, 1096),
    (646, 1010, 741, 1096),
]
_OPP_BAN_SLOTS = [
    (753, 1006, 842, 1095),
    (842, 1006, 937, 1095),
]

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
    # 存完整截图供坐标核验
    try:
        cv2.imwrite(os.path.join(_DEBUG_DIR, 'dbg_ban_full.png'),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    except Exception:
        pass
    result = []
    scores = []
    debug_crops = []
    labels = ['我方ban1', '我方ban2', '对手ban1', '对手ban2']
    all_slots = list(_MY_BAN_SLOTS) + list(_OPP_BAN_SLOTS)
    for i, (x1, y1, x2, y2) in enumerate(all_slots):
        crop = img[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        code, ban_score = _identify_ban_slot(gray)
        result.append(code)
        scores.append(ban_score)
        # 存 debug 图：放大3倍方便肉眼查看
        enlarged = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
                              (crop.shape[1]*3, crop.shape[0]*3),
                              interpolation=cv2.INTER_NEAREST)
        debug_crops.append((labels[i], code, enlarged))
    # 单独保存4张 + 拼一张横排
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
# 坐标由用户在 tp/10-12.png 标记
_LABEL_REGION = (632, 53, 862, 127)
_LABEL_SIZE   = (115, 37)

_MY_TURN_TMPL  = None
_OPP_TURN_TMPL = None
_POSTBAN_TMPL  = None


def _load_label_tmpls():
    import os
    global _MY_TURN_TMPL, _OPP_TURN_TMPL, _POSTBAN_TMPL
    x1, y1, x2, y2 = _LABEL_REGION
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
    x1, y1, x2, y2 = _LABEL_REGION
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
    """OCR 识别顶部回合文字，返回识别结果字符串（可能含 '你的回合' 或 '对手回合'）"""
    import io
    if img is None:
        img = capture()
    x1, y1, x2, y2 = _LABEL_REGION
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
    text = _read_turn_text(img)
    return '你的' in text


def is_opponent_turn_ocr(img: np.ndarray = None) -> bool:
    text = _read_turn_text(img)
    return '对手' in text or '对方' in text


def is_opponent_turn(img: np.ndarray = None) -> bool:
    _ensure_label_tmpls()
    if img is None:
        img = capture()
    return _opp_turn_score(img) >= 0.55


# ── 选秀后禁用 & 战斗开始检测 ────────────────────────────────
_POST_BAN_TMPL    = None
_BATTLE_RDY_TMPL  = None
_BANNER_SIZE      = (100, 13)


def _load_banner_tmpls():
    global _POST_BAN_TMPL, _BATTLE_RDY_TMPL
    x1, y1, x2, y2 = _BANNER_REGION
    import os
    for attr, path in [('_POST_BAN_TMPL',   os.path.join(_TMPL_BASE, 'phase', 'postban.png')),
                       ('_BATTLE_RDY_TMPL', os.path.join(_TMPL_BASE, 'phase', 'battle_ready.png'))]:
        img = np.array(Image.open(path).convert('L'))
        crop = img[y1:y2, x1:x2]
        tmpl = cv2.resize(crop, _BANNER_SIZE).astype(np.float32)
        globals()[attr] = tmpl


def _banner_ncc(img: np.ndarray, tmpl: np.ndarray) -> float:
    x1, y1, x2, y2 = _BANNER_REGION
    crop = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    query = cv2.resize(gray, _BANNER_SIZE).astype(np.float32)
    return _ncc(query, tmpl)


def is_in_draft(img: np.ndarray = None) -> bool:
    """检测是否在选秀阶段：OCR 读到"你的回合"或"对手回合" """
    if img is None:
        img = capture()
    text = _read_turn_text(img)
    return '你的' in text or '对手' in text or '对方' in text


def is_in_post_draft_ban(img: np.ndarray = None) -> bool:
    """检测顶部文字是否为"选择禁用英雄"（选秀后禁用阶段）"""
    if img is None:
        img = capture()
    text = _read_turn_text(img)
    return '禁用' in text and '你的' not in text and '对手' not in text and '对方' not in text


_PREBAN_REGION = (181, 137, 505, 203)   # 与preban.py保持一致

def is_battle_ready(img: np.ndarray = None) -> bool:
    """检测是否在准备战斗阶段：顶部"准备战斗" 且 preban区域无禁用文字（排除preban阶段）。"""
    if img is None:
        img = capture()
    if '准备' not in _read_turn_text(img):
        return False
    # preban阶段label也显示"准备战斗"，用preban区域排除
    preban_text = _ocr_region(img, _PREBAN_REGION)
    return '禁用' not in preban_text and '预先' not in preban_text


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

    cx, cy = _OPP_SLOT_CENTERS[ban_idx]
    click_at(cx, cy, delay=1.5)
    click_at(*_POST_BAN_BTN, delay=1.5)


def click_battle_start():
    """点击战斗开始按钮（与postban完成选择同坐标）"""
    click_at(*_POST_BAN_BTN, delay=1.5)


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

    # dwExtraInfo 必须用 c_size_t（指针大小），64-bit 下是 8 字节
    # 否则 sizeof(INPUT) 算错，SendInput 静默失败返回 0
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
            inp.type       = 1   # INPUT_KEYBOARD
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


# 搜索结果行区域（图70）
_SEARCH_RESULT_REGION  = (855, 402, 1264, 469)
# 点击结果后右侧状态文字区域（图80禁用/图81已选）
_SELECTED_HERO_REGION  = (1541, 389, 1699, 458)
# 右上角搜索/取消切换按钮区域（图73）：显示"搜索"=未打开，显示"取消"=已打开
_SEARCH_STATE_REGION  = (1747, 142, 1897, 197)
# 弹窗内绿色搜索执行按钮区域（图72）
_SEARCH_EXEC_REGION   = (1341, 227, 1538, 300)


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
        crop,                                                                 # 原图
        cv2.cvtColor(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY     + cv2.THRESH_OTSU)[1], cv2.COLOR_GRAY2RGB),  # 正向OTSU
        cv2.cvtColor(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1], cv2.COLOR_GRAY2RGB),  # 反向OTSU
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
    """检测搜索弹窗是否打开：72区域绿色像素占比足够高则认为绿色搜索按钮存在。"""
    if img is None:
        img = capture()
    x1, y1, x2, y2 = _SEARCH_EXEC_REGION
    patch = img[y1:y2, x1:x2].astype(float)
    r, g, b = patch[:, :, 0], patch[:, :, 1], patch[:, :, 2]
    # 绿色像素：G明显高于R和B
    green_mask = (g > 80) & (g > r * 1.3) & (g > b * 1.3)
    ratio = green_mask.mean()
    return ratio > 0.05   # 超过5%像素是绿色即认为按钮存在


def search_and_pick_candidates(candidates: list, log_fn=None, unavailable: set = None) -> str:
    """
    搜索并选人。candidates = [(code, name, prob), ...]
    unavailable: 可选集合，搜索到禁用/已选的英雄会写入，下次调用可提前过滤。
    """
    import pyautogui
    _log = log_fn or (lambda m: None)

    if not candidates:
        return ''

    # 点搜索按钮，等弹窗打开（OCR反色确认）
    _log('  [搜索] 点击搜索按钮')
    click_at(*_SEARCH_OPEN, delay=1.5)

    deadline = time.time() + 10
    while time.time() < deadline:
        img = capture()
        if _is_search_popup_open(img):
            _log('  [搜索] 弹窗已打开')
            break
        time.sleep(0.5)
    else:
        _log('  [搜索] 超时：弹窗未打开，放弃本轮')
        return ''

    # 弹窗打开后，点一次输入框让光标进去
    click_at(*_SEARCH_INPUT, delay=1.0)

    picked_code = ''
    seen_selected = []   # 搜到"已选择"的英雄code（可能是对手的）
    seen_banned   = []   # 搜到"禁用英雄"的英雄code（可能识别有误的禁用）
    for code, name, prob in candidates:
        # 每次搜索前确认绿色按钮还在，没有就尝试重新打开一次
        if not _is_search_popup_open():
            _log('  [搜索] 弹窗未打开，反复尝试重开...')
            opened = False
            for _attempt in range(30):
                click_at(*_SEARCH_OPEN, delay=1.5)
                if _is_search_popup_open():
                    _log(f'  [搜索] 弹窗已重新打开（第{_attempt+1}次）')
                    click_at(*_SEARCH_INPUT, delay=0.5)
                    opened = True
                    break
            if not opened:
                _log('  [搜索] 多次尝试仍无法打开弹窗，放弃本次选人')
                break

        _log(f'  [搜索] 搜索: {name!r}（{code}）')

        # 点X清空输入框，再输入
        click_at(1284, 264, delay=0.5)
        _type_unicode(name)
        time.sleep(3.0)   # 等输入后自动出结果再识别

        # OCR 读搜索结果行（鲁棒预处理）
        img = capture()
        # 保存调试截图
        try:
            x1, y1, x2, y2 = _SEARCH_RESULT_REGION
            cv2.imwrite(os.path.join(_DEBUG_DIR, 'dbg_result.png'),
                        cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_RGB2BGR))
        except Exception:
            pass
        result_text = _ocr_region_robust(img, _SEARCH_RESULT_REGION)
        _log(f'  [搜索] 结果OCR="{result_text}"')

        if _name_matches(name, result_text):
            # 点结果行，等右侧英雄卡出现
            click_at(*_FIRST_RESULT, delay=2.0)

            # OCR右侧英雄卡，检测是否可选
            img2 = capture()
            right_text = _ocr_region_robust(img2, _SELECTED_HERO_REGION)
            _log(f'  [搜索] 右侧状态OCR="{right_text}"')

            if '禁用' in right_text or '已选' in right_text:
                _log(f'  [搜索] 英雄不可选（{right_text.strip()}），换下一个')
                if unavailable is not None:
                    unavailable.add(code)
                if '已选' in right_text:
                    seen_selected.append(code)
                elif '禁用' in right_text:
                    seen_banned.append(code)
            else:
                # 可以选，点确认 → 点取消搜索复位状态
                click_at(*_CONFIRM_BTN, delay=2.0)
                click_at(*_SEARCH_OPEN, delay=1.5)
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
    for i, region in enumerate(MY_SLOTS):
        code, score, gap = identify_slot_debug(img, region)
        if log_fn:
            log_fn(f"  我方槽{i+1}: {_code_to_name.get(code, code)}（score={score:.3f}）")
        if code != 'unknown' and code not in my_picks:
            my_picks.append(code)
    for i, region in enumerate(OPP_SLOTS):
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
    enemy_pick_scores = [0.5] * len(enemy_picks)   # 预填充时给保守默认值
    banned_scores     = [0.5] * len(banned)         # 预填充时给保守默认值
    start_pos         = len(my_picks) + len(enemy_picks)

    # 跨轮不可用英雄（禁用 or 已选），避免重复搜索
    unavailable_codes: set = set(banned)

    # 先后手只在第一次OCR确认时锁定，中途接入则信任传入值
    my_first_locked = (start_pos > 0)

    label = f"（先后手待OCR确认）" if not my_first_locked else f"（{'先手' if my_first else '后手'}，中途接入）"
    if start_pos > 0:
        label += f" 从第{start_pos + 1}步继续"
    log_fn(f"选秀开始{label}")

    for pos in range(start_pos, 10):
        if _stopped():
            break
        total_before = len(my_picks) + len(enemy_picks)

        # 直接看屏幕判断是谁的回合，返回同一帧供后续使用
        whose, turn_img = _detect_current_turn(pos, stop_event=stop_event, log_fn=log_fn)
        if whose == 0:
            break

        # 第一次OCR成功：锁定先后手 + 等ban图标动画结束后截图识别
        if not my_first_locked:
            my_first = (whose == 1)
            my_first_locked = True
            log_fn(f'  {"先手" if my_first else "后手"}（OCR锁定，全程固定）')

            time.sleep(2.0)  # 等禁用图标动画结束
            ban_frame = capture()
            raw_bans, raw_ban_scores = identify_ban_slots(ban_frame)
            detected = [(c, s) for c, s in zip(raw_bans, raw_ban_scores) if c != 'empty']
            if detected:
                banned       = [c for c, s in detected]
                banned_scores= [s for c, s in detected]
                unavailable_codes.update(banned)
            ban_names = [_code_to_name.get(c, c) for c in banned]
            log_fn(f'  禁用: {ban_names or "无"}（{len(banned)}/4）')

        if whose == 1:   # 我的回合
            time.sleep(1.5)  # 等界面稳定

            # 扫描对手当前槽位
            img_scan = capture()
            _my_set = set(my_picks)
            for slot_i in range(len(enemy_picks), 5):
                # exclude=my_picks：我方选的不可能在对手槽，自动跳过取次高分
                code_s, score_s, gap_s = identify_slot_debug(img_scan, OPP_SLOTS[slot_i], exclude=_my_set)
                if code_s == 'unknown':
                    continue
                name_s = _code_to_name.get(code_s, code_s)
                if code_s in banned:
                    # 被禁英雄不可能出现在选人槽 → ban识别有误，选秀卡识别更可信
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

                # 已选择 → 确认为对手英雄，修正最低置信度条目
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

                # 禁用英雄 → 确认被禁，修正最低置信度禁用条目
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
                    # 搜索全部失败，仍须占位保持序列长度正确
                    placeholder = candidates[0][0] if candidates else (recs[0]['hero_code'] if recs else 'unknown')
                    log_fn(f"  ⚠ 搜索全部失败，以 {_code_to_name.get(placeholder, placeholder)} 占位")
                    my_picks.append(placeholder)
            else:
                all_used = set(my_picks + enemy_picks + banned)
                fallback = next((h for h in recommender.hero_list if h not in all_used), 'unknown')
                log_fn(f"  ⚠ 无推荐可用，以 {_code_to_name.get(fallback, fallback)} 兜底占位")
                my_picks.append(fallback)

            time.sleep(1.5)

        else:   # 对手回合
            _wait_after_opponent_pick(stop_event=stop_event, log_fn=log_fn)
            if _stopped():
                break
            opp_slot_idx = len(enemy_picks)
            if opp_slot_idx >= len(OPP_SLOTS):
                log_fn(f"  对手槽已满，跳过识别")
                continue
            code = 'unknown'
            pick_score = 0.0
            for attempt in range(1, 6):
                time.sleep(0.4)
                img_after = capture()
                code, score, gap = identify_slot_debug(img_after, OPP_SLOTS[opp_slot_idx], exclude=set(my_picks))
                name = _code_to_name.get(code, code)
                log_fn(f"  识别第{attempt}次: {name}（{code}）分数={score:.3f} gap={gap:.3f}")
                if code != 'unknown':
                    pick_score = score
                    break

            # 识别到的英雄已被占用（重复），视为识别失败走 fallback
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
                    # 推荐也拿不到，用已知英雄列表里第一个未被占用的代码占位
                    all_used = set(my_picks + enemy_picks + banned)
                    fallback = next((h for h in recommender.hero_list if h not in all_used), 'unknown')
                    code = fallback
                    log_fn(f"  ⚠ 识别失败且无推荐，以 {_code_to_name.get(code, code)} 兜底占位")

            log_fn(f"  → 对手选了: {_code_to_name.get(code, code)}（{code}）")
            enemy_picks.append(code)
            enemy_pick_scores.append(pick_score)

        log_fn(f"  进度 我方:{[_code_to_name.get(c,c) for c in my_picks]} "
               f"对方:{[_code_to_name.get(c,c) for c in enemy_picks]}")

    # 过渡到禁人阶段：只扫最后一个可能遗漏的槽位
    # 先手→对手最后选→检查OPP_SLOTS[4]；后手→我方最后选→检查MY_SLOTS[4]
    img_final = capture()
    if my_first and len(enemy_picks) < 5:
        slot_i = len(enemy_picks)
        code_new, score, _ = identify_slot_debug(img_final, OPP_SLOTS[slot_i], exclude=set(my_picks))
        trans_score = score
        all_used = set(my_picks + enemy_picks + banned)
        if code_new != 'unknown' and code_new in all_used:
            log_fn(f'  [过渡] {_code_to_name.get(code_new, code_new)} 已被占用，视为识别失败')
            code_new = 'unknown'
            trans_score = 0.0
        if code_new != 'unknown':
            log_fn(f'  [过渡] 对手槽{slot_i+1}: {_code_to_name.get(code_new, code_new)}（score={score:.3f}）')
        else:
            trans_score = 0.0
            fb = recommender.recommend(my_picks=my_picks, enemy_picks=enemy_picks,
                                       banned=banned, phase='pick5', my_first=my_first, top_k=1)
            code_new = fb[0]['hero_code'] if fb else next(
                (h for h in recommender.hero_list if h not in all_used), 'unknown')
            log_fn(f'  [过渡] 对手槽{slot_i+1} 识别失败，占位: {_code_to_name.get(code_new, code_new)}')
        enemy_picks.append(code_new)
        enemy_pick_scores.append(trans_score)
    elif not my_first and len(my_picks) < 5:
        slot_i = len(my_picks)
        code_new, score, _ = identify_slot_debug(img_final, MY_SLOTS[slot_i])
        all_used = set(my_picks + enemy_picks + banned)
        if code_new != 'unknown' and code_new in all_used:
            log_fn(f'  [过渡] {_code_to_name.get(code_new, code_new)} 已被占用，视为识别失败')
            code_new = 'unknown'
        if code_new != 'unknown':
            log_fn(f'  [过渡] 我方槽{slot_i+1}: {_code_to_name.get(code_new, code_new)}（score={score:.3f}）')
        else:
            fb = recommender.recommend(my_picks=my_picks, enemy_picks=enemy_picks,
                                       banned=banned, phase='pick5', my_first=my_first, top_k=1)
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
        for i, (x1, y1, x2, y2) in enumerate(MY_SLOTS):
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(out, f'MY{i+1}', (x1+2, y1+14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        for i, (x1, y1, x2, y2) in enumerate(OPP_SLOTS):
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
        if '你的' in text:
            _log(f'  [第{pos+1}步] 我的回合（OCR="{text}"）')
            return 1, img
        if '对手' in text or '对方' in text:
            _log(f'  [第{pos+1}步] 对手回合（OCR="{text}"）')
            return 2, img
        if '禁用' in text or '准备' in text:
            _log(f'  [第{pos+1}步] 检测到下一阶段（OCR="{text}"），退出')
            return 0, None
        time.sleep(0.5)
    _log(f'  [第{pos+1}步] 回合检测超时')
    return 0, None


def _wait_after_opponent_pick(timeout: int = 90, stop_event=None, log_fn=None):
    """已确认是对手回合，等对手选完（等到"你的回合"出现）。"""
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
        if '你的' in text:
            _log(f'  对手选完，我方回合（OCR="{text}"）')
            return
        if '禁用' in text or '准备' in text:
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
        if '你的' in text:
            _log(f'  我方回合开始（OCR="{text}"）')
            time.sleep(1.5)
            return
        preban_text = _ocr_region(img, _PREBAN_REGION)
        if '禁用' in preban_text or '预先' in preban_text:
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
        if '对手' in text or '对方' in text:
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
        if '你的' in text:
            _log(f'  对手选完，我方回合（OCR="{text}"）')
            return
        if '禁用' in text or '准备' in text:
            _log(f'  检测到下一阶段（OCR="{text}"），退出等待')
            return
        time.sleep(0.5)
    _log('  等待对手超时')


def detect_my_first(img: np.ndarray = None) -> bool:
    if img is None:
        img = capture()
    text = _read_turn_text(img)
    if '对手' in text or '对方' in text:
        return False
    # "你的回合"时，再确认对手第一槽是否已有英雄
    # 有的话说明对手先选了一个，我是后手（NCC>0.35即认为槽位非空）
    _, score, _ = identify_slot_debug(img, OPP_SLOTS[0])
    if score > 0.35:
        return False
    return True
