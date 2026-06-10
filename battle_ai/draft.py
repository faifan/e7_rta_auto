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
from battle_ai.perception import capture, is_battle_over, detect_opening_rule
from battle_ai.hero_config import is_unpracticed, is_priority, get_force_picks, get_excluded_by_picks, get_fallback_picks, get_counter_picks

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


# ── 坐标访问（从 cfg JSON 读取）──────────────────────────────
def _search_open():           return tuple(_dcfg()['search_open'])
def _search_input():          return tuple(_dcfg()['search_input'])
def _search_btn():            return tuple(_dcfg()['search_btn'])
def _first_result():          return tuple(_dcfg()['first_result'])
def _confirm_btn():           return tuple(_dcfg()['confirm_btn'])
def _post_ban_btn():          return tuple(_dcfg()['post_ban_btn'])
def _banner_region():         return tuple(_dcfg()['banner_region'])
def _label_region():          return tuple(_dcfg()['label_region'])
def _search_result_region():  return tuple(_dcfg()['search_result_region'])
def _selected_hero_region():  return tuple(_dcfg()['selected_hero_region'])
def _search_exec_region():    return tuple(_dcfg()['search_exec_region'])
def _search_clear_btn():      return tuple(_dcfg()['search_clear_btn'])
def _preban_region_draft():   return tuple(_dcfg()['preban_region'])
def _my_slots():   return [tuple(v) for v in _dcfg()['my_slots']]
def _opp_slots():  return [tuple(v) for v in _dcfg()['opp_slots']]
def _my_ban_slots():  return [tuple(v) for v in _dcfg()['my_ban_slots']]
def _opp_ban_slots(): return [tuple(v) for v in _dcfg()['opp_ban_slots']]
def _opp_slot_centers():
    return [((x1 + x2) // 2, (y1 + y2) // 2) for x1, y1, x2, y2 in _opp_slots()]
def _battle_slot_centers(): return [tuple(v) for v in _dcfg()['battle_slot_centers']]
def _battle_swap_regions():  return [tuple(v) for v in _dcfg()['battle_swap_regions']]
def _battle_swap_centers():  return [(int((r[0]+r[2])/2), int((r[1]+r[3])/2)) for r in _battle_swap_regions()]
def _battle_deselect():      return tuple(_dcfg()['battle_deselect'])
def _battle_yazuga_detect(): return tuple(_dcfg()['battle_yazuga_detect'])


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
_NCC_THRESHOLD   = 0.35

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


# ── 合成模板识别（新方法：合成立绘 + 职业/属性过滤 + NCC）─────────────────
# 覆盖 380+ 英雄，自动适配 1920×1080 / 1280×720 分辨率
_NEW_THRESHOLD   = 0.45
_NEW_CROP_LEFT   = 0.33     # 跳过左侧 UI 区（Lv.60/星级/图标），只比较右侧立绘
_REF_SLOT_H      = 125      # 1080p 基准槽高，用于计算分辨率缩放系数

_CARD_ASSETS_NEW = os.path.join(_ROOT, 'templates', 'card')
_ZYSX_DIR_NEW    = os.path.join(_CARD_ASSETS_NEW, 'zysx')
_L_CODE_RE_NEW   = _re.compile(r'^(.+?)(?:_s\d+)?_l(?:_\w+)?$')

_ATTR_BG_NEW = {
    'fire':  (150, 44,  38),
    'ice':   (139, 34,  32),
    'dark':  (151, 41,  41),
    'light': (152, 51,  39),
    'wind':  (149, 70,  50),
}
_ATTR_ICON_FILES = {
    'fire':  'cm_icon_profire.png',
    'ice':   'cm_icon_proice.png',
    'dark':  'cm_icon_promdark.png',
    'light': 'cm_icon_promlight.png',
    'wind':  'cm_icon_prowind.png',
}
_JOB_ICON_FILES = {
    'assassin': 'cm_icon_role_assassin.png',
    'knight':   'cm_icon_role_knight.png',
    'mage':     'cm_icon_role_mage.png',
    'manauser': 'cm_icon_role_manauser.png',
    'ranger':   'cm_icon_role_ranger.png',
    'warrior':  'cm_icon_role_warrior.png',
}

_E7_ATTRS: dict = {}          # code → (job_cd, attribute_cd)
_new_tmpls_cache: dict = {}   # (slot_w, slot_h) → {code: [gray ndarray]}
_JOB_ICON_RGB_NEW:  dict = {} # job_cd  → uint8 (44, 44, 3) @1080p 基准尺寸
_ATTR_ICON_RGB_NEW: dict = {} # attr_cd → uint8 (32, 32, 3) @1080p 基准尺寸


def _load_e7_attrs():
    if _E7_ATTRS:
        return
    try:
        with open(_E7_JSON, 'r', encoding='utf-8') as f:
            for h in json.load(f):
                _E7_ATTRS[h['code']] = (h.get('job_cd'), h.get('attribute_cd'))
    except Exception:
        pass


def _alpha_paste_new(canvas, icon_rgba, x, y):
    ih, iw = icon_rgba.shape[:2]
    ch, cw = canvas.shape[:2]
    x2, y2 = min(x + iw, cw), min(y + ih, ch)
    iw2, ih2 = x2 - x, y2 - y
    if iw2 <= 0 or ih2 <= 0:
        return
    src   = icon_rgba[:ih2, :iw2]
    alpha = src[:, :, 3:].astype(np.float32) / 255.0
    bg    = canvas[y:y2, x:x2].astype(np.float32)
    fg    = src[:, :, :3].astype(np.float32)
    canvas[y:y2, x:x2] = (bg * (1 - alpha) + fg * alpha).astype(np.uint8)


def _build_card_new(tp_path, attribute_cd, slot_w=426, slot_h=_REF_SLOT_H):
    bg     = _ATTR_BG_NEW.get(attribute_cd, (145, 43, 37))
    canvas = np.zeros((slot_h, slot_w, 3), dtype=np.uint8)
    canvas[:] = bg
    tpc    = Image.open(tp_path).convert('RGBA')
    arr    = np.array(tpc)
    scale  = slot_h / arr.shape[0]
    char_w = max(1, int(arr.shape[1] * scale))
    rgb_s  = cv2.resize(arr[:, :, :3], (char_w, slot_h), interpolation=cv2.INTER_LINEAR)
    alp_s  = cv2.resize(arr[:, :, 3],  (char_w, slot_h), interpolation=cv2.INTER_LINEAR)
    if char_w <= slot_w:
        x0   = slot_w - char_w
        mask = alp_s[:, :, np.newaxis].astype(np.float32) / 255.0
        reg  = canvas[:, x0:x0 + char_w].astype(np.float32)
        canvas[:, x0:x0 + char_w] = (reg * (1 - mask) + rgb_s.astype(np.float32) * mask).astype(np.uint8)
    else:
        crop_x   = char_w - slot_w
        rgb_crop = rgb_s[:, crop_x:]
        alp_crop = alp_s[:, crop_x:]
        mask     = alp_crop[:, :, np.newaxis].astype(np.float32) / 255.0
        reg      = canvas.astype(np.float32)
        canvas[:] = (reg * (1 - mask) + rgb_crop.astype(np.float32) * mask).astype(np.uint8)
    return canvas


def _get_new_templates(slot_w: int, slot_h: int) -> dict:
    dim = (slot_w, slot_h)
    if dim in _new_tmpls_cache:
        return _new_tmpls_cache[dim]
    _load_e7_attrs()
    tmpls = {}
    if not os.path.exists(_CARD_ASSETS_NEW):
        _new_tmpls_cache[dim] = tmpls
        return tmpls
    for fname in os.listdir(_CARD_ASSETS_NEW):
        if not fname.endswith('.png'):
            continue
        m = _L_CODE_RE_NEW.match(fname[:-4])
        if not m:
            continue
        code = m.group(1)
        if code not in _E7_ATTRS:
            continue
        _, attr_cd = _E7_ATTRS[code]
        try:
            rgb  = _build_card_new(
                os.path.join(_CARD_ASSETS_NEW, fname),
                attr_cd, slot_w=426, slot_h=slot_h)
            rgb  = rgb[:, (426 - slot_w):]          # 右对齐裁剪到槽宽
            lx   = int(slot_w * _NEW_CROP_LEFT)
            gray = cv2.cvtColor(rgb[:, lx:], cv2.COLOR_RGB2GRAY)
            tmpls.setdefault(code, []).append(
                cv2.resize(gray, _TMPL_SIZE).astype(np.float32))
        except Exception:
            pass
    _new_tmpls_cache[dim] = tmpls
    return tmpls


def _load_icon_rgb_new():
    if _JOB_ICON_RGB_NEW and _ATTR_ICON_RGB_NEW:
        return
    _bg = np.array([147, 42, 38], dtype=np.uint8)
    if not os.path.exists(_ZYSX_DIR_NEW):
        return
    for jcd, fname in _JOB_ICON_FILES.items():
        p = os.path.join(_ZYSX_DIR_NEW, fname)
        if os.path.exists(p):
            icon = np.array(Image.open(p).convert('RGBA').resize((44, 44)))
            mini = np.full((44, 44, 3), _bg, dtype=np.uint8)
            _alpha_paste_new(mini, icon, 0, 0)
            _JOB_ICON_RGB_NEW[jcd] = mini
    for acd, fname in _ATTR_ICON_FILES.items():
        p = os.path.join(_ZYSX_DIR_NEW, fname)
        if os.path.exists(p):
            icon = np.array(Image.open(p).convert('RGBA').resize((32, 32)))
            mini = np.full((32, 32, 3), _bg, dtype=np.uint8)
            _alpha_paste_new(mini, icon, 0, 0)
            _ATTR_ICON_RGB_NEW[acd] = mini


def _detect_job_attr_new(img_rgb, region):
    _load_icon_rgb_new()
    x1, y1, x2, y2 = region
    sh = y2 - y1
    sc = sh / _REF_SLOT_H   # 分辨率缩放系数（1080p=1.0，720p≈0.67）

    # ── 职业：3通道 NCC，图标按 sc 缩放 ──
    job_cd = None
    job_sz = max(4, int(44 * sc))
    job_sw = min(max(job_sz + 1, int(120 * sc)), x2 - x1)
    if job_sw >= job_sz and sh >= job_sz and _JOB_ICON_RGB_NEW:
        area = img_rgb[y1:y2, x1:x1 + job_sw].astype(np.float32)
        jsc  = {}
        for jcd_k, base in _JOB_ICON_RGB_NEW.items():
            tmpl = cv2.resize(base, (job_sz, job_sz)).astype(np.float32)
            jsc[jcd_k] = float(np.mean([
                cv2.matchTemplate(area[:, :, c], tmpl[:, :, c],
                                  cv2.TM_CCOEFF_NORMED).max()
                for c in range(3)]))
        best_j = max(jsc, key=jsc.get)
        if jsc[best_j] >= 0.40:
            job_cd = best_j

    # ── 属性：像素特征色计数，搜索区域按 sc 缩放 ──
    attr_cd = None
    ax1 = x1 + int(48 * sc)
    ax2 = min(ax1 + int(52 * sc), x2)
    ay2 = min(y1 + int(45 * sc), y2)
    if ax2 - ax1 >= 5 and ay2 > y1:
        af   = img_rgb[y1:ay2, ax1:ax2].astype(np.float32)
        R, G, B = af[:, :, 0], af[:, :, 1], af[:, :, 2]
        sc_a = {
            'fire':  int(np.sum((R > 175) & (G > 50)  & (R > 2.2*G) & (R > 3.0*B))),
            'ice':   int(np.sum((B > 170) & (B > 2.0*R) & (B > G))),
            'dark':  int(np.sum((R > 60)  & (B > 60)   & (G < 50))),
            'light': int(np.sum((R > 180) & (G > 150)  & (B < 80))),
            'wind':  int(np.sum((G > 130) & (G > R+30) & (G > B+30))),
        }
        best_a = max(sc_a, key=sc_a.get)
        if sc_a[best_a] >= 5:
            attr_cd = best_a

    return job_cd, attr_cd


def identify_slot(img: np.ndarray, region: tuple, exclude: set = None) -> str:
    code, _, _ = identify_slot_debug(img, region, exclude=exclude)
    return code


def identify_slot_debug(img: np.ndarray, region: tuple, exclude: set = None) -> tuple:
    """返回 (code, best_score, gap)；合成模板 NCC + 职业/属性过滤，覆盖 380+ 英雄。"""
    # ── 旧方法（仅 84 英雄实拍模板，已停用）──
    # if not _draft_templates:
    #     _load_draft_templates()
    # if not _draft_templates:
    #     return 'unknown', 0.0, 0.0
    # x1, y1, x2, y2 = region
    # crop = img[y1:y2, x1:x2]
    # gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    # query = cv2.resize(gray, _TMPL_SIZE).astype(np.float32)
    # best_code, best_score, second_score = 'unknown', -1.0, -1.0
    # for code, tmpls in _draft_templates.items():
    #     if exclude and code in exclude:
    #         continue
    #     s = max(_ncc_flat(query, t) for t in tmpls)
    #     if s > best_score:
    #         second_score = best_score
    #         best_score, best_code = s, code
    #     elif s > second_score:
    #         second_score = s
    # gap = best_score - second_score
    # if best_score >= _NCC_THRESHOLD:
    #     return best_code, best_score, gap
    # return 'unknown', best_score, gap

    # ── 新方法：合成立绘模板 + 职业/属性过滤 ──
    _load_e7_attrs()
    x1, y1, x2, y2 = region
    slot_w, slot_h  = x2 - x1, y2 - y1

    all_t = _get_new_templates(slot_w, slot_h)
    if not all_t:
        return 'unknown', 0.0, 0.0

    det_job, det_attr = _detect_job_attr_new(img, region)
    if det_job or det_attr:
        filtered = {
            code: ts for code, ts in all_t.items()
            if (exclude is None or code not in exclude)
            and (det_job  is None or _E7_ATTRS.get(code, (None, None))[0] == det_job)
            and (det_attr is None or _E7_ATTRS.get(code, (None, None))[1] == det_attr)
        }
        tmpls = filtered if filtered else {
            code: ts for code, ts in all_t.items()
            if exclude is None or code not in exclude
        }
    else:
        tmpls = {code: ts for code, ts in all_t.items()
                 if exclude is None or code not in exclude}

    if not tmpls:
        return 'unknown', 0.0, 0.0

    lx    = int(slot_w * _NEW_CROP_LEFT)
    crop  = img[y1:y2, x1:x2]
    gray  = cv2.cvtColor(crop[:, lx:], cv2.COLOR_RGB2GRAY)
    query = cv2.resize(gray, _TMPL_SIZE).astype(np.float32)

    scores = [(max(_ncc_flat(query, t) for t in ts), code)
              for code, ts in tmpls.items()]
    scores.sort(reverse=True)

    best_score, best_code = scores[0]
    second_score = scores[1][0] if len(scores) > 1 else -1.0
    gap = best_score - second_score

    if best_score >= _NEW_THRESHOLD:
        return best_code, best_score, gap
    return 'unknown', best_score, gap


# ── 禁用槽识别 ────────────────────────────────────────────────
# 多尺度 matchTemplate + hero_images（CDN头像，灰度，覆盖 380+ 英雄）
_HERO_IMAGES_DIR = os.path.join(_ROOT, 'templates', 'hero_images')
# _BAN_IMAGES_DIR  = os.path.join(_ROOT, 'templates', 'ban_images')  # 旧：实拍模板
# _BAN_TMPL_SIZE   = (64, 64)   # 旧：NCC resize 尺寸
# _BAN_SCALES    = [0.45, 0.52, 0.60, 0.68, 0.75]  # 旧：固定比例，720p 槽位小时大量 scale 被跳过
# 新：fill_pcts 表示模板占搜索区的比例，自动适配任意分辨率（1920/720 均有 6 个有效 scale）
_BAN_FILL_PCTS = [0.58, 0.66, 0.74, 0.82, 0.90, 0.97]
_BAN_THRESHOLD = 0.55   # 多尺度 matchTemplate 分数区间约 0.60~0.87，空槽 < 0.40

# code -> uint8 gray (112×112)，单张 CDN 头像
_ban_templates: dict = {}


def _load_ban_templates():
    # hero_images：RGBA→黑底→灰度，保持原始 112×112 供多尺度搜索
    if os.path.exists(_HERO_IMAGES_DIR):
        for fname in os.listdir(_HERO_IMAGES_DIR):
            if not fname.endswith('.png'):
                continue
            code = fname[:-4]
            path = os.path.join(_HERO_IMAGES_DIR, fname)
            try:
                h = Image.open(path).convert('RGBA')
                bg = Image.new('RGB', h.size, (0, 0, 0))
                bg.paste(h.convert('RGB'), mask=h.split()[3])
                _ban_templates[code] = np.array(bg.convert('L'))
            except Exception:
                pass
    # 旧：ban_images 实拍模板（NCC 方案）
    # _BAN_IMAGES_DIR = os.path.join(_ROOT, 'templates', 'ban_images')
    # if os.path.exists(_BAN_IMAGES_DIR):
    #     for fname in sorted(os.listdir(_BAN_IMAGES_DIR)):
    #         if not fname.endswith('.png'): continue
    #         stem = fname[:-4]
    #         code = stem.rsplit('_',1)[0] if '_' in stem and stem.rsplit('_',1)[1].isdigit() else stem
    #         img = np.array(Image.open(os.path.join(_BAN_IMAGES_DIR, fname)).convert('L'))
    #         tmpl = cv2.resize(img, (64,64)).astype(np.float32)
    #         _ban_templates.setdefault(code, []).append(tmpl)


def _ms_ban_score(search_gray: np.ndarray, tmpl_gray: np.ndarray) -> float:
    """多尺度 matchTemplate，ban 槽专用（灰度）。模板尺寸按搜索区比例计算，自适应分辨率。"""
    best = -1.0
    sh, sw = search_gray.shape
    for pct in _BAN_FILL_PCTS:
        tw = int(sw * pct)
        th = int(sh * pct)
        if tw >= sw or th >= sh or tw < 8 or th < 8:
            continue
        t = cv2.resize(tmpl_gray, (tw, th))
        r = cv2.matchTemplate(search_gray, t, cv2.TM_CCOEFF_NORMED)
        best = max(best, float(r.max()))
    return best


def _identify_ban_slot(crop_gray: np.ndarray) -> tuple:
    # 旧：NCC 64×64
    # query = cv2.resize(crop_gray, (64,64)).astype(np.float32)
    # for code, tmpls in _ban_templates.items():
    #     for tmpl in tmpls:
    #         s = _ncc_flat(query, tmpl)
    best_code, best_score = 'empty', -1.0
    for code, tmpl in _ban_templates.items():
        s = _ms_ban_score(crop_gray, tmpl)
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
    try:
        from config_loader import cfg
        r = cfg._profile.get('resolution', [1922, 1115]) if cfg.is_loaded() else [1922, 1115]
        ew, eh = int(r[0]), int(r[1])
    except Exception:
        ew, eh = 1922, 1115
    for var, fname in [('_MY_TURN_TMPL',  'my_turn.png'),
                       ('_OPP_TURN_TMPL', 'opp_turn.png'),
                       ('_POSTBAN_TMPL',  'postban.png')]:
        path = os.path.join(_TMPL_BASE, 'phase', fname)
        img  = np.array(Image.open(path).convert('L'))
        tmpl_h, tmpl_w = img.shape[:2]
        sx, sy = tmpl_w / ew, tmpl_h / eh
        crop = img[int(y1*sy):int(y2*sy), int(x1*sx):int(x2*sx)]
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
    try:
        from config_loader import cfg
        r = cfg._profile.get('resolution', [1922, 1115]) if cfg.is_loaded() else [1922, 1115]
        ew, eh = int(r[0]), int(r[1])
    except Exception:
        ew, eh = 1922, 1115
    for attr, path in [('_POST_BAN_TMPL',   os.path.join(_TMPL_BASE, 'phase', 'postban.png')),
                       ('_BATTLE_RDY_TMPL', os.path.join(_TMPL_BASE, 'phase', 'battle_ready.png'))]:
        img = np.array(Image.open(path).convert('L'))
        tmpl_h, tmpl_w = img.shape[:2]
        sx, sy = tmpl_w / ew, tmpl_h / eh
        crop = img[int(y1*sy):int(y2*sy), int(x1*sx):int(x2*sx)]
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

    # 规则2（永远生效）：绝不禁对手槽3（索引2）
    _FORBIDDEN_SLOT = 2

    # 规则1（有亚露嘉时生效）：确认后暂停5秒
    _has_yazuga = any('亚露嘉' in _code_to_name.get(c, '') for c in (my_picks or []))

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
            # 本地统计调整 finalban 优先级
            try:
                from config_loader import cfg as _cfg
                if _cfg.is_loaded() and getattr(_cfg, 'local_stats_enabled', False):
                    from battle_ai.local_stats import get_finalban_adjustments
                    _fadj = get_finalban_adjustments(enemy_picks)
                    if _fadj:
                        recs = [dict(r, probability=r['probability'] *
                                    _fadj.get(r['hero_code'], 1.0)) for r in recs]
                        recs.sort(key=lambda r: r['probability'], reverse=True)
            except Exception:
                pass
            for rec in recs:
                code = rec['hero_code']
                if code in enemy_picks:
                    idx = enemy_picks.index(code)
                    if idx == _FORBIDDEN_SLOT:
                        _log(f'  [禁用] 跳过槽3推荐: {_code_to_name.get(code, code)}，找下一个')
                        continue
                    ban_idx = idx
                    _log(f'  [禁用] 模型推荐禁: {_code_to_name.get(code, code)}（对手槽{ban_idx+1}）')
                    break
        except Exception as e:
            _log(f'  [禁用] 模型推荐异常: {e}')

    if ban_idx is None:
        candidates = [i for i in range(len(enemy_picks)) if i != _FORBIDDEN_SLOT]
        if not candidates:
            candidates = list(range(len(enemy_picks)))
        ban_idx = random.choice(candidates)
        code = enemy_picks[ban_idx] if enemy_picks else '?'
        _log(f'  [禁用] 随机禁: {_code_to_name.get(code, code)}（对手槽{ban_idx+1}）')

    cx, cy = _opp_slot_centers()[ban_idx]
    click_at(cx, cy, delay=1.5)
    click_at(*_post_ban_btn(), delay=1.5)

    banned_code = enemy_picks[ban_idx] if ban_idx < len(enemy_picks) else None
    return banned_code

    # 亚露嘉阵容位置调整在 battle_ready 阶段由 arrange_yazuga_first 处理


# ── 战斗前亚露嘉位置调整 ──────────────────────────────────────

# 战斗准备位置调整坐标均从配置文件读取，见 _battle_slot_centers() 等 accessor

_yazuga_battle_tmpl = None
_swap_btn_tmpl       = None


def _get_yazuga_tmpl():
    global _yazuga_battle_tmpl
    if _yazuga_battle_tmpl is None:
        _yazuga_battle_tmpl = cv2.imdecode(np.fromfile(os.path.join(_TMPL_BASE, 'yazuga_battle.png'), dtype=np.uint8), cv2.IMREAD_COLOR)
    return _yazuga_battle_tmpl


def _get_swap_btn_tmpl():
    global _swap_btn_tmpl
    if _swap_btn_tmpl is None:
        _swap_btn_tmpl = cv2.imdecode(np.fromfile(os.path.join(_TMPL_BASE, 'swap_btn.png'), dtype=np.uint8), cv2.IMREAD_COLOR)
    return _swap_btn_tmpl


def _region_score(img, region, tmpl) -> float:
    """对img的region区域和tmpl做NCC，返回最大相似度。"""
    x1, y1, x2, y2 = region
    patch = img[y1:y2, x1:x2]
    h, w = tmpl.shape[:2]
    if patch.shape[0] != h or patch.shape[1] != w:
        patch = cv2.resize(patch, (w, h))
    return float(cv2.matchTemplate(patch, tmpl, cv2.TM_CCOEFF_NORMED).max())


def _slot1_yazuga_score(img=None) -> float:
    tmpl = _get_yazuga_tmpl()
    if tmpl is None:
        return 0.0
    if img is None:
        img = capture()
    return _region_score(img, _battle_yazuga_detect(), tmpl)


def _swap_btn_visible(img, slot_i) -> float:
    """检测slot_i对应的交换按钮区域是否出现了交换按钮，返回相似度。"""
    tmpl = _get_swap_btn_tmpl()
    if tmpl is None:
        return 0.0
    return _region_score(img, _battle_swap_regions()[slot_i], tmpl)


def arrange_yazuga_first(my_picks, log_fn=None):
    """
    battle_ready阶段：若我方有亚露嘉，确保她在最右槽（前位）。
    穷举槽0→1→2：点击槽→等最右边交换按钮→点交换→等动画→检测最右槽。
    每槽只试一次，失败则点两下deselect重置后继续下一槽。
    """
    _log = log_fn or (lambda m: None)
    _load_hero_names()

    if not any('亚露嘉' in _code_to_name.get(c, '') for c in (my_picks or [])):
        return

    score = _slot1_yazuga_score()
    _log(f'  [阵容] 最右槽亚露嘉相似度: {score:.3f}')
    if score >= 0.70:
        _log('  [阵容] 亚露嘉已在最右槽，无需调整')
        return

    slot_centers = _battle_slot_centers()
    deselect     = _battle_deselect()
    # 交换按钮固定在最右边（索引2），无论点的是哪个槽
    _SWAP_IDX    = 2
    swap_center  = _battle_swap_centers()[_SWAP_IDX]

    for slot_i in range(3):   # 依次试槽0、1、2，每槽只试一次
        _log(f'  [阵容] 尝试槽{slot_i + 1}')
        click_at(*slot_centers[slot_i], delay=1.0)  # 点槽，等UI响应

        click_at(*swap_center, delay=0.5)           # 点交换按钮
        time.sleep(2.0)                             # 等换位动画（共2.5s）

        new_score = _slot1_yazuga_score()
        _log(f'  [阵容] 槽{slot_i + 1} 交换后最右槽相似度: {new_score:.3f}')
        if new_score >= 0.70:
            _log(f'  [阵容] 亚露嘉已移至最右槽（来自槽{slot_i + 1}）')
            return

        # 重置：点两下deselect，已试槽不再重试
        click_at(*deselect, delay=0.5)
        click_at(*deselect, delay=0.5)

    _log('  [阵容] 亚露嘉位置调整完成')


_KRIS_CODE = 'c4123'
_kris_battle_tmpl = None


def _get_kris_tmpl():
    global _kris_battle_tmpl
    if _kris_battle_tmpl is None:
        _kris_battle_tmpl = cv2.imdecode(np.fromfile(os.path.join(_TMPL_BASE, 'kris_battle.png'), dtype=np.uint8), cv2.IMREAD_COLOR)
    return _kris_battle_tmpl


def _last_detect_region():
    return tuple(_dcfg()['battle_last_detect'])


def _last_swap_point():
    return tuple(_dcfg()['battle_last_swap'])


def _slot_last_kris_score(img=None) -> float:
    tmpl = _get_kris_tmpl()
    if tmpl is None:
        return 0.0
    if img is None:
        img = capture()
    return _region_score(img, _last_detect_region(), tmpl)


def arrange_kris_not_last(my_picks, log_fn=None):
    """battle_ready阶段：若我方有银波克莉丝媞，确保她不在最后槽。"""
    _log = log_fn or (lambda m: None)

    if _KRIS_CODE not in (my_picks or []):
        return

    score = _slot_last_kris_score()
    _log(f'  [阵容] 最后槽克莉丝媞相似度: {score:.3f}')
    if score < 0.70:
        _log('  [阵容] 克莉丝媞不在最后槽，无需调整')
        return

    _log('  [阵容] 克莉丝媞在最后槽，执行换位')
    slot_cx, slot_cy = _battle_slot_centers()[1]   # 最后槽固定是索引1
    click_at(slot_cx, slot_cy, delay=1.0)
    click_at(*_last_swap_point(), delay=0.5)
    time.sleep(2.0)

    new_score = _slot_last_kris_score()
    _log(f'  [阵容] 换位后最后槽克莉丝媞相似度: {new_score:.3f}')
    if new_score < 0.70:
        _log('  [阵容] 克莉丝媞已移出最后槽')
    else:
        _log('  [阵容] ⚠ 克莉丝媞仍在最后槽，换位可能未成功')


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
    """用 SendInput KEYEVENTF_UNICODE 逐字发送，模拟真实键盘输入。"""
    import ctypes
    KEYEVENTF_UNICODE = 0x0004
    KEYEVENTF_KEYUP   = 0x0002
    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [('wVk', ctypes.c_ushort), ('wScan', ctypes.c_ushort),
                    ('dwFlags', ctypes.c_ulong), ('time', ctypes.c_ulong),
                    ('dwExtraInfo', ctypes.c_size_t)]
    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [('dx', ctypes.c_long), ('dy', ctypes.c_long),
                    ('mouseData', ctypes.c_ulong), ('dwFlags', ctypes.c_ulong),
                    ('time', ctypes.c_ulong), ('dwExtraInfo', ctypes.c_size_t)]
    class _INPUT_UNION(ctypes.Union):
        _fields_ = [('ki', KEYBDINPUT), ('mi', MOUSEINPUT)]
    class INPUT(ctypes.Structure):
        _fields_ = [('type', ctypes.c_ulong), ('_u', _INPUT_UNION)]
    user32 = ctypes.windll.user32
    sz = ctypes.sizeof(INPUT)
    for ch in text:
        for flag in (KEYEVENTF_UNICODE, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP):
            inp = INPUT()
            inp.type = 1; inp._u.ki.wVk = 0
            inp._u.ki.wScan = ord(ch); inp._u.ki.dwFlags = flag
            user32.SendInput(1, ctypes.byref(inp), sz)
        time.sleep(0.05)

# 备用方法（PostMessage WM_CHAR），如需切回取消注释并替换上方函数体
# def _type_unicode(text: str):
#     import ctypes
#     from battle_ai.executor import _find_main_hwnd, get_window_title
#     WM_CHAR = 0x0102
#     hwnd = _find_main_hwnd(get_window_title())
#     if not hwnd:
#         return
#     user32 = ctypes.windll.user32
#     for ch in text:
#         user32.PostMessageW(hwnd, WM_CHAR, ord(ch), 0)
#         time.sleep(0.05)


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


_OCR_SIMILAR = {
    '人': '入',  '入': '人',
    '末': '未',  '未': '末',
    '莉': '利',  '利': '莉',
    '丽': '俪',  '俪': '丽',
    '亚': '芒',  '芒': '亚',
    '尔': '示',  '示': '尔',
    '己': '已',  '已': '己',
    '土': '士',  '士': '土',
}


def _name_matches(hero_name: str, ocr_text: str) -> bool:
    """任意一个字（或其相似字）出现在OCR结果中即算匹配。"""
    if not ocr_text:
        return False
    for ch in hero_name:
        if ch in ocr_text:
            return True
        for similar in _OCR_SIMILAR.get(ch, ''):
            if similar in ocr_text:
                return True
    return False


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
    _last_click = time.time()

    deadline = time.time() + 15
    while time.time() < deadline:
        img = capture()
        if _is_search_popup_open(img):
            _log('  [搜索] 弹窗已打开')
            break
        if time.time() - _last_click >= 3.0:
            click_at(*_search_open(), delay=0.5)
            _last_click = time.time()
        time.sleep(0.5)
    else:
        _log('  [搜索] 超时：弹窗未打开，放弃本轮')
        return '', [], []

    click_at(*_search_input(), delay=1.5)

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

        click_at(*_search_clear_btn(), delay=0.3)
        click_at(*_search_input(), delay=0.8)
        try:
            from config_loader import cfg
            if cfg.is_loaded() and getattr(cfg, 'input_method', 'emulator') == 'pc':
                from battle_ai.executor import type_text_chinese
                type_text_chinese(name)
            else:
                _type_unicode(name)
        except Exception:
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
    my_preban         = []
    enemy_preban      = []

    unavailable_codes: set = set(banned)

    my_first_locked = (start_pos > 0)
    opening_rule_id = 0  # preban结束后由detect_opening_rule()赋值，pick阶段开始时已知

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

            time.sleep(4.5)
            ban_frame = capture()
            opening_rule_id = detect_opening_rule(ban_frame)
            _rule_names = {1: '攻击', 2: '防御', 3: '抵抗', 4: '支援'}
            log_fn(f'  开局规则: {_rule_names.get(opening_rule_id, "未知")}（ID={opening_rule_id}）')
            raw_bans, raw_ban_scores = identify_ban_slots(ban_frame)
            # raw_bans 顺序固定：[我方ban1, 我方ban2, 对手ban1, 对手ban2]
            my_preban    = [c for c in raw_bans[:2] if c != 'empty']
            enemy_preban = [c for c in raw_bans[2:] if c != 'empty']
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

            # 用我方已选次数推算对方期望人数，避免对方光速选人导致pos失真
            # 先手: 我方第N选（0起）前对方应有人数 [0,2,2,4,4]
            # 后手: 我方第N选（0起）前对方应有人数 [1,1,3,3,5]
            _my_pick_idx = len(my_picks) - len(init_my_picks or [])
            if my_first:
                _exp_enemy = [0, 2, 2, 4, 4][min(_my_pick_idx, 4)]
            else:
                _exp_enemy = [1, 1, 3, 3, 5][min(_my_pick_idx, 4)]

            _my_set = set(my_picks)
            _new_slots = list(range(len(enemy_picks), min(_exp_enemy, 5)))
            if _new_slots:
                # 等对方槽位渲染完成，最多等8秒（时间控制，兼容ADB截图延迟）
                _scan_deadline = time.time() + 8.0
                while time.time() < _scan_deadline:
                    img_scan = capture()
                    if all(identify_slot_debug(img_scan, cur_opp_slots[si], exclude=_my_set)[0] != 'unknown'
                           for si in _new_slots):
                        break
                    time.sleep(0.3)
            else:
                img_scan = capture()
            for slot_i in _new_slots:
                code_s, score_s, gap_s = identify_slot_debug(img_scan, cur_opp_slots[slot_i], exclude=_my_set)
                if code_s == 'unknown':
                    log_fn(f"  扫描对手槽{slot_i+1}: 识别失败，占位 unknown")
                    enemy_picks.append('unknown')
                    enemy_pick_scores.append(0.0)
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
                opening_rule_id=opening_rule_id,
            )

            # 未练跳过 + 优先加权（系数叠加，统一排序）
            _PRIORITY_BOOST = 5.0
            _filtered = []
            for _rec in recs:
                _rname = _code_to_name.get(_rec['hero_code'], '')
                if is_unpracticed(_rname):
                    log_fn(f"  → 跳过未练: {_rname}（{_rec['hero_code']}）")
                    continue
                if is_priority(_rname):
                    log_fn(f"  → 优先加权: {_rname}（×{_PRIORITY_BOOST}）")
                    _filtered.append(dict(_rec, probability=_rec['probability'] * _PRIORITY_BOOST))
                else:
                    _filtered.append(_rec)
            recs = _filtered

            # 本地统计调整（在优先系数基础上叠加）
            try:
                from config_loader import cfg as _cfg
                if _cfg.is_loaded() and getattr(_cfg, 'local_stats_enabled', False):
                    from battle_ai.local_stats import get_pick_adjustments
                    _adj = get_pick_adjustments([r['hero_code'] for r in recs])
                    if _adj:
                        recs = [dict(r, probability=r['probability'] *
                                    _adj.get(r['hero_code'], 1.0)) for r in recs]
                        for r in recs:
                            if r['hero_code'] in _adj:
                                log_fn(f"  [本地] {_code_to_name.get(r['hero_code'], r['hero_code'])}"
                                       f" ×{_adj[r['hero_code']]:.2f}")
            except Exception:
                pass

            # 优先系数 × 本地系数叠加后统一排序
            recs.sort(key=lambda r: r['probability'], reverse=True)

            # 互斥组规则排除（只看我方已选）
            _rule_excluded = get_excluded_by_picks(my_picks, _code_to_name, log_fn=log_fn)
            unavailable_codes.update(_rule_excluded)

            # 对位必选：检测对手已选，触发 counter_picks 配置
            _counter_names = get_counter_picks([_code_to_name.get(c, '') for c in enemy_picks])
            _counter_cands = []
            for _ct_name in _counter_names:
                for _fc, _fn in _code_to_name.items():
                    if _fn == _ct_name and (_fc not in my_picks and _fc not in enemy_picks
                                             and _fc not in banned and _fc not in unavailable_codes):
                        _counter_cands.append((_fc, _fn, 1.0))
                        log_fn(f"  → 对位必选: {_fn}（因对手已选对应英雄）")
                        break

            # 强制选取：不管模型推不推，只要可用就排第一
            _force_names = get_force_picks()
            _force_cands = []
            for _fc, _fn in _code_to_name.items():
                if _fn in _force_names:
                    if (_fc not in my_picks and _fc not in enemy_picks
                            and _fc not in banned and _fc not in unavailable_codes):
                        _force_cands.append((_fc, _fn, 1.0))
                        log_fn(f"  → 强制选取: {_fn}（{_fc}）")

            if recs or _force_cands or _counter_cands:
                candidates = list(prepend_candidates or []) + _counter_cands + _force_cands
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

                _enemy_corrected = False
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
                            _enemy_corrected = True

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

                # 未选上 + 有新对手信息 → 重调模型再搜一次
                if not picked_code and _enemy_corrected:
                    log_fn('  [重推] 对手信息已修正，重新推荐并搜索')
                    _retry_recs = recommender.recommend(
                        my_picks=my_picks, enemy_picks=enemy_picks,
                        banned=banned, phase=phase, my_first=my_first, top_k=5,
                        opening_rule_id=opening_rule_id)
                    _retry_filtered = []
                    for _rec in _retry_recs:
                        _rn = _code_to_name.get(_rec['hero_code'], '')
                        if is_unpracticed(_rn):
                            continue
                        if is_priority(_rn):
                            _retry_filtered.append(dict(_rec, probability=_rec['probability'] * _PRIORITY_BOOST))
                        else:
                            _retry_filtered.append(_rec)
                    _retry_filtered.sort(key=lambda r: r['probability'], reverse=True)
                    _retry_cands = list(prepend_candidates or [])
                    for _rec in _retry_filtered[:5]:
                        _c = _rec['hero_code']
                        if _c in unavailable_codes:
                            continue
                        _n = _code_to_name.get(_c, '')
                        if not _n:
                            continue
                        _retry_cands.append((_c, _n, _rec['probability']))
                        log_fn(f"  → [重推] {_n}（{_c}，{_rec['probability']:.3f}）")
                    if _retry_cands:
                        picked_code, _s2, _b2 = search_and_pick_candidates(
                            _retry_cands, log_fn=log_fn, unavailable=unavailable_codes)
                        for _cc in _s2:
                            if (_cc not in my_picks and _cc not in banned
                                    and _cc not in enemy_picks):
                                _unc = [(i, s) for i, s in enumerate(enemy_pick_scores) if s < 1.0]
                                if _unc:
                                    _mi = min(_unc, key=lambda x: x[1])[0]
                                    enemy_picks[_mi] = _cc
                                    enemy_pick_scores[_mi] = 1.0
                                    unavailable_codes.add(_cc)
                        for _cc in _b2:
                            if _cc not in banned:
                                _unc = [(i, s) for i, s in enumerate(banned_scores) if s < 1.0]
                                if _unc:
                                    _mi = min(_unc, key=lambda x: x[1])[0]
                                    banned[_mi] = _cc
                                    banned_scores[_mi] = 1.0
                                    unavailable_codes.add(_cc)

                if picked_code:
                    time.sleep(1.0)
                    my_picks.append(picked_code)
                else:
                    placeholder = candidates[0][0] if candidates else (recs[0]['hero_code'] if recs else 'unknown')
                    log_fn(f"  ⚠ 搜索全部失败，以 {_code_to_name.get(placeholder, placeholder)} 占位")
                    my_picks.append(placeholder)
            else:
                # 模型推荐全部在未练名单，尝试 fallback_picks 配置
                _fallback_names = get_fallback_picks()
                _fallback_cands = []
                all_used = set(my_picks + enemy_picks + banned)
                for _fb_name in _fallback_names:
                    for _fc, _fn in _code_to_name.items():
                        if _fn == _fb_name and _fc not in all_used and _fc not in unavailable_codes:
                            _fallback_cands.append((_fc, _fn, 0.5))
                            break
                if _fallback_cands:
                    log_fn(f"  → 推荐全为未练，走兜底配置: {[c[1] for c in _fallback_cands]}")
                    picked_code, _, _ = search_and_pick_candidates(_fallback_cands, log_fn=log_fn, unavailable=unavailable_codes)
                    if picked_code:
                        time.sleep(1.0)
                        my_picks.append(picked_code)
                    else:
                        fallback = next((h for h in recommender.hero_list if h not in all_used), 'unknown')
                        log_fn(f"  ⚠ 兜底配置搜索失败，以 {_code_to_name.get(fallback, fallback)} 占位")
                        my_picks.append(fallback)
                else:
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
                log_fn(f"  ⚠ 识别失败，以 unknown 占位（不猜测，避免污染后续识别）")

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
                                       banned=banned, phase='pick5', my_first=my_first, top_k=1,
                                       opening_rule_id=opening_rule_id)
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
                                       banned=banned, phase='pick5', my_first=my_first, top_k=1,
                                       opening_rule_id=opening_rule_id)
            all_used = set(my_picks + enemy_picks + banned)
            code_new = fb[0]['hero_code'] if fb else next(
                (h for h in recommender.hero_list if h not in all_used), 'unknown')
            log_fn(f'  [过渡] 我方槽{slot_i+1} 识别失败，占位: {_code_to_name.get(code_new, code_new)}')
        my_picks.append(code_new)

    log_fn(f"选秀完成！我方: {[_code_to_name.get(c,c) for c in my_picks]} "
           f"对方: {[_code_to_name.get(c,c) for c in enemy_picks]}")
    _save_debug_slots()
    return {'my_picks': my_picks, 'enemy_picks': enemy_picks,
            'my_first': my_first, 'banned': banned,
            'my_preban': my_preban, 'enemy_preban': enemy_preban}


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
    next_phase_count = 0  # 防抖：连续3次才确认是真实阶段切换，避免过渡帧噪声误触发
    while time.time() < deadline:
        if stop_event and stop_event.is_set():
            return 0, None
        img = capture()
        if is_battle_over(img):
            _log(f'  [第{pos+1}步] 选秀中检测到战斗结束（对手投降），退出')
            return 0, None
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
            next_phase_count += 1
            if next_phase_count >= 3:
                _log(f'  [第{pos+1}步] 检测到下一阶段（OCR="{text}"），退出')
                return 0, None
            _log(f'  [第{pos+1}步] 疑似下一阶段（OCR="{text}"），确认中{next_phase_count}/3')
        else:
            next_phase_count = 0
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
        img = capture()
        if is_battle_over(img):
            _log('  选秀中检测到战斗结束（对手投降），退出等待')
            return
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


def _wait_my_turn(timeout: int = 60, stop_event=None, log_fn=None):
    _log = log_fn or (lambda m: None)
    deadline = time.time() + timeout
    last_log = 0
    while time.time() < deadline:
        if stop_event and stop_event.is_set():
            return
        img = capture()
        if is_battle_over(img):
            _log('  选秀中检测到战斗结束（对手投降），退出等待')
            return
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
        img = capture()
        if is_battle_over(img):
            return
        text = _read_turn_text(img)
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
        if is_battle_over(img):
            return
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
