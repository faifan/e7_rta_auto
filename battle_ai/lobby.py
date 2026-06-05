"""大厅导航：战斗结算 → 申请战斗"""
import os
import time
import numpy as np
import cv2
from PIL import Image
from battle_ai.executor import click_at, focus_game_window
from battle_ai.perception import capture

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_DEFAULT_APPLY_BTN          = (1292, 1009)
_DEFAULT_CONFIRM_BTN        = (1755, 1026)
_DEFAULT_CONFIRM_BTN_LEVELUP = (961,  1023)
_DEFAULT_BTN_REGION         = (1120, 947, 1464, 1072)
_TMPL_SIZE                  = (128, 46)
_TMPL_DIR                   = os.path.join(_ROOT, 'templates', 'phase')


def _lcfg() -> dict:
    try:
        from config_loader import cfg
        if cfg.is_loaded():
            return cfg.section('lobby')
    except ImportError:
        pass
    return {}

def _apply_btn():
    p = _lcfg()
    return tuple(p['apply_btn']) if 'apply_btn' in p else _DEFAULT_APPLY_BTN

def _confirm_btn():
    p = _lcfg()
    return tuple(p['confirm_btn']) if 'confirm_btn' in p else _DEFAULT_CONFIRM_BTN

def _confirm_btn_levelup():
    p = _lcfg()
    return tuple(p['confirm_btn_levelup']) if 'confirm_btn_levelup' in p else _DEFAULT_CONFIRM_BTN_LEVELUP

def _match_accept_btn():
    p = _lcfg()
    return tuple(p['match_accept_btn']) if 'match_accept_btn' in p else None

def _result_unknown_click():
    p = _lcfg()
    return tuple(p['result_unknown_click']) if 'result_unknown_click' in p else None

_DEFAULT_INTIMACY_DISMISS = (1125, 839)  # 点此关闭亲密度弹窗，弹窗消失后可检测到底层胜利界面

def _intimacy_dismiss_btn():
    p = _lcfg()
    return tuple(p['intimacy_dismiss_btn']) if 'intimacy_dismiss_btn' in p else _DEFAULT_INTIMACY_DISMISS



def _btn_region():
    p = _lcfg()
    return tuple(p['btn_region']) if 'btn_region' in p else _DEFAULT_BTN_REGION


_lobby_tmpl      = None
_waiting_tmpl    = None
_main_menu_tmpl  = None
_arena_menu_tmpl = None


def _crop_btn(img_path: str) -> np.ndarray:
    img = np.array(Image.open(img_path).convert('L'))
    tmpl_h, tmpl_w = img.shape[:2]
    x1, y1, x2, y2 = _btn_region()
    try:
        from config_loader import cfg
        r = cfg._profile.get('resolution', [1922, 1115]) if cfg.is_loaded() else [1922, 1115]
        ew, eh = int(r[0]), int(r[1])
    except Exception:
        ew, eh = 1922, 1115
    sx, sy = tmpl_w / ew, tmpl_h / eh
    crop = img[int(y1*sy):int(y2*sy), int(x1*sx):int(x2*sx)]
    return cv2.resize(crop, _TMPL_SIZE).astype(np.float32)


_DEFAULT_SIGNIN_CONFIRM_BTN = (1609, 998)

def _signin_confirm_btn():
    p = _lcfg()
    return tuple(p['signin_confirm_btn']) if 'signin_confirm_btn' in p else _DEFAULT_SIGNIN_CONFIRM_BTN


_DEFAULT_MAIN_MENU_REGION  = (1744, 563, 1861, 608)
_DEFAULT_ARENA_MENU_REGION = (1113, 978, 1320, 1030)
_DEFAULT_MAIN_MENU_ARENA_BTN  = (1802, 585)
_DEFAULT_ARENA_MENU_WORLD_BTN = (1216, 1004)
_MENU_TMPL_SIZE = (128, 46)

def _main_menu_region():
    p = _lcfg()
    return tuple(p['main_menu_region']) if 'main_menu_region' in p else _DEFAULT_MAIN_MENU_REGION

def _arena_menu_region():
    p = _lcfg()
    return tuple(p['arena_menu_region']) if 'arena_menu_region' in p else _DEFAULT_ARENA_MENU_REGION

def _main_menu_arena_btn():
    p = _lcfg()
    return tuple(p['main_menu_arena_btn']) if 'main_menu_arena_btn' in p else _DEFAULT_MAIN_MENU_ARENA_BTN

def _arena_menu_world_btn():
    p = _lcfg()
    return tuple(p['arena_menu_world_btn']) if 'arena_menu_world_btn' in p else _DEFAULT_ARENA_MENU_WORLD_BTN

def _crop_region(img_path: str, region_fn) -> np.ndarray:
    img = np.array(Image.open(img_path).convert('L'))
    tmpl_h, tmpl_w = img.shape[:2]
    x1, y1, x2, y2 = region_fn()
    try:
        from config_loader import cfg
        r = cfg._profile.get('resolution', [1922, 1115]) if cfg.is_loaded() else [1922, 1115]
        ew, eh = int(r[0]), int(r[1])
    except Exception:
        ew, eh = 1922, 1115
    sx, sy = tmpl_w / ew, tmpl_h / eh
    crop = img[int(y1*sy):int(y2*sy), int(x1*sx):int(x2*sx)]
    return cv2.resize(crop, _MENU_TMPL_SIZE).astype(np.float32)

def _load_templates():
    global _lobby_tmpl, _waiting_tmpl, _main_menu_tmpl, _arena_menu_tmpl
    _lobby_tmpl      = _crop_btn(os.path.join(_TMPL_DIR, 'lobby_apply.png'))
    _waiting_tmpl    = _crop_btn(os.path.join(_TMPL_DIR, 'lobby_waiting.png'))
    _main_menu_tmpl  = _crop_region(os.path.join(_TMPL_DIR, 'main_menu.png'),  _main_menu_region)
    _arena_menu_tmpl = _crop_region(os.path.join(_TMPL_DIR, 'arena_menu.png'), _arena_menu_region)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    denom = np.sqrt((a**2).sum() * (b**2).sum())
    return float(np.sum(a * b) / denom) if denom > 1e-6 else 0.0


def _btn_ncc(img: np.ndarray, tmpl: np.ndarray) -> float:
    x1, y1, x2, y2 = _btn_region()
    crop  = img[y1:y2, x1:x2]
    gray  = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    query = cv2.resize(gray, _TMPL_SIZE).astype(np.float32)
    return _ncc(query, tmpl)


def _region_ncc(img: np.ndarray, region_fn, tmpl: np.ndarray) -> float:
    x1, y1, x2, y2 = region_fn()
    crop  = img[y1:y2, x1:x2]
    gray  = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    query = cv2.resize(gray, _MENU_TMPL_SIZE).astype(np.float32)
    return _ncc(query, tmpl)


def dismiss_signin_reward():
    """关闭签到奖励弹窗，点击确认按钮。"""
    click_at(*_signin_confirm_btn(), delay=1.0)


def is_in_main_menu(img: np.ndarray = None) -> bool:
    if _main_menu_tmpl is None:
        _load_templates()
    if img is None:
        img = capture()
    return _region_ncc(img, _main_menu_region, _main_menu_tmpl) >= 0.5


def is_in_arena_menu(img: np.ndarray = None) -> bool:
    if _arena_menu_tmpl is None:
        _load_templates()
    if img is None:
        img = capture()
    return _region_ncc(img, _arena_menu_region, _arena_menu_tmpl) >= 0.5


def click_arena_btn():
    click_at(*_main_menu_arena_btn(), delay=1.5)


def click_world_arena_btn():
    click_at(*_arena_menu_world_btn(), delay=2.0)


def is_in_lobby(img: np.ndarray = None) -> bool:
    if _lobby_tmpl is None:
        _load_templates()
    if img is None:
        img = capture()
    return _btn_ncc(img, _lobby_tmpl) >= 0.5


def is_waiting_for_match(img: np.ndarray = None) -> bool:
    if _waiting_tmpl is None:
        _load_templates()
    if img is None:
        img = capture()
    return _btn_ncc(img, _waiting_tmpl) >= 0.5


def click_match_accept():
    """点击匹配确认界面的确认按钮（waiting→preban 之间的中间界面）。"""
    btn = _match_accept_btn()
    if btn:
        click_at(*btn, delay=1.0)

def click_result_unknown():
    """点击结算后未知中间界面的屏幕中心。"""
    pos = _result_unknown_click()
    if pos:
        click_at(*pos, delay=1.0)

def confirm_battle_result():
    click_at(*_confirm_btn())
    time.sleep(1.5)

def confirm_levelup_result():
    click_at(*_confirm_btn_levelup())
    time.sleep(1.5)

def dismiss_intimacy_dialog():
    """关闭亲密度等级上升弹窗，弹窗消失后底层胜利界面可被正常检测。"""
    click_at(*_intimacy_dismiss_btn(), delay=1.0)


def apply_for_battle():
    click_at(*_apply_btn())
    time.sleep(1.0)


def wait_for_lobby(timeout: int = 30) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if is_in_lobby():
            return True
        time.sleep(1.0)
    return False
