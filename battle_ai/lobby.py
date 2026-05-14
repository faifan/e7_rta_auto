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

def _btn_region():
    p = _lcfg()
    return tuple(p['btn_region']) if 'btn_region' in p else _DEFAULT_BTN_REGION


_lobby_tmpl   = None
_waiting_tmpl = None


def _crop_btn(img_path: str) -> np.ndarray:
    img = np.array(Image.open(img_path).convert('L'))
    x1, y1, x2, y2 = _btn_region()
    crop = img[y1:y2, x1:x2]
    return cv2.resize(crop, _TMPL_SIZE).astype(np.float32)


def _load_templates():
    global _lobby_tmpl, _waiting_tmpl
    _lobby_tmpl   = _crop_btn(os.path.join(_TMPL_DIR, 'lobby_apply.png'))
    _waiting_tmpl = _crop_btn(os.path.join(_TMPL_DIR, 'lobby_waiting.png'))


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


def confirm_battle_result():
    from battle_ai.perception import capture, is_levelup_screen
    img = capture()
    btn = _confirm_btn_levelup() if is_levelup_screen(img) else _confirm_btn()
    click_at(*btn)
    time.sleep(1.5)


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
