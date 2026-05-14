import pyautogui
import time
import ctypes
import ctypes.wintypes as wt

# 默认值（cfg 未加载时使用）
_DEFAULT_TITLE = '第七史诗'
_DEFAULT_SKILL_POS = {'S1': (1543, 1029), 'S2': (1685, 1027), 'S3': (1836, 1029)}
_DEFAULT_BURN_POS  = (1021, 1020)
_DEFAULT_ENEMY_POS = [(1280, 619), (1479, 486), (1525, 770), (1711, 594)]

# 向后兼容：其他模块可能直接 import WINDOW_TITLE / SKILL_POS
WINDOW_TITLE = _DEFAULT_TITLE
SKILL_POS    = _DEFAULT_SKILL_POS


def _get_cfg_exec() -> dict:
    try:
        from config_loader import cfg
        if cfg.is_loaded():
            return cfg.section('executor')
    except ImportError:
        pass
    return {}


def get_window_title() -> str:
    try:
        from config_loader import cfg
        if cfg.is_loaded():
            return cfg.window_title
    except ImportError:
        pass
    return _DEFAULT_TITLE


def _get_skill_pos() -> dict:
    sec = _get_cfg_exec()
    if 'skill_pos' in sec:
        return {k: tuple(v) for k, v in sec['skill_pos'].items()}
    return _DEFAULT_SKILL_POS


def _get_burn_pos() -> tuple:
    sec = _get_cfg_exec()
    return tuple(sec['burn_pos']) if 'burn_pos' in sec else _DEFAULT_BURN_POS


def _get_enemy_pos() -> list:
    sec = _get_cfg_exec()
    return [tuple(p) for p in sec['enemy_pos']] if 'enemy_pos' in sec else _DEFAULT_ENEMY_POS


_u32 = ctypes.windll.user32

class _POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

_win_offset = None


def focus_game_window():
    """把游戏窗口拉到前台，缓存客户区屏幕偏移量。"""
    global _win_offset, WINDOW_TITLE, SKILL_POS
    title = get_window_title()
    WINDOW_TITLE = title  # 同步向后兼容常量
    SKILL_POS    = _get_skill_pos()

    hwnd = _u32.FindWindowW(None, title)
    if not hwnd:
        raise RuntimeError(f"找不到窗口：{title!r}，请确认游戏已打开")
    _u32.ShowWindow(hwnd, 9)
    _u32.keybd_event(0x12, 0, 0, 0)
    _u32.SetForegroundWindow(hwnd)
    _u32.keybd_event(0x12, 0, 0x0002, 0)
    time.sleep(0.5)
    pt = _POINT(0, 0)
    _u32.ClientToScreen(hwnd, ctypes.byref(pt))
    _win_offset = (pt.x, pt.y)
    return _win_offset


def _click(x, y, delay=0.4):
    if _win_offset is None:
        raise RuntimeError("请先调用 focus_game_window()")
    ox, oy = _win_offset
    pyautogui.click(ox + x, oy + y)
    time.sleep(delay)


def click_skill(skill: str):
    pos = _get_skill_pos()
    assert skill in pos, f"未知技能: {skill}"
    _click(*pos[skill])


def click_target(idx: int):
    enemy_pos = _get_enemy_pos()
    assert 0 <= idx < len(enemy_pos), f"目标索引超范围: {idx}"
    _click(*enemy_pos[idx])


def click_burn():
    _click(*_get_burn_pos())


def do_action(skill: str, target_idx: int, burn: bool = False):
    """单体技能：(可选)烧魂 → 点技能 → 点目标"""
    if burn:
        click_burn()
    click_skill(skill)
    click_target(target_idx)


def do_aoe(skill: str, burn: bool = False):
    """群体技能：(可选)烧魂 → 快速双击技能按钮"""
    if burn:
        click_burn()
    pos = _get_skill_pos()
    x, y = pos[skill]
    if _win_offset is None:
        raise RuntimeError("请先调用 focus_game_window()")
    ox, oy = _win_offset
    pyautogui.doubleClick(ox + x, oy + y, interval=0.05)
    time.sleep(0.4)


def click_at(x: int, y: int, delay: float = 0.4):
    """点击窗口内任意坐标（客户区相对坐标）"""
    _click(x, y, delay)


def type_text(text: str):
    pyautogui.hotkey('ctrl', 'a')
    time.sleep(0.1)
    pyautogui.typewrite(text, interval=0.05)


def type_text_chinese(text: str):
    _set_clipboard(text)
    time.sleep(0.1)
    pyautogui.hotkey('ctrl', 'a')
    time.sleep(0.05)
    pyautogui.hotkey('ctrl', 'v')
    time.sleep(0.2)


def _set_clipboard(text: str):
    import ctypes as c
    k = c.windll.kernel32
    u = c.windll.user32
    k.GlobalAlloc.restype   = c.c_void_p
    k.GlobalAlloc.argtypes  = [c.c_uint, c.c_size_t]
    k.GlobalLock.restype    = c.c_void_p
    k.GlobalLock.argtypes   = [c.c_void_p]
    k.GlobalUnlock.argtypes = [c.c_void_p]
    u.SetClipboardData.restype  = c.c_void_p
    u.SetClipboardData.argtypes = [c.c_uint, c.c_void_p]

    _clipboard_lock = __import__('threading').Lock()
    with _clipboard_lock:
        buf = (text + '\0').encode('utf-16-le')
        h = k.GlobalAlloc(0x0002, len(buf))
        if not h:
            raise RuntimeError('GlobalAlloc 失败')
        p = k.GlobalLock(h)
        if not p:
            raise RuntimeError('GlobalLock 失败')
        c.memmove(p, buf, len(buf))
        k.GlobalUnlock(h)
        for _ in range(20):
            if u.OpenClipboard(0):
                break
            time.sleep(0.05)
        else:
            raise RuntimeError('OpenClipboard 失败')
        u.EmptyClipboard()
        if not u.SetClipboardData(13, h):
            u.CloseClipboard()
            raise RuntimeError('SetClipboardData 失败')
        u.CloseClipboard()
