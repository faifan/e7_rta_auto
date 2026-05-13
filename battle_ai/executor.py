import pyautogui
import time
import ctypes
import ctypes.wintypes as wt

WINDOW_TITLE = '第七史诗'

# 坐标基于游戏窗口客户区左上角 (0, 0)
SKILL_POS = {
    'S1': (1543, 1029),
    'S2': (1685, 1027),
    'S3': (1836, 1029),
}
BURN_POS = (1021, 1020)

ENEMY_POS = [
    (1280, 619),
    (1479, 486),
    (1525, 770),
    (1711, 594),
]

_u32 = ctypes.windll.user32
_clipboard_lock = __import__('threading').Lock()

class _POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

_win_offset = None  # (left, top) 屏幕绝对坐标


def focus_game_window():
    """把游戏窗口拉到前台，缓存客户区屏幕偏移量。"""
    global _win_offset
    hwnd = _u32.FindWindowW(None, WINDOW_TITLE)
    if not hwnd:
        raise RuntimeError(f"找不到窗口：{WINDOW_TITLE!r}，请确认游戏已打开")
    _u32.ShowWindow(hwnd, 9)          # SW_RESTORE
    _u32.keybd_event(0x12, 0, 0, 0)  # 解除前台锁定
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
    assert skill in SKILL_POS, f"未知技能: {skill}"
    _click(*SKILL_POS[skill])


def click_target(idx: int):
    assert 0 <= idx <= 3, f"目标索引超范围: {idx}"
    _click(*ENEMY_POS[idx])


def click_burn():
    _click(*BURN_POS)


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
    x, y = SKILL_POS[skill]
    if _win_offset is None:
        raise RuntimeError("请先调用 focus_game_window()")
    ox, oy = _win_offset
    pyautogui.doubleClick(ox + x, oy + y, interval=0.05)
    time.sleep(0.4)


def click_at(x: int, y: int, delay: float = 0.4):
    """点击窗口内任意坐标（客户区相对坐标）"""
    _click(x, y, delay)


def type_text(text: str):
    """输入文字（需先点击输入框）"""
    pyautogui.hotkey('ctrl', 'a')
    time.sleep(0.1)
    pyautogui.typewrite(text, interval=0.05)


def type_text_chinese(text: str):
    """输入中文（ctypes 写剪贴板 → Ctrl+A+V 粘贴）"""
    _set_clipboard(text)
    time.sleep(0.1)
    pyautogui.hotkey('ctrl', 'a')
    time.sleep(0.05)
    pyautogui.hotkey('ctrl', 'v')
    time.sleep(0.2)


def _set_clipboard(text: str):
    """直接用ctypes写剪贴板，支持中文，加锁防多线程冲突。"""
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
