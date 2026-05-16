import pyautogui
import time
import ctypes
import ctypes.wintypes as wt

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(0)
except Exception:
    pass

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

class _RECT(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

# 游戏渲染区左上角在屏幕上的物理坐标（由 focus_game_window 缓存）
# 点击时加上这个偏移，坐标才是屏幕绝对位置
_win_offset = None

# 物理像素 / 逻辑像素比
# 例：屏幕 150% 缩放，游戏渲染区实际物理宽 2880，profile 逻辑宽 1920 → dpi_scale=1.5
# 用途：把 profile 逻辑坐标 × dpi_scale → 物理坐标，再加 _win_offset 才能正确点击
_dpi_scale  = 1.0


def _get_expected_resolution() -> tuple:
    try:
        from config_loader import cfg
        if cfg.is_loaded():
            r = cfg._profile.get('resolution', [1922, 1115])
            return (int(r[0]), int(r[1]))
    except Exception:
        pass
    return (1922, 1115)


def _find_game_child(parent_hwnd, expected_w, expected_h):
    """遍历子窗口，返回 client 尺寸与期望匹配的那个，找不到返回 None。"""
    found = []
    WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    def _cb(hwnd, _):
        rc = _RECT()
        _u32.GetClientRect(hwnd, ctypes.byref(rc))
        if rc.right == expected_w and rc.bottom == expected_h:
            found.append(hwnd)
        return True

    _u32.EnumChildWindows(parent_hwnd, WNDENUMPROC(_cb), 0)
    return found[0] if found else None


_FALLBACK_TITLES = ['MuMu安卓设备', 'MuMu模拟器']

def _drawn_title_h(hwnd, client_h: int) -> int:
    """检测 client 区顶部自绘标题栏高度（如 MuMu 的 Chrome 风标题栏）。"""
    largest_h = [0]
    _EnumProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def _cb(child, _):
        r = _RECT()
        _u32.GetClientRect(child, ctypes.byref(r))
        if r.bottom > largest_h[0]:
            largest_h[0] = r.bottom
        return True
    _proc = _EnumProc(_cb)
    _u32.EnumChildWindows(hwnd, _proc, 0)
    if 0 < largest_h[0] < client_h:
        return client_h - largest_h[0]
    return 0


def _resize_to_profile(hwnd):
    """把游戏窗口的游戏内容区调整到 profile 分辨率大小。"""
    ew, eh = _get_expected_resolution()
    wr = _RECT(); _u32.GetWindowRect(hwnd, ctypes.byref(wr))
    cr = _RECT(); _u32.GetClientRect(hwnd, ctypes.byref(cr))
    os_deco_w = (wr.right - wr.left) - cr.right
    os_deco_h = (wr.bottom - wr.top) - cr.bottom
    drawn_h   = _drawn_title_h(hwnd, cr.bottom)
    outer_w   = ew + os_deco_w
    outer_h   = eh + os_deco_h + drawn_h
    print(f"[resize] os_deco={os_deco_w}x{os_deco_h}  drawn_title={drawn_h}"
          f"  → resize to {outer_w}x{outer_h}  (位置不变)")
    SWP_NOMOVE   = 0x0002
    SWP_NOZORDER = 0x0004
    _u32.SetWindowPos(hwnd, 0, 0, 0, outer_w, outer_h, SWP_NOMOVE | SWP_NOZORDER)


def _find_main_hwnd(title: str):
    hwnd = _u32.FindWindowW(None, title)
    if not hwnd:
        for fb in _FALLBACK_TITLES:
            hwnd = _u32.FindWindowW(None, fb)
            if hwnd:
                print(f"[focus] '{title}' 未找到，使用回退标题 '{fb}'")
                break
    return hwnd


def focus_game_window():
    """把游戏窗口拉到前台，缓存客户区屏幕偏移量。"""
    global _win_offset, WINDOW_TITLE, SKILL_POS
    title = get_window_title()
    WINDOW_TITLE = title
    SKILL_POS    = _get_skill_pos()

    hwnd = _find_main_hwnd(title)
    if not hwnd:
        raise RuntimeError(f"找不到窗口：{title!r}，请确认游戏已打开")

    # 最小化时先还原
    if _u32.IsIconic(hwnd):
        _u32.ShowWindow(hwnd, 9)
        time.sleep(0.5)

    # 无论是否最小化，都检查窗口是否超出屏幕 → 移回
    wr = _RECT()
    _u32.GetWindowRect(hwnd, ctypes.byref(wr))
    sw = _u32.GetSystemMetrics(0)   # SM_CXSCREEN（物理）
    sh = _u32.GetSystemMetrics(1)   # SM_CYSCREEN（物理）
    ww = wr.right - wr.left
    wh = wr.bottom - wr.top
    nx = max(0, min(wr.left, sw - ww))
    ny = max(0, min(wr.top,  sh - wh))
    if nx != wr.left or ny != wr.top:
        print(f"[focus] 窗口超出屏幕 ({wr.left},{wr.top})，移到 ({nx},{ny})")
        _u32.SetWindowPos(hwnd, 0, nx, ny, 0, 0, 0x0001 | 0x0004)
        time.sleep(0.5)

    _u32.keybd_event(0x12, 0, 0, 0)
    _u32.SetForegroundWindow(hwnd)
    _u32.keybd_event(0x12, 0, 0x0002, 0)
    time.sleep(0.3)

    # 找最大子窗口（游戏渲染区），用其屏幕坐标作为点击基准
    best_area = [0]; best_child = [0]; best_cw = [0]
    _EnumProc2 = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def _cb2(child, _):
        cr = _RECT(); _u32.GetClientRect(child, ctypes.byref(cr))
        area = cr.right * cr.bottom
        if area > best_area[0]:
            best_area[0] = area; best_child[0] = child; best_cw[0] = cr.right
        return True
    _u32.EnumChildWindows(hwnd, _EnumProc2(_cb2), 0)

    # ── 坐标基准计算 ──────────────────────────────────────────────
    # Profile 里存的是逻辑游戏坐标（相对于游戏画面左上角，范围 0→宽/0→高）
    # 要点击屏幕上正确的位置，需要：
    #   屏幕物理坐标 = _win_offset + 逻辑坐标 × _dpi_scale
    #
    # _win_offset：游戏渲染区左上角的屏幕物理坐标，用 ClientToScreen 获得
    # _dpi_scale ：渲染区实际物理宽 ÷ profile 逻辑宽
    #              同一台机器上只要游戏窗口大小不变，这个值就固定

    global _dpi_scale
    ew, _ = _get_expected_resolution()   # profile 中声明的逻辑宽
    if best_child[0]:
        # 找到了游戏真实渲染子窗口（例如 Unity/UE 的渲染层）
        # 用子窗口客户区(0,0)转屏幕坐标，比主窗口更准确（不含标题栏/边框）
        cpt = _POINT(0, 0)
        _u32.ClientToScreen(best_child[0], ctypes.byref(cpt))
        _win_offset = (cpt.x, cpt.y)
        # 子窗口实际物理宽 ÷ profile 逻辑宽 = dpi_scale
        _dpi_scale  = best_cw[0] / ew if best_cw[0] > 0 else 1.0
    else:
        # 没有子窗口（原生窗口），直接用主窗口客户区
        # 主窗口客户区可能包含手绘标题栏，需额外跳过其高度
        pt = _POINT(0, 0); _u32.ClientToScreen(hwnd, ctypes.byref(pt))
        rc = _RECT(); _u32.GetClientRect(hwnd, ctypes.byref(rc))
        tab_bar = _drawn_title_h(hwnd, rc.bottom)
        _win_offset = (pt.x, pt.y + tab_bar)
        _dpi_scale  = 1.0   # 无子窗口时无法判断缩放，默认 1:1

    print(f"[focus] offset={_win_offset}  dpi_scale={_dpi_scale:.2f}")
    return _win_offset


def _send_input_click(sx: int, sy: int):
    """用 SendInput 归一化绝对坐标点击，DPI 无关。"""
    vw = _u32.GetSystemMetrics(78)   # SM_CXVIRTUALSCREEN
    vh = _u32.GetSystemMetrics(79)   # SM_CYVIRTUALSCREEN
    vl = _u32.GetSystemMetrics(76)   # SM_XVIRTUALSCREEN（多显示器左偏移）
    vt = _u32.GetSystemMetrics(77)   # SM_YVIRTUALSCREEN（多显示器上偏移）
    nx = int((sx - vl) * 65535 / vw)
    ny = int((sy - vt) * 65535 / vh)

    MOVE  = 0x0001; DOWN = 0x0002; UP = 0x0004
    ABS   = 0x8000; VDESK = 0x4000
    flags_move = MOVE | ABS | VDESK
    flags_dn   = DOWN | ABS | VDESK
    flags_up   = UP   | ABS | VDESK

    class _MI(ctypes.Structure):
        _fields_ = [('dx',ctypes.c_long),('dy',ctypes.c_long),
                    ('mouseData',ctypes.c_ulong),('dwFlags',ctypes.c_ulong),
                    ('time',ctypes.c_ulong),('dwExtraInfo',ctypes.c_void_p)]
    class _IN(ctypes.Structure):
        _fields_ = [('type',ctypes.c_ulong),('mi',_MI)]

    arr = (_IN * 3)(
        _IN(0, _MI(nx, ny, 0, flags_move, 0, None)),
        _IN(0, _MI(nx, ny, 0, flags_dn,   0, None)),
        _IN(0, _MI(nx, ny, 0, flags_up,   0, None)),
    )
    ctypes.windll.user32.SendInput(3, arr, ctypes.sizeof(_IN))


def _click(x, y, delay=0.4):
    # x, y 是 profile 里的逻辑游戏坐标
    # 转屏幕物理坐标：窗口偏移 + 逻辑坐标 × dpi_scale
    if _win_offset is None:
        raise RuntimeError("请先调用 focus_game_window()")
    ox, oy = _win_offset
    _send_input_click(int(ox + x * _dpi_scale), int(oy + y * _dpi_scale))
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
    sx = int(ox + x * _dpi_scale)
    sy = int(oy + y * _dpi_scale)
    _send_input_click(sx, sy)
    time.sleep(0.05)
    _send_input_click(sx, sy)
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
