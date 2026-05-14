"""
全局配置加载器。
GUI 启动后用户选择窗口/语言，调用 cfg.load() 一次。
各模块通过 cfg 对象取值，cfg 未加载时返回硬编码默认值。
"""
import json
import os
import ctypes

_ROOT = os.path.dirname(os.path.abspath(__file__))

_DEFAULT_PROFILE = os.path.join(_ROOT, 'profiles', '1922x1115.json')
_DEFAULT_LANG    = os.path.join(_ROOT, 'lang', 'zh_cn.json')
_DEFAULT_TITLE   = '第七史诗'


class _Cfg:
    def __init__(self):
        self._profile: dict = {}
        self._lang:    dict = {}
        self.window_title:  str  = _DEFAULT_TITLE
        self.profile_path:  str  = _DEFAULT_PROFILE
        self.lang_path:     str  = _DEFAULT_LANG
        self._loaded = False

    def load(self, window_title: str,
             profile_path: str = _DEFAULT_PROFILE,
             lang_path:    str = _DEFAULT_LANG):
        with open(profile_path, encoding='utf-8') as f:
            self._profile = json.load(f)
        with open(lang_path, encoding='utf-8') as f:
            self._lang = json.load(f)
        self.window_title = window_title
        self.profile_path = profile_path
        self.lang_path    = lang_path
        self._loaded = True

    def is_loaded(self) -> bool:
        return self._loaded

    # ── 坐标访问 ──────────────────────────────────────────────

    def section(self, sec: str) -> dict:
        """返回 profile 中某个 section 的完整字典。"""
        return self._profile.get(sec, {})

    def coord(self, sec: str, key: str):
        """
        取坐标，返回 tuple。
        列表长度 ≤ 4 → tuple，否则返回原始列表。
        """
        val = self._profile.get(sec, {}).get(key)
        if val is None:
            raise KeyError(f"profile 中找不到 [{sec}][{key}]")
        if isinstance(val, list) and len(val) <= 4:
            return tuple(val)
        return val

    # ── 语言关键词访问 ────────────────────────────────────────

    def lang(self, key: str, default=None):
        return self._lang.get(key, default)


cfg = _Cfg()


# ── 工具：枚举当前所有可见窗口 ────────────────────────────────

def list_windows() -> list:
    """返回当前桌面所有可见窗口标题列表（去重、去空）。"""
    titles = []
    EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_int, ctypes.c_int)

    def _cb(hwnd, _):
        if ctypes.windll.user32.IsWindowVisible(hwnd):
            length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
            if length > 0:
                buf = ctypes.create_unicode_buffer(length + 1)
                ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
                t = buf.value.strip()
                if t:
                    titles.append(t)
        return True

    ctypes.windll.user32.EnumWindows(EnumWindowsProc(_cb), 0)
    return sorted(set(titles))


# ── 工具：列出可用 profile 和 lang 文件 ───────────────────────

def list_profiles() -> list:
    d = os.path.join(_ROOT, 'profiles')
    if not os.path.isdir(d):
        return []
    return [f for f in os.listdir(d) if f.endswith('.json')]


def list_langs() -> dict:
    """返回 {显示名: 文件路径} 字典。"""
    d = os.path.join(_ROOT, 'lang')
    mapping = {
        'zh_cn.json': '简体中文',
        'zh_tw.json': '繁體中文',
    }
    result = {}
    if os.path.isdir(d):
        for f in os.listdir(d):
            if f.endswith('.json'):
                label = mapping.get(f, f)
                result[label] = os.path.join(d, f)
    return result
