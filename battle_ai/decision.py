import json
import os
import difflib
import zhconv

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PRIORITY_FILE = os.path.join(_ROOT, 'skill_priority.json')

_DEFAULT_PRIORITY = ['S3', 'S2', 'S1']
_DEFAULT_S3_SKIP  = 2


def _load_db() -> dict:
    if not os.path.exists(_PRIORITY_FILE):
        return {}
    with open(_PRIORITY_FILE, encoding='utf-8') as f:
        return json.load(f)

_db = _load_db()

# 每场战斗内存状态：{normalized_name: s3_skip_remaining}
_s3_skip: dict[str, int] = {}

# 首回合强制烧魂（队伍含特定角色时触发）
_FORCE_FIRST_BURN_CHARS  = {'黑暗牧者迪埃妮'}
_NO_SOUL_BURN_SELF       = {'黑暗牧者迪埃妮'}   # 这些角色自己永不烧魂
_force_first_burn_armed  = False
_force_first_burn_done   = False

_my_team_names: list[str] = []


def set_my_team(names: list[str]):
    """战斗开始前传入己方5个英雄中文名，用于 OCR 名字匹配优化。"""
    global _my_team_names
    _my_team_names = [zhconv.convert(n, 'zh-hans') for n in names if n]


def check_force_first_burn_pick(name: str) -> bool:
    """检查该英雄名是否触发首回合强制烧魂（繁简均支持）。"""
    return zhconv.convert(name, 'zh-hans') in _FORCE_FIRST_BURN_CHARS


def arm_force_first_burn():
    """选秀后确认迪埃妮在阵容时调用，armed首回合强制烧魂。"""
    global _force_first_burn_armed, _force_first_burn_done
    _force_first_burn_armed = True
    _force_first_burn_done  = False


def is_force_first_burn_pending() -> bool:
    """返回True表示首回合强制烧魂尚未触发。"""
    return _force_first_burn_armed and not _force_first_burn_done


def mark_force_first_burn_done():
    """首回合逻辑执行后调用，无论是否成功。"""
    global _force_first_burn_done
    _force_first_burn_done = True


def _edit_dist(a: str, b: str) -> int:
    """Levenshtein 编辑距离（替换/插入/删除各计1步）。"""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            prev, dp[j] = dp[j], prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
    return dp[n]


def _fuzzy_key(name: str) -> str | None:
    """在JSON键中找最相似的，相似度>=0.6才返回。"""
    if not name or not _db:
        return None
    matches = difflib.get_close_matches(name, _db.keys(), n=1, cutoff=0.6)
    return matches[0] if matches else None


def _norm(name: str | None) -> str | None:
    """把OCR名字归一化为JSON键（或原始名）。繁体先转简体再匹配。"""
    if not name:
        return None
    name_s = zhconv.convert(name, 'zh-hans')

    # 优先在己方已知阵容（5人）里用编辑距离匹配
    # 编辑距离比 SequenceMatcher 更能处理多字同时认错的情况
    # 阈值0.75：最多允许75%字符需要更改（3字名最多2字错）
    if _my_team_names:
        best_key, best_ratio = None, 1.0
        for team_name in _my_team_names:
            truncated = name_s[:len(team_name)]
            dist = _edit_dist(truncated, team_name)
            ratio = dist / max(len(truncated), len(team_name))
            if ratio < best_ratio:
                best_ratio, best_key = ratio, team_name
        if best_key and best_ratio <= 0.75:
            return _fuzzy_key(best_key) or best_key

    # fallback: 全局 fuzzy match
    key = _fuzzy_key(name_s)
    return key if key else name_s


def _get_priority(key: str | None) -> list[str]:
    if key and key in _db:
        entry = _db[key]
        if isinstance(entry, list):
            return list(entry)
        if isinstance(entry, dict):
            return list(entry.get('priority', _DEFAULT_PRIORITY))
    return list(_DEFAULT_PRIORITY)


def _get_s3_skip_cfg(key: str | None) -> int:
    if key and key in _db:
        entry = _db[key]
        if isinstance(entry, dict):
            return int(entry.get('s3_skip', _DEFAULT_S3_SKIP))
    return _DEFAULT_S3_SKIP


def team_has_soul_burn() -> bool:
    """队伍里是否有配置了烧魂技能的角色。"""
    return any(get_soul_burn_skill(name) for name in _my_team_names)


def get_soul_burn_skill(char_name: str | None) -> str | None:
    """返回该角色配置的烧魂技能，未配置或配置格式不含soul_burn则返回None。"""
    key = _norm(char_name)
    if key in _NO_SOUL_BURN_SELF:
        return None
    if key and key in _db:
        entry = _db[key]
        if isinstance(entry, dict):
            return entry.get('soul_burn')
    return None


def get_candidates(char_name: str | None) -> list[str]:
    """
    返回本回合候选技能有序列表（仅过滤s3_skip，不检测亮度/冷却）。
    """
    key = _norm(char_name)
    priority = _get_priority(key)

    # s3_skip 检查与递减
    skip = _s3_skip.get(key or char_name or '', 0)
    s3_ready = (skip == 0)
    if not s3_ready and (key or char_name):
        _s3_skip[key or char_name] = skip - 1

    return [s for s in priority if not (s == 'S3' and not s3_ready)]


def on_s3_success(char_name: str | None):
    """S3成功触发后调用，设置该角色的s3_skip计数。"""
    key = _norm(char_name)
    target = key or char_name
    if not target:
        return
    _s3_skip[target] = _get_s3_skip_cfg(key)


def reset_battle():
    """每场战斗开始时调用，清空所有角色状态。"""
    _s3_skip.clear()
    global _force_first_burn_armed, _force_first_burn_done
    _force_first_burn_armed = False
    _force_first_burn_done  = False
