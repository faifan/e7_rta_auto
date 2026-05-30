import json
import os
import difflib
import zhconv

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PRIORITY_FILE = os.path.join(_ROOT, 'config', 'skill_priority.json')

_DEFAULT_PRIORITY = ['S3', 'S2', 'S1']
_DEFAULT_S3_SKIP  = 2


def _load_db() -> dict:
    if not os.path.exists(_PRIORITY_FILE):
        return {}
    with open(_PRIORITY_FILE, encoding='utf-8') as f:
        return json.load(f)

_db = _load_db()

# 每场战斗内存状态
_s3_skip:            dict[str, int]  = {}
_s2_fail_streak:     dict[str, int]  = {}   # S2 连续无响应次数
_s2_disabled:        dict[str, bool] = {}   # 满2次后本场禁用S2
_pending_extra_turn: dict[str, str]  = {}   # char_key -> 'force_burn'|'soul_burn'|'normal'
_first_action_done:  set[str]        = set() # 已完成首次行动（burn_timing用）

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

    # 队伍已知：OCR结果必定是队伍中某一个，找字符重叠比例最高的
    if _my_team_names:
        best_key, best_ratio = None, 0.0
        for team_name in _my_team_names:
            overlap = sum(1 for c in team_name if c in name_s)
            ratio = overlap / len(team_name)
            if ratio > best_ratio:
                best_ratio, best_key = ratio, team_name
        if best_key and best_ratio > 0:   # 有任意重叠即取最优
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


def get_skill_type(char_name: str | None, skill: str) -> str:
    """返回技能类型 'single' 或 'aoe'。未配置角色默认 'aoe'（不需要点目标，更安全）。"""
    key = _norm(char_name)
    if key and key in _db:
        entry = _db[key]
        if isinstance(entry, dict):
            return entry.get(skill, {}).get('type', 'aoe')
    return 'aoe'


_DEAD_HP_RATIO = 0.02  # 血量比例低于此视为已死亡


def get_attack_target(hp_ratios: list[float], enemy_has_yazuga: bool,
                      front_row_indices: list[int]) -> int:
    """根据血量和亚露嘉状态，返回应攻击的目标索引。"""
    n = len(hp_ratios)
    if n == 0:
        return 0

    alive = [i for i in range(n) if hp_ratios[i] > _DEAD_HP_RATIO]
    if not alive:
        return 0

    all_full = all(hp_ratios[i] > 0.95 for i in alive)

    # if enemy_has_yazuga:
    #     front_alive = [i for i in front_row_indices if i in alive]
    #     if front_alive:
    #         return min(front_alive, key=lambda i: hp_ratios[i])

    # if all_full:
    #     return alive[0]

    # return min(alive, key=lambda i: hp_ratios[i])
    return 0


def get_burn_timing(char_name: str | None) -> str | None:
    """返回角色配置的烧魂时机：'first'|'second'|None。"""
    key = _norm(char_name)
    if key and key in _db:
        entry = _db[key]
        if isinstance(entry, dict):
            return entry.get('burn_timing')
    return None


def is_first_action_done(char_name: str | None) -> bool:
    key = _norm(char_name) or char_name
    return key in _first_action_done if key else True


def mark_first_action_done(char_name: str | None):
    key = _norm(char_name) or char_name
    if key:
        _first_action_done.add(key)


def get_extra_turn_skill(char_name: str | None) -> str | None:
    """返回该角色配置的额外回合技能代号，未配置返回None。"""
    key = _norm(char_name)
    if key and key in _db:
        entry = _db[key]
        if isinstance(entry, dict):
            return entry.get('extra_turn')
    return None


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


def _is_s2_passive(key: str | None) -> bool:
    """已录入配置且S2为被动，返回True。"""
    if key and key in _db:
        entry = _db[key]
        if isinstance(entry, dict):
            return entry.get('S2', {}).get('type') == 'passive'
    return False


def get_candidates(char_name: str | None) -> list[str]:
    """返回本回合候选技能有序列表（过滤 s3_skip、s2_disabled、被动S2）。
    未配置角色返回默认 ['S3','S2','S1']。
    """
    key = _norm(char_name)

    # 未配置：S3→S1，不试S2
    if not key or key not in _db:
        return ['S3', 'S1']

    priority = _get_priority(key)

    skip = _s3_skip.get(key, 0)
    s3_ready = (skip == 0)
    if not s3_ready:
        _s3_skip[key] = skip - 1

    # s2_off     = _s2_disabled.get(key, False)
    s2_passive = _is_s2_passive(key)

    return [s for s in priority
            if not (s == 'S3' and not s3_ready)
            and not (s == 'S2' and s2_passive)]


def on_s3_success(char_name: str | None):
    """S3成功触发后调用，设置该角色的s3_skip计数。"""
    key = _norm(char_name)
    target = key or char_name
    if not target:
        return
    _s3_skip[target] = _get_s3_skip_cfg(key)


def on_s2_success(char_name: str | None):
    """S2成功：重置连续失败计数，允许继续使用。"""
    # key = _norm(char_name)
    # target = key or char_name
    # if target:
    #     _s2_fail_streak.pop(target, None)
    #     _s2_disabled.pop(target, None)
    pass


def on_s2_fail(char_name: str | None):
    """S2无响应：累计次数，连续失败2次后本场禁用。"""
    # key = _norm(char_name)
    # target = key or char_name
    # if not target:
    #     return
    # streak = _s2_fail_streak.get(target, 0) + 1
    # _s2_fail_streak[target] = streak
    # if streak >= 2:
    #     _s2_disabled[target] = True
    pass


# ── 额外回合 pending 状态 ─────────────────────────────────────

def set_pending_extra_turn(char_name: str | None, mode: str):
    """mode: 'force_burn' | 'soul_burn' | 'normal'"""
    key = _norm(char_name) or char_name
    if key:
        _pending_extra_turn[key] = mode


def get_pending_extra_turn(char_name: str | None) -> str | None:
    key = _norm(char_name) or char_name
    return _pending_extra_turn.get(key) if key else None


def clear_pending_extra_turn(char_name: str | None):
    key = _norm(char_name) or char_name
    if key:
        _pending_extra_turn.pop(key, None)


def reset_battle():
    """每场战斗开始时调用，清空所有角色状态。"""
    _s3_skip.clear()
    _s2_fail_streak.clear()
    _s2_disabled.clear()
    _pending_extra_turn.clear()
    _first_action_done.clear()
    global _force_first_burn_armed, _force_first_burn_done
    _force_first_burn_armed = False
    _force_first_burn_done  = False
