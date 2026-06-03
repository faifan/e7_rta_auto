"""英雄个人配置：未练跳过 + 优先选择 + 互斥组规则"""
import json
import os
import zhconv

_ROOT             = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_FILE      = os.path.join(_ROOT, 'config', 'hero_config.json')
_RULES_FILE       = os.path.join(_ROOT, 'config', 'pick_rules.json')
_COUNTER_FILE     = os.path.join(_ROOT, 'config', 'counter_picks.json')
_ATK_PRIORITY_FILE = os.path.join(_ROOT, 'config', 'attack_priority.json')

_config        = None
_rules         = None
_counter       = None
_atk_priority  = None


def _load() -> dict:
    global _config
    if _config is None:
        if os.path.exists(_CONFIG_FILE):
            with open(_CONFIG_FILE, encoding='utf-8') as f:
                _config = json.load(f)
        else:
            _config = {}
    return _config


def _n(name: str) -> str:
    return zhconv.convert(name, 'zh-hans')


def is_unpracticed(name: str) -> bool:
    """返回 True 表示该角色在未练名单中，选秀时跳过。"""
    cfg = _load()
    target = _n(name)
    return target in {_n(n) for n in cfg.get('unpracticed', [])}


def is_priority(name: str) -> bool:
    """返回 True 表示该角色在优先名单中，选秀时挪到候选列表最前面。"""
    cfg = _load()
    target = _n(name)
    return target in {_n(n) for n in cfg.get('priority', [])}


def get_fallback_picks() -> list:
    """返回兜底候选角色名列表，当模型推荐全部被未练跳过时按序搜索选人。"""
    cfg = _load()
    return [_n(name) for name in cfg.get('fallback_picks', [])]


def get_force_picks() -> set:
    """返回强制选取角色名集合（简体），不管模型推不推、只要可用就排第一候选。"""
    cfg = _load()
    return {_n(name) for name in cfg.get('force_pick', [])}


def _load_counter() -> dict:
    global _counter
    if _counter is None:
        if os.path.exists(_COUNTER_FILE):
            with open(_COUNTER_FILE, encoding='utf-8') as f:
                _counter = json.load(f)
        else:
            _counter = {}
    return _counter


def get_counter_picks(enemy_names: list) -> list:
    """根据对手已选英雄名列表，返回应对位必选的我方英雄名列表（简体，保持配置顺序）。"""
    cfg = _load_counter()
    if not cfg:
        return []
    enemy_set = {_n(name) for name in enemy_names if name}
    result = []
    for enemy_name, my_name in cfg.items():
        if _n(enemy_name) in enemy_set:
            result.append(_n(my_name))
    return result


def _load_rules() -> list:
    global _rules
    if _rules is None:
        if os.path.exists(_RULES_FILE):
            with open(_RULES_FILE, encoding='utf-8') as f:
                _rules = json.load(f)
        else:
            _rules = []
    return _rules


def _rule_name_in(rule_name: str, name_set: set) -> bool:
    """rule_name 是否与 name_set 中某个名字子串匹配（双向）。"""
    rn = _n(rule_name)
    return any(rn in _n(n) or _n(n) in rn for n in name_set if n)


def get_excluded_by_picks(my_pick_codes: list, code_to_name: dict, log_fn=None) -> set:
    """
    根据 pick_rules.json 互斥组规则，返回应排除的英雄 code 集合。
    只考虑我方已选（my_pick_codes），对手阵容不参与判断。
    """
    _log = log_fn or (lambda m: None)
    rules = _load_rules()
    if not rules:
        return set()

    my_names = {code_to_name.get(c, '') for c in my_pick_codes}
    excluded  = set()

    for rule in rules:
        group = rule.get('group', [])
        if not group:
            continue
        max_count = rule.get('max', len(group) - 1)

        picked_in_group = [gn for gn in group if _rule_name_in(gn, my_names)]
        if len(picked_in_group) < max_count:
            continue

        not_picked = [gn for gn in group if not _rule_name_in(gn, my_names)]
        if not not_picked:
            continue

        _log(f'  [互斥规则] 已选 {picked_in_group}，排除: {not_picked}')
        for gn in not_picked:
            gn_s = _n(gn)
            for code, name in code_to_name.items():
                if gn_s in _n(name) or _n(name) in gn_s:
                    excluded.add(code)

    return excluded


def _load_atk_priority() -> list:
    global _atk_priority
    if _atk_priority is None:
        if os.path.exists(_ATK_PRIORITY_FILE):
            with open(_ATK_PRIORITY_FILE, encoding='utf-8') as f:
                _atk_priority = json.load(f)
        else:
            _atk_priority = []
    return _atk_priority


def get_attack_priority() -> list:
    """返回攻击优先级英雄名列表（简体，按配置顺序）。"""
    return [_n(name) for name in _load_atk_priority()]


def resolve_attack_target(pos_map: dict, code_to_name: dict,
                          alive_slots: 'list | None' = None) -> 'int | None':
    """根据攻击优先级配置和站位识别结果，返回应攻击的槽位 index。

    pos_map:     detect_enemy_positions 的返回值 {slot_idx: code}
    code_to_name: code → 英雄名（简体）
    alive_slots: 当前存活的槽位列表，None 表示全部存活
    返回目标槽位 index，无匹配返回 None（调用方走默认逻辑）。
    """
    priority = get_attack_priority()
    if not priority or not pos_map:
        return None
    valid = set(alive_slots) if alive_slots is not None else set(pos_map.keys())
    for name in priority:
        for slot, code in pos_map.items():
            if slot not in valid:
                continue
            if _n(code_to_name.get(code, '')) == name:
                return slot
    return None
