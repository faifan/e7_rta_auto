"""本地对局统计：保存每局数据，计算调整系数干预AI推荐。"""
import json
import math
import os
from datetime import datetime, timedelta

_ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR   = os.path.join(_ROOT, 'data')
_STATS_FILE = os.path.join(_DATA_DIR, 'bot_games.jsonl')

_PRIOR       = 5     # 贝叶斯先验（等效5赢5负起步）
_HALFLIFE    = 15    # 置信度半衰期：出场15局时置信度约63%
_WINDOW_DAYS = 30    # 只统计最近30天
_MIN_GAMES   = 3     # 低于此出场数不干扰模型
_MAX_ADJ     = 1.8   # 最大放大上限
_MIN_ADJ     = 0.4   # 最大压缩下限


def save_game(draft_result: dict, iswin: int,
              finalban_code: str = None, enemy_finalban_code: str = None):
    """
    追加一局完整记录到 bot_games.jsonl。
    iswin: 1=赢，2=输（与训练数据一致）。
    my_picks 或 enemy_picks 含 unknown 时跳过，避免污染统计。
    """
    my_picks    = draft_result.get('my_picks',    [])
    enemy_picks = draft_result.get('enemy_picks', [])
    if 'unknown' in my_picks or 'unknown' in enemy_picks:
        return

    os.makedirs(_DATA_DIR, exist_ok=True)
    record = {
        'ts':             datetime.now().isoformat(timespec='seconds'),
        'iswin':          iswin,
        'my_picks':       my_picks,
        'enemy_picks':    enemy_picks,
        'my_preban':      draft_result.get('my_preban',    []),
        'enemy_preban':   draft_result.get('enemy_preban', []),
        'my_first':       draft_result.get('my_first',     True),
        'finalban':       finalban_code,
        'enemy_finalban': enemy_finalban_code,
    }
    with open(_STATS_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')


def _load_recent() -> list:
    """读取最近 _WINDOW_DAYS 天的对局记录。"""
    if not os.path.exists(_STATS_FILE):
        return []
    cutoff = datetime.now() - timedelta(days=_WINDOW_DAYS)
    games = []
    with open(_STATS_FILE, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if datetime.fromisoformat(rec['ts']) >= cutoff:
                    games.append(rec)
            except Exception:
                pass
    return games


def _bayesian_wr(wins: int, games: int) -> float:
    return (_PRIOR + wins) / (2 * _PRIOR + games)


def _confidence(games: int) -> float:
    return 1.0 - math.exp(-games / _HALFLIFE)


def _clamp(v: float) -> float:
    return max(_MIN_ADJ, min(_MAX_ADJ, v))


def get_pick_adjustments(candidate_codes: list) -> dict:
    """
    计算候选英雄的pick调整系数。
    返回 {code: factor}，factor>1加分，<1减分，不含系数的英雄数据不足，不干扰。
    """
    games = _load_recent()
    if not games:
        return {}

    total_wins = sum(1 for g in games if g.get('iswin') == 1)
    global_wr  = _bayesian_wr(total_wins, len(games))

    result = {}
    for code in candidate_codes:
        with_hero = [g for g in games if code in g.get('my_picks', [])]
        n = len(with_hero)
        if n < _MIN_GAMES:
            continue
        wins      = sum(1 for g in with_hero if g.get('iswin') == 1)
        local_wr  = _bayesian_wr(wins, n)
        conf      = _confidence(n)
        ratio     = local_wr / global_wr if global_wr > 0 else 1.0
        result[code] = _clamp(1.0 + conf * (ratio - 1.0))

    return result


def get_finalban_adjustments(enemy_codes: list) -> dict:
    """
    计算对手英雄的finalban威胁调整系数。
    返回 {code: factor}，factor>1表示遇到该英雄时胜率偏低，应优先禁。
    """
    games = _load_recent()
    if not games:
        return {}

    total_losses = sum(1 for g in games if g.get('iswin') == 2)
    global_loss  = _bayesian_wr(total_losses, len(games))

    result = {}
    for code in enemy_codes:
        vs_hero = [g for g in games if code in g.get('enemy_picks', [])]
        n = len(vs_hero)
        if n < _MIN_GAMES:
            continue
        losses     = sum(1 for g in vs_hero if g.get('iswin') == 2)
        local_loss = _bayesian_wr(losses, n)
        conf       = _confidence(n)
        ratio      = local_loss / global_loss if global_loss > 0 else 1.0
        result[code] = _clamp(1.0 + conf * (ratio - 1.0))

    return result
