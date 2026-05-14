import time
from battle_ai.executor import focus_game_window, do_aoe
from battle_ai.perception import capture, is_battle_over, read_turn_badge, read_char_name
from battle_ai.decision import get_candidates, on_s3_success, reset_battle

POLL_INTERVAL       = 1.0   # 主循环轮询间隔（秒）
_SKILL_POLL_SEC     = 2.0   # 技能成功检测间隔
_SKILL_MAX_POLLS    = 3     # 最多检测3次


def _execute_skill(skill: str, char_name: str | None, turn: int, log_fn) -> bool:
    """
    点击技能后轮询中央徽章：
      - 文字消失('none') 或 变为'enemy_turn' → 触发成功，返回True
      - 3次仍为'my_turn' → 无响应，返回False
    """
    do_aoe(skill)
    for _ in range(_SKILL_MAX_POLLS):
        time.sleep(_SKILL_POLL_SEC)
        badge = read_turn_badge()
        if badge != 'my_turn':
            log_fn(f"[回合 {turn}] {char_name or '?'} {skill} ✓")
            return True
    log_fn(f"[回合 {turn}] {char_name or '?'} {skill} 无响应")
    return False


def run(stop_event=None, log_fn=None):
    _log = log_fn or print
    _log("切换到游戏窗口...")
    focus_game_window()
    reset_battle()
    _log("开始战斗循环")

    turn = 0
    while True:
        if stop_event and stop_event.is_set():
            _log("战斗AI：收到停止信号，退出")
            break

        img = capture()

        if is_battle_over(img):
            time.sleep(1.0)
            if is_battle_over():
                _log(f"战斗结束！共行动 {turn} 次")
                break
            continue

        badge = read_turn_badge(img)

        if badge != 'my_turn':
            time.sleep(POLL_INTERVAL)
            continue

        # 0.5秒二次确认，防止过渡帧误触
        time.sleep(0.5)
        img = capture()
        if read_turn_badge(img) != 'my_turn':
            continue

        turn += 1
        char_name = read_char_name(img)
        candidates = get_candidates(char_name, img)
        _log(f"[回合 {turn}] 角色={char_name or '未知'} 候选={candidates}")

        if is_battle_over(img):
            _log(f"战斗结束！共行动 {turn - 1} 次")
            break

        executed = False
        for skill in candidates:
            success = _execute_skill(skill, char_name, turn, _log)
            if success:
                if skill == 'S3':
                    on_s3_success(char_name)
                executed = True
                break
            # S3失败（说明被对手加了CD）→ 跳过S2直接兜底S1
            if skill == 'S3':
                _log(f"[回合 {turn}] S3无响应，跳过S2")
                break

        if not executed:
            _log(f"[回合 {turn}] 兜底S1")
            do_aoe('S1')
            time.sleep(_SKILL_POLL_SEC)
