import time
from battle_ai.executor import focus_game_window, do_aoe, click_burn
from battle_ai.perception import (capture, is_battle_over, read_turn_badge, read_char_name,
                                   is_soul_burn_available, is_soul_burn_activated)
from battle_ai.decision import (get_candidates, on_s3_success, reset_battle, get_soul_burn_skill,
                                 is_force_first_burn_pending, mark_force_first_burn_done,
                                 arm_force_first_burn)

POLL_INTERVAL       = 1.0   # 主循环轮询间隔（秒）
_SKILL_POLL_SEC     = 2.0   # 技能成功检测间隔
_SKILL_MAX_POLLS    = 3     # 最多检测3次


def _execute_with_burn_try_all(turn: int, log_fn) -> bool:
    """首回合强制烧魂：激活一次burn后依次试 S3→S2→S1，哪个成功用哪个。"""
    click_burn()
    time.sleep(1.5)                    # 等动画+卡顿
    if not is_soul_burn_activated():   # OCR轮询最多3s
        log_fn(f"[回合 {turn}] 强制烧魂未激活")
        return False

    for skill in ('S3', 'S2', 'S1'):
        do_aoe(skill)
        for _ in range(_SKILL_MAX_POLLS):
            time.sleep(_SKILL_POLL_SEC)
            if read_turn_badge() != 'my_turn':
                log_fn(f"[回合 {turn}] 首回合 {skill} 烧魂 ✓")
                return True
        log_fn(f"[回合 {turn}] {skill} 无响应，尝试下一技能")

    return False


def _execute_with_burn(skill: str, char_name: str | None, turn: int, log_fn) -> bool:
    """
    烧魂路径：click_burn → 验证Cancel激活 → 双击技能 → 轮询徽章。
    激活验证失败最多重试一次，仍失败则返回False退回普通流程。
    """
    click_burn()
    time.sleep(1.5)                    # 等动画+卡顿
    if not is_soul_burn_activated():   # OCR轮询最多3s
        log_fn(f"[回合 {turn}] 烧魂未激活，退回普通流程")
        return False

    do_aoe(skill)
    for _ in range(_SKILL_MAX_POLLS):
        time.sleep(_SKILL_POLL_SEC)
        badge = read_turn_badge()
        if badge != 'my_turn':
            log_fn(f"[回合 {turn}] {char_name or '?'} {skill} 烧魂 ✓")
            return True
    log_fn(f"[回合 {turn}] {char_name or '?'} {skill} 烧魂无响应")
    return False


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


def run(stop_event=None, log_fn=None, arm_force_burn=False):
    _log = log_fn or print
    _log("切换到游戏窗口...")
    focus_game_window()
    reset_battle()
    if arm_force_burn:
        arm_force_first_burn()
        _log("迪埃妮首回合烧魂已预装")
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

        # 立刻采样烧魂帧（在0.5s确认延迟前，覆盖首回合快速闪烁窗口）
        _early_burn_avail = is_soul_burn_available(img)

        # 0.5秒二次确认，防止过渡帧误触
        time.sleep(0.5)
        img = capture()
        if read_turn_badge(img) != 'my_turn':
            continue

        turn += 1
        char_name = read_char_name(img)

        if is_battle_over(img):
            _log(f"战斗结束！共行动 {turn - 1} 次")
            break

        executed = False

        # 首回合强制烧魂（队伍含迪埃妮时，第一个我方回合无论谁行动）
        # 迪埃妮自己不触发（她是触发方不是施法方）
        if is_force_first_burn_pending():
            mark_force_first_burn_done()
            if '迪埃妮' not in (char_name or ''):
                burn_avail = _early_burn_avail or is_soul_burn_available(img)
                if not burn_avail:
                    burn_avail = is_soul_burn_available()
                if burn_avail:
                    _log(f"[回合 {turn}] 首回合强制烧魂（迪埃妮在队）")
                    executed = _execute_with_burn_try_all(turn, _log)
                else:
                    _log(f"[回合 {turn}] 首回合强制烧魂：未检测到烧魂按钮")

        # 烧魂优先：仅对配置了soul_burn的角色检测（迪埃妮在decision.py层已屏蔽）
        soul_burn_skill = get_soul_burn_skill(char_name)

        if soul_burn_skill and not executed:
            burn_avail = _early_burn_avail or is_soul_burn_available(img)
            if not burn_avail:
                burn_avail = is_soul_burn_available()
            if burn_avail:
                _log(f"[回合 {turn}] 角色={char_name or '未知'} 烧魂→{soul_burn_skill}")
                success = _execute_with_burn(soul_burn_skill, char_name, turn, _log)
                if success:
                    if soul_burn_skill == 'S3':
                        on_s3_success(char_name)
                    executed = True
            else:
                _log(f"[回合 {turn}] 角色={char_name or '未知'} 烧魂按钮未检测到")

        if not executed:
            candidates = get_candidates(char_name, img)
            _log(f"[回合 {turn}] 角色={char_name or '未知'} 候选={candidates}")
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
