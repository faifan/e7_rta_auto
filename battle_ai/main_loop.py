import time
from battle_ai.executor import focus_game_window, do_aoe, click_burn
from battle_ai.perception import (capture, is_battle_over, is_intimacy_levelup,
                                   read_turn_badge, read_char_name,
                                   is_soul_burn_available, is_soul_burn_activated)
from battle_ai.decision import (
    get_candidates, on_s3_success, on_s2_success, on_s2_fail,
    reset_battle, get_soul_burn_skill, get_extra_turn_skill,
    is_force_first_burn_pending, mark_force_first_burn_done,
    arm_force_first_burn, set_my_team, team_has_soul_burn,
    set_pending_extra_turn, get_pending_extra_turn, clear_pending_extra_turn,
)

POLL_INTERVAL       = 1.0   # 主循环轮询间隔（秒）
_SKILL_POLL_SEC     = 1.3   # 普通技能：单次等待后检测徽章（需覆盖技能动画延迟）
_BURN_POLL_SEC      = 2.0   # 烧魂技能：检测间隔（动画较长）
_BURN_MAX_POLLS     = 3     # 烧魂最多检测次数


def _execute_with_burn_try_all(turn: int, log_fn) -> bool:
    """首回合强制烧魂：激活一次burn后依次试 S3→S2→S1，哪个成功用哪个。"""
    click_burn()
    time.sleep(1.5)                    # 等动画+卡顿
    if not is_soul_burn_activated():   # OCR轮询最多3s
        log_fn(f"[回合 {turn}] 强制烧魂未激活")
        return False

    for attempt in range(2):
        for skill in ('S3', 'S2', 'S1'):
            do_aoe(skill)
            time.sleep(1.5)
            if read_turn_badge() != 'my_turn':
                log_fn(f"[回合 {turn}] 首回合 {skill} 烧魂 ✓（第{attempt+1}轮）")
                return True
            log_fn(f"[回合 {turn}] {skill} 无响应，尝试下一技能")

    return False


def _execute_with_burn(skill: str, char_name: str | None, turn: int, log_fn) -> str:
    """
    烧魂路径。返回：
      'success'    — 技能生效，回合结束
      'extra_turn' — 技能生效，且获得额外回合（badge仍在）
      'failed'     — 烧魂未激活或技能无响应
    """
    click_burn()
    time.sleep(1.5)
    if not is_soul_burn_activated():
        log_fn(f"[回合 {turn}] 烧魂未激活，退回普通流程")
        return 'failed'

    do_aoe(skill)
    for _ in range(_BURN_MAX_POLLS):
        time.sleep(_BURN_POLL_SEC)
        if read_turn_badge() != 'my_turn':
            log_fn(f"[回合 {turn}] {char_name or '?'} {skill} 烧魂 ✓")
            return 'success'

    if get_extra_turn_skill(char_name) == skill:
        log_fn(f"[回合 {turn}] {char_name or '?'} {skill} 烧魂 ✓ (额外回合)")
        return 'extra_turn'

    log_fn(f"[回合 {turn}] {char_name or '?'} {skill} 烧魂无响应")
    return 'failed'


def _execute_skill(skill: str, char_name: str | None, turn: int, log_fn) -> str:
    """
    普通技能路径。返回：
      'success'    — 技能生效，回合结束
      'extra_turn' — 技能生效，且获得额外回合（badge仍在）
      'failed'     — 技能无响应（被动/冷却）
    """
    do_aoe(skill)
    time.sleep(_SKILL_POLL_SEC)
    if read_turn_badge() != 'my_turn':
        # S3 动画可能 >1.3s，badge 短暂消失后再回来表示额外回合，补等一次
        if get_extra_turn_skill(char_name) == skill:
            time.sleep(1.5)
            if read_turn_badge() == 'my_turn':
                log_fn(f"[回合 {turn}] {char_name or '?'} {skill} ✓ (额外回合)")
                return 'extra_turn'
        log_fn(f"[回合 {turn}] {char_name or '?'} {skill} ✓")
        return 'success'
    log_fn(f"[回合 {turn}] {char_name or '?'} {skill} 无响应")
    return 'failed'


def run(stop_event=None, log_fn=None, arm_force_burn=False, my_team_names=None):
    _log = log_fn or print
    _log("切换到游戏窗口...")
    offset = focus_game_window()
    _log(f"[focus] 窗口偏移={offset}")
    reset_battle()
    if my_team_names:
        set_my_team(my_team_names)
        _log(f"己方阵容：{my_team_names}")
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

        if is_battle_over(img) or is_intimacy_levelup(img):
            time.sleep(1.0)
            if is_battle_over() or is_intimacy_levelup():
                _log(f"战斗结束！共行动 {turn} 次")
                break
            continue

        badge = read_turn_badge(img)

        if badge != 'my_turn':
            time.sleep(POLL_INTERVAL)
            continue

        # 每回合等1秒让烧魂按钮出现后再采样
        time.sleep(1.0)
        img = capture()

        # 采样烧魂帧（在0.5s确认延迟前，覆盖首回合快速闪烁窗口）
        _early_burn_avail = is_soul_burn_available(img)

        # 0.5秒二次确认，防止过渡帧误触
        time.sleep(0.5)
        img = capture()
        if read_turn_badge(img) != 'my_turn':
            continue

        turn += 1
        char_name = read_char_name(img)

        if is_battle_over(img) or is_intimacy_levelup(img):
            _log(f"战斗结束！共行动 {turn - 1} 次")
            break

        executed = False

        # ── Step 0: 额外回合处理 ──────────────────────────────────
        extra_turn_mode = get_pending_extra_turn(char_name)
        is_extra_turn   = extra_turn_mode is not None
        if is_extra_turn:
            clear_pending_extra_turn(char_name)

        if is_extra_turn and extra_turn_mode == 'force_burn':
            burn_avail = _early_burn_avail or is_soul_burn_available(img) or is_soul_burn_available()
            if burn_avail:
                _log(f"[回合 {turn}] 额外回合·强制烧魂（迪埃妮）")
                executed = _execute_with_burn_try_all(turn, _log)
            else:
                _log(f"[回合 {turn}] 额外回合·强制烧魂：按钮未检测到")

        elif is_extra_turn and extra_turn_mode == 'soul_burn':
            soul_burn_skill = get_soul_burn_skill(char_name)
            if soul_burn_skill:
                burn_avail = _early_burn_avail or is_soul_burn_available(img) or is_soul_burn_available()
                if burn_avail:
                    _log(f"[回合 {turn}] 额外回合·烧魂→{soul_burn_skill}")
                    result = _execute_with_burn(soul_burn_skill, char_name, turn, _log)
                    if result != 'failed':
                        if soul_burn_skill == 'S3':
                            on_s3_success(char_name)
                        executed = True
                        if result == 'extra_turn':
                            set_pending_extra_turn(char_name, 'normal')

        # extra_turn_mode == 'normal'：跳过Steps1-2，直接落到Step3普通技能

        # ── Step 1: 首回合强制烧魂（迪埃妮在队，非迪埃妮自己行动）──
        if not executed and not is_extra_turn and is_force_first_burn_pending():
            mark_force_first_burn_done()
            if '迪埃妮' not in (char_name or ''):
                extra_turn_skill  = get_extra_turn_skill(char_name)
                soul_burn_skill_c = get_soul_burn_skill(char_name)
                burn_avail = _early_burn_avail or is_soul_burn_available(img) or is_soul_burn_available()

                if extra_turn_skill and burn_avail:
                    if extra_turn_skill != soul_burn_skill_c:
                        # Case A: 先放额外回合技能（不烧魂），额外回合再强制烧魂
                        _log(f"[回合 {turn}] 首回合：先放 {extra_turn_skill}，额外回合后烧魂")
                        result = _execute_skill(extra_turn_skill, char_name, turn, _log)
                        if result == 'extra_turn':
                            set_pending_extra_turn(char_name, 'force_burn')
                            if extra_turn_skill == 'S3':
                                on_s3_success(char_name)
                            executed = True
                        elif result == 'success':
                            executed = True  # 成功但无额外回合，烧魂时机丢失
                    else:
                        # Case B: extra_turn == soul_burn，直接烧魂（强化+额外回合）
                        _log(f"[回合 {turn}] 首回合：烧魂额外回合技能 {extra_turn_skill}")
                        result = _execute_with_burn(extra_turn_skill, char_name, turn, _log)
                        if result != 'failed':
                            if extra_turn_skill == 'S3':
                                on_s3_success(char_name)
                            executed = True
                            if result == 'extra_turn':
                                set_pending_extra_turn(char_name, 'normal')

                if not executed:
                    # Case C: 无extra_turn配置或上述路径失败 → 原逻辑
                    if burn_avail:
                        _log(f"[回合 {turn}] 首回合强制烧魂（迪埃妮在队）")
                        executed = _execute_with_burn_try_all(turn, _log)
                    else:
                        _log(f"[回合 {turn}] 首回合强制烧魂：未检测到烧魂按钮")

        # 候选列表只算一次（get_candidates 有副作用：递减 s3_skip）
        _candidates   = get_candidates(char_name)
        _et_skill     = get_extra_turn_skill(char_name)
        _et_available = bool(_et_skill and _et_skill in _candidates)

        # ── Step 2: 普通烧魂 ─────────────────────────────────────
        # 若角色配置了 extra_turn 技能且该技能当前可用，跳过烧魂留给额外回合
        if not executed and not is_extra_turn and not _et_available:
            soul_burn_skill = get_soul_burn_skill(char_name)
            if soul_burn_skill:
                burn_avail = _early_burn_avail or is_soul_burn_available(img) or is_soul_burn_available()
                if burn_avail:
                    _log(f"[回合 {turn}] 角色={char_name or '未知'} 烧魂→{soul_burn_skill}")
                    result = _execute_with_burn(soul_burn_skill, char_name, turn, _log)
                    if result != 'failed':
                        if soul_burn_skill == 'S3':
                            on_s3_success(char_name)
                        executed = True
                        if result == 'extra_turn':
                            set_pending_extra_turn(char_name, 'normal')
                else:
                    _log(f"[回合 {turn}] 角色={char_name or '未知'} 烧魂按钮未检测到")
            elif not team_has_soul_burn():
                # 队伍无烧魂配置 → 激活一次烧魂，逐个试 S3→S2→S1
                burn_avail = _early_burn_avail or is_soul_burn_available(img) or is_soul_burn_available()
                if burn_avail:
                    _log(f"[回合 {turn}] {char_name or '未知'} 通用烧魂（逐个试）")
                    executed = _execute_with_burn_try_all(turn, _log)

        # ── Step 3: 普通技能 ─────────────────────────────────────
        if not executed:
            # 额外回合(normal模式)时跳过 extra_turn 技能，防止循环触发额外回合
            if is_extra_turn and extra_turn_mode == 'normal' and _et_skill:
                cands = [s for s in _candidates if s != _et_skill]
            else:
                cands = _candidates
            _log(f"[回合 {turn}] 角色={char_name or '未知'} 候选={cands}")
            for skill in cands:
                result = _execute_skill(skill, char_name, turn, _log)
                if result != 'failed':
                    if skill == 'S3':
                        on_s3_success(char_name)
                    if skill == 'S2':
                        on_s2_success(char_name)
                    executed = True
                    if result == 'extra_turn':
                        sb = get_soul_burn_skill(char_name)
                        if sb and sb != skill:
                            set_pending_extra_turn(char_name, 'soul_burn')
                        else:
                            set_pending_extra_turn(char_name, 'normal')
                    break
                if skill == 'S2':
                    on_s2_fail(char_name)

        # ── Step 4: 兜底 ─────────────────────────────────────────
        if not executed:
            _log(f"[回合 {turn}] 兜底S1")
            do_aoe('S1')
            time.sleep(_SKILL_POLL_SEC)
