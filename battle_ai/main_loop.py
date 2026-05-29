import time
from battle_ai.executor import focus_game_window, do_aoe, do_action, click_burn
from battle_ai.perception import (capture, is_battle_over, is_intimacy_levelup,
                                   read_turn_badge, read_char_name,
                                   is_soul_burn_available,
                                   get_enemy_hp_ratios)
from battle_ai.lobby import is_in_lobby, is_waiting_for_match
from battle_ai.decision import (
    get_candidates, on_s3_success, on_s2_success, on_s2_fail,
    reset_battle, get_soul_burn_skill, get_extra_turn_skill,
    is_force_first_burn_pending, mark_force_first_burn_done,
    arm_force_first_burn, set_my_team,
    get_burn_timing, is_first_action_done, mark_first_action_done,
    get_skill_type, get_attack_target,
    set_pending_extra_turn, get_pending_extra_turn, clear_pending_extra_turn,
)

POLL_INTERVAL       = 1.0   # 主循环轮询间隔（秒）
_SKILL_POLL_SEC     = 1.3   # 普通技能：单次等待后检测徽章（需覆盖技能动画延迟）
_BURN_POLL_SEC      = 2.0   # 烧魂技能：检测间隔（动画较长）
_BURN_MAX_POLLS     = 3     # 烧魂最多检测次数



def _do_skill(skill: str, target_idx: int | None):
    """统一技能点击：单体用 do_action，群体用 do_aoe。"""
    if target_idx is not None:
        do_action(skill, target_idx)
    else:
        do_aoe(skill)


def _execute_with_burn(skill: str, char_name: str | None, turn: int, log_fn,
                       target_idx: int | None = None) -> str:
    """
    烧魂路径。返回：
      'success'    — 技能生效，回合结束
      'extra_turn' — 技能生效，且获得额外回合（badge仍在）
      'failed'     — 烧魂未激活或技能无响应
    """
    click_burn()
    time.sleep(0.3)   # 短暂等待点击生效
    _do_skill(skill, target_idx)
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


def _execute_skill(skill: str, char_name: str | None, turn: int, log_fn,
                   target_idx: int | None = None) -> str:
    """
    普通技能路径。返回：
      'success'    — 技能生效，回合结束
      'extra_turn' — 技能生效，且获得额外回合（badge仍在）
      'failed'     — 技能无响应（被动/冷却）
    """
    _do_skill(skill, target_idx)
    time.sleep(_SKILL_POLL_SEC)
    _is_et_skill = (get_extra_turn_skill(char_name) == skill)
    # S3释放后动画较长，最多等4s（1.3+2.7）再判
    if read_turn_badge() == 'my_turn' and skill == 'S3':
        time.sleep(2.7)
    if read_turn_badge() != 'my_turn':
        if _is_et_skill:
            time.sleep(1.5)
            if read_turn_badge() == 'my_turn':
                log_fn(f"[回合 {turn}] {char_name or '?'} {skill} ✓ (额外回合)")
                return 'extra_turn'
        log_fn(f"[回合 {turn}] {char_name or '?'} {skill} ✓")
        return 'success'
    log_fn(f"[回合 {turn}] {char_name or '?'} {skill} 无响应")
    return 'failed'



def run(stop_event=None, log_fn=None, arm_force_burn=False, my_team_names=None,
        enemy_has_yazuga=False):
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

        if is_in_lobby(img) or is_waiting_for_match(img):
            _log(f"检测到大厅，战斗已结束（共行动 {turn} 次）")
            break

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

        # 烧魂按钮可用性：早采样 + 确认帧，算一次全程复用
        _burn_avail = _early_burn_avail or is_soul_burn_available(img)

        # 候选列表只算一次（get_candidates 有副作用：递减 s3_skip）
        _candidates   = get_candidates(char_name)

        # 单体目标：血量只读一次，整回合复用
        _hp_ratios = get_enemy_hp_ratios(img)
        try:
            from config_loader import cfg
            _front_row = cfg.section('executor').get('front_row_indices', [2, 3])
        except Exception:
            _front_row = [2, 3]

        def _tgt(skill: str) -> 'int | None':
            if get_skill_type(char_name, skill) != 'single':
                return None
            if not _hp_ratios:
                return 0
            return get_attack_target(_hp_ratios, enemy_has_yazuga, _front_row)
        _et_skill     = get_extra_turn_skill(char_name)
        _et_available = bool(_et_skill and _et_skill in _candidates)

        # ── Step 0: 额外回合处理 ──────────────────────────────────
        extra_turn_mode = get_pending_extra_turn(char_name)
        is_extra_turn   = extra_turn_mode is not None
        if is_extra_turn:
            clear_pending_extra_turn(char_name)

        if is_extra_turn and extra_turn_mode == 'force_burn':
            soul_burn_skill = get_soul_burn_skill(char_name)
            burn_avail = _burn_avail
            if soul_burn_skill and burn_avail:
                _log(f"[回合 {turn}] 额外回合·强制烧魂→{soul_burn_skill}")
                result = _execute_with_burn(soul_burn_skill, char_name, turn, _log, target_idx=_tgt(soul_burn_skill))
                if result != 'failed':
                    if soul_burn_skill == 'S3':
                        on_s3_success(char_name)
                    executed = True
                    if result == 'extra_turn':
                        set_pending_extra_turn(char_name, 'normal')
            else:
                _log(f"[回合 {turn}] 额外回合·强制烧魂：{'无烧魂配置' if not soul_burn_skill else '按钮未检测到'}")

        elif is_extra_turn and extra_turn_mode == 'soul_burn':
            soul_burn_skill = get_soul_burn_skill(char_name)
            if soul_burn_skill:
                burn_avail = _burn_avail
                if burn_avail:
                    _log(f"[回合 {turn}] 额外回合·烧魂→{soul_burn_skill}")
                    result = _execute_with_burn(soul_burn_skill, char_name, turn, _log, target_idx=_tgt(soul_burn_skill))
                    if result != 'failed':
                        if soul_burn_skill == 'S3':
                            on_s3_success(char_name)
                        executed = True
                        if result == 'extra_turn':
                            set_pending_extra_turn(char_name, 'normal')

        # extra_turn_mode == 'normal'：跳过 Step1/2，直接落到 Step3 普通技能

        # ── Step 1: 团队强制烧魂（迪埃妮在队，非迪埃妮自己行动）──
        if not executed and not is_extra_turn and is_force_first_burn_pending():
            mark_force_first_burn_done()
            if '迪埃妮' not in (char_name or ''):
                soul_burn_skill = get_soul_burn_skill(char_name)
                burn_avail = _burn_avail

                if soul_burn_skill and burn_avail:
                    if _et_available and _et_skill != soul_burn_skill:
                        # Case A: 先放额外回合技能，额外回合里烧魂
                        _log(f"[回合 {turn}] 团队强制：先放{_et_skill}，额外回合烧{soul_burn_skill}")
                        result = _execute_skill(_et_skill, char_name, turn, _log, target_idx=_tgt(_et_skill))
                        if result != 'failed':
                            if _et_skill == 'S3':
                                on_s3_success(char_name)
                            if result == 'extra_turn':
                                set_pending_extra_turn(char_name, 'force_burn')
                        executed = (result != 'failed')
                    else:
                        # Case B/C: 直接烧魂（含 extra_turn==soul_burn 或无 extra_turn）
                        _log(f"[回合 {turn}] 团队强制：烧魂→{soul_burn_skill}")
                        result = _execute_with_burn(soul_burn_skill, char_name, turn, _log, target_idx=_tgt(soul_burn_skill))
                        if result != 'failed':
                            if soul_burn_skill == 'S3':
                                on_s3_success(char_name)
                            executed = True
                            if result == 'extra_turn':
                                set_pending_extra_turn(char_name, 'normal')
                else:
                    _log(f"[回合 {turn}] 团队强制烧魂：{'无烧魂配置' if not soul_burn_skill else '按钮未检测到'}，走普通技能")

        # ── Step 2: 角色自身烧魂逻辑 ─────────────────────────────
        if not executed and not is_extra_turn:
            soul_burn_skill = get_soul_burn_skill(char_name)
            burn_timing     = get_burn_timing(char_name)

            if burn_timing == 'first' and not is_first_action_done(char_name):
                mark_first_action_done(char_name)
                if soul_burn_skill:
                    burn_avail = _burn_avail
                    if burn_avail:
                        if _et_available and _et_skill != soul_burn_skill:
                            # 先额外回合，额外回合再烧
                            _log(f"[回合 {turn}] 首次：先放{_et_skill}，额外回合烧{soul_burn_skill}")
                            result = _execute_skill(_et_skill, char_name, turn, _log, target_idx=_tgt(_et_skill))
                            if result != 'failed':
                                if _et_skill == 'S3':
                                    on_s3_success(char_name)
                                if result == 'extra_turn':
                                    set_pending_extra_turn(char_name, 'soul_burn')
                            executed = (result != 'failed')
                        else:
                            # 直接烧（含 extra_turn==soul_burn 或无 extra_turn）
                            _log(f"[回合 {turn}] 首次：烧魂→{soul_burn_skill}")
                            result = _execute_with_burn(soul_burn_skill, char_name, turn, _log, target_idx=_tgt(soul_burn_skill))
                            if result != 'failed':
                                if soul_burn_skill == 'S3':
                                    on_s3_success(char_name)
                                executed = True
                                if result == 'extra_turn':
                                    set_pending_extra_turn(char_name, 'normal')
                    # burn不可用：fall through 到 Step3 普通技能

            elif burn_timing == 'second' and not is_first_action_done(char_name):
                # 首次行动不烧，标记后走 Step3
                mark_first_action_done(char_name)

            elif soul_burn_skill:
                # 正常烧魂逻辑（无 burn_timing，或 'second' 首次已完成）
                # S3冷却好了且烧魂目标不是S3：先让Step3放S3，不抢烧魂
                s3_ready = 'S3' in _candidates and soul_burn_skill != 'S3'
                burn_avail = _burn_avail and not s3_ready
                if burn_avail:
                    if _et_available and _et_skill == soul_burn_skill:
                        # extra_turn == soul_burn：直接烧（修复原BUG）
                        _log(f"[回合 {turn}] 烧魂→{soul_burn_skill}（强化+额外回合）")
                        result = _execute_with_burn(soul_burn_skill, char_name, turn, _log, target_idx=_tgt(soul_burn_skill))
                        if result != 'failed':
                            if soul_burn_skill == 'S3':
                                on_s3_success(char_name)
                            executed = True
                            if result == 'extra_turn':
                                set_pending_extra_turn(char_name, 'normal')
                    elif _et_available and _et_skill != soul_burn_skill:
                        # 先放额外回合技能，额外回合再烧
                        _log(f"[回合 {turn}] 先放{_et_skill}，额外回合烧{soul_burn_skill}")
                        result = _execute_skill(_et_skill, char_name, turn, _log, target_idx=_tgt(_et_skill))
                        if result != 'failed':
                            if _et_skill == 'S3':
                                on_s3_success(char_name)
                            if result == 'extra_turn':
                                set_pending_extra_turn(char_name, 'soul_burn')
                        executed = (result != 'failed')
                    else:
                        # 无额外回合竞争，直接烧
                        _log(f"[回合 {turn}] 烧魂→{soul_burn_skill}")
                        result = _execute_with_burn(soul_burn_skill, char_name, turn, _log, target_idx=_tgt(soul_burn_skill))
                        if result != 'failed':
                            if soul_burn_skill == 'S3':
                                on_s3_success(char_name)
                            executed = True
                            if result == 'extra_turn':
                                set_pending_extra_turn(char_name, 'normal')

        # ── Step 3: 普通技能 ─────────────────────────────────────
        if not executed:
            if is_extra_turn and extra_turn_mode == 'normal' and _et_skill:
                cands = [s for s in _candidates if s != _et_skill]
            else:
                cands = _candidates
            _log(f"[回合 {turn}] 角色={char_name or '未知'} 候选={cands}")
            for skill in cands:
                result = _execute_skill(skill, char_name, turn, _log, target_idx=_tgt(skill))
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
            _do_skill('S1', _tgt('S1'))
            time.sleep(_SKILL_POLL_SEC)
