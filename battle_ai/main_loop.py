import time
import random
from battle_ai.executor import focus_game_window, do_action, do_aoe, ENEMY_POS
from battle_ai.perception import (capture, is_my_turn, is_battle_over,
                                   img_similarity, skill_area_unchanged)
from battle_ai.decision import get_candidates

_STATIC_THRESHOLD = 0.95   # 全屏相似度>=此值认为技能未生效
POLL_INTERVAL = 1.0


def _execute_skill(skill, stype, turn, log_fn):
    """
    尝试执行一个技能。
    返回 True=成功发动，False=未响应（被动或漏检冷却）。
    """
    target = random.randint(0, len(ENEMY_POS) - 1)
    img_before = capture()

    if stype == '群体':
        do_aoe(skill)
        time.sleep(5.0)
        img_after = capture()
        if img_similarity(img_before, img_after) < _STATIC_THRESHOLD:
            log_fn(f"[回合 {turn}] {skill} 群体 ✓")
            return True
        log_fn(f"[回合 {turn}] {skill} 群体无响应（被动）")
        return False

    elif stype == '单体':
        do_action(skill, target)
        time.sleep(3.0)
        img_after = capture()
        if img_similarity(img_before, img_after) < _STATIC_THRESHOLD:
            log_fn(f"[回合 {turn}] {skill} 单体→目标{target} ✓")
            return True
        log_fn(f"[回合 {turn}] {skill} 单体无响应（被动）")
        return False

    else:  # 自动：先试单体，技能按钮无变化则直接判定被动
        do_action(skill, target)
        time.sleep(1.5)
        img_mid = capture()

        # 技能按钮区域未变化 → 被动，跳过（省去后续AOE等待）
        if skill_area_unchanged(img_before, img_mid, skill):
            log_fn(f"[回合 {turn}] {skill} 按钮无响应（被动），跳过")
            return False

        # 全屏有变化 → 单体命中
        if img_similarity(img_before, img_mid) < _STATIC_THRESHOLD:
            log_fn(f"[回合 {turn}] {skill} 自动→单体成功（相似度{img_similarity(img_before, img_mid):.3f}）✓")
            time.sleep(2.0)
            return True

        # 按钮有变化但全屏无变化 → 技能选中但需AOE双击
        log_fn(f"[回合 {turn}] {skill} 自动→单点无效，改AOE")
        do_aoe(skill)
        time.sleep(4.0)
        img_after = capture()
        if img_similarity(img_before, img_after) < _STATIC_THRESHOLD:
            log_fn(f"[回合 {turn}] {skill} 自动→AOE成功 ✓")
            return True
        log_fn(f"[回合 {turn}] {skill} AOE也无响应")
        return False


def run(stop_event=None, log_fn=None):
    _log = log_fn or print
    _log("切换到游戏窗口...")
    focus_game_window()
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

        if is_my_turn(img):
            time.sleep(0.5)
            if not is_my_turn():
                continue

            turn += 1
            candidates = get_candidates(img)

            if is_battle_over():
                _log(f"战斗结束！共行动 {turn - 1} 次")
                break

            executed = False
            for skill, stype in candidates:
                if _execute_skill(skill, stype, turn, _log):
                    executed = True
                    break

            if not executed:
                # 所有候选均失败，兜底强制S1单体
                _log(f"[回合 {turn}] 所有候选技能无响应，兜底S1")
                do_action('S1', random.randint(0, len(ENEMY_POS) - 1))
                time.sleep(3.0)
        else:
            time.sleep(POLL_INTERVAL)
