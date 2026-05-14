"""
第七史诗 全自动RTA 主控界面
运行此文件启动 GUI，选择窗口和语言后点击"开始"启动自动化。
"""
import os
import sys
import time
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext
import queue

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

LOG_FILE = os.path.join(_HERE, 'run.log')
_log_fh  = open(LOG_FILE, 'w', encoding='utf-8', buffering=1)

MODEL_PATH     = os.path.join(_HERE, 'draft_transformer.pth')
HERO_LIST_PATH = os.path.join(_HERE, 'hero_list_146.json')

CN_FONT  = ('Microsoft YaHei', 10)
CN_BOLD  = ('Microsoft YaHei', 13, 'bold')
CN_SMALL = ('Microsoft YaHei', 9)
MONO     = ('Consolas', 9)


class AutoRunApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("第七史诗 全自动RTA")
        self.root.configure(bg='#1e1e1e')
        self.root.geometry('720x600')
        self.root.resizable(True, True)

        self._stop_event  = threading.Event()
        self._thread      = None
        self._log_queue   = queue.Queue()
        self._recommender = None

        self._build_ui()
        self._poll_log()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI ───────────────────────────────────────────────────
    def _build_ui(self):
        tk.Label(self.root, text='第七史诗 全自动RTA',
                 font=CN_BOLD, fg='#61dafb', bg='#1e1e1e').pack(pady=(14, 4))

        # ── 配置区 ────────────────────────────────────────────
        cfg_frame = tk.LabelFrame(self.root, text=' 启动配置 ',
                                  font=CN_SMALL, fg='#888', bg='#1e1e1e',
                                  labelanchor='nw')
        cfg_frame.pack(fill=tk.X, padx=14, pady=(0, 6))

        # 窗口选择行
        row1 = tk.Frame(cfg_frame, bg='#1e1e1e')
        row1.pack(fill=tk.X, padx=8, pady=(6, 2))
        tk.Label(row1, text='游戏窗口:', font=CN_FONT,
                 fg='#c9d1d9', bg='#1e1e1e', width=8, anchor='e').pack(side=tk.LEFT)

        self._window_var = tk.StringVar()
        self._window_cb  = ttk.Combobox(row1, textvariable=self._window_var,
                                        font=CN_FONT, width=38, state='readonly')
        self._window_cb.pack(side=tk.LEFT, padx=(4, 4))

        tk.Button(row1, text='刷新', font=CN_SMALL,
                  bg='#30363d', fg='#c9d1d9', relief=tk.FLAT,
                  cursor='hand2', command=self._refresh_windows).pack(side=tk.LEFT)

        # 语言 + 坐标方案行
        row2 = tk.Frame(cfg_frame, bg='#1e1e1e')
        row2.pack(fill=tk.X, padx=8, pady=(2, 6))

        tk.Label(row2, text='语言:', font=CN_FONT,
                 fg='#c9d1d9', bg='#1e1e1e', width=8, anchor='e').pack(side=tk.LEFT)
        self._lang_var = tk.StringVar(value='简体中文')
        self._lang_cb  = ttk.Combobox(row2, textvariable=self._lang_var,
                                      font=CN_FONT, width=14, state='readonly')
        self._lang_cb.pack(side=tk.LEFT, padx=(4, 16))

        tk.Label(row2, text='分辨率:', font=CN_FONT,
                 fg='#c9d1d9', bg='#1e1e1e').pack(side=tk.LEFT)
        self._profile_var = tk.StringVar(value='1922x1115.json')
        self._profile_cb  = ttk.Combobox(row2, textvariable=self._profile_var,
                                         font=CN_FONT, width=18, state='readonly')
        self._profile_cb.pack(side=tk.LEFT, padx=(4, 0))

        # 填充下拉选项
        self._refresh_windows()
        self._refresh_lang_profile()

        # ── 状态行 ────────────────────────────────────────────
        self._status_var = tk.StringVar(value='就绪 — 选择窗口后点击"开始"')
        self._status_lbl = tk.Label(self.root, textvariable=self._status_var,
                                    font=CN_FONT, fg='#888888', bg='#1e1e1e')
        self._status_lbl.pack()

        # ── 按钮行 ────────────────────────────────────────────
        btn_frame = tk.Frame(self.root, bg='#1e1e1e')
        btn_frame.pack(pady=8)

        self._start_btn = tk.Button(
            btn_frame, text='▶  开始', font=CN_FONT,
            bg='#2ea043', fg='white', width=14, height=2,
            relief=tk.FLAT, cursor='hand2', command=self._on_start)
        self._start_btn.pack(side=tk.LEFT, padx=12)

        self._stop_btn = tk.Button(
            btn_frame, text='⏹  停止', font=CN_FONT,
            bg='#da3633', fg='white', width=14, height=2,
            relief=tk.FLAT, cursor='hand2', state=tk.DISABLED,
            command=self._on_stop)
        self._stop_btn.pack(side=tk.LEFT, padx=12)

        # ── 日志区 ────────────────────────────────────────────
        log_frame = tk.Frame(self.root, bg='#1e1e1e')
        log_frame.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 14))
        tk.Label(log_frame, text='运行日志', font=CN_FONT,
                 fg='#555', bg='#1e1e1e').pack(anchor=tk.W)

        self._log = scrolledtext.ScrolledText(
            log_frame, font=MONO, bg='#0d1117', fg='#c9d1d9',
            insertbackground='white', state=tk.DISABLED, wrap=tk.WORD)
        self._log.pack(fill=tk.BOTH, expand=True)

        self._log.tag_config('info',  foreground='#c9d1d9')
        self._log.tag_config('ok',    foreground='#3fb950')
        self._log.tag_config('warn',  foreground='#d29922')
        self._log.tag_config('error', foreground='#f85149')
        self._log.tag_config('phase', foreground='#61dafb')

    # ── 下拉刷新 ──────────────────────────────────────────────
    def _refresh_windows(self):
        from config_loader import list_windows
        titles = list_windows()
        self._window_cb['values'] = titles
        # 优先预选第七史诗
        for t in titles:
            if '第七史诗' in t or '史诗' in t:
                self._window_var.set(t)
                break
        else:
            if titles:
                self._window_var.set(titles[0])

    def _refresh_lang_profile(self):
        from config_loader import list_langs, list_profiles

        langs = list_langs()
        self._lang_cb['values'] = list(langs.keys())
        if self._lang_var.get() not in langs:
            self._lang_var.set(list(langs.keys())[0] if langs else '')
        self._lang_map = langs

        profiles = list_profiles()
        self._profile_cb['values'] = profiles
        if self._profile_var.get() not in profiles and profiles:
            self._profile_var.set(profiles[0])

    # ── 日志 ─────────────────────────────────────────────────
    def log(self, msg: str, tag: str = 'info'):
        ts   = time.strftime('%H:%M:%S')
        line = f'[{ts}] {msg}\n'
        self._log_queue.put((line, tag))
        _log_fh.write(line)

    def _poll_log(self):
        try:
            while True:
                text, tag = self._log_queue.get_nowait()
                self._log.config(state=tk.NORMAL)
                self._log.insert(tk.END, text, tag)
                self._log.see(tk.END)
                self._log.config(state=tk.DISABLED)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_log)

    def _set_status(self, text: str, color: str = '#888888'):
        self._status_var.set(text)
        self._status_lbl.config(fg=color)

    # ── 按钮回调 ─────────────────────────────────────────────
    def _on_start(self):
        window_title = self._window_var.get().strip()
        if not window_title:
            self._set_status('请先选择游戏窗口！', '#f85149')
            return

        # 加载配置
        from config_loader import cfg, _ROOT
        import os
        lang_name    = self._lang_var.get()
        lang_path    = self._lang_map.get(lang_name,
                                          os.path.join(_ROOT, 'lang', 'zh_cn.json'))
        profile_name = self._profile_var.get()
        profile_path = os.path.join(_ROOT, 'profiles', profile_name)
        cfg.load(window_title, profile_path, lang_path)
        self.log(f'配置加载完成：窗口={window_title}  语言={lang_name}  坐标={profile_name}', 'ok')

        self._stop_event.clear()
        self._start_btn.config(state=tk.DISABLED)
        self._stop_btn.config(state=tk.NORMAL)
        self._set_status('运行中...', '#3fb950')
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _on_close(self):
        self._stop_event.set()
        self.root.destroy()

    def _on_stop(self):
        self._stop_event.set()
        self.log('用户请求停止，当前步骤完成后停止...', 'warn')
        self._stop_btn.config(state=tk.DISABLED)

    def _on_stopped(self):
        self._start_btn.config(state=tk.NORMAL)
        self._stop_btn.config(state=tk.DISABLED)
        self._set_status('已停止 — 点击"开始"继续', '#d29922')

    # ── 等待辅助 ─────────────────────────────────────────────
    def _wait_for(self, check_fn, timeout: int, label: str) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._stop_event.is_set():
                raise _StopLoop()
            if check_fn():
                return True
            time.sleep(1.0)
        self.log(f'  等待"{label}"超时（{timeout}秒）', 'warn')
        return False

    def _check_stop(self):
        if self._stop_event.is_set():
            raise _StopLoop()

    # ── 主循环 ───────────────────────────────────────────────
    def _run_loop(self):
        try:
            self._load_model()
        except Exception as e:
            self.log(f'模型加载失败: {e}', 'error')
            self.root.after(0, self._on_stopped)
            return

        round_num = 0
        try:
            while not self._stop_event.is_set():
                round_num += 1
                self.log(f'═══ 第 {round_num} 场 ═══', 'phase')
                self._run_one_round()
                time.sleep(2.0)
        except _StopLoop:
            self.log('已停止', 'warn')
        except Exception as e:
            self.log(f'未处理异常: {e}', 'error')
        finally:
            self.root.after(0, self._on_stopped)

    def _load_model(self):
        if self._recommender is not None:
            return
        from transformer_inference import DraftRecommender
        self.log('加载 Transformer 模型...', 'info')
        self._recommender = DraftRecommender(MODEL_PATH, HERO_LIST_PATH)
        self.log('模型加载完成', 'ok')

    def _run_one_round(self):
        from battle_ai.executor   import focus_game_window
        from battle_ai.perception import capture, is_battle_over, is_in_battle
        from battle_ai.lobby      import (confirm_battle_result, apply_for_battle,
                                          is_in_lobby, is_waiting_for_match)
        from battle_ai.preban     import is_in_preban, do_preban
        from battle_ai.draft      import (run_draft, scan_existing_picks,
                                          is_in_draft,
                                          is_in_post_draft_ban, do_post_draft_ban,
                                          is_battle_ready, click_battle_start)
        from battle_ai.main_loop  import run as run_battle

        focus_game_window()

        preban_done  = False
        draft_done   = False
        postban_done = False
        draft_result = {'my_picks': [], 'enemy_picks': []}

        while not self._stop_event.is_set():
            img = capture()

            if   is_battle_over(img):              phase = 'result'
            elif is_in_lobby(img):                 phase = 'lobby'
            elif is_in_preban(img):                phase = 'preban'
            elif is_waiting_for_match(img):        phase = 'waiting'
            elif is_in_post_draft_ban(img):        phase = 'postban'
            elif is_battle_ready(img):             phase = 'battle_ready'
            elif draft_done and is_in_battle(img): phase = 'battle'
            elif preban_done and not draft_done:   phase = 'draft'
            elif is_in_draft(img):                 phase = 'draft'
            else:                                  phase = 'wait'

            h, w = img.shape[:2]
            self.log(f'[阶段] {phase}  ({w}x{h})', 'phase')

            if phase == 'result':
                self.log('战斗结算，点击确认', 'info')
                confirm_battle_result()
                time.sleep(2.0)
                return

            elif phase == 'lobby':
                self.log('大厅，点击申请战斗', 'info')
                apply_for_battle()
                time.sleep(2.0)

            elif phase == 'waiting':
                time.sleep(2.0)

            elif phase == 'preban':
                if not preban_done:
                    self.log('预禁用，点击英雄并确认', 'info')
                    do_preban()
                    preban_done = True
                time.sleep(1.0)

            elif phase == 'draft':
                total = len(draft_result['my_picks']) + len(draft_result['enemy_picks'])
                if total < 10:
                    img2     = capture()
                    init_my  = list(draft_result['my_picks'])
                    init_opp = list(draft_result['enemy_picks'])
                    if init_my or init_opp:
                        self.log(f'  中途接入：我方{len(init_my)}个 对手{len(init_opp)}个', 'info')
                    draft_result = run_draft(
                        self._recommender,
                        log_fn=lambda msg: self.log(msg, 'info'),
                        stop_event=self._stop_event,
                        init_my_picks=init_my,
                        init_enemy_picks=init_opp,
                    )
                    self.log('选秀阶段结束', 'ok')
                    draft_done = True
                time.sleep(1.0)

            elif phase == 'postban':
                if not postban_done:
                    self.log('选秀后禁用，模型推荐禁用', 'info')
                    do_post_draft_ban(
                        enemy_picks=draft_result['enemy_picks'],
                        recommender=self._recommender,
                        my_picks=draft_result['my_picks'],
                        banned=draft_result.get('banned', []),
                        my_first=draft_result.get('my_first', True),
                        log_fn=lambda msg: self.log(msg, 'info'),
                    )
                    postban_done = True
                time.sleep(1.0)

            elif phase == 'battle_ready':
                self.log('点击战斗开始', 'info')
                click_battle_start()
                time.sleep(4.0)

            elif phase == 'battle':
                self.log('战斗AI运行中...', 'phase')
                try:
                    run_battle(stop_event=self._stop_event,
                               log_fn=lambda msg: self.log(msg, 'info'))
                    self.log('战斗结束', 'ok')
                except Exception as e:
                    self.log(f'战斗异常: {e}', 'error')
                return

            time.sleep(1.0)


class _StopLoop(Exception):
    pass


if __name__ == '__main__':
    root = tk.Tk()
    AutoRunApp(root)
    root.mainloop()
