import os
import csv
import time
import glob
import json
import random
import numpy as np
import sounddevice as sd
from psychopy import visual, core, event, gui

# Participant-facing text in three languages, selected at startup.
# NOTE: This is a TEMPLATE / demonstration framework. The consent text below is
# an ethics/legal DRAFT only — the framework has no IRB/ethics approval and is not
# used to collect data. Anyone adapting it for real use must obtain ethics-board
# approval, fill in the [placeholders], and have a native speaker verify the
# English and Japanese versions.
LANG = {
    'English': {
        'consent_title': 'Informed Consent — TEMPLATE',
        'consent_text': (
            "=== TEMPLATE / DEMONSTRATION ONLY — NOT FOR ACTIVE DATA COLLECTION ===\n"
            "This program is released as an open methodological framework. It has\n"
            "NOT received ethics-committee / IRB approval, and the framework author\n"
            "is not collecting any data with it. The form below is a CONSENT\n"
            "TEMPLATE. Before any use with human participants you must (1) obtain\n"
            "approval from your institution's ethics board and (2) adapt this form\n"
            "(investigator, contact, data handling) to your local laws.\n"
            "Framework author: Andrey Mayer (fragerin4iks@gmail.com).\n"
            "==================================================================\n\n"
            "CONDITIONS OF PARTICIPATION\n"
            "You may take part in this study if you are over 18 years of age.\n\n"
            "VOLUNTARY PARTICIPATION\n"
            "1. Your participation is entirely voluntary.\n"
            "2. You may stop at any stage without any negative consequences.\n\n"
            "ANONYMITY\n"
            "Participation is anonymous. Data are stored on secure media and\n"
            "accessible only to the researcher.\n\n"
            "PROCEDURE\n"
            "A two-stage choice task (61 image pairs), about 40 minutes.\n"
            "BENEFIT AND RISK\n"
            "Risks are minimal. For any questions: [Principal Investigator —\n"
            "name] ([contact email]).\n\n"
            "CONFIRMATION\n"
            "I confirm that I have read and understood the aims and procedure.\n"
            "By pressing 'OK', I confirm my consent.\n\n"

        ),
        'consent_field': 'I confirm my consent',
        'warn_title': 'Warning',
        'warn_msg': 'Error: you must check the box to continue.',
        'intake_title': 'Participant information',
        'click_to_start': "\n\n(Click the left mouse button to begin)",
        'calibration': "STAGE 1: CALIBRATION\n\nClick EXACTLY ON the squares as fast as you can.",
        'test': "STAGE 2: TEST\n\nChoose a side intuitively.\n\nStay fully focused!",
        'break': ("BREAK\n\nLet's take a break!\nRest...\n\n"
                  "When you are ready to continue:\n\nPress the left mouse button"),
    },
    'Русский': {
        'consent_title': 'Информированное согласие — ШАБЛОН',
        'consent_text': (
            "=== ШАБЛОН / ТОЛЬКО ДЕМОНСТРАЦИЯ — НЕ ДЛЯ РЕАЛЬНОГО СБОРА ДАННЫХ ===\n"
            "Эта программа опубликована как открытый методологический фреймворк.\n"
            "Она НЕ получила одобрения этического комитета / IRB, и автор\n"
            "фреймворка не собирает с её помощью никаких данных. Форма ниже —\n"
            "ШАБЛОН согласия. Перед любым использованием с участием людей нужно\n"
            "(1) получить одобрение этического комитета вашего учреждения и\n"
            "(2) адаптировать форму (исследователь, контакты, данные) под местное\n"
            "законодательство.\n"
            "Автор фреймворка: Андрей Майер (fragerin4iks@gmail.com).\n"
            "==================================================================\n\n"
            "УСЛОВИЯ УЧАСТИЯ В ИССЛЕДОВАНИИ\n"
            "Вы можете принять участие, если вам больше 18 лет.\n\n"
            "ДОБРОВОЛЬНОСТЬ УЧАСТИЯ\n"
            "1. Ваше участие исключительно добровольно.\n"
            "2. Вы можете прекратить на любом этапе без негативных последствий.\n\n"
            "АНОНИМНОСТЬ\n"
            "Участие анонимно. Данные хранятся на защищённых носителях\n"
            "и доступны только исследователю.\n\n"
            "ПРОЦЕДУРА\n"
            "Тест на выбор в два этапа (61 пара изображений), около 40 минут.\n"
            "ВЫГОДА И РИСК\n"
            "Риски минимальны. По всем вопросам: [ФИО исследователя]\n"
            "([контактный email]).\n\n"
            "ПОДТВЕРЖДЕНИЕ\n"
            "Я подтверждаю, что прочитал(а) и понял(а) цели и процедуру.\n"
            "Нажимая «ОК», я подтверждаю своё согласие.\n\n"

        ),
        'consent_field': 'Я подтверждаю своё согласие',
        'warn_title': 'Внимание',
        'warn_msg': 'Ошибка: нужно поставить галочку, чтобы продолжить.',
        'intake_title': 'Информация об участнике',
        'click_to_start': "\n\n(Нажмите левую кнопку мыши, чтобы начать)",
        'calibration': "ЭТАП 1: КАЛИБРОВКА\n\nМаксимально быстро кликайте ТОЧНО ПО квадратам.",
        'test': "ЭТАП 2: ТЕСТ\n\nВыбирайте сторону интуитивно.\n\nБудьте предельно внимательны!",
        'break': ("ПАУЗА\n\nСделаем паузу!\nОтдохните...\n\n"
                  "Как будете готовы продолжить:\n\nНажмите левую кнопку мыши"),
    },
    '日本語': {
        'consent_title': 'インフォームド・コンセント — テンプレート',
        'consent_text': (
            "=== テンプレート／デモ専用 — 実際のデータ収集には使用不可 ===\n"
            "本プログラムはオープンな方法論的フレームワークとして公開されています。\n"
            "倫理委員会／IRBの承認は受けておらず、作成者はこれを用いていかなる\n"
            "データも収集していません。以下は同意書のテンプレートです。人を対象に\n"
            "使用する前に、(1) 所属機関の倫理委員会の承認を得て、(2) 本書（研究者・\n"
            "連絡先・データの取り扱い）を各地域の法令に合わせて改訂してください。\n"
            "フレームワーク作成者：Andrey Mayer (fragerin4iks@gmail.com)。\n"
            "==================================================================\n\n"
            "研究参加の条件\n"
            "18歳以上の方が参加できます。\n\n"
            "参加の任意性\n"
            "1. 参加は完全に任意です。\n"
            "2. いつでも不利益なく中止できます。\n\n"
            "匿名性\n"
            "参加は匿名です。データは安全な媒体に保管され、\n"
            "研究者のみがアクセスできます。\n\n"
            "手続き\n"
            "2段階の選択課題（画像ペア61組）、約40分。\n"
            "利益とリスク\n"
            "リスクは最小限です。お問い合わせ：［研究責任者の氏名］\n"
            "（［連絡先メール］）。\n\n"
            "確認\n"
            "目的と手続きを読み、理解したことを確認します。\n"
            "「OK」を押すことで同意を確認します。\n\n"

        ),
        'consent_field': '同意します',
        'warn_title': '注意',
        'warn_msg': 'エラー：続行するにはチェックを入れてください。',
        'intake_title': '参加者情報',
        'click_to_start': "\n\n（左クリックで開始）",
        'calibration': "ステージ1：キャリブレーション\n\nできるだけ速く、四角を正確にクリックしてください。",
        'test': "ステージ2：テスト\n\n直感で側を選んでください。\n\n集中してください！",
        'break': ("休憩\n\n少し休みましょう。\n休んでください…\n\n"
                  "続ける準備ができたら：\n\n左クリックしてください"),
    },
}


class NeuroScienceSystem:
    """
    Core experimental system for the DAMP paradigm
    (Dynamic Affective Micro-Preference Paradigm).

    Handles the full experimental pipeline:
    - Informed consent and participant registration
    - Motor calibration
    - Sequential ultra-brief stimulus presentation (18-30 ms)
    - Continuous mouse trajectory recording at 100 Hz
    - SWAP trial management
    - Data export (results.csv, trajectories.csv)
    """

    def __init__(self):

        # Select interface language first — consent and all on-screen prompts use it
        lang_field = {'Language / Язык / 言語': list(LANG.keys())}
        lang_dlg = gui.DlgFromDict(lang_field, title='Language')
        if not lang_dlg.OK:
            core.quit()
        self.lang = lang_field['Language / Язык / 言語']
        self.L = LANG[self.lang]
        self.show_consent()

        # Stimulus presentation mode:
        # True  = fixed pilot template (same stimuli and timing for all participants)
        # False = full randomization (each session independently randomized)
        self.use_fixed_stimuli = True
        self.pilot_template_file = "pilot_template.json"

        # Mouse trajectory sampling interval (seconds) — corresponds to 100 Hz
        self.sampling_rate = 0.01

        # Initial stimulus size and horizontal offset (overridden after window creation)
        self.stim_size = (700, 700)
        self.stim_pos_offset = 500

        # Stimulus exposure duration range (seconds)
        # 18-30 ms = 3-5 frames at 165 Hz (1 frame = 6.06 ms; bounds chosen to land on whole frames)
        self.stim_duration_min_s = 0.018
        self.stim_duration_max_s = 0.030

        # Inter-stimulus interval range (seconds)
        # Jittered to prevent rhythmic adaptation and motor anticipation
        self.isi_min_s = 0.08
        self.isi_max_s = 0.15

        # Participant metadata — collected via GUI dialog before session start
        self.info = {
            'ID': '',
            'Age': '',
            'Gender': ['Male', 'Female', 'Other'],
        }
        dlg = gui.DlgFromDict(self.info, title=self.L['intake_title'])
        if not dlg.OK: core.quit()

        # Create unique session ID and output directory
        self.session_id = f"{self.info['ID']}_S{time.strftime('%Y%m%d_%H%M%S')}"
        self.path = f"data/{self.session_id}"
        if not os.path.exists(self.path): os.makedirs(self.path)

        # Initialize full-screen black window (units in pixels)
        self.win = visual.Window(fullscr=True, units="pix", color=[-1, -1, -1], checkTiming=False)

        # Scale stimulus size and position offset to actual screen resolution
        stim_px = int(min(self.win.size) * 0.90)
        self.stim_size = (stim_px, stim_px)
        self.stim_pos_offset = int(self.win.size[0] * 0.2)

        # Load stimulus paths from categorized directories
        self.incentives_pool = self.load_incentives()

        # Input devices and audio
        self.mouse = event.Mouse(win=self.win)
        # Warning tone (1000 Hz, 150 ms) rendered to a buffer and played via sounddevice,
        # which is reliable on Windows (PsychoPy's audio backend can fail silently)
        _fs = 44100
        _t = np.linspace(0, 0.15, int(_fs * 0.15), endpoint=False)
        _tone = 0.5 * np.sin(2 * np.pi * 1000 * _t)
        _fade = np.linspace(0, 1, int(_fs * 0.005))   # 5 ms fade in/out to avoid clicks
        _tone[:len(_fade)] *= _fade
        _tone[-len(_fade):] *= _fade[::-1]
        self.beep_tone = _tone.astype('float32')
        self.beep_fs = _fs

        # Global session clock (used for timestamps and RT measurement)
        self.global_clock = core.Clock()

        # Frame-level timing constants derived from monitor refresh rate (165 Hz default)
        # All stimulus durations are expressed in frames for sub-millisecond precision
        self.monitor_fps = 165
        self.frame_duration = 1.0 / self.monitor_fps
        self.frames_stim_min = max(1, int(round(self.stim_duration_min_s / self.frame_duration)))
        self.frames_stim_max = max(1, int(round(self.stim_duration_max_s / self.frame_duration)))
        self.frames_isi_min = max(1, int(round(self.isi_min_s / self.frame_duration)))
        self.frames_isi_max = max(1, int(round(self.isi_max_s / self.frame_duration)))

        # Session data containers
        self.session_results = []   # trial-level behavioral data
        self.trajectory_log = []    # time-resolved mouse coordinates

        self.current_status = "STARTUP"
        self.base_rt = 0            # individual baseline RT from calibration (ms)

        # Memory of first 10 trials for subsequent SWAP re-presentation
        self.swap_memory = []

        # Tracks used stimuli to ensure uniqueness (random mode only)
        self.used_stimuli = {'positive': set(), 'negative': set(), 'neutral': set()}

        # Trials where participant exceeded response window — repeated at session end
        self.missed_trials = []

        # Pilot template loaded or generated at runtime
        self.pilot_template = None

    def show_consent(self):
        """
        Display the informed-consent form (in the selected language) before the
        session begins. The participant must explicitly confirm consent to proceed;
        the session terminates if consent is declined. No external page is opened.
        """
        while True:
            consent_dlg = gui.Dlg(title=self.L['consent_title'])
            consent_dlg.addText(self.L['consent_text'])
            consent_dlg.addField(self.L['consent_field'], initial=False)

            res = consent_dlg.show()
            if not consent_dlg.OK:
                core.quit()
            if res[0] is True:
                break
            msg = gui.Dlg(title=self.L['warn_title'])
            msg.addText(self.L['warn_msg'])
            msg.show()

    def load_incentives(self):
        """
        Scan stimulus directories and return categorized file path lists.
        Requires minimum 50 images per category (Positive, Negative, Neutral).
        Terminates session if any category is insufficient.
        """
        paths = {
            'neutral': glob.glob("Database_2026/final_standardized_stimuli/Neutral/*.*"),
            'positive': glob.glob("Database_2026/final_standardized_stimuli/Positive/*.*"),
            'negative': glob.glob("Database_2026/final_standardized_stimuli/Negative/*.*"),
        }
        for key, val in paths.items():
            if len(val) < 50:
                print(
                    f"\n!!! CRITICAL ERROR !!!\nNot enough files in Database_2026/final_standardized_stimuli/{key.capitalize()}/ (minimum 50 required)")
                core.quit()
        return paths

    def generate_pilot_template(self):
        """
        Generate a fixed randomized experiment template and save it to disk.

        Uses a fixed random seed (42) to ensure the template is reproducible
        across sessions. All trial parameters are pre-determined:
        stimulus identities, spatial arrangement, frame counts, and inter-trial intervals.

        The template is saved as pilot_template.json and reused for all
        subsequent participants, ensuring identical presentation conditions.
        """
        print("\n" + "=" * 70)
        print("GENERATING NEW PILOT TEMPLATE")
        print("=" * 70)

        random.seed(42)

        template = {
            'version': '1.0',
            'created': time.strftime('%Y-%m-%d %H:%M:%S'),
            'trials': []
        }

        # Generate balanced set of 51 trials: 17 per condition
        base_conditions = (
                [('positive', 'neutral')] * 17 +
                [('negative', 'neutral')] * 17 +
                [('positive', 'negative')] * 17
        )
        random.shuffle(base_conditions)

        for i in range(1, 52):
            cat1, cat2 = base_conditions[i - 1]
            img1 = random.choice(self.incentives_pool[cat1])
            img2 = random.choice(self.incentives_pool[cat2])

            t_type = 'NORMAL'

            # Randomize left/right spatial position of each stimulus
            is_cat1_left = random.random() > 0.5

            # Randomize frame counts within allowed ranges
            stim1_frames = random.randint(self.frames_stim_min, self.frames_stim_max)
            isi_frames = random.randint(self.frames_isi_min, self.frames_isi_max)
            stim2_frames = random.randint(self.frames_stim_min, self.frames_stim_max)

            # Randomize fixation and inter-trial interval durations
            fixation_duration = 2.0 + random.random() * 1.5
            iti_duration = 1.0 + random.random() * 1.5

            trial_data = {
                'trial_n': i,
                'cat1': cat1,
                'cat2': cat2,
                'img1': img1,
                'img2': img2,
                'trial_type': t_type,
                'is_cat1_left': is_cat1_left,
                'stim1_frames': stim1_frames,
                'isi_frames': isi_frames,
                'stim2_frames': stim2_frames,
                'fixation_duration': round(fixation_duration, 3),
                'iti_duration': round(iti_duration, 3)
            }

            template['trials'].append(trial_data)

        # Restore system randomness after template generation
        random.seed()

        with open(self.pilot_template_file, 'w', encoding='utf-8') as f:
            json.dump(template, f, indent=2, ensure_ascii=False)

        print(f"✓ Template saved to: {self.pilot_template_file}")
        print(f"✓ Generated {len(template['trials'])} trials")
        print("=" * 70 + "\n")

        return template

    def load_pilot_template(self):
        """
        Load existing pilot template from disk.
        If no template file is found, generates a new one automatically.
        """
        if not os.path.exists(self.pilot_template_file):
            print(f"\n⚠ Template file '{self.pilot_template_file}' not found!")
            print("→ Generating a new template...\n")
            return self.generate_pilot_template()

        try:
            with open(self.pilot_template_file, 'r', encoding='utf-8') as f:
                template = json.load(f)

            print("\n" + "=" * 70)
            print("LOADED EXISTING PILOT TEMPLATE")
            print("=" * 70)
            print(f"✓ Version: {template.get('version', 'unknown')}")
            print(f"✓ Created: {template.get('created', 'unknown')}")
            print(f"✓ Number of trials: {len(template['trials'])}")
            print("=" * 70 + "\n")

            return template

        except Exception as e:
            print(f"\n⚠ ERROR loading template: {e}")
            print("→ Generating a new template...\n")
            return self.generate_pilot_template()

    def play_beep(self):
        """Play the trial-onset warning tone via sounddevice (non-blocking)."""
        try:
            sd.play(self.beep_tone, self.beep_fs)
        except Exception as e:
            print(f"Beep error: {e}")

    def wait_frames(self, n_frames):
        """Advance exactly n_frames monitor frames by calling win.flip() each frame."""
        for _ in range(n_frames):
            self.win.flip()

    def wait_for_click(self, text):
        """Display instruction text and wait for left mouse button click to proceed."""
        instr = visual.TextStim(self.win, text=text + self.L['click_to_start'],
                                height=30, font='Arial', color='white')
        instr.draw()
        self.win.flip()
        event.clearEvents()
        while True:
            if self.mouse.getPressed()[0]:
                self.wait_frames(int(round(0.2 / self.frame_duration)))
                break
            if event.getKeys(['escape']): self.finalize()

    def run_calibration(self):
        """
        Motor calibration phase — 7 trials of simple click-on-square task.

        Participant clicks squares appearing randomly left or right.
        Only clicks within the square boundary are accepted.
        The median RT across all 7 trials is stored as base_rt,
        used to normalize reaction times throughout the main experiment.
        """
        self.current_status = "CALIBRATION"
        self.wait_for_click(self.L['calibration'])

        rts = []
        fix = visual.TextStim(self.win, text="+", height=60, font='Arial')

        for i in range(1, 8):
            # Fixation cross with jittered pre-trial delay
            fix.draw()
            self.win.flip()
            core.wait(0.5)
            core.wait(1.0 + random.random())

            # Present square on randomly chosen side
            side = random.choice([-self.stim_pos_offset, self.stim_pos_offset])
            square_size = 250
            sq = visual.Rect(self.win, size=(square_size, square_size), pos=(side, 0), fillColor='gray')

            # Reset cursor to bottom center before each trial
            self.mouse.setPos((0, -500))

            sq.draw()
            self.win.flip()
            t0 = self.global_clock.getTime()

            while True:
                sq.draw()
                self.win.flip()

                if self.mouse.getPressed()[0]:
                    click_pos = self.mouse.getPos()

                    # Validate that click landed within square boundaries
                    x_in_range = abs(click_pos[0] - side) <= square_size / 2
                    y_in_range = abs(click_pos[1] - 0) <= square_size / 2

                    if x_in_range and y_in_range:
                        rt = (self.global_clock.getTime() - t0) * 1000
                        rts.append(rt)
                        print(f"✓ Calibration {i}/7: RT = {int(rt)} ms")
                        core.wait(0.2)
                        break
                    else:
                        print(f"✗ Miss! Click EXACTLY ON the square")
                        core.wait(0.3)

                if event.getKeys(['escape']):
                    self.finalize()

            self.win.flip()
            core.wait(0.5)

        # Median is used to reduce sensitivity to outlier trials
        self.base_rt = np.median(rts) if rts else 500
        print(f"→ Baseline RT: {int(self.base_rt)} ms")
        return self.base_rt

    def get_unique_stimulus(self, category):
        """
        Select a stimulus from the given category that has not yet been used.
        Falls back to full pool if all stimuli have been exhausted.
        Used in random mode only — pilot template handles uniqueness at generation time.
        """
        available = [s for s in self.incentives_pool[category] if s not in self.used_stimuli[category]]

        if not available:
            print(f"Warning: all stimuli in category {category} have been used!")
            available = self.incentives_pool[category]

        chosen = random.choice(available)
        self.used_stimuli[category].add(chosen)
        return chosen

    def present_trial(self, p1, p2, t_type, n, trial_params=None):
        """
        Execute a single experimental trial.

        Trial sequence:
        1. Audio warning beep + fixation cross
        2. Sequential stimulus presentation: stim1 → ISI → stim2 (frame-precise)
        3. Blank screen response window (3.0 s)
        4. Mouse trajectory recorded at 100 Hz until click or timeout

        Parameters:
            p1, p2      : file paths to left and right stimulus images
            t_type      : trial type — 'NORMAL' or 'SWAP'
            n           : trial number (used for trajectory logging)
            trial_params: dict with pre-determined frame counts and fixation duration
                          (from pilot template); if None, values are randomized

        Returns:
            dict with trial-level behavioral data including RT, choice, colors,
            valence categories, and session metadata
        """
        self.current_status = f"T_{n}_{t_type}"

        # Pre-load stimulus images at fixed positions (left = -offset, right = +offset)
        img1 = visual.ImageStim(self.win, image=p1, size=self.stim_size, pos=(-self.stim_pos_offset, 0),
                                opacity=1.0, contrast=1.0, interpolate=False)
        img2 = visual.ImageStim(self.win, image=p2, size=self.stim_size, pos=(self.stim_pos_offset, 0),
                                opacity=1.0, contrast=1.0, interpolate=False)
        fix = visual.TextStim(self.win, text="+", height=60, font='Arial')

        # Warning beep + fixation
        self.play_beep()
        fix.draw()
        self.win.flip()

        # Fixation duration: from template (fixed mode) or jittered (random mode)
        if trial_params and 'fixation_duration' in trial_params:
            core.wait(trial_params['fixation_duration'])
        else:
            core.wait(2.0 + random.random() * 1.5)

        t_flash = -1

        # Hide cursor during stimulus presentation
        self.mouse.setVisible(0)
        t_stim = self.global_clock.getTime()

        # Resolve frame counts from template or generate randomly
        if trial_params:
            stim1_frames = trial_params['stim1_frames']
            isi_frames = trial_params['isi_frames']
            stim2_frames = trial_params['stim2_frames']
        else:
            stim1_frames = random.randint(self.frames_stim_min, self.frames_stim_max)
            isi_frames = random.randint(self.frames_isi_min, self.frames_isi_max)
            stim2_frames = random.randint(self.frames_stim_min, self.frames_stim_max)

        # Sequential stimulus presentation: stim1 → blank ISI → stim2 → blank
        img1.draw()
        self.win.flip()
        self.wait_frames(stim1_frames)
        self.win.flip()

        self.wait_frames(isi_frames)

        img2.draw()
        self.win.flip()
        self.wait_frames(stim2_frames)
        self.win.flip()

        # Reset cursor to bottom center and begin response window
        self.mouse.setPos((0, -500))
        self.mouse.setVisible(1)
        self.win.flip()
        t_resp = self.global_clock.getTime()
        choice = None
        last_s = 0

        # Response window: 3.0 seconds
        # Mouse position sampled at 100 Hz; click accepted only beyond ±200 px threshold
        while self.global_clock.getTime() - t_resp < 3.0:
            now = self.global_clock.getTime()
            pos = self.mouse.getPos()

            # Append trajectory sample at 100 Hz
            if now - last_s >= self.sampling_rate:
                self.trajectory_log.append({
                    'n': n,
                    'type': t_type,
                    't': round(now, 3),
                    't_rel': round(now - t_resp, 3),
                    'x': int(pos[0]),
                    'y': int(pos[1])
                })
                last_s = now

            # Register click only if cursor has moved beyond central dead zone
            if self.mouse.getPressed()[0]:
                if pos[0] < -200: choice = 'left'; break
                if pos[0] > 200: choice = 'right'; break

            if event.getKeys(['escape']): self.finalize()

        t_end = self.global_clock.getTime()

        rt = (t_end - t_resp) * 1000 if choice else -1

        rt_normalized = (rt / self.base_rt) if (choice and self.base_rt > 0) else -1

        def get_emotion_category(path):
            path_normalized = str(path).replace('\\', '/').lower()
            if '/positive/' in path_normalized:
                return 'positive'
            elif '/negative/' in path_normalized:
                return 'negative'
            elif '/neutral/' in path_normalized:
                return 'neutral'
            return 'unknown'

        emo_l = get_emotion_category(p1)
        emo_r = get_emotion_category(p2)

        color_l = os.path.splitext(os.path.basename(p1))[0].split('_')[-1]
        color_r = os.path.splitext(os.path.basename(p2))[0].split('_')[-1]

        chosen_color = "none"
        if choice == 'left':
            chosen_color = color_l
        elif choice == 'right':
            chosen_color = color_r

        return {
            'n': n, 'type': t_type, 'choice': choice,
            'chosen_color': chosen_color,
            'color_l': color_l,
            'color_r': color_r,
            'emo_l': emo_l,
            'emo_r': emo_r,
            'rt': round(rt, 1),
            'rt_normalized': round(rt_normalized, 3),
            'base_rt': round(self.base_rt, 1),
            'img_l': os.path.basename(p1), 'img_r': os.path.basename(p2),
            't_stim': round(t_stim, 3), 't_flash': round(t_flash, 3) if t_flash > 0 else -1,
            't_resp': round(t_resp, 3), 't_end': round(t_end, 3),
            'age': self.info['Age'],
            'gender': self.info['Gender'],
            'consent': 'Confirmed'
        }

    def run(self):
        """
        Main experiment runner.

        Sequence:
        1. Measure actual monitor frame rate and recalculate all frame counts
        2. Motor calibration (7 trials)
        3. Main block: 51 trials (fixed or random mode)
           - Break after trial 31
           - Missed trials (timeouts) stored for later repetition
           - First 10 valid trials stored in swap_memory for SWAP re-presentation
        4. Missed trial repetition block (if any)
        5. SWAP block: re-present first 10 pairs with mirrored spatial arrangement
        6. Save all data via finalize()
        """
        # Warm up display and measure actual frame rate
        for _ in range(10): self.win.flip()
        try:
            measured_fps = self.win.getActualFrameRate()
            if measured_fps and measured_fps > 0:
                self.monitor_fps = measured_fps
                self.frame_duration = 1.0 / self.monitor_fps
                self.frames_stim_min = max(1, int(round(self.stim_duration_min_s / self.frame_duration)))
                self.frames_stim_max = max(1, int(round(self.stim_duration_max_s / self.frame_duration)))
                self.frames_isi_min = max(1, int(round(self.isi_min_s / self.frame_duration)))
                self.frames_isi_max = max(1, int(round(self.isi_max_s / self.frame_duration)))
                print(f"→ Monitor FPS: {int(self.monitor_fps)} Hz")
                print(
                    f"→ Stimulus exposure: {self.frames_stim_min}-{self.frames_stim_max} frames "
                    f"({int(self.frames_stim_min * self.frame_duration * 1000)}-"
                    f"{int(self.frames_stim_max * self.frame_duration * 1000)} ms)")
        except:
            pass

        try:
            self.global_clock.reset()
            self.run_calibration()

            # Load or generate stimulus presentation template
            if self.use_fixed_stimuli:
                print("\n" + "🔒" * 35)
                print("MODE: PILOT (FIXED TEMPLATE)")
                print("🔒" * 35)
                self.pilot_template = self.load_pilot_template()
                trials_source = self.pilot_template['trials']
            else:
                print("\n" + "🎲" * 35)
                print("MODE: MAIN (FULL RANDOMIZATION)")
                print("🎲" * 35)
                base_conditions = (
                        [('positive', 'neutral')] * 17 +
                        [('negative', 'neutral')] * 17 +
                        [('positive', 'negative')] * 17
                )
                random.shuffle(base_conditions)
                trials_source = None

            self.wait_for_click(self.L['test'])

            # Main trial block (51 trials)
            for i in range(1, 52):
                try:
                    if i == 32:
                        self.wait_for_click(self.L['break'])

                    if self.use_fixed_stimuli:
                        # Fixed mode: retrieve all parameters from pre-generated template
                        trial_template = trials_source[i - 1]
                        cat1 = trial_template['cat1']
                        cat2 = trial_template['cat2']
                        img1 = trial_template['img1']
                        img2 = trial_template['img2']
                        t_type = trial_template['trial_type']
                        is_cat1_left = trial_template['is_cat1_left']

                        trial_params = {
                            'stim1_frames': trial_template['stim1_frames'],
                            'isi_frames': trial_template['isi_frames'],
                            'stim2_frames': trial_template['stim2_frames'],
                            'fixation_duration': trial_template['fixation_duration']
                        }

                        p_l, p_r = (img1, img2) if is_cat1_left else (img2, img1)

                    else:
                        cat1, cat2 = base_conditions[i - 1]
                        img1 = self.get_unique_stimulus(cat1)
                        img2 = self.get_unique_stimulus(cat2)
                        t_type = 'NORMAL'
                        is_cat1_left = random.random() > 0.5
                        p_l, p_r = (img1, img2) if is_cat1_left else (img2, img1)
                        trial_params = None

                    data = self.present_trial(p_l, p_r, t_type, i, trial_params)
                    data['is_swap'] = 0  # 0 = not a SWAP trial

                    if data['choice'] == 'left':
                        data['chosen_emo'] = cat1 if is_cat1_left else cat2
                    elif data['choice'] == 'right':
                        data['chosen_emo'] = cat1 if not is_cat1_left else cat2
                    else:

                        data['chosen_emo'] = 'timeout'
                        self.missed_trials.append({
                            'p_l': p_l, 'p_r': p_r, 't_type': t_type,
                            'cat1': cat1, 'cat2': cat2, 'is_cat1_left': is_cat1_left,
                            'trial_params': trial_params
                        })
                        print(f"⚠ Trial {i} missed — will be repeated later")

                    # Store first 10 valid trials for SWAP re-presentation
                    if i <= 10 and data['choice']:
                        self.swap_memory.append({
                            'l': p_l, 'r': p_r, 'choice': data['choice']
                        })

                    self.session_results.append(data)

                    # Brief confirmation dot between trials
                    confirm_dot = visual.Circle(self.win, radius=5, fillColor='gray', pos=(0, 0))
                    confirm_dot.draw()
                    self.win.flip()

                    if trial_params and 'iti_duration' in trial_template:
                        core.wait(trial_template['iti_duration'])
                    else:
                        core.wait(1.0 + random.random() * 1.5)

                    self.win.flip()
                    core.wait(0.2)

                except Exception as e:
                    print(f"ERROR in trial {i}: {e}")

            # Missed trial repetition block
            if self.missed_trials:
                print(f"\n→ Repeating {len(self.missed_trials)} missed trials...")
                random.shuffle(self.missed_trials)

                missed_idx = 52
                for missed in self.missed_trials:
                    try:
                        data = self.present_trial(
                            missed['p_l'], missed['p_r'],
                            missed['t_type'], missed_idx,
                            missed.get('trial_params')
                        )
                        data['is_swap'] = 0

                        if data['choice'] == 'left':
                            data['chosen_emo'] = missed['cat1'] if missed['is_cat1_left'] else missed['cat2']
                        elif data['choice'] == 'right':
                            data['chosen_emo'] = missed['cat1'] if not missed['is_cat1_left'] else missed['cat2']
                        else:
                            data['chosen_emo'] = 'timeout'
                            print(f"⚠ Trial {missed_idx} missed again!")

                        self.session_results.append(data)

                        confirm_dot = visual.Circle(self.win, radius=5, fillColor='gray', pos=(0, 0))
                        confirm_dot.draw()
                        self.win.flip()
                        core.wait(1.0 + random.random() * 1.5)

                        missed_idx += 1
                    except Exception as e:
                        print(f"ERROR in repeated trial: {e}")

            # SWAP block: re-present first 10 pairs with mirrored left/right positions
            # is_swap = 0 if participant maintained original choice, 1 if switched
            if self.swap_memory:
                random.shuffle(self.swap_memory)

                swap_idx = 52 if not self.missed_trials else 52 + len(self.missed_trials)
                for old_pair in self.swap_memory:
                    try:
                        # Mirror spatial positions
                        p_l, p_r = old_pair['r'], old_pair['l']
                        t_type = 'SWAP'

                        data = self.present_trial(p_l, p_r, t_type, swap_idx, trial_params=None)

                        if data['choice'] and old_pair['choice']:
                            old_chosen_img = old_pair['l'] if old_pair['choice'] == 'left' else old_pair['r']
                            new_chosen_img = p_l if data['choice'] == 'left' else p_r
                            # 0 = same stimulus chosen (stable preference)
                            # 1 = different stimulus chosen (switched)
                            data['is_swap'] = 0 if old_chosen_img == new_chosen_img else 1
                        else:
                            data['is_swap'] = -1  # indeterminate (timeout in one of the trials)

                        data['chosen_emo'] = 'swap_check'

                        self.session_results.append(data)

                        confirm_dot = visual.Circle(self.win, radius=5, fillColor='gray', pos=(0, 0))
                        confirm_dot.draw()
                        self.win.flip()
                        core.wait(1.0 + random.random() * 1.5)

                        swap_idx += 1
                    except Exception as e:
                        print(f"ERROR in SWAP trial: {e}")

        finally:
            self.finalize()

    def finalize(self):
        """
        Save all session data and close the experiment.

        Writes:
        - results.csv: trial-level behavioral data
        - trajectories.csv: time-resolved mouse coordinates at 100 Hz
        """
        self.current_status = "EXITING"

        if self.session_results:
            with open(f"{self.path}/results.csv", 'w', newline='', encoding='utf-8') as f:
                dw = csv.DictWriter(f, fieldnames=self.session_results[0].keys())
                dw.writeheader()
                dw.writerows(self.session_results)
            print(f"Results saved to: {self.path}/results.csv")
        else:
            print("Error: results list is empty. Nothing to save.")

        if self.trajectory_log:
            with open(f"{self.path}/trajectories.csv", 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['n', 'type', 't', 't_rel', 'x', 'y']
                dw = csv.DictWriter(f, fieldnames=fieldnames)
                dw.writeheader()
                dw.writerows(self.trajectory_log)
            print(f"Trajectories saved to: {self.path}/trajectories.csv")

        self.win.close()
        core.quit()


if __name__ == "__main__":
    app = NeuroScienceSystem()
    app.run()