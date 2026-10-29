"""Camera-first desktop layout for the local demonstration collector."""
from __future__ import annotations

import queue
import subprocess
import threading
from pathlib import Path

from orca_teleop.collector_app import CAMERAS, PHASES, camera_sources


def launch(collector):
    import fcntl
    import tkinter as tk
    from tkinter import filedialog, font as tkfont, messagebox, ttk
    from PIL import Image, ImageTk

    app = tk.Tk()
    app.title('ORCA · 화투 시연 수집' + (' [UI 시험 모드]' if collector.demo else ''))
    app.tk.call('tk', 'scaling', 1.333333333)
    lock = (collector.config_dir / ('demo.lock' if collector.demo else 'app.lock')).open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        messagebox.showinfo('ORCA 수집 앱', '이미 수집 앱이 실행 중입니다.')
        app.destroy()
        return
    app.geometry('1560x940')
    app.minsize(1080, 680)
    app.configure(bg='#eef2f6')
    try:
        app.state('zoomed')
    except tk.TclError:
        pass

    fonts = []
    def font(size=11, bold=False, family='Noto Sans CJK KR'):
        value = tkfont.Font(app, family=family, size=11, weight='bold' if bold else 'normal')
        value.configure(size=max(1, round(11 * size * 1.8 / value.metrics('linespace'))))
        fonts.append(value)
        return value
    for name in ('TkDefaultFont', 'TkTextFont', 'TkMenuFont', 'TkHeadingFont',
                 'TkCaptionFont', 'TkSmallCaptionFont', 'TkIconFont', 'TkTooltipFont'):
        value = tkfont.nametofont(name)
        value.configure(family='Noto Sans CJK KR', size=11)
        value.configure(size=max(1, round(11 * 21 / value.metrics('linespace'))))

    style = ttk.Style(app)
    style.theme_use('clam')
    style.configure('TFrame', background='#eef2f6')
    style.configure('TLabel', background='#eef2f6', foreground='#152536', font=font(11))
    style.configure('Muted.TLabel', foreground='#647488', font=font(10))
    style.configure('Title.TLabel', font=font(22, True))
    style.configure('Status.TLabel', font=font(13, True), foreground='#16765f')
    style.configure('Count.TLabel', font=font(32, True))
    style.configure('TButton', font=font(11), padding=(12, 10), background='#e1e9f1',
                    foreground='#34485f', borderwidth=0, relief='flat')
    style.map('TButton', background=[('disabled', '#e7ecf1'), ('active', '#d2deea')],
              foreground=[('disabled', '#8a97a5')])
    style.configure('TRadiobutton', background='#eef2f6', foreground='#34485f', padding=4, font=font(10))
    style.configure('TCheckbutton', background='#eef2f6', foreground='#34485f', padding=4, font=font(10))
    style.map('TRadiobutton', background=[('active', '#e0e8f0')])
    style.map('TCheckbutton', background=[('active', '#e0e8f0')])
    style.configure('TEntry', padding=6, foreground='#1c334b')
    style.configure('TCombobox', padding=6)
    style.configure('TNotebook', background='#eef2f6', borderwidth=0)
    style.configure('TNotebook.Tab', padding=(13, 10), font=font(11))
    style.map('TNotebook.Tab', background=[('selected', '#f9fbfd'), ('!selected', '#e1e9f1')],
              foreground=[('selected', '#152536'), ('!selected', '#607388')])
    style.configure('TProgressbar', background='#16816a', troughcolor='#dce5ed', borderwidth=0)
    for name, color in (('Record', '#ca414b'), ('Save', '#147d64'), ('Stop', '#34485f')):
        style.configure(name + '.TButton', background=color, foreground='white',
                        font=font(14, True), padding=(20, 14))
        style.map(name + '.TButton', background=[('disabled', '#e0e6ec'), ('active', color)],
                  foreground=[('disabled', '#83909e')])
    style.configure('Retry.TButton', font=font(13, True), padding=(18, 14))

    outer = ttk.Frame(app, padding=(18, 14))
    outer.pack(fill='both', expand=True)
    outer.rowconfigure(1, weight=1)
    outer.columnconfigure(0, weight=1)
    top = ttk.Frame(outer)
    top.grid(row=0, column=0, sticky='ew', pady=(0, 12))
    title = ttk.Frame(top)
    title.pack(side='left')
    ttk.Label(title, text='화투 시연 수집', style='Title.TLabel').pack(anchor='w')
    ttk.Label(title, text='Quest  /  AM-ARM200 + ORCA Hand', style='Muted.TLabel').pack(anchor='w')
    phase_var = tk.StringVar(value='연결 전')
    badge = ttk.Label(top, textvariable=phase_var, style='Status.TLabel')
    badge.pack(side='right', padx=12)
    if collector.demo:
        ttk.Label(top, text='UI 시험 모드 · 학습용 데이터 아님', foreground='#b45324').pack(side='right', padx=20)

    content = ttk.Panedwindow(outer, orient='horizontal')
    content.grid(row=1, column=0, sticky='nsew')
    workspace, sidebar = ttk.Frame(content), ttk.Frame(content, width=330)
    sidebar.pack_propagate(False)
    content.add(workspace, weight=5)
    content.add(sidebar, weight=0)
    workspace.rowconfigure(1, weight=1)
    workspace.columnconfigure(0, weight=1)

    focus = tk.StringVar(value='wrist')
    equal_views = tk.BooleanVar(value=False)
    fullscreen = tk.BooleanVar(value=False)
    toolbar = ttk.Frame(workspace)
    toolbar.grid(row=0, column=0, sticky='ew', pady=(0, 9))
    ttk.Label(toolbar, text='카메라', font=font(12, True)).pack(side='left', padx=(0, 14))
    camera_grid = ttk.Frame(workspace)
    camera_grid.grid(row=1, column=0, sticky='nsew')
    panels, latest, enlarged = {}, {}, {}
    pending_click = [None]

    class VideoPanel:
        def __init__(self, parent, name, popup=False):
            self.name, self.popup = name, popup
            self.frame = tk.Frame(parent, background='#172334', highlightthickness=1,
                                  highlightbackground='#26394e')
            head = tk.Frame(self.frame, background='#172334')
            head.pack(fill='x', padx=14, pady=(10, 7))
            self.title = tk.Label(head, text=CAMERAS[name], bg='#172334', fg='#f1f5fa', font=font(12, True))
            self.title.pack(side='left')
            self.indicator = tk.Label(head, text='영상 대기', bg='#172334', fg='#92a4b8', font=font(10))
            self.indicator.pack(side='right')
            self.canvas = tk.Canvas(self.frame, background='#0b131e', highlightthickness=0,
                                    width=1, height=1, cursor='hand2' if not popup else '')
            self.canvas.pack(fill='both', expand=True, padx=1, pady=(0, 1))
            self.image = self.canvas.create_image(0, 0, anchor='center')
            self.message = self.canvas.create_text(0, 0, fill='#a8b8cb', font=font(12),
                                                    justify='center', text='카메라를 연결해 주세요')
            self.photo = None
            self.canvas.bind('<Configure>', lambda event: self.draw())
            if not popup:
                def single_click(event):
                    if pending_click[0] is not None:
                        app.after_cancel(pending_click[0])
                    pending_click[0] = app.after(300, lambda: select_camera(name))
                def double_click(event):
                    if pending_click[0] is not None:
                        app.after_cancel(pending_click[0])
                        pending_click[0] = None
                    enlarge(name)
                for widget in (self.canvas, self.title):
                    widget.bind('<Button-1>', single_click)
                    widget.bind('<Double-Button-1>', double_click)
        def draw(self):
            width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
            if width < 8 or height < 8:
                return
            self.canvas.coords(self.image, width / 2, height / 2)
            self.canvas.coords(self.message, width / 2, height / 2)
            self.canvas.itemconfigure(self.message, width=max(1, width - 36))
            source, age, error = latest.get(self.name, (None, None, '카메라 미리보기로 구도를 확인하세요'))
            if source is None or (age is not None and age > 2):
                message = (f'영상 수신 대기\n마지막 수신 {age:.0f}초 전' if age is not None and age > 2
                           else error or '카메라 연결·입력 선택을 확인해 주세요')
                self.canvas.itemconfigure(self.image, image='')
                self.canvas.itemconfigure(self.message, text=message, state='normal')
                self.indicator.configure(text='영상 대기', fg='#a0aec0')
                return
            # Fit the actual available area, allowing enlargement. Do not crop
            # fingers or stretch a 4:3 camera feed to the window's aspect ratio.
            scale = min(width / source.width, height / source.height)
            size = (max(1, round(source.width * scale)), max(1, round(source.height * scale)))
            image = source.resize(size, Image.Resampling.BILINEAR)
            self.photo = ImageTk.PhotoImage(image)
            self.canvas.itemconfigure(self.image, image=self.photo)
            self.canvas.itemconfigure(self.message, state='hidden')
            self.indicator.configure(text=f'{source.width} × {source.height}  ·  실시간', fg='#62d7ad')

    def arrange():
        for panel in panels.values():
            panel.frame.grid_forget()
        for column in range(3):
            camera_grid.columnconfigure(column, weight=0, minsize=0)
        for row in range(2):
            camera_grid.rowconfigure(row, weight=1)
        if equal_views.get():
            for col, name in enumerate(CAMERAS):
                camera_grid.columnconfigure(col, weight=1, uniform='equal')
                panels[name].frame.grid(row=0, column=col, rowspan=2, sticky='nsew',
                                        padx=(0 if col == 0 else 5, 0 if col == 2 else 5))
        else:
            for column, weight in ((0, 2), (1, 1)):
                camera_grid.columnconfigure(column, weight=weight, uniform='focus')
            panels[focus.get()].frame.grid(row=0, column=0, rowspan=2, sticky='nsew', padx=(0, 8))
            rest = [name for name in CAMERAS if name != focus.get()]
            for row, name in enumerate(rest):
                panels[name].frame.grid(row=row, column=1, sticky='nsew', pady=(0 if row == 0 else 4, 4 if row == 0 else 0))
        for name, panel in panels.items():
            selected = not equal_views.get() and focus.get() == name
            panel.frame.configure(highlightbackground='#45b998' if selected else '#26394e',
                                  highlightthickness=2 if selected else 1)
        app.after_idle(lambda: [panel.draw() for panel in panels.values()])

    def select_camera(name):
        focus.set(name)
        equal_views.set(False)
        arrange()

    def enlarge(name):
        if name in enlarged:
            enlarged[name][0].lift()
            return
        window = tk.Toplevel(app)
        window.title(f'{CAMERAS[name]} 카메라 확대 · Esc로 닫기')
        window.geometry('1280x960')
        window.minsize(600, 450)
        panel = VideoPanel(window, name, popup=True)
        panel.frame.pack(fill='both', expand=True)
        enlarged[name] = (window, panel)
        def dismiss(event=None):
            enlarged.pop(name, None)
            window.destroy()
        window.protocol('WM_DELETE_WINDOW', dismiss)
        window.bind('<Escape>', dismiss)
        panel.draw()

    for name in CAMERAS:
        ttk.Radiobutton(toolbar, text=f'{CAMERAS[name]} 크게', variable=focus, value=name,
                        command=lambda n=name: select_camera(n)).pack(side='left', padx=(0, 10))
        panels[name] = VideoPanel(camera_grid, name)
    ttk.Checkbutton(toolbar, text='같은 크기로', variable=equal_views, command=arrange).pack(side='left', padx=8)
    def toggle_fullscreen(event=None):
        fullscreen.set(not fullscreen.get())
        app.attributes('-fullscreen', fullscreen.get())
        return 'break'
    def leave_fullscreen(event=None):
        fullscreen.set(False)
        app.attributes('-fullscreen', False)
    ttk.Button(toolbar, text='전체화면  F11', command=toggle_fullscreen).pack(side='right')
    app.bind('<F11>', toggle_fullscreen)
    app.bind('<Escape>', leave_fullscreen)
    arrange()

    ttk.Label(workspace, text='화면 클릭: 크게 보기 전환     더블클릭: 별도 확대 창     경계선 드래그: 설정 영역 너비 조절',
              style='Muted.TLabel').grid(row=2, column=0, sticky='w', pady=(8, 0))

    # Keep configuration and logs off the camera workspace.
    side = ttk.Frame(sidebar, padding=(16, 0, 0, 0))
    side.pack(fill='both', expand=True)
    count_var, timing_var, task_var = tk.StringVar(), tk.StringVar(), tk.StringVar(value=collector.settings['task'])
    target_var = tk.StringVar(value=str(collector.settings['target']))
    velocity_var = tk.StringVar(value=str(collector.settings['velocity']))
    acceleration_var = tk.StringVar(value=str(collector.settings['acceleration']))
    folder_var = tk.StringVar(value=collector.settings['data_root'])
    source_vars = {name: tk.StringVar(value=str(value)) for name, value in collector.settings['sources'].items()}
    ttk.Label(side, text='성공한 시연', style='Muted.TLabel').pack(anchor='w')
    ttk.Label(side, textvariable=count_var, style='Count.TLabel').pack(anchor='w')
    progress = ttk.Progressbar(side, maximum=50)
    progress.pack(fill='x', pady=(3, 8))
    ttk.Label(side, textvariable=timing_var, style='Muted.TLabel').pack(anchor='w', pady=(0, 12))

    notebook = ttk.Notebook(side)
    notebook.pack(fill='both', expand=True)
    collection_page, device_page = (ttk.Frame(notebook) for _ in range(2))
    log_tab = ttk.Frame(notebook, padding=12)
    notebook.add(collection_page, text='수집 설정')
    notebook.add(device_page, text='장치')
    notebook.add(log_tab, text='로그')
    def scrollable(page):
        canvas = tk.Canvas(page, bg='#eef2f6', highlightthickness=0, width=1, height=1)
        bar = ttk.Scrollbar(page, command=canvas.yview)
        canvas.configure(yscrollcommand=bar.set)
        bar.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)
        body = ttk.Frame(canvas, padding=12)
        item = canvas.create_window(0, 0, window=body, anchor='nw')
        body.bind('<Configure>', lambda event: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>', lambda event: canvas.itemconfigure(item, width=event.width))
        def wheel(event):
            widget = event.widget
            while widget is not None:
                if widget == body:
                    direction = -1 if getattr(event, 'num', None) == 4 or getattr(event, 'delta', 0) > 0 else 1
                    canvas.yview_scroll(direction * 2, 'units')
                    return 'break'
                widget = getattr(widget, 'master', None)
        for event in ('<Button-4>', '<Button-5>', '<MouseWheel>'):
            app.bind(event, wheel, add='+')
        return body
    collection_tab, device_tab = scrollable(collection_page), scrollable(device_page)
    fields, source_boxes, buttons = [], [], {}
    ttk.Label(collection_tab, text='작업 지시', font=font(11, True)).pack(anchor='w')
    entry = ttk.Entry(collection_tab, textvariable=task_var)
    entry.pack(fill='x', pady=(6, 12))
    fields.append(entry)
    for label, variable in (('성공 목표 (회)', target_var), ('팔 속도 (°/초)', velocity_var),
                            ('가속도 (°/초²)', acceleration_var)):
        row = ttk.Frame(collection_tab)
        row.pack(fill='x', pady=5)
        ttk.Label(row, text=label).pack(side='left')
        entry = ttk.Entry(row, textvariable=variable, width=7)
        entry.pack(side='right')
        fields.append(entry)
    ttk.Separator(collection_tab).pack(fill='x', pady=16)
    ttk.Label(collection_tab, text='데이터 저장 위치', font=font(11, True)).pack(anchor='w')
    folder_entry = ttk.Entry(collection_tab, textvariable=folder_var)
    folder_entry.pack(fill='x', pady=(7, 8))
    fields.append(folder_entry)
    folder_buttons = ttk.Frame(collection_tab)
    folder_buttons.pack(fill='x')
    def choose_folder():
        path = filedialog.askdirectory(initialdir=folder_var.get())
        if path:
            folder_var.set(path)
    choose_button = ttk.Button(folder_buttons, text='폴더 변경', command=choose_folder)
    choose_button.pack(side='left')
    def open_folder():
        path = collector.dataset_root or Path(folder_var.get())
        if path.exists():
            subprocess.Popen(['xdg-open', str(path)])
    ttk.Button(folder_buttons, text='저장 위치 열기', command=open_folder).pack(side='right')
    ttk.Label(collection_tab, text='성공 저장만 횟수에 포함됩니다.\n실패와 종료 시 미저장 시도는 폐기합니다.\n시도 사이에는 직접 자세·카드를 정리하세요.',
              style='Muted.TLabel', wraplength=280, justify='left').pack(anchor='w', pady=16)

    for name in CAMERAS:
        ttk.Label(device_tab, text=f'{CAMERAS[name]} 카메라', font=font(11, True)).pack(anchor='w', pady=(8, 4))
        box = ttk.Combobox(device_tab, textvariable=source_vars[name], values=camera_sources())
        box.pack(fill='x', pady=(0, 8))
        fields.append(box)
        source_boxes.append(box)
    ttk.Label(device_tab, text='새 PC에서는 입력을 지정한 뒤\n미리보기로 실제 구도를 확인하세요.',
              style='Muted.TLabel').pack(anchor='w', pady=8)
    diagnostics = tk.Text(device_tab, height=8, bg='#e4eaf0', fg='#34485f', relief='flat',
                          wrap='word', padx=8, pady=8, font=font(9), state='disabled')
    diagnostics.pack(fill='both', expand=True, pady=(6, 0))
    log_widget = tk.Text(log_tab, height=6, width=1, bg='#142031', fg='#c9d7e8', relief='flat',
                         wrap='word', padx=10, pady=8, font=font(10, family='DejaVu Sans Mono'), state='disabled')
    scrollbar = ttk.Scrollbar(log_tab, command=log_widget.yview)
    log_widget.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side='right', fill='y')
    log_widget.pack(fill='both', expand=True)

    setup = ttk.Frame(side)
    # Reserve the connection controls before giving the remaining space to
    # the scrollable settings notebook, including on a 1280x800 laptop.
    setup.pack(side='bottom', fill='x', pady=(12, 0), before=notebook)
    setup.columnconfigure(0, weight=1)
    setup.columnconfigure(1, weight=1)
    busy = [False]
    results = queue.Queue()
    def capture_settings():
        collector.save_settings({'task': task_var.get(), 'target': target_var.get(),
                                 'velocity': velocity_var.get(), 'acceleration': acceleration_var.get(),
                                 'data_root': folder_var.get(),
                                 'sources': {name: var.get() for name, var in source_vars.items()}})
    def run_background(fn):
        if busy[0]:
            return
        busy[0] = True
        def work():
            try:
                result = fn()
                if isinstance(result, str):
                    results.put(result)
                    collector.logs.append(result)
            except Exception as exc:
                collector.error, collector.phase = str(exc), 'error'
            finally:
                busy[0] = False
        threading.Thread(target=work, daemon=True).start()
    def prepare(fn):
        try:
            capture_settings()
            collector.error = ''
            run_background(fn)
        except (ValueError, OSError) as exc:
            messagebox.showerror('설정 확인', str(exc))
    def refresh_devices():
        for box in source_boxes:
            box.configure(values=camera_sources())
        notebook.select(device_page)
        prepare(collector.diagnostics)
    for key, label, callback in (
        ('check', '연결 확인', refresh_devices),
        ('preview', '카메라 미리보기', lambda: prepare(collector.open_preview)),
        ('connect', '텔레옵 연결', lambda: prepare(collector.connect)),
    ):
        button = ttk.Button(setup, text=label, command=callback)
        if key == 'connect':
            button.grid(row=1, column=0, columnspan=2, sticky='ew', pady=(6, 0))
        else:
            button.grid(row=0, column=0 if key == 'check' else 1, sticky='ew',
                        padx=(0, 3) if key == 'check' else (3, 0))
        buttons[key] = button

    # Recording actions stay visible regardless of selected settings/log tab.
    footer = ttk.Frame(outer, padding=(0, 12, 0, 0))
    footer.grid(row=2, column=0, sticky='ew')
    detail_var = tk.StringVar(value='카메라 구도를 확인한 뒤 텔레옵을 연결하세요.')
    detail_label = ttk.Label(footer, textvariable=detail_var, style='Muted.TLabel', anchor='w')
    detail_label.pack(fill='x', pady=(0, 9))
    action_row = ttk.Frame(footer)
    action_row.pack(fill='x')
    def command(name):
        try:
            collector.command(name)
        except ValueError as exc:
            messagebox.showinfo('수집 상태', str(exc))
    for key, label, callback, button_style in (
        ('record', '● 녹화 시작', lambda: command('record'), 'Record.TButton'),
        ('save', '✓ 성공 저장', lambda: command('save'), 'Save.TButton'),
        ('discard', '↻ 실패 · 재시도', lambda: command('discard'), 'Retry.TButton'),
        ('stop', '세션 종료', lambda: command('stop'), 'Stop.TButton'),
    ):
        button = ttk.Button(action_row, text=label, command=callback, style=button_style)
        button.pack(side='left', padx=(0, 12))
        buttons[key] = button
    ttk.Label(action_row, text='왼손 핀치\n팔 이동 클러치', style='Muted.TLabel', justify='right').pack(side='right', padx=8)

    last_log = ['']
    closing = [False]
    def refresh():
        state = collector.status()
        phase, active = state['phase'], state['active']
        phase_var.set(PHASES.get(phase, phase))
        badge.configure(foreground='#ca414b' if phase in ('error', 'recording') else '#16765f')
        saved = state.get('saved', 0)
        count_var.set(f'성공 {saved} / {target_var.get()}')
        seconds = int(state.get('seconds', 0))
        timing_var.set(f'현재 시도  {seconds // 60:02d}:{seconds % 60:02d}   ·   {state.get("frames", 0)} 프레임')
        try:
            progress.configure(maximum=max(1, int(target_var.get())), value=saved)
        except ValueError:
            pass
        for key in ('check', 'preview', 'connect'):
            buttons[key].configure(state='disabled' if active or busy[0] else 'normal')
        for field in fields:
            field.configure(state='disabled' if active or busy[0] else 'normal')
        choose_button.configure(state='disabled' if active or busy[0] else 'normal')
        for key in ('record', 'save', 'discard', 'stop'):
            allowed = active and (key == 'stop' or (key == 'record' and phase == 'ready')
                                 or (key in ('save', 'discard') and phase == 'recording'))
            buttons[key].configure(state='normal' if allowed else 'disabled')
        if closing[0]:
            detail_var.set('미저장 시도를 폐기하고 데이터를 정리하는 중입니다…')
        elif state.get('error'):
            detail_var.set(state['error'].splitlines()[0][:180] + '  ·  자세한 내용은 로그에서 확인하세요.')
        elif collector.demo:
            detail_var.set('UI 시험 전용 · 모터와 카메라를 사용하지 않으며 더미 관절값만 저장합니다.')
        elif phase in ('connecting', 'warming_up'):
            detail_var.set('Quest Browser: http://localhost:8765 → Start hand tracking · 오른손을 보이게 해주세요.')
        elif phase == 'recording':
            detail_var.set('시연을 마치면 성공 저장, 실패하면 실패 · 재시도를 누르세요.')
        elif phase == 'ready':
            detail_var.set('텔레옵으로 자세·카드를 준비한 뒤 녹화 시작을 누르세요. 현재는 기록하지 않습니다.')
        else:
            detail_var.set('카메라 구도를 확인한 뒤 텔레옵을 연결하세요. 영상은 창 크기에 맞춰 표시됩니다.')
        detail_label.configure(wraplength=max(600, outer.winfo_width() - 20))
        for name, panel in panels.items():
            try:
                rgb, age = collector.frame(name)
                latest[name] = (None if rgb is None else Image.fromarray(rgb), age,
                                '카메라 미리보기로 구도를 확인하세요' if phase in ('idle', 'finished')
                                else '카메라 연결·입력 선택을 확인해 주세요')
            except Exception as exc:
                latest[name] = (None, None, f'영상 확인 필요\n{str(exc)[:120]}')
            panel.draw()
            if name in enlarged:
                enlarged[name][1].draw()
        while not results.empty():
            diagnostics.configure(state='normal')
            diagnostics.delete('1.0', 'end')
            diagnostics.insert('end', results.get_nowait())
            diagnostics.configure(state='disabled')
        text = '\n'.join(list(collector.logs))
        if text != last_log[0]:
            last_log[0] = text
            log_widget.configure(state='normal')
            log_widget.delete('1.0', 'end')
            log_widget.insert('end', text)
            log_widget.see('end')
            log_widget.configure(state='disabled')
        app.after(200, refresh)

    def close():
        if closing[0]:
            return
        if collector.active():
            if not messagebox.askyesno('수집 종료', '현재 미저장 시도는 폐기하고 세션을 종료할까요?'):
                return
            closing[0] = True
            collector.command('stop')
            def wait_closed():
                if collector.active():
                    app.after(250, wait_closed)
                else:
                    app.destroy()
            wait_closed()
        else:
            app.destroy()
    app.protocol('WM_DELETE_WINDOW', close)
    def fit_sidebar(event=None):
        width = content.winfo_width()
        if width > 800:
            # A maximized window can move to a smaller monitor. A stale sash
            # coordinate must not push the counter/settings completely offscreen.
            position = content.sashpos(0)
            content.sashpos(0, max(480, min(position, width - 330)))
    content.bind('<Configure>', lambda event: app.after_idle(fit_sidebar))
    app.after_idle(lambda: content.sashpos(0, max(480, content.winfo_width() - 350)))
    refresh()
    try:
        app.mainloop()
    finally:
        collector.close_preview()
        if collector.active():
            collector.command('stop')
