# fish_ui_glass_plus_persist.py
import sys, time, threading, os, json
from math import sqrt
from dataclasses import dataclass

from PyQt6 import QtCore, QtGui, QtWidgets

import mss
from pynput import mouse as pyn_mouse, keyboard as pyn_keyboard
from PIL import Image, ImageOps, ImageFilter
import pytesseract

def _lev(a: str, b: str) -> int:
    # tiny Levenshtein (edit distance) for fuzzy match
    if a == b: return 0
    if not a:  return len(b)
    if not b:  return len(a)
    dp = list(range(len(b)+1))
    for i, ca in enumerate(a, 1):
        prev = dp[0]
        dp[0] = i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            dp[j] = min(
                dp[j] + 1,          # deletion
                dp[j-1] + 1,        # insertion
                prev + (ca != cb),  # substitution
            )
            prev = cur
    return dp[-1]

def detect_mash(img: Image.Image, min_conf: int = 55, debug_log=None) -> bool:
    """
    Robustly detect the word 'MASH' in img (ROI).
    - Upscales, autocontrast, sharpens, binarizes
    - Tesseract constrained to a single word + whitelist
    - Fuzzy match allows small OCR errors (distance <= 1)
    """
    # 1) preprocess
    proc = img.convert("L").resize(
        (max(1, img.width*2), max(1, img.height*2)),
        Image.Resampling.BICUBIC
    )
    proc = ImageOps.autocontrast(proc, cutoff=2)
    proc = proc.filter(ImageFilter.UnsharpMask(radius=1.2, percent=150, threshold=3))
    # Simple fixed threshold (works well for UI text). Adjust 150..190 if needed.
    thr = proc.point(lambda p: 255 if p > 170 else 0)

    # 2) OCR (single word; restrict characters)
    cfg = "--oem 3 --psm 8 -c tessedit_char_whitelist=MASH!"
    data = pytesseract.image_to_data(
        thr,
        output_type=pytesseract.Output.DICT,
        config=cfg,
        lang="eng"
    )

    tokens = []
    for t, c in zip(data.get("text", []), data.get("conf", [])):
        t = (t or "").strip()
        try: conf = float(c)
        except: conf = -1.0
        if t and conf >= float(min_conf):
            tokens.append(t)

    # also try a raw read for cases where data dict is empty
    raw = pytesseract.image_to_string(thr, config=cfg, lang="eng") or ""
    if raw.strip():
        tokens.append(raw.strip())

    # 3) normalize + fuzzy check
    cand = "".join(tokens).replace(" ", "").upper()
    tokens_up = [t.replace(" ", "").upper() for t in tokens]
    haystack = tokens_up + ([cand] if cand else [])

    hit = any(
        ("MASH" in s) or (_lev(s, "MASH") <= 1)  # allow 1 edit (e.g. MAS H, M4SH, MASH!)
        for s in haystack if s
    )

    if debug_log is not None:
        debug_log(f"[ocr] tokens={tokens_up}  joined={cand}  hit={hit}")

    return hit

# --------------------- Config Manager ---------------------
class ConfigManager:
    def __init__(self, path=None):
        if path is None:
            # store next to script
            base = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(base, "config.json")
        self.path = path
        self.data = {}
        if not os.path.exists(self.path):
            self.data = self._default_config()
            self.save()
        else:
            self.load()
            self._ensure_defaults()

    def _default_config(self):
        return {
            "tesseract_path": r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            "poll_interval": 0.01,
            "min_conf": 55,
            "tol_a": 5.0,
            "tol_b": 50.0,
            "break_dB": 10.0,
            "pre_hold_delay": 1.0,
            "post_a_delay": 1.0,
            "burst_count": 1,
            "burst_delay": 0.05,
            "debug": False,
            "pix_a": [1794, 578],
            "pix_b": [833, 972],
            "roi": [796, 955, 950, 1008]
        }

    def _ensure_defaults(self):
        defaults = self._default_config()
        changed = False
        for k, v in defaults.items():
            if k not in self.data:
                self.data[k] = v
                changed = True
        if changed:
            self.save()

    def load(self):
        with open(self.path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value
        self.save()

# --------------------- Types ---------------------
@dataclass
class Rect:
    left: int
    top: int
    right: int
    bottom: int
    @property
    def width(self):  return max(0, self.right - self.left)
    @property
    def height(self): return max(0, self.bottom - self.top)
    def normalized(self):
        l = min(self.left, self.right); r = max(self.left, self.right)
        t = min(self.top, self.bottom); b = max(self.top, self.bottom)
        return Rect(l, t, r, b)

@dataclass
class RunConfig:
    tesseract_path: str
    poll_interval: float
    min_conf: int
    tol_a: float
    tol_b: float
    break_dB: float
    pre_hold_delay: float
    post_a_delay: float
    burst_count: int
    burst_delay: float

# --------------------- Defaults (for first run & fallbacks) ---------------------
DEFAULT_TESS = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
DEFAULTS = RunConfig(
    tesseract_path=DEFAULT_TESS,
    poll_interval=0.01,
    min_conf=55,
    tol_a=5.0,
    tol_b=50.0,
    break_dB=10.0,
    pre_hold_delay=1.0,
    post_a_delay=1.0,
    burst_count=1,
    burst_delay=0.05
)

DEFAULT_PIX_A = (1794, 578)
DEFAULT_PIX_B = (833, 972)
DEFAULT_ROI   = Rect(796, 955, 950, 1008).normalized()

CLICK_BUTTON  = pyn_mouse.Button.left

# --------------------- Utils ---------------------
def color_distance(a, b):
    return sqrt((int(a[0])-int(b[0]))**2 + (int(a[1])-int(b[1]))**2 + (int(a[2])-int(b[2]))**2)

def get_pixel_rgb(x, y, sct):
    mon = {"left": int(x), "top": int(y), "width": 1, "height": 1}
    img = sct.grab(mon)  # BGRA
    return (img.pixel(0, 0)[2], img.pixel(0, 0)[1], img.pixel(0, 0)[0])  # -> RGB

def grab_roi_image(sct: mss.mss, r: Rect) -> Image.Image:
    r = r.normalized()
    mon = {"left": int(r.left), "top": int(r.top), "width": int(r.width), "height": int(r.height)}
    if mon["width"] <= 0 or mon["height"] <= 0:
        raise ValueError("ROI has zero size")
    raw = sct.grab(mon)
    img = Image.frombytes("RGB", (raw.width, raw.height), raw.rgb)
    return img

def ocr_words_with_conf(img: Image.Image):
    out = []
    try:
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        n = len(data.get("text", []))
        for i in range(n):
            candidate = (data["text"][i] or "").strip()
            if not candidate:
                continue
            try:
                conf = float(data.get("conf", ["-1"])[i])
            except Exception:
                conf = -1.0
            out.append((candidate, conf))
    except Exception as e:
        out.append(("[ocr-error]", -1.0))
        out.append((str(e), -1.0))
    return out

def first_confident_word(img: Image.Image, min_conf: int):
    for w, c in ocr_words_with_conf(img):
        if c >= float(min_conf):
            return w
    return None

# --------------------- Worker Thread ---------------------
class FisherWorker(QtCore.QThread):
    log = QtCore.pyqtSignal(str)
    countChanged = QtCore.pyqtSignal(int)
    stateChanged = QtCore.pyqtSignal(str)

    def __init__(self, pix_a, pix_b, roi_rect: Rect, cfg: RunConfig, debug_mode: bool, parent=None):
        super().__init__(parent)
        self._pix_a = pix_a
        self._pix_b = pix_b
        self._roi = roi_rect
        self._cfg = cfg
        self._debug = debug_mode

        self._stop_event = threading.Event()
        self._quit_event = threading.Event()
        self._left_down = False
        self._mouse = pyn_mouse.Controller()
        self._fish_caught = 0

    def stop(self): self._stop_event.set()
    @property
    def fish_caught(self): return self._fish_caught

    def _press(self):
        if not self._left_down:
            self._mouse.press(CLICK_BUTTON); self._left_down = True
    def _release(self):
        if self._left_down:
            self._mouse.release(CLICK_BUTTON); self._left_down = False

    def _check_stop(self):
        if self._stop_event.is_set() or self._quit_event.is_set():
            raise RuntimeError("stopped")

    def _wait_until_pixel_change(self, x, y, sct, tolerance, label="PIX"):
        base = get_pixel_rgb(x, y, sct)
        self.log.emit(f"{label} base RGB: {base}")
        while True:
            self._check_stop()
            rgb = get_pixel_rgb(x, y, sct)
            dist = color_distance(rgb, base)
            if self._debug:
                self.log.emit(f"[dbg] {label} Δ={dist:.2f} rgb={rgb}")
            if dist > tolerance:
                self.log.emit(f"{label} changed (Δ={dist:.2f})")
                return rgb
            time.sleep(self._cfg.poll_interval)

    def _burst_clicks(self, n, delay):
        for _ in range(n):
            self._check_stop()
            self._mouse.click(CLICK_BUTTON, 1)
            time.sleep(delay)

    def run(self):
        pytesseract.pytesseract.tesseract_cmd = self._cfg.tesseract_path
        self._stop_event.clear()
        self.stateChanged.emit("running")
        self.log.emit(f"Sequence started with A={self._pix_a} B={self._pix_b} ROI={self._roi}")
        self.log.emit(
            f"Config: poll={self._cfg.poll_interval}s, min_conf={self._cfg.min_conf}, "
            f"tolA={self._cfg.tol_a}, tolB={self._cfg.tol_b}, breakΔB={self._cfg.break_dB}, "
            f"preHold={self._cfg.pre_hold_delay}s, postA={self._cfg.post_a_delay}s, "
            f"burst={self._cfg.burst_count}@{self._cfg.burst_delay}s, debug={self._debug}"
        )
        with mss.mss() as sct:
            try:
                while True:
                    self._check_stop()
                    self.log.emit("Start cycle")

                    time.sleep(self._cfg.pre_hold_delay)
                    self._press()
                    time.sleep(self._cfg.post_a_delay)

                    self._wait_until_pixel_change(*self._pix_a, sct, tolerance=self._cfg.tol_a, label="A")
                    self._release()

                    self._wait_until_pixel_change(*self._pix_b, sct, tolerance=self._cfg.tol_b, label="B")
                    time.sleep(self._cfg.pre_hold_delay)
                    self.log.emit("Now Holding")
                    self._press()
                    ref_b = get_pixel_rgb(*self._pix_b, sct)

                    while True:
                        self._check_stop()
                        new_b = get_pixel_rgb(*self._pix_b, sct)
                        dB = color_distance(ref_b, new_b)
                        if self._debug:
                            self.log.emit(f"[dbg] ΔB={dB:.2f} rgbB={new_b}")
                        if dB > self._cfg.break_dB:
                            self.log.emit(f"B changed > {self._cfg.break_dB:.2f}, break hold")
                            break

                        if self._roi is not None:
                            try:
                                img = grab_roi_image(sct, self._roi)
                                hit = detect_mash(img, self._cfg.min_conf, debug_log=(self.log.emit if self._debug else None))
                                if hit:
                                    self.log.emit("OCR 'MASH' -> tap burst")
                                    self._release()
                                    self._burst_clicks(self._cfg.burst_count, self._cfg.burst_delay)
                                    self._press()

                            except Exception as e:
                                self.log.emit(f"[roi-error] {e}")

                        time.sleep(self._cfg.poll_interval)

                    self._release()
                    self._fish_caught += 1
                    self.countChanged.emit(self._fish_caught)
                    self.log.emit("Catched ✓")
                    time.sleep(self._cfg.post_a_delay)

            except RuntimeError:
                self.log.emit("Sequence stopped.")
            finally:
                self._release()
                self.stateChanged.emit("stopped")

# --------------------- Global Hotkeys (F8/F6/F7/ESC) ---------------------
class HotkeyThread(QtCore.QThread):
    startRequested = QtCore.pyqtSignal()
    stopRequested = QtCore.pyqtSignal()
    quitRequested = QtCore.pyqtSignal()
    setCoordRequested = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self._listener = None

    def run(self):
        def on_press(key):
            try:
                if key == pyn_keyboard.Key.f8:
                    self.startRequested.emit()
                elif key == pyn_keyboard.Key.f6:
                    self.stopRequested.emit()
                elif key == pyn_keyboard.Key.f7:
                    self.setCoordRequested.emit()
            except Exception:
                return False

        from pynput import keyboard as _kb
        self._listener = _kb.Listener(on_press=on_press)
        self._listener.start()
        self._listener.join()

    def stop(self):
        if self._listener:
            self._listener.stop()

# --------------------- Clickable label ---------------------
class ClickableLabel(QtWidgets.QLabel):
    clicked = QtCore.pyqtSignal()
    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent):
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mouseReleaseEvent(ev)

# --------------------- Overlay Window ---------------------
BG_ALPHA = 200
RADIUS   = 14

class GlassWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.cfg = ConfigManager()  # persists values
        self.setWindowFlags(
            QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowSystemMenuHint
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setWindowFlag(QtCore.Qt.WindowType.WindowStaysOnTopHint, True)
        self.resize(1040, 640)

        self._drag_pos = None
        self._panel_color = QtGui.QColor(28, 30, 36, BG_ALPHA)
        self._card_color = QtGui.QColor(22, 24, 30, int(BG_ALPHA*0.95))
        self._snap_qimage = None
        self._snap_path = None

        # UI
        self.titleBar = self._make_titlebar("Fishing Helper — Glass+Config+Debug (persist)")
        self.content = QtWidgets.QWidget()
        self.content.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.content.setLayout(self._make_content_layout())
        self._load_config_into_widgets()
        self._wire_persistence()

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)
        root.addWidget(self.titleBar)
        root.addWidget(self.content, 1)

        # hotkeys
        self.hotkeys = HotkeyThread()
        self.hotkeys.startRequested.connect(self._start_worker)
        self.hotkeys.stopRequested.connect(self._stop_worker)
        self.hotkeys.quitRequested.connect(self.close)
        self.hotkeys.setCoordRequested.connect(self._capture_f7_pair)
        self.hotkeys.start()

        self.worker: FisherWorker | None = None

    # ---------- UI builders ----------
    def _make_titlebar(self, title: str) -> QtWidgets.QWidget:
        bar = QtWidgets.QWidget(); bar.setFixedHeight(44)
        bar.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        lay = QtWidgets.QHBoxLayout(bar); lay.setContentsMargins(10, 8, 8, 8); lay.setSpacing(8)

        lbl = QtWidgets.QLabel(title)
        f = lbl.font(); f.setPointSize(11); f.setBold(True)
        lbl.setFont(f); lbl.setStyleSheet("color:white;")

        btn_min = QtWidgets.QPushButton("—")
        btn_cls = QtWidgets.QPushButton("✕")
        for b in (btn_min, btn_cls):
            b.setFixedSize(36, 28)
            b.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            b.setStyleSheet("""
                QPushButton {
                    color: #e6e8f0; background: rgba(255,255,255,0.08);
                    border: none; border-radius: 6px;
                }
                QPushButton:hover { background: rgba(255,255,255,0.15); }
                QPushButton:pressed { background: rgba(255,255,255,0.22); }
            """)
        btn_min.clicked.connect(self.showMinimized)
        btn_cls.clicked.connect(self.close)
        lay.addWidget(lbl, 1); lay.addWidget(btn_min); lay.addWidget(btn_cls)
        return bar

    def _make_card(self, title: str | None, body: QtWidgets.QWidget) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        w.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        v = QtWidgets.QVBoxLayout(w); v.setContentsMargins(14,14,14,14); v.setSpacing(8)
        if title:
            lbl = QtWidgets.QLabel(title)
            f = lbl.font(); f.setPointSize(12); f.setBold(True)
            lbl.setFont(f); lbl.setStyleSheet("color:#e6e8f0;")
            v.addWidget(lbl)
        v.addWidget(body, 1)
        def paint_event(e, _w=w):
            p = QtGui.QPainter(_w); p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            r = _w.rect()
            path = QtGui.QPainterPath()
            path.addRoundedRect(QtCore.QRectF(r.adjusted(0,0,-1,-1)), 12, 12)
            p.fillPath(path, self._card_color)
        w.paintEvent = paint_event
        return w

    def _make_xy_row(self, label_text: str, x_spin: QtWidgets.QSpinBox, y_spin: QtWidgets.QSpinBox):
        row = QtWidgets.QHBoxLayout(); row.setSpacing(6)
        lbl = QtWidgets.QLabel(label_text); lbl.setStyleSheet("color:#e6e8f0;")
        for sp, suffix in ((x_spin, "x"), (y_spin, "y")):
            sp.setRange(0, 20000); sp.setFixedWidth(110)
            sp.setObjectName(f"{label_text.lower().replace(' ','_')}_{suffix}")
        btn_xy = QtWidgets.QToolButton(); btn_xy.setText("📍 XY"); btn_xy.setToolTip(f"Capture {label_text} X+Y")
        btn_xy.clicked.connect(lambda: (self._capture_pair(x_spin, y_spin), self._save_coords_to_config()))
        row.addWidget(lbl)
        row.addWidget(x_spin); row.addWidget(y_spin); row.addWidget(btn_xy)
        row.addStretch(1)
        return row

    def _make_lr_tb_row(self, label_text: str, l_spin, t_spin, r_spin, b_spin):
        grid = QtWidgets.QGridLayout(); grid.setHorizontalSpacing(6); grid.setVerticalSpacing(4)
        title = QtWidgets.QLabel(label_text); title.setStyleSheet("color:#e6e8f0;")
        grid.addWidget(title, 0, 0, 1, 6)
        for sp, name in ((l_spin,"left"),(t_spin,"top"),(r_spin,"right"),(b_spin,"bottom")):
            sp.setRange(0, 20000); sp.setFixedWidth(110)
            sp.setObjectName(f"roi_{name}")
        btn_tl = QtWidgets.QToolButton(); btn_tl.setText("📍 TL"); btn_tl.setToolTip("Capture Top-Left (left+top)")
        btn_br = QtWidgets.QToolButton(); btn_br.setText("📍 BR"); btn_br.setToolTip("Capture Bottom-Right (right+bottom)")
        btn_tl.clicked.connect(lambda: (self._capture_pair(l_spin, t_spin), self._save_coords_to_config()))
        btn_br.clicked.connect(lambda: (self._capture_pair(r_spin, b_spin), self._save_coords_to_config()))

        grid.addWidget(QtWidgets.QLabel("left (x)"),   1, 0); grid.addWidget(l_spin, 1, 1)
        grid.addWidget(QtWidgets.QLabel("top (y)"),    2, 0); grid.addWidget(t_spin, 2, 1)
        grid.addWidget(QtWidgets.QLabel("right (x)"),  3, 0); grid.addWidget(r_spin, 3, 1)
        grid.addWidget(QtWidgets.QLabel("bottom (y)"), 4, 0); grid.addWidget(b_spin, 4, 1)
        grid.addWidget(btn_tl, 1, 2); grid.addWidget(btn_br, 3, 2)
        return grid

    def _make_config_panel(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(w); form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)

        # tesseract path
        self.ed_tess = QtWidgets.QLineEdit()
        btn_browse = QtWidgets.QToolButton(); btn_browse.setText("…")
        btn_browse.clicked.connect(self._browse_tesseract)
        h1 = QtWidgets.QHBoxLayout(); h1.addWidget(self.ed_tess, 1); h1.addWidget(btn_browse)
        p1 = QtWidgets.QWidget(); p1.setLayout(h1); form.addRow("Tesseract:", p1)

        def dspin(minv, maxv, step, val, w=110):
            ds = QtWidgets.QDoubleSpinBox(); ds.setRange(minv, maxv); ds.setSingleStep(step); ds.setValue(val); ds.setFixedWidth(w); return ds
        def ispin(minv, maxv, step, val, w=110):
            s = QtWidgets.QSpinBox(); s.setRange(minv, maxv); s.setSingleStep(step); s.setValue(val); s.setFixedWidth(w); return s

        # defaults for initial creation; actual values loaded from config after UI build
        self.sp_poll  = dspin(0.001, 1.0, 0.001, DEFAULTS.poll_interval)
        self.sp_conf  = ispin(0, 100, 1, DEFAULTS.min_conf)
        self.sp_tolA  = dspin(0.0, 255.0, 0.5, DEFAULTS.tol_a)
        self.sp_tolB  = dspin(0.0, 255.0, 0.5, DEFAULTS.tol_b)
        self.sp_break = dspin(0.0, 255.0, 0.5, DEFAULTS.break_dB)
        self.sp_pre   = dspin(0.0, 3.0, 0.05, DEFAULTS.pre_hold_delay)
        self.sp_postA = dspin(0.0, 3.0, 0.05, DEFAULTS.post_a_delay)
        self.sp_bcnt  = ispin(1, 10, 1, DEFAULTS.burst_count)
        self.sp_bdel  = dspin(0.0, 0.5, 0.01, DEFAULTS.burst_delay)

        form.addRow("Poll Interval (s):", self.sp_poll)
        form.addRow("Min OCR Conf:", self.sp_conf)
        form.addRow("Tolerance A:", self.sp_tolA)
        form.addRow("Tolerance B:", self.sp_tolB)
        form.addRow("Break ΔB:", self.sp_break)
        form.addRow("Pre-Hold Delay (s):", self.sp_pre)
        form.addRow("Post-A Delay (s):", self.sp_postA)
        form.addRow("Burst Count:", self.sp_bcnt)
        form.addRow("Burst Delay (s):", self.sp_bdel)

        self.cb_debug = QtWidgets.QCheckBox("Debug mode (logs Δcolor & OCR word each tick)")
        form.addRow("", self.cb_debug)
        return w

    def _make_content_layout(self) -> QtWidgets.QLayout:
        # Left: Controls + Log + Preview
        left = QtWidgets.QWidget(); lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(0,0,0,0); lv.setSpacing(8)

        stats = QtWidgets.QHBoxLayout()
        self.count_lbl = QtWidgets.QLabel("Caught: 0")
        self.state_lbl = QtWidgets.QLabel("State: idle")
        for lbl in (self.count_lbl, self.state_lbl):
            lbl.setStyleSheet("color:#e6e8f0;")
        stats.addWidget(self.count_lbl, 1); stats.addWidget(self.state_lbl, 1)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_stop  = QtWidgets.QPushButton("Stop")
        self.btn_debugsnap = QtWidgets.QPushButton("Debug Snapshot")
        for b in (self.btn_start, self.btn_stop, self.btn_debugsnap):
            b.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            b.setMinimumHeight(36)
            b.setStyleSheet("""
                QPushButton {
                    color:#e6e8f0; background: rgba(255,255,255,0.08);
                    border:none; border-radius:10px; padding:6px 12px;
                }
                QPushButton:hover { background: rgba(255,255,255,0.15); }
                QPushButton:pressed { background: rgba(255,255,255,0.22); }
            """)
        self.btn_start.clicked.connect(self._start_worker)
        self.btn_stop.clicked.connect(self._stop_worker)
        self.btn_debugsnap.clicked.connect(self._debug_snapshot)
        btn_row.addWidget(self.btn_start); btn_row.addWidget(self.btn_stop); btn_row.addWidget(self.btn_debugsnap)

        self.log_edit = QtWidgets.QPlainTextEdit(); self.log_edit.setReadOnly(True)
        self.log_edit.setStyleSheet("QPlainTextEdit{background:transparent;color:#dfe2ea;border:none;}")

        lv.addLayout(stats); lv.addLayout(btn_row); lv.addWidget(self.log_edit, 1)

        # preview area
        self.snap_label = ClickableLabel()
        self.snap_label.setMinimumHeight(220)
        self.snap_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.snap_label.setStyleSheet("background: rgba(255,255,255,0.04); border-radius: 8px;")
        self.snap_label.clicked.connect(self._open_full_snapshot)
        lv.addWidget(self.snap_label, 0)

        left_card = self._make_card("Controls & Log", left)

        # Right: Coordinates + Config tabs
        tabs = QtWidgets.QTabWidget()
        tabs.setStyleSheet("QTabWidget::pane{border:0;} QTabBar::tab{padding:6px 10px;}")

        # Coordinates tab
        coord = QtWidgets.QWidget(); rv = QtWidgets.QVBoxLayout(coord); rv.setContentsMargins(0,0,0,0); rv.setSpacing(8)
        self.a_x = QtWidgets.QSpinBox(); self.a_y = QtWidgets.QSpinBox()
        self.b_x = QtWidgets.QSpinBox(); self.b_y = QtWidgets.QSpinBox()
        for sp in (self.a_x, self.a_y, self.b_x, self.b_y):
            sp.setRange(0, 20000); sp.setFixedWidth(110)
        rv.addLayout(self._make_xy_row("PIX_A", self.a_x, self.a_y))
        rv.addLayout(self._make_xy_row("PIX_B", self.b_x, self.b_y))

        self.roi_l = QtWidgets.QSpinBox(); self.roi_t = QtWidgets.QSpinBox()
        self.roi_r = QtWidgets.QSpinBox(); self.roi_b = QtWidgets.QSpinBox()
        for sp in (self.roi_l, self.roi_t, self.roi_r, self.roi_b):
            sp.setRange(0, 20000); sp.setFixedWidth(110)
        rv.addLayout(self._make_lr_tb_row("ROI", self.roi_l, self.roi_t, self.roi_r, self.roi_b))
        rv.addStretch(1)

        tabs.addTab(self._make_card(None, coord), "Coordinates")
        tabs.addTab(self._make_card(None, self._make_config_panel()), "Config")

        right_card = self._make_card("Settings", tabs)

        split = QtWidgets.QHBoxLayout()
        split.setContentsMargins(0,0,0,0); split.setSpacing(8)
        split.addWidget(left_card, 6); split.addWidget(right_card, 6)
        return split

    # ---------- Config load & persist wiring ----------
    def _load_config_into_widgets(self):
        # Coords
        pa = self.cfg.get("pix_a", list(DEFAULT_PIX_A))
        pb = self.cfg.get("pix_b", list(DEFAULT_PIX_B))
        self.a_x.setValue(int(pa[0])); self.a_y.setValue(int(pa[1]))
        self.b_x.setValue(int(pb[0])); self.b_y.setValue(int(pb[1]))

        r = self.cfg.get("roi", [DEFAULT_ROI.left, DEFAULT_ROI.top, DEFAULT_ROI.right, DEFAULT_ROI.bottom])
        self.roi_l.setValue(int(r[0])); self.roi_t.setValue(int(r[1]))
        self.roi_r.setValue(int(r[2])); self.roi_b.setValue(int(r[3]))

        # Configs
        self.ed_tess.setText(self.cfg.get("tesseract_path", DEFAULTS.tesseract_path))
        self.sp_poll.setValue(float(self.cfg.get("poll_interval", DEFAULTS.poll_interval)))
        self.sp_conf.setValue(int(self.cfg.get("min_conf", DEFAULTS.min_conf)))
        self.sp_tolA.setValue(float(self.cfg.get("tol_a", DEFAULTS.tol_a)))
        self.sp_tolB.setValue(float(self.cfg.get("tol_b", DEFAULTS.tol_b)))
        self.sp_break.setValue(float(self.cfg.get("break_dB", DEFAULTS.break_dB)))
        self.sp_pre.setValue(float(self.cfg.get("pre_hold_delay", DEFAULTS.pre_hold_delay)))
        self.sp_postA.setValue(float(self.cfg.get("post_a_delay", DEFAULTS.post_a_delay)))
        self.sp_bcnt.setValue(int(self.cfg.get("burst_count", DEFAULTS.burst_count)))
        self.sp_bdel.setValue(float(self.cfg.get("burst_delay", DEFAULTS.burst_delay)))
        self.cb_debug.setChecked(bool(self.cfg.get("debug", False)))

    def _wire_persistence(self):
        # Coords: save both values on change for each pair
        def save_pix_a(): self.cfg.set("pix_a", [self.a_x.value(), self.a_y.value()])
        def save_pix_b(): self.cfg.set("pix_b", [self.b_x.value(), self.b_y.value()])
        def save_roi():   self.cfg.set("roi", [self.roi_l.value(), self.roi_t.value(), self.roi_r.value(), self.roi_b.value()])

        self.a_x.valueChanged.connect(save_pix_a); self.a_y.valueChanged.connect(save_pix_a)
        self.b_x.valueChanged.connect(save_pix_b); self.b_y.valueChanged.connect(save_pix_b)
        for sp in (self.roi_l, self.roi_t, self.roi_r, self.roi_b):
            sp.valueChanged.connect(save_roi)

        # Config widgets
        self.ed_tess.textChanged.connect(lambda s: self.cfg.set("tesseract_path", s))
        self.sp_poll.valueChanged.connect(lambda v: self.cfg.set("poll_interval", float(v)))
        self.sp_conf.valueChanged.connect(lambda v: self.cfg.set("min_conf", int(v)))
        self.sp_tolA.valueChanged.connect(lambda v: self.cfg.set("tol_a", float(v)))
        self.sp_tolB.valueChanged.connect(lambda v: self.cfg.set("tol_b", float(v)))
        self.sp_break.valueChanged.connect(lambda v: self.cfg.set("break_dB", float(v)))
        self.sp_pre.valueChanged.connect(lambda v: self.cfg.set("pre_hold_delay", float(v)))
        self.sp_postA.valueChanged.connect(lambda v: self.cfg.set("post_a_delay", float(v)))
        self.sp_bcnt.valueChanged.connect(lambda v: self.cfg.set("burst_count", int(v)))
        self.sp_bdel.valueChanged.connect(lambda v: self.cfg.set("burst_delay", float(v)))
        self.cb_debug.toggled.connect(lambda b: self.cfg.set("debug", bool(b)))

    # Helper to save coords explicitly (used by capture buttons)
    def _save_coords_to_config(self):
        self.cfg.set("pix_a", [self.a_x.value(), self.a_y.value()])
        self.cfg.set("pix_b", [self.b_x.value(), self.b_y.value()])
        self.cfg.set("roi",   [self.roi_l.value(), self.roi_t.value(), self.roi_r.value(), self.roi_b.value()])

    # ---------- Capture helpers ----------
    def _capture_pair(self, spin_x: QtWidgets.QSpinBox, spin_y: QtWidgets.QSpinBox):
        x, y = map(int, pyn_mouse.Controller().position)
        spin_x.setValue(x); spin_y.setValue(y)

    @QtCore.pyqtSlot()
    def _capture_f7_pair(self):
        """F7 sets both X+Y for the relevant group based on focused widget."""
        w = QtWidgets.QApplication.focusWidget()
        x, y = map(int, pyn_mouse.Controller().position)
        name = (w.objectName() if isinstance(w, QtWidgets.QWidget) else "").lower()
        if name.startswith("pix_a_"):
            self.a_x.setValue(x); self.a_y.setValue(y)
        elif name.startswith("pix_b_"):
            self.b_x.setValue(x); self.b_y.setValue(y)
        elif name in ("roi_left", "roi_top"):
            self.roi_l.setValue(x); self.roi_t.setValue(y)
        elif name in ("roi_right", "roi_bottom"):
            self.roi_r.setValue(x); self.roi_b.setValue(y)
        else:
            # fallback: set PIX_A
            self.a_x.setValue(x); self.a_y.setValue(y)
        self._save_coords_to_config()

    def _browse_tesseract(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select tesseract.exe",
                                                        os.path.dirname(self.cfg.get("tesseract_path", DEFAULT_TESS)) or "C:\\",
                                                        "Executables (*.exe)")
        if path:
            self.ed_tess.setText(path)  # textChanged will persist

    # ---------- Worker control ----------
    def _collect_cfg(self) -> RunConfig:
        return RunConfig(
            tesseract_path=self.ed_tess.text().strip(),
            poll_interval=float(self.sp_poll.value()),
            min_conf=int(self.sp_conf.value()),
            tol_a=float(self.sp_tolA.value()),
            tol_b=float(self.sp_tolB.value()),
            break_dB=float(self.sp_break.value()),
            pre_hold_delay=float(self.sp_pre.value()),
            post_a_delay=float(self.sp_postA.value()),
            burst_count=int(self.sp_bcnt.value()),
            burst_delay=float(self.sp_bdel.value())
        )

    def _start_worker(self):
        if self.worker and self.worker.isRunning():
            self._append_log("[info] already running"); return

        pix_a = (self.a_x.value(), self.a_y.value())
        pix_b = (self.b_x.value(), self.b_y.value())
        roi_rect = Rect(self.roi_l.value(), self.roi_t.value(), self.roi_r.value(), self.roi_b.value()).normalized()
        cfg = self._collect_cfg()

        pytesseract.pytesseract.tesseract_cmd = cfg.tesseract_path

        self.worker = FisherWorker(pix_a, pix_b, roi_rect, cfg, self.cb_debug.isChecked())
        self.worker.log.connect(self._append_log)
        self.worker.countChanged.connect(self._set_count)
        self.worker.stateChanged.connect(self._set_state)
        self.worker.start()
        self._append_log(f"[start] A={pix_a} B={pix_b} ROI={roi_rect}")

    def _stop_worker(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop(); self._append_log("[stop] requested")
        else:
            self._append_log("[info] nothing to stop")

    # ---------- Debug snapshot ----------
    def _debug_snapshot(self):
        import tempfile
        from PIL import ImageDraw, ImageFont, Image

        cfg = self._collect_cfg()
        pytesseract.pytesseract.tesseract_cmd = cfg.tesseract_path

        pix_a = (self.a_x.value(), self.a_y.value())
        pix_b = (self.b_x.value(), self.b_y.value())
        roi_rect = Rect(self.roi_l.value(), self.roi_t.value(),
                        self.roi_r.value(), self.roi_b.value()).normalized()

        try:
            with mss.mss() as sct:
                # ---- Original logs (unchanged) ----
                ba = get_pixel_rgb(*pix_a, sct); bb = get_pixel_rgb(*pix_b, sct)
                self._append_log(f"[snap] A rgb={ba}  B rgb={bb}")
                time.sleep(0.01)
                ba2 = get_pixel_rgb(*pix_a, sct); bb2 = get_pixel_rgb(*pix_b, sct)
                self._append_log(f"[snap] A Δ={color_distance(ba2, ba):.2f}  B Δ={color_distance(bb2, bb):.2f}")

                img_roi = grab_roi_image(sct, roi_rect)
                words = ocr_words_with_conf(img_roi)
                if not words:
                    self._append_log("[snap] OCR: (no words)")
                else:
                    self._append_log("[snap] OCR words (word, conf):")
                    max_show = 25
                    for i, (w, c) in enumerate(words[:max_show]):
                        self._append_log(f"   {i+1:02d}. {w!r}  conf={c:.1f}")
                    if len(words) > max_show:
                        self._append_log(f"   ... and {len(words) - max_show} more")

                # ---- Fullscreen capture and annotation ----
                raw = sct.grab(sct.monitors[1])  # primary monitor
                base = Image.frombytes("RGB", (raw.width, raw.height), raw.rgb)

                draw = ImageDraw.Draw(base)
                font = ImageFont.load_default()

                # Direct screen coords
                ax, ay = pix_a
                bx, by = pix_b
                rlx, rty = roi_rect.left, roi_rect.top
                rrx, rby = roi_rect.right, roi_rect.bottom

                # ROI rectangle (green)
                draw.rectangle([(rlx, rty), (rrx, rby)], outline=(0, 220, 120), width=3)
                draw.text((rlx + 4, rty + 4), "ROI", fill=(0, 220, 120), font=font)

                def crosshair(x, y, color, label):
                    r = 8
                    draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=3)
                    draw.line((x - 12, y, x + 12, y), fill=color, width=2)
                    draw.line((x, y - 12, x, y + 12), fill=color, width=2)
                    draw.text((x + 10, y - 14), label, fill=color, font=font)

                # PIX_A (red) and PIX_B (blue)
                crosshair(ax, ay, (230, 70, 70),   "PIX_A")
                crosshair(bx, by, (80, 160, 255),  "PIX_B")

                # Legend
                legend_y = 6
                draw.text((6, legend_y), "Legend:", fill=(240, 240, 240), font=font)
                draw.text((64, legend_y), "ROI (green), PIX_A (red), PIX_B (blue)",
                          fill=(200, 200, 200), font=font)

                # Save annotated image (temp) and remember path for cleanup
                out_path = os.path.join(tempfile.gettempdir(), "debug_snapshot_annotated.png")
                base.save(out_path)
                self._snap_path = out_path
                self._append_log(f"[snap] Annotated screenshot saved: {out_path}")

                # ---- Embed into left preview (scaled) ----
                qimg = QtGui.QImage(base.tobytes(), base.width, base.height,
                                    base.width * 3, QtGui.QImage.Format.Format_RGB888)
                self._snap_qimage = qimg.copy()  # keep our own copy

                if self.snap_label.size().width() > 0 and self.snap_label.size().height() > 0:
                    pix = QtGui.QPixmap.fromImage(self._snap_qimage).scaled(
                        self.snap_label.size(),
                        QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                        QtCore.Qt.TransformationMode.SmoothTransformation
                    )
                    self.snap_label.setPixmap(pix)

        except Exception as e:
            self._append_log(f"[snap-error] {e}")

    # ---------- Open full snapshot ----------
    def _open_full_snapshot(self):
        if self._snap_qimage is None:
            return
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Full Debug Snapshot")
        dlg.resize(1200, 800)

        scroll = QtWidgets.QScrollArea(dlg)
        scroll.setWidgetResizable(True)
        lbl = QtWidgets.QLabel()
        lbl.setPixmap(QtGui.QPixmap.fromImage(self._snap_qimage))
        lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        scroll.setWidget(lbl)

        lay = QtWidgets.QVBoxLayout(dlg)
        lay.addWidget(scroll)
        btn = QtWidgets.QPushButton("Close")
        btn.clicked.connect(dlg.accept)
        lay.addWidget(btn, alignment=QtCore.Qt.AlignmentFlag.AlignRight)

        dlg.exec()

    # ---------- UI plumbing ----------
    @QtCore.pyqtSlot(str)
    def _append_log(self, text: str):
        self.log_edit.appendPlainText(text)
        sb = self.log_edit.verticalScrollBar(); sb.setValue(sb.maximum())

    @QtCore.pyqtSlot(int)
    def _set_count(self, n: int):
        self.count_lbl.setText(f"Caught: {n}")

    @QtCore.pyqtSlot(str)
    def _set_state(self, s: str):
        self.state_lbl.setText(f"State: {s}")

    # ---------- Drag + Paint + Resize ----------
    def mousePressEvent(self, e: QtGui.QMouseEvent):
        if e.button() == QtCore.Qt.MouseButton.LeftButton:
            self._drag_pos = e.globalPosition().toPoint() - self.frameGeometry().topLeft(); e.accept()
    def mouseMoveEvent(self, e: QtGui.QMouseEvent):
        if self._drag_pos and e.buttons() & QtCore.Qt.MouseButton.LeftButton:
            self.move(e.globalPosition().toPoint() - self._drag_pos); e.accept()
    def mouseReleaseEvent(self, e: QtGui.QMouseEvent):
        self._drag_pos = None

    def paintEvent(self, e: QtGui.QPaintEvent):
        p = QtGui.QPainter(self); p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        r = self.rect()
        path = QtGui.QPainterPath()
        path.addRoundedRect(QtCore.QRectF(r.adjusted(0,0,-1,-1)), RADIUS, RADIUS)
        p.fillPath(path, self._panel_color)

    def resizeEvent(self, e: QtGui.QResizeEvent):
        super().resizeEvent(e)
        if getattr(self, "snap_label", None) and self._snap_qimage is not None:
            pix = QtGui.QPixmap.fromImage(self._snap_qimage).scaled(
                self.snap_label.size(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation
            )
            self.snap_label.setPixmap(pix)

    def closeEvent(self, e: QtGui.QCloseEvent):
        try:
            if self.worker and self.worker.isRunning():
                self.worker.stop()
                self.worker.wait(1500)
        finally:
            try:
                self.hotkeys.stop()
            except Exception:
                pass
            # remove temp screenshot if present
            try:
                if self._snap_path and os.path.exists(self._snap_path):
                    os.remove(self._snap_path)
            except Exception:
                pass
        sys.exit(0)   # immediately kills program

# --------------------- main ---------------------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = GlassWindow()
    w.show()
    sys.exit(app.exec())
