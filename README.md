# AutoFish – Overlay + OCR Fishing Helper (PyQt6)

A glassy, always-on-top overlay for desktop fishing minigames.  
It watches two pixels for color changes, OCRs a small ROI for the word **“MASH”**, and automates clicks with configurable timing. All settings persist in `config.json`.

---

## ✨ Features

- Transparent, draggable overlay UI (PyQt6)
- Persistent settings via `config.json` (auto-created on first run)
- Pixel-change detection on two probes: **PIX_A** and **PIX_B**
- OCR of a **ROI** using Tesseract to detect “MASH” (with preprocessing + fuzzy match)
- Burst/tap automation when “MASH” is seen
- Global hotkeys: **F8** start · **F6** stop · **F7** capture cursor XY  
- Debug snapshot: one-click full-screen capture with ROI/points annotated
- Live log + caught counter

---

## 🧰 Requirements

- **Windows 10/11** (tested)
- **Python 3.10+**
- **Tesseract OCR** (Windows installer; set path in UI or `config.json`)
- Packages:
  - `PyQt6`
  - `mss`
  - `pynput`
  - `pillow`
  - `pytesseract`

Install Python deps:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```
PyQt6
mss
pynput
pillow
pytesseract
```

### Install Tesseract (Windows)

1. Download and install: https://github.com/tesseract-ocr/tesseract (choose Windows installer).
2. Note the install path (commonly `C:\Program Files\Tesseract-OCR\tesseract.exe`).
3. In the app, set **Tesseract** path (right panel → Config) or edit `config.json`.

---

## 🚀 Running

```bash
python AutoFish.py
```

**Hotkeys (global):**
- **F8** – Start
- **F6** – Stop
- **F7** – Contextual XY capture (sets the focused pair in the UI, e.g., PIX_A X/Y or ROI edges)

---

## 🖱️ Quick Setup

1. Position your game/window where the UI elements are visible.
2. In **Settings → Coordinates**:
   - Click **📍 XY** beside **PIX_A** and **PIX_B** to capture the two probe points (or focus a spinbox and press **F7**).
   - For **ROI**: use **📍 TL** (top-left) and **📍 BR** (bottom-right).
3. In **Settings → Config**:
   - Verify **Tesseract** path.
   - Adjust **Min OCR Conf** (start ~55) and pixel tolerances (**Tolerance A/B**, **Break ΔB**) as needed.
4. Press **Start** or **F8**.

**Tip:** Use **Debug Snapshot** to verify ROI and points visually; it also logs OCR tokens/confidence.

---

## ⚙️ What the Bot Does

1. Waits, then holds mouse (pre-hold + post-A delays).
2. Watches **PIX_A** for a significant color change → releases.
3. Watches **PIX_B** for a change, then re-holds to “fish”.
4. While holding, OCRs the **ROI** each poll:
   - If “MASH” is detected (fuzzy match, conf ≥ threshold), it quickly releases, performs a **burst** of taps, then resumes hold.
5. On exit of hold (big ΔB), it counts a catch.

---

## 📄 Config Persistence

A `config.json` is created next to `AutoFish.py` on first run. Example:

```json
{
  "tesseract_path": "C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
  "poll_interval": 0.01,
  "min_conf": 55,
  "tol_a": 5.0,
  "tol_b": 50.0,
  "break_dB": 10.0,
  "pre_hold_delay": 1.0,
  "post_a_delay": 1.0,
  "burst_count": 1,
  "burst_delay": 0.05,
  "debug": false,
  "pix_a": [1794, 578],
  "pix_b": [833, 972],
  "roi": [796, 955, 950, 1008]
}
```

All values can be edited in the UI; changes are saved immediately.

---

## 🧪 OCR & Fuzzy Match

- Preprocessing: grayscale → upscaling → autocontrast → unsharp → threshold.
- Tesseract is run in single-word mode (`--psm 8`) with a **whitelist** (`MASH!`).
- Fuzzy match allows small OCR errors (up to edit distance 1), e.g., `MAS H`, `M4SH`, `MASH!`.

Tune:
- **Min OCR Conf**: raise if false positives; lower if misses.
- **ROI**: tight box around the word improves accuracy.
- **Threshold** (code uses fixed 170) can be tweaked if you edit the file.

---

## 🪟 Always-on-Top Notes (Windows)

- The app sets `WindowStaysOnTopHint`. Some full-screen/exclusive apps may still cover it.
- If needed, run both the game and this app **non-elevated** (or both elevated).
- Borderless windowed mode for the game helps.
- Optional Win32 “hard-topmost” helper can be added (see code comments in issues section).

---

## 🐛 Troubleshooting

**Tesseract not found / `tesseract is not in your PATH`**
- Install Tesseract and set its full path in the UI or `config.json`.
- Test in terminal: `"<path>\tesseract.exe" --version`.

**Overlay not on top**
- Use borderless windowed game mode.
- Toggle start/stop to bump z-order.
- As a last resort, add a timer that re-asserts topmost via Win32 (`SetWindowPos(HWND_TOPMOST)`).

**OCR misses “MASH”**
- Tighten ROI to the exact word.
- Lower **Min OCR Conf** slightly (e.g., 45–50).
- Increase UI contrast (in-game color settings).
- Check debug logs; use **Debug Snapshot** to see what Tesseract sees.

**Clicks not happening**
- Some games ignore synthetic input or require window focus.
- Try running Python as admin (only if your game runs as admin).
- Verify **burst_count** / **burst_delay**.

---

## 🔑 Hotkeys Recap

- **F8** – Start sequence  
- **F6** – Stop sequence  
- **F7** – Capture XY for the currently focused spinbox group (PIX_A / PIX_B / ROI edges)

Buttons:
- **Start / Stop** (same as hotkeys)
- **Debug Snapshot** – saves annotated screenshot and shows a scaled preview

---

## 📸 Debug Snapshot

- Saves an annotated PNG (temp folder) with:
  - ROI rectangle (green)
  - PIX_A crosshair (red)
  - PIX_B crosshair (blue)
- Path is printed in the log. Clicking the preview opens a full-size viewer.

---

## ⚠️ Disclaimer

This tool generates automated input. Use only where permitted by the game’s terms of service and local laws. You are responsible for compliance.

---

## 🤝 Contributing

Issues and PRs welcome. Please include:
- Python & OS version
- A small log excerpt (enable **Debug mode**)
- A debug snapshot (redact sensitive info if needed)

---

## 📄 License

MIT
