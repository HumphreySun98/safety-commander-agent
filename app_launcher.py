"""
app_launcher.py — run SafetyCommander as a desktop APP window (not a browser tab).

Wraps the existing Flask dashboard in a native window via pywebview. The autonomous
shift loop runs in the background exactly as in dashboard.py — this only changes the
shell so it looks like a product on the big screen.

    pip install pywebview
    python app_launcher.py
    # video mode (live YOLO on the 4090):
    SC_VIDEO=demo_clips python app_launcher.py

NOTE: pywebview needs a GUI backend (Cocoa on macOS — works out of the box; on
Windows it uses Edge WebView2; on Linux it needs GTK/Qt). Test on your demo machine.

Zero-setup fallback (no extra deps) — a chromeless browser window:
    python dashboard.py &
    #  macOS:  open -na "Google Chrome" --args --app=http://localhost:8000
    #  or just open http://localhost:8000 and press F11 (fullscreen)
"""
import threading
import time

import dashboard   # reuse the Flask app + the background shift loop


def _serve():
    dashboard.app.run(host="127.0.0.1", port=8000, threaded=True, debug=False)


def main():
    import webview                         # pip install pywebview
    dashboard.start_shift()                # kick off the autonomous agent loop
    threading.Thread(target=_serve, daemon=True).start()
    time.sleep(1.0)                        # let the server bind
    webview.create_window(
        "SafetyCommander — Autonomous Safety Officer",
        "http://127.0.0.1:8000",
        width=1480, height=940,
    )
    webview.start()                        # blocks until the window closes


if __name__ == "__main__":
    main()
