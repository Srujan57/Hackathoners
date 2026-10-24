"""
web_server.py - browser front end for Foot/Air Piano. Standard library only; reuses foot_piano.App unchanged.

It runs the same camera -> pose -> Pico loop as foot_piano.py, but headless, and serves ../web:
  GET  /              the UI
  GET  /video         the annotated camera view (MJPEG; the same picture foot_piano.py draws)
  GET  /events        live state (server-sent events): held keys, mode, calibration, Pico link
  GET  /api/presets   the preset profiles
  POST /api/profile   a profile JSON (validated by InputProfile.from_dict) -> applied live
  POST /api/action    {"name": "MODE" | "SONG" | "RECAL" | "STOP"}
  POST /api/key       {"k": 0-7}  test-tap one piano key

Run:   python web_server.py --port COM7          (or --no-pico), then open http://localhost:8000
Only listens on 127.0.0.1. The Pico's USB port is held by this process, so close foot_piano.py/Thonny first.
"""

import argparse
import json
import mimetypes
import os
import queue
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from pianolib.layout import NUM_KEYS                                   # noqa: E402
from pianolib.profile import PRESETS, InputProfile                     # noqa: E402

WEB = os.path.realpath(os.path.join(HERE, "..", "web"))
MAX_BODY = 65536
ACTIONS = ("MODE", "SONG", "RECAL", "STOP")


class Shared:
    def __init__(self):
        self.cond = threading.Condition()
        self.jpeg = None
        self.seq = 0
        self.state = {"starting": True}
        self.cmds = queue.Queue()
        self.stop = threading.Event()

    def publish(self, jpeg, state):
        with self.cond:
            self.jpeg, self.state = jpeg, state
            self.seq += 1
            self.cond.notify_all()


def make_handler(sh, port):
    allowed_hosts = {"localhost:%d" % port, "127.0.0.1:%d" % port}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            if not isinstance(body, bytes):
                body = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _host_ok(self):
            return self.headers.get("Host", "") in allowed_hosts      # blocks DNS-rebinding

        def do_GET(self):
            if not self._host_ok():
                return self._send(403, {"error": "bad host"})
            path = urlparse(self.path).path
            try:
                if path == "/video":
                    return self._video()
                if path == "/events":
                    return self._events()
                if path == "/api/presets":
                    return self._send(200, {k: p.to_dict() for k, p in PRESETS.items()})
                rel = "index.html" if path == "/" else path.lstrip("/")
                full = os.path.realpath(os.path.join(WEB, rel))
                if not full.startswith(WEB + os.sep) or not os.path.isfile(full):
                    return self._send(404, {"error": "not found"})
                with open(full, "rb") as f:
                    self._send(200, f.read(), mimetypes.guess_type(full)[0] or "application/octet-stream")
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _video(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            last = -1
            while not sh.stop.is_set():
                with sh.cond:
                    sh.cond.wait_for(lambda: sh.seq != last or sh.stop.is_set(), timeout=1.0)
                    jpeg, last = sh.jpeg, sh.seq
                if jpeg is None:
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpeg))
                self.wfile.write(jpeg + b"\r\n")

        def _events(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            while not sh.stop.is_set():
                self.wfile.write(("data: %s\n\n" % json.dumps(sh.state)).encode())
                self.wfile.flush()
                time.sleep(0.08)

        def do_POST(self):
            origin = self.headers.get("Origin")                      # refuse other websites poking the servos
            if not self._host_ok() or (origin and urlparse(origin).netloc not in allowed_hosts):
                return self._send(403, {"error": "forbidden"})
            try:
                n = int(self.headers.get("Content-Length", 0))
                if not 0 < n <= MAX_BODY:
                    return self._send(400, {"error": "bad body size"})
                body = json.loads(self.rfile.read(n))
                path = urlparse(self.path).path
                if path == "/api/profile":
                    sh.cmds.put(("profile", InputProfile.from_dict(body)))
                elif path == "/api/action" and isinstance(body, dict) and body.get("name") in ACTIONS:
                    sh.cmds.put(("action", body["name"]))
                elif path == "/api/key" and isinstance(body, dict) and isinstance(body.get("k"), int) \
                        and 0 <= body["k"] < NUM_KEYS and not isinstance(body["k"], bool):
                    sh.cmds.put(("tap", body["k"]))
                else:
                    return self._send(400, {"error": "unknown or invalid request"})
                self._send(200, {"ok": True})
            except (ValueError, TypeError) as e:                     # includes bad JSON and invalid profiles
                self._send(400, {"error": str(e)[:200]})

    return Handler


def vision_loop(args, sh):
    import cv2
    from foot_piano import App
    from pianolib.player import SONGS, PianoLink
    from pianolib.pose import LatestFrameCamera, PoseBackend
    from pianolib.profile import load_profile
    from pianolib.ui import draw

    def make_backend(need_hands):
        print("[model] loading RTMPose (%s) - the first run downloads the weights..." % args.model_mode)
        return PoseBackend(need_hands, mode=args.model_mode, device=args.device,
                           refine_hands=not args.no_hand_refine)

    link = PianoLink(None if args.no_pico else args.port)
    app = App(load_profile(args.profile), link, make_backend, args.width / float(args.height))
    cam = LatestFrameCamera(args.camera, args.width, args.height)
    error, test_release = "", {}
    last, fps, lat_ms = time.monotonic(), 0.0, 0.0
    try:
        while not sh.stop.is_set():
            frame, tcap = cam.read(timeout=2.0)
            if frame is None:
                error = "No camera frames. Check --camera or close other apps using the camera."
                sh.publish(None, {"error": error})
                break
            while True:
                try:
                    kind, val = sh.cmds.get_nowait()
                except queue.Empty:
                    break
                try:
                    if kind == "profile":
                        app.player.stop_all()
                        app.set_profile(val)
                        error = ""
                    elif kind == "action":
                        app.press_button(val)
                    elif kind == "tap":
                        link.press(val)
                        test_release[val] = time.monotonic() + 0.2
                except Exception as e:                               # e.g. the hand model could not load
                    error = "Could not apply that: %s" % str(e)[:120]
            now = time.monotonic()
            for k, t in list(test_release.items()):
                if now >= t:
                    link.release(k)
                    del test_release[k]

            app.process(frame, tcap)
            H, W = frame.shape[:2]
            btn = app.menu.update(time.monotonic(), app.pointers_px(H), W, H)
            if btn:
                app.press_button(btn)
            now = time.monotonic()
            lat_ms = 0.9 * lat_ms + 0.1 * (now - tcap) * 1000.0
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-3))
            last = now

            disp = cv2.flip(frame, 1)
            draw(disp, cv2, app.rig, app.player, link, app.menu, fps, lat_ms)
            if W > 960:
                disp = cv2.resize(disp, (960, int(H * 960 / W)))
            ok, buf = cv2.imencode(".jpg", disp, [cv2.IMWRITE_JPEG_QUALITY, 72])
            pl = app.player
            sh.publish(buf.tobytes() if ok else None, {
                "profile": app.profile.to_dict(), "mode": pl.mode, "modes": list(pl.modes),
                "calibrating": app.rig.calibrating, "status": app.rig.status(),
                "warnings": list(app.rig.warnings[:3]),
                "held": [k for k in range(NUM_KEYS) if link.held(k)],
                "pico": link.ser is not None, "link_error": link.error,
                "fps": round(fps, 1), "lat": round(lat_ms), "song": SONGS[pl.song][0],
                "next": pl.next_note_name(), "error": error})
    except Exception as e:
        sh.publish(None, {"error": "Tracking stopped: %s" % e})
        raise
    finally:
        link.close()
        cam.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", help="Pico serial port, e.g. COM7 or /dev/cu.usbmodem1101")
    ap.add_argument("--no-pico", action="store_true", help="run without hardware")
    ap.add_argument("--http-port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--profile", default="two_feet", help="profile to start with (the UI replaces it)")
    ap.add_argument("--model-mode", default="balanced", choices=["lightweight", "balanced", "performance"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-hand-refine", action="store_true")
    args = ap.parse_args()
    if not args.port and not args.no_pico:
        ap.error("give --port <serial port>, or --no-pico to try it without hardware")

    sh = Shared()
    server = ThreadingHTTPServer(("127.0.0.1", args.http_port), make_handler(sh, args.http_port))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = "http://localhost:%d" % args.http_port
    print("[web] UI at", url)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        vision_loop(args, sh)
    except KeyboardInterrupt:
        pass
    finally:
        sh.stop.set()
        server.shutdown()


if __name__ == "__main__":
    main()