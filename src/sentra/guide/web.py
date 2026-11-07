"""Phone/browser speech page for Sentra Guide. Owner: Swechha (docs/TEAM_PLAN.md, Swechha task 3).

The laptop serves a small page. A phone on the same hotspot opens it, taps "Tap to turn on sound" once (iOS needs a
tap before a page may speak), and then hears every cue in the browser's own voice (speechSynthesis), close to the
person instead of from the laptop speakers.

What is sent, over Server-Sent Events (one long HTTP response, standard library only):
- ``alert``: ``GuideAlert.to_dict()`` (the contract in docs/ARCHITECTURE.md) **minus ``source``**. The source is the
  camera's address (e.g. a DroidCam URL): anyone who has it can watch the live camera, so it stays in the laptop's
  event log and never reaches the page.
- ``status``: about once a second: fps, latency, distance mode, whether the laptop is muted. Doubles as a heartbeat,
  so the page can say "Disconnected" within a few seconds.
- ``hello``: on connect, the last few alerts (shown, never spoken) and the latest status.
Never video, images or boxes. There is no route that could return a frame.

Routes: ``/`` ``/app.js`` ``/style.css`` (the page), ``/events`` (the stream), ``/health`` (JSON). Everything
else is 404.

Network rule (docs/ARCHITECTURE.md, Section 6): listen only where the phone is. The default host is 127.0.0.1 (this
laptop only). For a phone, pass the laptop's address on the phone's hotspot, e.g. 172.20.10.2 on an iPhone hotspot.
Listening on every network (0.0.0.0) and public addresses are refused.
"""
from __future__ import annotations

import ipaddress
import json
import logging
import queue
import socket
import threading
import time
from collections import deque
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from sentra.guide.schema import SCHEMA_VERSION, GuideAlert, GuideSink, GuideUpdate

log = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
STATIC_DIR = Path(__file__).with_name("static")
KEEPALIVE_S = 15.0       # comment line on an idle stream, so proxies and phones don't drop it
RECONNECT_MS = 2000      # tells the browser how soon to reconnect after a drop
HISTORY = 5              # recent alerts a newly connected page shows (never speaks)
MAX_PENDING = 50         # per viewer; a stuck viewer loses its oldest messages, never blocks the camera loop

_STATIC_ROUTES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}
_SECURITY_HEADERS = {
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    # Only this laptop's own files and stream; nothing inline, nothing from the internet.
    "Content-Security-Policy": "default-src 'self'; img-src 'none'; media-src 'none'; frame-ancestors 'none'",
}


def check_host(host: str) -> str:
    """Return `host` if it's a safe address to listen on, else raise ValueError with what to do instead."""
    if host in ("", "0.0.0.0", "::", "*"):
        raise ValueError(
            "The phone page won't listen on every network (0.0.0.0). Pass this laptop's address on the phone's "
            f"hotspot instead, e.g. --web-host 172.20.10.2. This laptop's addresses: {', '.join(local_ipv4()) or '?'}"
        )
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        raise ValueError(f"--web-host must be an IP address like 172.20.10.2, not {host!r}") from None
    if ip.is_global:
        raise ValueError(f"{host} is a public internet address. The phone page must stay on the phone's hotspot.")
    return host


# Address ranges that phones and Windows use for their own hotspots. Anything else private (campus Wi-Fi, a home
# router) works, but then everyone on that network can open the page, so it gets a warning.
KNOWN_HOTSPOT_NETWORKS = {
    "iPhone hotspot": ipaddress.ip_network("172.20.10.0/28"),
    "Android hotspot (older phones)": ipaddress.ip_network("192.168.43.0/24"),
    "Windows Mobile Hotspot": ipaddress.ip_network("192.168.137.0/24"),
}


def hotspot_warning(host: str) -> str | None:
    """A warning to show when `host` isn't loopback or a known phone-hotspot address; None when it's fine."""
    ip = ipaddress.ip_address(host)
    if ip.is_loopback or any(ip in net for net in KNOWN_HOTSPOT_NETWORKS.values()):
        return None
    return (f"{host} isn't a known phone-hotspot address (an iPhone hotspot gives the laptop 172.20.10.x). If this "
            "is campus or shared Wi-Fi, anyone on that network can open the phone page and read the cues. Use the "
            "phone's own hotspot for real sessions. (Newer Android hotspots use random addresses; if you're on "
            "one, this warning is expected.)")


def local_ipv4() -> list[str]:
    """This laptop's IPv4 addresses (not 127.x), to help pick --web-host. Best effort; may be empty."""
    found: list[str] = []
    try:
        infos = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
    except OSError:
        return found
    for info in infos:
        address = str(info[4][0])
        if not address.startswith("127.") and address not in found:
            found.append(address)
    return found


def hotspot_hint() -> str:
    """One line telling the user which --web-host to use, based on this laptop's addresses."""
    addresses = local_ipv4()
    iphone = [a for a in addresses if a.startswith("172.20.10.")]
    if iphone:
        return f"On the iPhone hotspot, use --web-host {iphone[0]}"
    if addresses:
        return f"Use --web-host with this laptop's address on the phone's hotspot (this laptop: {', '.join(addresses)})"
    return "Use --web-host with this laptop's address on the phone's hotspot (see ipconfig)"


PAGE_HIDDEN_FIELDS = frozenset({"source"})  # camera address: watching the stream needs nothing else


def page_alert(alert: GuideAlert) -> dict[str, Any]:
    """What the page gets for one alert: GuideAlert.to_dict() without the camera address."""
    return {key: value for key, value in alert.to_dict().items() if key not in PAGE_HIDDEN_FIELDS}


def _put_newest(q: queue.Queue, item: Any) -> None:
    """Put without blocking; if the viewer's queue is full, drop its oldest message (newest news wins)."""
    while True:
        try:
            q.put_nowait(item)
            return
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass


class Broadcaster:
    """Fans messages out to every connected page. Thread-safe; `send()` never blocks the camera loop."""

    def __init__(self, history: int = HISTORY, max_pending: int = MAX_PENDING) -> None:
        self._lock = threading.Lock()
        self._subscribers: set[queue.Queue] = set()
        self._recent: deque[dict[str, Any]] = deque(maxlen=history)
        self._status: dict[str, Any] | None = None
        self._max_pending = max_pending
        self._closed = False

    @property
    def viewers(self) -> int:
        with self._lock:
            return len(self._subscribers)

    def subscribe(self) -> queue.Queue:
        """A new page connected. Its queue starts with a `hello` (recent alerts + latest status)."""
        q: queue.Queue = queue.Queue(maxsize=self._max_pending)
        with self._lock:
            if self._closed:
                q.put_nowait(None)
                return q
            hello = {"schema": SCHEMA_VERSION, "type": "hello", "recent": list(self._recent), "status": self._status}
            q.put_nowait(("hello", hello))
            self._subscribers.add(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            self._subscribers.discard(q)

    def send(self, event: str, data: dict[str, Any]) -> None:
        with self._lock:
            if self._closed:
                return
            if event == "alert":
                self._recent.append(data)
            elif event == "status":
                self._status = data
            for q in self._subscribers:
                _put_newest(q, (event, data))

    def close(self) -> None:
        """End every open stream."""
        with self._lock:
            self._closed = True
            for q in self._subscribers:
                _put_newest(q, None)


class _Server(ThreadingHTTPServer):
    daemon_threads = True          # open streams never keep the app alive
    allow_reuse_address = False    # don't silently share a port with another copy of the app

    def __init__(self, address: tuple[str, int], broadcaster: Broadcaster) -> None:
        self.broadcaster = broadcaster
        super().__init__(address, _Handler)


class _Handler(BaseHTTPRequestHandler):
    server: _Server
    server_version = "SentraGuide"
    sys_version = ""

    def do_GET(self) -> None:  # noqa: N802 (http.server naming)
        path = urlsplit(self.path).path
        if path in _STATIC_ROUTES:
            self._static(*_STATIC_ROUTES[path])
        elif path == "/events":
            self._events()
        elif path == "/health":
            body = {"ok": True, "schema": SCHEMA_VERSION, "viewers": self.server.broadcaster.viewers}
            self._send(200, json.dumps(body).encode(), "application/json")
        else:
            self._send(404, b"Not found", "text/plain; charset=utf-8")

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 (http.server signature)
        log.debug("web %s - %s", self.client_address[0], format % args)

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for name, value in _SECURITY_HEADERS.items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def _static(self, name: str, content_type: str) -> None:
        try:
            body = (STATIC_DIR / name).read_bytes()
        except OSError:
            self._send(500, b"Page file missing", "text/plain; charset=utf-8")
            return
        self._send(200, body, content_type)

    def _events(self) -> None:
        broadcaster = self.server.broadcaster
        q = broadcaster.subscribe()
        log.info("Phone page connected from %s (%d open)", self.client_address[0], broadcaster.viewers)
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            for name, value in _SECURITY_HEADERS.items():
                self.send_header(name, value)
            self.end_headers()
            self._write(f"retry: {RECONNECT_MS}\n\n")
            while True:
                try:
                    item = q.get(timeout=KEEPALIVE_S)
                except queue.Empty:
                    self._write(": keepalive\n\n")
                    continue
                if item is None:
                    break
                event, data = item
                self._write(f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n")
        except OSError:
            pass  # the phone went away (locked screen, closed tab, lost Wi-Fi); it reconnects by itself
        finally:
            broadcaster.unsubscribe(q)
            self.close_connection = True
            log.info("Phone page disconnected from %s (%d open)", self.client_address[0], broadcaster.viewers)

    def _write(self, text: str) -> None:
        self.wfile.write(text.encode("utf-8"))
        self.wfile.flush()


class WebSpeechPage:
    """Implements GuideSink: sends each GuideAlert (and a status heartbeat) to phone pages. Never sends frames.

    `controls` is the same GuideControls the debug view flips with `m`: when the laptop is muted, the page shows
    cues but doesn't speak them either.
    """

    def __init__(
        self,
        controls: Any | None = None,
        *,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        status_every_s: float = 1.0,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        check_host(host)
        warning = hotspot_warning(host)
        if warning:
            log.warning(warning)
        self.controls = controls
        self.status_every_s = status_every_s
        self._clock = clock
        self._last_status = float("-inf")
        self.broadcaster = Broadcaster()
        self._server = _Server((host, port), self.broadcaster)  # binds now, so a busy port fails at start-up
        self.host, self.port = self._server.server_address[:2]
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.2},
                                         name="WebSpeechPage", daemon=True)
        self._thread.start()
        self._closed = False

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def publish(self, update: GuideUpdate) -> None:
        now = self._clock()
        if update.alerts or now - self._last_status >= self.status_every_s:
            self.broadcaster.send("status", self.status(update))  # first, so the page knows if it's muted
            self._last_status = now
        for alert in update.alerts:
            self.broadcaster.send("alert", page_alert(alert))

    def status(self, update: GuideUpdate) -> dict[str, Any]:
        # No "source": the camera address would let anyone with the page open the live stream.
        return {
            "schema": SCHEMA_VERSION,
            "type": "status",
            "fps": round(update.fps, 1),
            "latency_ms": round(update.latency_ms, 1),
            "depth_status": update.depth_status,
            "muted": bool(getattr(self.controls, "muted", False)),
            "stale": update.stale,
            "objects": len(update.objects),
        }

    def should_stop(self) -> bool:
        return False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.broadcaster.close()
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5.0)


class TeeSink:
    """Sends every update to several sinks (e.g. the debug view and the phone page). Stops when any one says so."""

    def __init__(self, *sinks: GuideSink) -> None:
        self.sinks = sinks

    def publish(self, update: GuideUpdate) -> None:
        for sink in self.sinks:
            sink.publish(update)

    def idle(self, message: str = "") -> None:
        """No new frame (switching cameras, stalled stream): pass it on to the sinks that can show it."""
        for sink in self.sinks:
            idle = getattr(sink, "idle", None)
            if idle is not None:
                idle(message)

    def should_stop(self) -> bool:
        return any(sink.should_stop() for sink in self.sinks)

    def close(self) -> None:
        for sink in self.sinks:
            try:
                sink.close()
            except Exception:
                log.exception("Closing %s failed", type(sink).__name__)
