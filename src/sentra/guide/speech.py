"""Speak alerts without blocking the video loop, and never speak old news.

Speaker runs a backend on its own thread. Messages wait in a small queue; when it is full the oldest is
dropped, and a message older than `max_age_s` by the time the backend is free is dropped too, because the
scene has probably changed.

Backends:
- WindowsSpeech: the laptop's built-in Windows voice (System.Speech through one long-running PowerShell
  process, so there is no extra Python package and no start-up delay per message). Laptop speakers are a
  demo fallback; speech on the phone, near the user, is a planned web-page feature (docs/TEAM_PLAN.md).
- PrintSpeech: prints "[voice] ..." for headless runs and machines without the Windows voice.
"""
from __future__ import annotations

import base64
import logging
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from typing import Protocol

log = logging.getLogger(__name__)


class SpeechBackend(Protocol):
    name: str

    def start(self) -> None:
        """Get ready to speak (called once, on the speech thread)."""
        ...

    def speak(self, text: str) -> None:
        """Say `text` and return when finished. Raise on failure."""
        ...

    def close(self) -> None: ...


class PrintSpeech:
    name = "print"

    def start(self) -> None:
        pass

    def speak(self, text: str) -> None:
        print(f"[voice] {text}", flush=True)

    def close(self) -> None:
        pass


_PS_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$voice = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voice.Rate = __RATE__
[Console]::Out.WriteLine('ready')
while ($true) {
    $line = [Console]::In.ReadLine()
    if ($null -eq $line) { break }
    if ($line.Length -gt 0) { $voice.Speak($line) }
    [Console]::Out.WriteLine('done')
}
"""


def _powershell() -> str:
    found = shutil.which("powershell")
    if found:
        return found
    root = os.environ.get("SystemRoot", r"C:\Windows")
    return os.path.join(root, "System32", "WindowsPowerShell", "v1.0", "powershell.exe")


class WindowsSpeech:
    """Windows' built-in voice. `rate` is -10 (slow) .. 10 (fast); 1 is a little quicker than default."""

    name = "windows"

    def __init__(self, rate: int = 1, start_timeout_s: float = 15.0) -> None:
        if not -10 <= rate <= 10:
            raise ValueError("rate must be between -10 and 10")
        self.rate = rate
        self.start_timeout_s = start_timeout_s
        self._proc: subprocess.Popen[str] | None = None

    def command(self) -> list[str]:
        script = _PS_SCRIPT.replace("__RATE__", str(self.rate))
        encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
        return [_powershell(), "-NoLogo", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                "-EncodedCommand", encoded]

    def start(self) -> None:
        if sys.platform != "win32":
            raise RuntimeError("The Windows voice needs Windows")
        self._proc = subprocess.Popen(
            self.command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="ascii",
            errors="replace",
            bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self._expect("ready", self.start_timeout_s)

    def speak(self, text: str) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("WindowsSpeech is not started")
        line = " ".join(text.split()).encode("ascii", "replace").decode("ascii")
        self._proc.stdin.write(line + "\n")
        self._proc.stdin.flush()
        self._expect("done", 30.0)

    def _expect(self, word: str, timeout_s: float) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        result: list[str] = []
        reader = threading.Thread(target=lambda: result.append(self._proc.stdout.readline()), daemon=True)
        reader.start()
        reader.join(timeout_s)
        if not result or result[0].strip() != word:
            got = repr(result[0].strip()) if result else "nothing"
            raise RuntimeError(f"Windows voice: expected {word!r}, got {got}")

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()  # the PowerShell loop reads end-of-input and exits
            proc.wait(timeout=3)
        except Exception:
            proc.kill()


def default_backend() -> SpeechBackend:
    return WindowsSpeech() if sys.platform == "win32" else PrintSpeech()


class Speaker:
    """Queue + thread around a SpeechBackend. If the backend fails, it falls back to printing."""

    def __init__(self, backend: SpeechBackend, *, max_age_s: float = 2.0, max_pending: int = 2) -> None:
        self.backend = backend
        self.max_age_s = max_age_s
        self._queue: queue.Queue[tuple[str, float] | None] = queue.Queue(maxsize=max_pending)
        self.spoken: list[str] = []
        self.dropped = 0
        self.failed = False
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="Speaker", daemon=True)
        self._thread.start()

    def say(self, text: str, created: float | None = None) -> None:
        """Queue `text`. `created` is the perf_counter time the message became true (default: now)."""
        item = (text, time.perf_counter() if created is None else created)
        with self._lock:
            while True:
                try:
                    self._queue.put_nowait(item)
                    return
                except queue.Full:
                    try:
                        self._queue.get_nowait()  # newest news wins
                        self.dropped += 1
                    except queue.Empty:
                        pass

    def close(self) -> None:
        with self._lock:
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            self._queue.put_nowait(None)
        self._thread.join(timeout=5.0)

    def _run(self) -> None:
        backend = self.backend
        try:
            backend.start()
        except Exception as exc:
            log.warning("Voice %r unavailable (%s); printing alerts instead.", backend.name, exc)
            backend, self.failed = PrintSpeech(), True
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    break
                text, created = item
                if time.perf_counter() - created > self.max_age_s:
                    self.dropped += 1
                    continue
                try:
                    backend.speak(text)
                except Exception as exc:
                    log.warning("Voice %r failed (%s); printing alerts instead.", backend.name, exc)
                    backend.close()
                    backend, self.failed = PrintSpeech(), True
                    backend.speak(text)
                self.spoken.append(text)
        finally:
            backend.close()
