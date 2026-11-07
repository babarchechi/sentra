// Sentra Guide phone page: receives cue JSON from the laptop (Server-Sent Events) and speaks it with the browser's
// own voice. Owner: Swechha. Served by src/sentra/guide/web.py. Never receives video or images.
"use strict";

const MAX_WAIT_MS = 2000;   // a cue that waited longer than this to be spoken is dropped: the scene has changed
                            // (same rule as the laptop voice, Speaker.max_age_s)
const SILENT_MS = 4000;     // the laptop sends a status every second; this long without anything = disconnected
const STUCK_SPEECH_MS = 6000; // some phones never fire "end"; don't let that silence every later cue
const HISTORY = 5;

const $ = (id) => document.getElementById(id);
const els = {
  connection: $("connection"), enable: $("enable"), noSpeech: $("no-speech"), cue: $("cue"),
  cueAge: $("cue-age"), cueBox: document.querySelector(".cue"), voice: $("voice"), laptop: $("laptop"),
  recent: $("recent"),
};

const canSpeak = "speechSynthesis" in window && "SpeechSynthesisUtterance" in window;
const state = {
  audioOn: false,        // set by the first tap (iOS won't speak before one)
  voiceOn: true,         // the Voice on/off toggle
  laptopMuted: false,    // the laptop's m key mutes the page too
  lastMessageAt: 0,
  lastCueAt: 0,
  pending: null,         // newest cue waiting to be spoken; a newer one replaces it
  speakingSince: 0,
  wakeLock: null,
};

// ---- Connection --------------------------------------------------------------------------------------------

function setConnection(kind, text) {
  if (els.connection.textContent === text && els.connection.classList.contains(`connection--${kind}`)) return;
  els.connection.className = `connection connection--${kind}`;
  els.connection.textContent = text;
}

function heardFromLaptop() {
  state.lastMessageAt = Date.now();
  setConnection("ok", "Connected to the laptop");
}

function connect() {
  const source = new EventSource("/events");
  source.addEventListener("hello", (event) => {
    const hello = JSON.parse(event.data);
    heardFromLaptop();
    els.recent.replaceChildren();
    (hello.recent || []).forEach((alert) => addRecent(alert, true));
    if (hello.status) applyStatus(hello.status);
  });
  source.addEventListener("status", (event) => {
    heardFromLaptop();
    applyStatus(JSON.parse(event.data));
  });
  source.addEventListener("alert", (event) => {
    heardFromLaptop();
    onAlert(JSON.parse(event.data));
  });
  source.addEventListener("error", () => {
    // EventSource reconnects by itself (the server asks for every 2 s).
    setConnection("bad", "Disconnected. Trying again… Cues are NOT coming through.");
  });
}

setInterval(() => {
  if (state.lastMessageAt && Date.now() - state.lastMessageAt > SILENT_MS) {
    setConnection("bad", "No signal from the laptop. Cues are NOT coming through.");
  }
  updateCueAge();
  if (state.speakingSince && Date.now() - state.speakingSince > STUCK_SPEECH_MS) finishSpeaking();
}, 1000);

// ---- Status from the laptop --------------------------------------------------------------------------------

function applyStatus(status) {
  state.laptopMuted = Boolean(status.muted);
  const parts = [];
  if (typeof status.fps === "number") parts.push(`${status.fps.toFixed(0)} fps`);
  // "Distance: off (camera only)" -> "Distance off"
  if (status.depth_status) parts.push(status.depth_status.replace(": ", " ").replace(/\s*\(.*\)$/, ""));
  if (state.laptopMuted) parts.push("Muted on the laptop (m): cues shown, not spoken");
  if (status.stale) parts.push("Camera frames are delayed");
  els.laptop.textContent = parts.length ? `Laptop · ${parts.join(" · ")}` : "";
  updateLiveRegion(); // muting on the laptop stops this page speaking, so screen readers must take over
}

// ---- Cues --------------------------------------------------------------------------------------------------

function onAlert(alert) {
  if (!alert || typeof alert.text !== "string") return;
  state.lastCueAt = Date.now();
  els.cue.textContent = alert.text;
  els.cueBox.classList.add("cue--fresh");
  els.cueBox.classList.toggle("cue--high", alert.severity === "high");
  updateCueAge();
  addRecent(alert, false);
  if (shouldSpeak()) {
    state.pending = { text: alert.text, receivedAt: Date.now() };
    speakNext();
  }
}

function addRecent(alert, earlier) {
  const item = document.createElement("li");
  const time = document.createElement("time");
  const now = new Date();
  time.dateTime = now.toISOString();
  time.textContent = earlier ? "Earlier" : now.toLocaleTimeString([], { hour: "numeric", minute: "2-digit", second: "2-digit" });
  item.append(time, document.createTextNode(alert.text));
  if (earlier) item.className = "earlier";
  els.recent.prepend(item);
  while (els.recent.children.length > HISTORY) els.recent.lastElementChild.remove();
}

function updateCueAge() {
  if (!state.lastCueAt) return;
  const seconds = Math.round((Date.now() - state.lastCueAt) / 1000);
  els.cueAge.textContent = seconds < 2 ? "just now" : `${seconds} s ago`;
  if (seconds >= 3) els.cueBox.classList.remove("cue--fresh", "cue--high");
}

// ---- Speech ------------------------------------------------------------------------------------------------

function shouldSpeak() {
  return canSpeak && state.audioOn && state.voiceOn && !state.laptopMuted;
}

function speakNext() {
  if (state.speakingSince || !state.pending) return;
  const item = state.pending;
  state.pending = null;
  if (Date.now() - item.receivedAt > MAX_WAIT_MS || !shouldSpeak()) return speakNext();
  const utterance = new SpeechSynthesisUtterance(item.text);
  utterance.lang = "en-US";
  utterance.rate = 1.05;
  utterance.onend = finishSpeaking;
  utterance.onerror = finishSpeaking;
  state.speakingSince = Date.now();
  window.speechSynthesis.speak(utterance);
}

function finishSpeaking() {
  state.speakingSince = 0;
  speakNext();
}

// The cue text is a live region so screen readers (VoiceOver, TalkBack) announce it, but only while this page
// isn't speaking it itself; otherwise a screen-reader user would hear every cue twice.
function updateLiveRegion() {
  els.cue.setAttribute("aria-live", shouldSpeak() ? "off" : "polite");
}

async function keepScreenOn() {
  try {
    if ("wakeLock" in navigator && document.visibilityState === "visible") {
      state.wakeLock = await navigator.wakeLock.request("screen");
    }
  } catch (err) {
    // Not supported or refused (e.g. low battery). The footer already asks to keep the screen on.
  }
}

els.enable.addEventListener("click", () => {
  if (!canSpeak) {
    els.noSpeech.hidden = false;
    els.enable.hidden = true;
    return;
  }
  // This first utterance, inside the tap, is what unlocks speech on iOS.
  window.speechSynthesis.cancel();
  const hello = new SpeechSynthesisUtterance("Sentra Guide sound on.");
  hello.lang = "en-US";
  window.speechSynthesis.speak(hello);
  state.audioOn = true;
  els.enable.hidden = true;
  els.voice.hidden = false;
  updateLiveRegion();
  keepScreenOn();
});

els.voice.addEventListener("click", () => {
  state.voiceOn = !state.voiceOn;
  els.voice.setAttribute("aria-pressed", String(state.voiceOn));
  els.voice.textContent = state.voiceOn ? "Voice: on" : "Voice: off";
  if (!state.voiceOn) {
    state.pending = null;
    window.speechSynthesis.cancel();
  }
  updateLiveRegion();
});

document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible" && state.audioOn) keepScreenOn();
});

if (!canSpeak) {
  els.enable.textContent = "Text only on this browser";
  els.noSpeech.hidden = false;
}
updateLiveRegion();
connect();
