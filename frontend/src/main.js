import { pickSentence } from './sentences.js';
import './style.css';

// TODO: wire to backend /enroll and /identify endpoints

(function () {
  const promptEl = document.getElementById("prompt");
  const inputEl = document.getElementById("input");
  const progressEl = document.getElementById("progress");
  const shuffleBtn = document.getElementById("shuffle");
  const recordBtn = document.getElementById("record");
  const resetBtn = document.getElementById("reset");
  const resultsEl = document.getElementById("results");
  const rhythmEl = document.getElementById("rhythm");
  const toggleRawBtn = document.getElementById("toggle-raw");
  const copyRawBtn = document.getElementById("copy-raw");
  const rawLogEl = document.getElementById("raw-log");

  const statCount = document.getElementById("stat-count");
  const statDuration = document.getElementById("stat-duration");
  const statWpm = document.getElementById("stat-wpm");
  const statDwell = document.getElementById("stat-dwell");
  const statFlight = document.getElementById("stat-flight");

  let currentSentence = "";
  let events = []; // raw {code, key, type, t} log, t = performance.now()
  let keystrokeCount = 0;
  let minKeystrokes = 15;

  function newSentence() {
    currentSentence = pickSentence(currentSentence);
    promptEl.textContent = currentSentence;
    minKeystrokes = Math.min(15, Math.ceil(currentSentence.replace(/\s/g, "").length * 0.6));
    resetCapture();
  }

  function resetCapture() {
    events = [];
    keystrokeCount = 0;
    inputEl.value = "";
    recordBtn.disabled = true;
    progressEl.textContent = "0 keystrokes";
    resultsEl.classList.add("hidden");
    rawLogEl.classList.add("hidden");
    copyRawBtn.classList.add("hidden");
    toggleRawBtn.textContent = "Show raw event log";
    inputEl.focus();
  }

  function onKeyEvent(e) {
    if (e.repeat) return;
    events.push({ code: e.code, key: e.key, type: e.type, t: performance.now() });

    if (e.type === "keydown") {
      keystrokeCount += 1;
      const remaining = minKeystrokes - keystrokeCount;
      progressEl.textContent =
        remaining > 0
          ? `${keystrokeCount} keystrokes · ${remaining} more to record`
          : `${keystrokeCount} keystrokes · ready`;
      recordBtn.disabled = keystrokeCount < minKeystrokes;
    }
  }

  // Pair keydown/keyup by physical key (code), FIFO per key, to get dwell time.
  // This is a display-only computation for the demo — the real pipeline computes
  // HL/IL/PL/RL server-side from the raw event log, identically for training and live capture.
  function pairEvents(rawEvents) {
    const pending = {};
    const pairs = [];
    for (const e of rawEvents) {
      if (e.type === "keydown") {
        (pending[e.code] ??= []).push({ key: e.key, code: e.code, press: e.t });
      } else if (e.type === "keyup") {
        const queue = pending[e.code];
        if (queue && queue.length) {
          const p = queue.shift();
          p.release = e.t;
          pairs.push(p);
        }
      }
    }
    return pairs.filter((p) => p.release !== undefined).sort((a, b) => a.press - b.press);
  }

  function buildKeystrokeRecords(pairs) {
    return pairs.map((p, i) => {
      const dwell = p.release - p.press;
      const flight = i === 0 ? null : p.press - pairs[i - 1].release;
      return {
        key: p.key,
        code: p.code,
        dwell_ms: Math.round(dwell * 100) / 100,
        flight_ms: flight === null ? null : Math.round(flight * 100) / 100,
      };
    });
  }

  function mean(values) {
    if (!values.length) return 0;
    return values.reduce((a, b) => a + b, 0) / values.length;
  }

  function renderResults() {
    const pairs = pairEvents(events);
    const records = buildKeystrokeRecords(pairs);

    const dwellValues = records.map((r) => r.dwell_ms);
    const flightValues = records.map((r) => r.flight_ms).filter((v) => v !== null && v >= 0);

    const firstPress = pairs[0]?.press ?? 0;
    const lastRelease = pairs[pairs.length - 1]?.release ?? 0;
    const durationSec = (lastRelease - firstPress) / 1000;
    const typedChars = inputEl.value.trim().length;
    const wpm = durationSec > 0 ? (typedChars / 5) / (durationSec / 60) : 0;

    statCount.textContent = records.length;
    statDuration.textContent = `${durationSec.toFixed(1)}s`;
    statWpm.textContent = Math.round(wpm);
    statDwell.textContent = `${Math.round(mean(dwellValues))}ms`;
    statFlight.textContent = `${Math.round(mean(flightValues))}ms`;

    rhythmEl.innerHTML = "";
    const maxDwell = Math.max(...dwellValues, 1);
    for (const d of dwellValues) {
      const bar = document.createElement("div");
      bar.className = "flex-1 min-w-[2px] bg-[var(--color-accent)] rounded-t-sm opacity-85";
      bar.style.height = `${Math.max((d / maxDwell) * 100, 6)}%`;
      rhythmEl.appendChild(bar);
    }

    rawLogEl.textContent = JSON.stringify(
      {
        sentence: currentSentence,
        typed: inputEl.value,
        keystrokes: records,
      },
      null,
      2
    );

    resultsEl.classList.remove("hidden");
  }

  inputEl.addEventListener("keydown", onKeyEvent);
  inputEl.addEventListener("keyup", onKeyEvent);
  inputEl.addEventListener("paste", (e) => e.preventDefault());

  shuffleBtn.addEventListener("click", newSentence);
  resetBtn.addEventListener("click", newSentence);

  recordBtn.addEventListener("click", () => {
    inputEl.blur();
    renderResults();
  });

  toggleRawBtn.addEventListener("click", () => {
    const isHidden = rawLogEl.classList.contains("hidden");
    rawLogEl.classList.toggle("hidden", !isHidden);
    copyRawBtn.classList.toggle("hidden", !isHidden);
    toggleRawBtn.textContent = isHidden ? "Hide raw event log" : "Show raw event log";
  });

  copyRawBtn.addEventListener("click", () => {
    navigator.clipboard.writeText(rawLogEl.textContent);
    copyRawBtn.textContent = "Copied";
    setTimeout(() => (copyRawBtn.textContent = "Copy JSON"), 1200);
  });

  newSentence();
})();
