import { useEffect, useRef, useState } from "react";
import { pickSentence } from "./sentences.js";

const API = "/api"; // proxied to the backend by vite.config.js
const ENROLL_SESSIONS = 5; // eval: more enrollment sentences -> markedly better accuracy
// Backend rejects fewer than 26 paired keystrokes (features.extract.MIN_KEYSTROKES + 1); leave margin.
const MIN_KEYSTROKES = 30;

async function postJson(path, body) {
  const res = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail));
  return data;
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

export default function App() {
  const inputRef = useRef(null);
  const eventsRef = useRef([]);
  const keystrokeCountRef = useRef(0);
  const enrollSessionsRef = useRef([]);
  const usedSentencesRef = useRef([]); // enrollment never repeats a sentence

  const [mode, setMode] = useState("enroll");
  const [name, setName] = useState("");
  const [nameLocked, setNameLocked] = useState(false);
  const [sessionCount, setSessionCount] = useState(0);
  const [enrolled, setEnrolled] = useState(false);
  const [busy, setBusy] = useState(false);
  const [apiStatus, setApiStatus] = useState("");
  const [matches, setMatches] = useState(null);

  const [sentence, setSentence] = useState("");
  const [progressText, setProgressText] = useState("0 keystrokes");
  const [recordDisabled, setRecordDisabled] = useState(true);
  const [showResults, setShowResults] = useState(false);
  const [showRaw, setShowRaw] = useState(false);
  const [copyLabel, setCopyLabel] = useState("Copy JSON");
  const [stats, setStats] = useState({ count: 0, duration: "0.0s", wpm: 0, dwell: "0ms", flight: "0ms" });
  const [dwellValues, setDwellValues] = useState([]);
  const [rawText, setRawText] = useState("");

  // The typing box only exists once we know whose typing it is (enroll) or who to look up (identify).
  const capturing = mode === "identify" || (nameLocked && !enrolled);

  function resetCapture() {
    eventsRef.current = [];
    keystrokeCountRef.current = 0;
    if (inputRef.current) inputRef.current.value = "";
    setRecordDisabled(true);
    setProgressText("0 keystrokes");
    setShowResults(false);
    setShowRaw(false);
    setCopyLabel("Copy JSON");
    inputRef.current?.focus();
  }

  function newSentence(accumulate) {
    const next = pickSentence(usedSentencesRef.current);
    usedSentencesRef.current = accumulate ? [...usedSentencesRef.current, next] : [next];
    setSentence(next);
    resetCapture();
  }

  function lockName() {
    if (!name.trim()) return;
    setName(name.trim());
    setNameLocked(true);
    usedSentencesRef.current = [];
    newSentence(true);
  }

  function resetEnrollment() {
    enrollSessionsRef.current = [];
    usedSentencesRef.current = [];
    setName("");
    setNameLocked(false);
    setEnrolled(false);
    setSessionCount(0);
    setApiStatus("");
  }

  function switchMode(next) {
    setMode(next);
    setMatches(null);
    resetEnrollment();
    if (next === "identify") newSentence(false);
  }

  function onKeyDown(e) {
    // Enter submits (same as the button) and is never part of the typing sample.
    if (e.key === "Enter") {
      e.preventDefault();
      if (!recordDisabled && !busy) submit();
      return;
    }
    onKeyEvent(e);
  }

  function onKeyUp(e) {
    if (e.key !== "Enter") onKeyEvent(e);
  }

  function onKeyEvent(e) {
    if (e.repeat) return;
    eventsRef.current.push({ code: e.code, key: e.key, type: e.type, t: performance.now() });

    if (e.type === "keydown") {
      keystrokeCountRef.current += 1;
      const count = keystrokeCountRef.current;
      const remaining = MIN_KEYSTROKES - count;
      setProgressText(
        remaining > 0 ? `${count} keystrokes · ${remaining} more to record` : `${count} keystrokes · ready`
      );
      setRecordDisabled(count < MIN_KEYSTROKES);
    }
  }

  function renderResults() {
    const pairs = pairEvents(eventsRef.current);
    const records = buildKeystrokeRecords(pairs);

    const dwells = records.map((r) => r.dwell_ms);
    const flights = records.map((r) => r.flight_ms).filter((v) => v !== null && v >= 0);

    const firstPress = pairs[0]?.press ?? 0;
    const lastRelease = pairs[pairs.length - 1]?.release ?? 0;
    const durationSec = (lastRelease - firstPress) / 1000;
    const typedChars = inputRef.current.value.trim().length;
    const wpm = durationSec > 0 ? (typedChars / 5) / (durationSec / 60) : 0;

    setStats({
      count: records.length,
      duration: `${durationSec.toFixed(1)}s`,
      wpm: Math.round(wpm),
      dwell: `${Math.round(mean(dwells))}ms`,
      flight: `${Math.round(mean(flights))}ms`,
    });
    setDwellValues(dwells);
    setRawText(JSON.stringify({ sentence, typed: inputRef.current.value, keystrokes: records }, null, 2));
    setShowResults(true);
  }

  async function submit() {
    const events = eventsRef.current;
    if (mode === "enroll") {
      const sessions = [...enrollSessionsRef.current, { sentence, events }];
      if (sessions.length < ENROLL_SESSIONS) {
        enrollSessionsRef.current = sessions;
        setSessionCount(sessions.length);
        setApiStatus("");
        newSentence(true);
        return;
      }
      setBusy(true);
      try {
        await postJson("/enroll", { person_id: name.toLowerCase(), name, sessions });
        enrollSessionsRef.current = [];
        setEnrolled(true);
        setApiStatus(`Enrolled ${name}. Switch to Identify to test.`);
      } catch (err) {
        setApiStatus(`Error: ${err.message}`);
        resetCapture(); // retype this last sentence
      } finally {
        setBusy(false);
      }
    } else {
      inputRef.current?.blur();
      renderResults();
      setBusy(true);
      try {
        const res = await postJson("/identify", { sentence, events });
        setMatches(res);
        setApiStatus(res.matched ? "" : "No confident match.");
      } catch (err) {
        setApiStatus(`Error: ${err.message}`);
      } finally {
        setBusy(false);
      }
    }
  }

  useEffect(() => {
    if (capturing) inputRef.current?.focus();
  }, [capturing, sentence]);

  const maxDwell = Math.max(...dwellValues, 1);
  const lastEnrollSentence = sessionCount === ENROLL_SESSIONS - 1;

  return (
    <main className="w-full max-w-[560px] bg-[var(--color-card)] rounded-3xl shadow-[var(--shadow-card)] p-12">
      <div className="flex gap-2 mb-6">
        {["enroll", "identify"].map((m) => (
          <button
            key={m}
            onClick={() => switchMode(m)}
            className={`px-4 py-1.5 rounded-full border-0 text-sm font-semibold capitalize cursor-pointer ${
              mode === m ? "bg-[var(--color-text)] text-[var(--color-card)]" : "bg-[var(--color-bg)] text-[var(--color-text-secondary)]"
            }`}
          >
            {m}
          </button>
        ))}
        <a
          href={`${API}/map`}
          target="_blank"
          rel="noreferrer"
          className="ml-auto px-4 py-1.5 rounded-full text-sm font-semibold no-underline bg-[var(--color-bg)] text-[var(--color-text-secondary)]"
        >
          Map &#8599;
        </a>
      </div>

      {mode === "enroll" && !nameLocked && (
        <>
          <input
            autoFocus
            value={name}
            onChange={(e) => setName(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && lockName()}
            placeholder="Your name"
            className="w-full mb-6 border-0 border-b border-[var(--color-border)] bg-transparent text-[var(--color-text)] text-[17px] py-2 outline-none focus:border-b-[var(--color-accent)]"
          />
          <button
            onClick={lockName}
            disabled={!name.trim()}
            className="w-full py-3.5 px-7 border-0 rounded-full bg-[var(--color-text)] text-[var(--color-card)] font-sans text-base font-semibold cursor-pointer transition disabled:opacity-35 disabled:cursor-not-allowed enabled:hover:opacity-85 enabled:active:scale-[0.98]"
          >
            Start enrollment
          </button>
        </>
      )}

      {mode === "enroll" && nameLocked && (
        <div className="mb-6 text-[17px] font-medium">Enrolling: {name}</div>
      )}

      {capturing && (
        <>
          <div className="text-xs font-semibold tracking-widest uppercase text-[var(--color-text-secondary)] mb-4">
            {mode === "enroll" ? `Enrollment sentence ${sessionCount + 1} of ${ENROLL_SESSIONS}` : "Typing sample"}
          </div>

          <div className="flex items-start gap-3 mb-7">
            <p className="flex-1 m-0 text-2xl leading-snug font-medium tracking-tight">{sentence}</p>
            <button
              onClick={() => newSentence(mode === "enroll")}
              className="shrink-0 w-9 h-9 rounded-full border-0 bg-[var(--color-bg)] text-[var(--color-text-secondary)] text-base cursor-pointer transition duration-150 hover:bg-[var(--color-border)] hover:rotate-45"
              title="New sentence"
              aria-label="New sentence"
            >
              &#8635;
            </button>
          </div>

          <textarea
            ref={inputRef}
            onKeyDown={onKeyDown}
            onKeyUp={onKeyUp}
            onPaste={(e) => e.preventDefault()}
            className="w-full resize-none border-0 border-b border-[var(--color-border)] bg-transparent text-[var(--color-text)] placeholder:text-[var(--color-text-secondary)] font-sans text-[17px] leading-relaxed py-2 outline-none transition-colors duration-150 focus:border-b-[var(--color-accent)]"
            rows="3"
            placeholder="Start typing…"
            autoComplete="off"
            autoCorrect="off"
            autoCapitalize="off"
            spellCheck="false"
          />

          <div className="mt-2.5 text-[13px] text-[var(--color-text-secondary)]">
            <span>{progressText}</span>
          </div>

          <button
            onClick={submit}
            disabled={recordDisabled || busy}
            className="mt-7 w-full py-3.5 px-7 border-0 rounded-full bg-[var(--color-text)] text-[var(--color-card)] font-sans text-base font-semibold cursor-pointer transition disabled:opacity-35 disabled:cursor-not-allowed enabled:hover:opacity-85 enabled:active:scale-[0.98]"
          >
            {mode === "identify" ? "Identify me" : lastEnrollSentence ? "Finish enrollment" : "Next sentence"}
          </button>
        </>
      )}

      {enrolled && (
        <button
          onClick={resetEnrollment}
          className="mt-6 w-full py-3 px-7 border border-[var(--color-border)] rounded-full bg-transparent text-[var(--color-text)] text-[15px] font-medium cursor-pointer transition-colors duration-150 hover:bg-[var(--color-bg)]"
        >
          Enroll another person
        </button>
      )}

      {apiStatus && <p className="mt-4 text-sm text-[var(--color-text-secondary)]">{apiStatus}</p>}
      {mode === "identify" && matches?.matched && (
        <div className="mt-4 text-[15px]">
          {matches.results.map((r, i) => (
            <div key={r.person_id}>
              {i + 1}. {r.name} ({r.similarity.toFixed(3)})
            </div>
          ))}
        </div>
      )}

      <section className={`${mode === "identify" && showResults ? "" : "hidden "}mt-9 pt-8 border-t border-[var(--color-border)]`}>
        <h2 className="m-0 mb-5 text-lg font-semibold">Typing sample</h2>

        <div className="grid grid-cols-3 gap-x-3 gap-y-5">
          <div className="flex flex-col">
            <span className="text-[22px] font-semibold tracking-tight">{stats.count}</span>
            <span className="mt-0.5 text-xs text-[var(--color-text-secondary)]">keystrokes</span>
          </div>
          <div className="flex flex-col">
            <span className="text-[22px] font-semibold tracking-tight">{stats.duration}</span>
            <span className="mt-0.5 text-xs text-[var(--color-text-secondary)]">duration</span>
          </div>
          <div className="flex flex-col">
            <span className="text-[22px] font-semibold tracking-tight">{stats.wpm}</span>
            <span className="mt-0.5 text-xs text-[var(--color-text-secondary)]">wpm</span>
          </div>
          <div className="flex flex-col">
            <span className="text-[22px] font-semibold tracking-tight">{stats.dwell}</span>
            <span className="mt-0.5 text-xs text-[var(--color-text-secondary)]">avg dwell</span>
          </div>
          <div className="flex flex-col">
            <span className="text-[22px] font-semibold tracking-tight">{stats.flight}</span>
            <span className="mt-0.5 text-xs text-[var(--color-text-secondary)]">avg flight</span>
          </div>
        </div>

        <div className="mt-7 h-14 flex items-end gap-0.5" aria-label="Dwell time per keystroke">
          {dwellValues.map((d, i) => (
            <div
              key={i}
              className="flex-1 min-w-[2px] bg-[var(--color-accent)] rounded-t-sm opacity-85"
              style={{ height: `${Math.max((d / maxDwell) * 100, 6)}%` }}
            />
          ))}
        </div>

        <div className="mt-5 flex gap-4">
          <button
            onClick={() => setShowRaw((prev) => !prev)}
            className="border-0 bg-transparent text-[var(--color-accent)] font-sans text-[13px] font-medium cursor-pointer p-0"
          >
            {showRaw ? "Hide raw event log" : "Show raw event log"}
          </button>
          <button
            onClick={() => {
              navigator.clipboard.writeText(rawText);
              setCopyLabel("Copied");
              setTimeout(() => setCopyLabel("Copy JSON"), 1200);
            }}
            className={`${showRaw ? "" : "hidden "}border-0 bg-transparent text-[var(--color-accent)] font-sans text-[13px] font-medium cursor-pointer p-0`}
          >
            {copyLabel}
          </button>
        </div>
        <pre
          className={`${showRaw ? "" : "hidden "}mt-3 max-h-[220px] overflow-y-auto overflow-x-hidden whitespace-pre-wrap break-words bg-[var(--color-bg)] rounded-xl p-4 font-mono text-xs leading-relaxed text-[var(--color-text)]`}
        >
          {rawText}
        </pre>

        <button
          onClick={() => {
            setMatches(null);
            setApiStatus("");
            newSentence(false);
          }}
          className="mt-6 w-full py-3 px-7 border border-[var(--color-border)] rounded-full bg-transparent text-[var(--color-text)] text-[15px] font-medium cursor-pointer transition-colors duration-150 hover:bg-[var(--color-bg)]"
        >
          Try another sentence
        </button>
      </section>
    </main>
  );
}
