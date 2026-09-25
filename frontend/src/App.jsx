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
  if (!res.ok) {
    const err = new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail));
    err.status = res.status;
    throw err;
  }
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

// Bands are anchored to the backend's own decision threshold, never to invented absolutes.
function confidenceBand(similarity, threshold) {
  if (similarity >= threshold + 0.15) return { label: "strong match", color: "var(--color-accent)" };
  if (similarity >= threshold) return { label: "match", color: "var(--color-accent-deep)" };
  return { label: "below threshold", color: "var(--color-text-secondary)" };
}

const stroke = {
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.6,
  strokeLinecap: "round",
  strokeLinejoin: "round",
};

function Icon({ path, size = 16, className = "" }) {
  return (
    <svg width={size} height={size} viewBox="0 0 16 16" aria-hidden="true" className={className} {...stroke}>
      {path}
    </svg>
  );
}

const icons = {
  refresh: <path d="M13.5 8a5.5 5.5 0 1 1-1.6-3.9M13.5 2v3h-3" />,
  external: <path d="M6 10l5-5M6.5 5H11v4.5" />,
  check: <path d="M3 8.5l3.2 3.2L13 5" />,
  alert: <path d="M8 4.5v4M8 11.5h.01M8 1.8l6 11H2z" />,
  declined: <path d="M8 14A6 6 0 1 0 8 2a6 6 0 0 0 0 12zM5.2 5.2l5.6 5.6" />,
  back: <path d="M9.5 3.5L5 8l4.5 4.5" />,
};

// The mark's concentric rings, reused as the working indicator: the logo scans while the model does.
function ScanRings({ size = 18, active = false }) {
  const spin = { transformBox: "fill-box", transformOrigin: "center" };
  return (
    <svg width={size} height={size} viewBox="0 0 48 48" aria-hidden="true" className="shrink-0">
      <circle
        cx="24"
        cy="24"
        r="19"
        fill="none"
        stroke="var(--color-accent-deep)"
        strokeWidth="4"
        strokeLinecap="round"
        strokeDasharray="21 8.85"
        style={{ ...spin, animation: active ? "scan-cw 2.4s linear infinite" : undefined }}
      />
      <circle
        cx="24"
        cy="24"
        r="11.5"
        fill="none"
        stroke="var(--color-accent)"
        strokeWidth="3.5"
        strokeLinecap="round"
        strokeDasharray="13 5.85"
        style={{ ...spin, animation: active ? "scan-ccw 1.8s linear infinite" : undefined }}
      />
    </svg>
  );
}

const statusStyles = {
  success: { color: "var(--color-accent)", icon: icons.check },
  error: { color: "var(--color-danger)", icon: icons.alert },
  declined: { color: "var(--color-caution)", icon: icons.declined },
};

// Destructive steps ask first, in-world, rather than through a browser confirm().
function Choice({ question, actions }) {
  return (
    <div className="mt-6 rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] p-4 animate-[rise_280ms_cubic-bezier(0.16,1,0.3,1)]">
      <p className="m-0 flex items-start gap-2.5 text-[14.5px] leading-snug text-[var(--color-caution)]">
        <Icon path={icons.alert} className="mt-[3px] shrink-0" />
        <span>{question}</span>
      </p>
      <div className="mt-3.5 flex flex-wrap gap-2">
        {actions.map((a) => (
          <button
            key={a.label}
            onClick={a.onClick}
            className={`px-4 py-2 rounded-full border text-[14px] font-semibold cursor-pointer transition-colors duration-150 ${
              a.destructive
                ? "border-transparent bg-[var(--color-danger)] text-[var(--color-card)] hover:opacity-85"
                : "border-[var(--color-border)] bg-transparent text-[var(--color-text)] hover:bg-[var(--color-card)]"
            }`}
          >
            {a.label}
          </button>
        ))}
      </div>
    </div>
  );
}

function Status({ status }) {
  const style = status ? statusStyles[status.kind] : null;
  return (
    <div aria-live="polite" className={status ? "mt-6" : ""}>
      {status && (
        <p
          className="m-0 flex items-start gap-2.5 text-[14.5px] leading-snug animate-[rise_320ms_cubic-bezier(0.16,1,0.3,1)]"
          style={{ color: style.color }}
        >
          <Icon path={style.icon} className="mt-[3px] shrink-0" />
          <span>{status.text}</span>
        </p>
      )}
    </div>
  );
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
  const [status, setStatus] = useState(null); // { kind: success | error | declined, text }
  const [matches, setMatches] = useState(null);
  const [pendingDiscard, setPendingDiscard] = useState(null); // { run } — awaiting confirmation
  const [collision, setCollision] = useState(null); // { sessions } — name already enrolled

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
    setStatus(null);
    setPendingDiscard(null);
    setCollision(null);
  }

  // Recorded samples are unrecoverable once dropped, so anything that would drop them asks first.
  function guardDiscard(run) {
    if (enrollSessionsRef.current.length > 0) setPendingDiscard({ run });
    else run();
  }

  function switchMode(next) {
    if (next === mode) return;
    guardDiscard(() => {
      setMode(next);
      setMatches(null);
      resetEnrollment();
      if (next === "identify") newSentence(false);
    });
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

  async function sendEnrollment(sessions, replace) {
    setBusy(true);
    setCollision(null);
    try {
      await postJson("/enroll", { person_id: name.toLowerCase(), name, sessions, replace });
      enrollSessionsRef.current = [];
      setEnrolled(true);
      setStatus({
        kind: "success",
        text: `${name} is enrolled from ${ENROLL_SESSIONS} typing samples. Switch to Identify to test it.`,
      });
    } catch (err) {
      if (err.status === 409) {
        // Hold the samples so choosing "replace" doesn't cost the user five sentences of retyping.
        setCollision({ sessions });
        setStatus(null);
      } else {
        setStatus({ kind: "error", text: `${err.message} Retype this sentence to try again.` });
        resetCapture(); // retype this last sentence
      }
    } finally {
      setBusy(false);
    }
  }

  async function submit() {
    const events = eventsRef.current;
    if (mode === "enroll") {
      const sessions = [...enrollSessionsRef.current, { sentence, events }];
      if (sessions.length < ENROLL_SESSIONS) {
        enrollSessionsRef.current = sessions;
        setSessionCount(sessions.length);
        setStatus(null);
        newSentence(true);
        return;
      }
      await sendEnrollment(sessions, false);
    } else {
      inputRef.current?.blur();
      renderResults();
      setBusy(true);
      try {
        const res = await postJson("/identify", { sentence, events });
        setMatches(res);
        setStatus(
          res.matched
            ? null
            : {
                kind: "declined",
                text: `No confident match. Every enrolled person scored below the ${res.min_confidence.toFixed(
                  2
                )} similarity threshold, so TypeID declined to guess rather than force a top pick.`,
              }
        );
      } catch (err) {
        setStatus({ kind: "error", text: err.message });
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
  const pillBase = "px-4 py-1.5 rounded-full border-0 text-sm font-semibold cursor-pointer transition-colors duration-150";
  // Disabled goes flat rather than faded: a translucent accent muddies into an unreadable teal.
  const primaryButton =
    "w-full py-3.5 px-7 border-0 rounded-full bg-[var(--color-accent)] text-[var(--color-card)] font-sans text-base font-semibold cursor-pointer transition disabled:bg-[var(--color-surface)] disabled:text-[var(--color-text-secondary)] disabled:cursor-not-allowed enabled:hover:bg-[var(--color-accent-deep)] enabled:hover:text-[var(--color-text)] enabled:active:scale-[0.98]";
  const ghostButton =
    "w-full py-3 px-7 border border-[var(--color-border)] rounded-full bg-transparent text-[var(--color-text)] text-[15px] font-medium cursor-pointer transition-colors duration-150 hover:bg-[var(--color-surface)]";

  return (
    <div className="min-h-screen flex flex-col items-center justify-center gap-5 px-4 py-10 sm:px-6">
      <header className="w-full max-w-[560px] flex items-center gap-3.5">
        <img src="/mark.png" alt="" width="46" height="46" className="rounded-[14px] shrink-0" />
        <div className="min-w-0">
          <h1 className="m-0 font-display text-[22px] font-bold tracking-tight leading-none">TypeID</h1>
          <p className="m-0 mt-1.5 text-[13.5px] leading-snug text-[var(--color-text-secondary)]">
            Recognizes you by <span className="font-semibold text-[var(--color-text)]">how</span> you type, not
            what you type.
          </p>
        </div>
      </header>

      <main className="w-full max-w-[560px] bg-[var(--color-card)] rounded-3xl shadow-[var(--shadow-card)] border border-[var(--color-border)] p-6 sm:p-10">
        <div className="flex gap-2 mb-7">
          {["enroll", "identify"].map((m) => (
            <button
              key={m}
              onClick={() => switchMode(m)}
              aria-pressed={mode === m}
              className={`${pillBase} capitalize ${
                mode === m
                  ? "bg-[var(--color-accent)] text-[var(--color-card)]"
                  : "bg-[var(--color-surface)] text-[var(--color-text-secondary)] hover:text-[var(--color-text)]"
              }`}
            >
              {m}
            </button>
          ))}
          <a
            href={`${API}/map`}
            target="_blank"
            rel="noreferrer"
            className={`${pillBase} ml-auto no-underline inline-flex items-center gap-1.5 bg-[var(--color-surface)] text-[var(--color-text-secondary)] hover:text-[var(--color-text)]`}
          >
            Map
            <Icon path={icons.external} size={14} />
          </a>
        </div>

        {mode === "enroll" && !nameLocked && (
          <>
            <p className="m-0 mb-6 text-[15px] leading-relaxed text-[var(--color-text-secondary)]">
              Enrolling records your rhythm across {ENROLL_SESSIONS} short sentences. Nothing you type is
              stored as text — only the timing between keys.
            </p>
            <label htmlFor="person-name" className="block mb-2 text-[13px] font-medium text-[var(--color-text-secondary)]">
              Your name
            </label>
            <input
              id="person-name"
              autoFocus
              value={name}
              onChange={(e) => setName(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && lockName()}
              placeholder="e.g. Andrei"
              className="w-full mb-7 border-0 border-b border-[var(--color-border)] bg-transparent text-[var(--color-text)] placeholder:text-[var(--color-text-secondary)] text-[17px] py-2 outline-none focus:border-b-[var(--color-accent)]"
            />
            <button onClick={lockName} disabled={!name.trim()} className={primaryButton}>
              Start enrollment
            </button>
          </>
        )}

        {mode === "enroll" && nameLocked && !enrolled && (
          <div className="mb-7 flex items-center justify-between gap-4">
            <div className="min-w-0">
              <div className="text-[17px] font-medium truncate">Enrolling {name}</div>
              <div className="mt-2 flex items-center gap-1.5">
                {Array.from({ length: ENROLL_SESSIONS }, (_, i) => (
                  <span
                    key={i}
                    className="h-1 w-7 rounded-full transition-colors duration-300"
                    style={{
                      background:
                        i < sessionCount
                          ? "var(--color-accent)"
                          : i === sessionCount
                            ? "var(--color-accent-deep)"
                            : "var(--color-border)",
                    }}
                  />
                ))}
                <span className="ml-2 text-[13px] text-[var(--color-text-secondary)]">
                  {sessionCount + 1} of {ENROLL_SESSIONS}
                </span>
              </div>
            </div>
            <button
              onClick={() => guardDiscard(resetEnrollment)}
              className="shrink-0 inline-flex items-center gap-1 border-0 bg-transparent py-3 pl-3 text-[13px] font-medium text-[var(--color-text-secondary)] cursor-pointer transition-colors duration-150 hover:text-[var(--color-text)]"
            >
              <Icon path={icons.back} size={14} />
              Change name
            </button>
          </div>
        )}

        {capturing && (
          <>
            {mode === "identify" && (
              <p className="m-0 mb-6 text-[15px] leading-relaxed text-[var(--color-text-secondary)]">
                Type the sentence below and TypeID will rank it against everyone enrolled.
              </p>
            )}

            <div className="flex items-start gap-3 mb-7">
              <p className="flex-1 m-0 font-display text-[22px] sm:text-2xl leading-snug font-medium tracking-tight">
                {sentence}
              </p>
              <button
                onClick={() => newSentence(mode === "enroll")}
                className="shrink-0 grid place-items-center w-9 h-9 rounded-full border-0 bg-[var(--color-surface)] text-[var(--color-text-secondary)] cursor-pointer transition duration-150 hover:text-[var(--color-text)] hover:rotate-45"
                title="Swap in a different sentence"
                aria-label="Swap in a different sentence"
              >
                <Icon path={icons.refresh} />
              </button>
            </div>

            <label htmlFor="capture" className="sr-only">
              Type the sentence shown above
            </label>
            <textarea
              id="capture"
              ref={inputRef}
              onKeyDown={onKeyDown}
              onKeyUp={onKeyUp}
              onPaste={(e) => e.preventDefault()}
              className="w-full resize-none border-0 border-b border-[var(--color-border)] bg-transparent text-[var(--color-text)] placeholder:text-[var(--color-text-secondary)] font-sans text-[17px] leading-relaxed py-2 outline-none transition-colors duration-150 focus:border-b-[var(--color-accent)]"
              rows="3"
              placeholder="Start typing… (pasting is disabled — the rhythm is the point)"
              autoComplete="off"
              autoCorrect="off"
              autoCapitalize="off"
              spellCheck="false"
            />

            <div className="mt-2.5 text-[13px] tabular-nums text-[var(--color-text-secondary)]">{progressText}</div>

            <button onClick={submit} disabled={recordDisabled || busy} className={`mt-7 ${primaryButton}`}>
              <span className="inline-flex items-center justify-center gap-2.5">
                {busy && <ScanRings size={17} active />}
                {busy
                  ? mode === "identify"
                    ? "Comparing against the gallery…"
                    : "Building your profile…"
                  : mode === "identify"
                    ? "Identify me"
                    : lastEnrollSentence
                      ? "Finish enrollment"
                      : "Next sentence"}
              </span>
            </button>
          </>
        )}

        {pendingDiscard && (
          <Choice
            question={`Discard ${enrollSessionsRef.current.length} recorded ${
              enrollSessionsRef.current.length === 1 ? "sample" : "samples"
            }? They can't be recovered.`}
            actions={[
              { label: "Keep recording", onClick: () => setPendingDiscard(null) },
              {
                label: "Discard",
                destructive: true,
                onClick: () => {
                  const { run } = pendingDiscard;
                  setPendingDiscard(null);
                  enrollSessionsRef.current = [];
                  run();
                },
              },
            ]}
          />
        )}

        {collision && (
          <Choice
            question={`Someone is already enrolled as ${name}. Replacing overwrites their stored typing profile for good.`}
            actions={[
              { label: "Use a different name", onClick: resetEnrollment },
              {
                label: `Replace ${name}`,
                destructive: true,
                onClick: () => sendEnrollment(collision.sessions, true),
              },
            ]}
          />
        )}

        <Status status={status} />

        {mode === "identify" && matches?.matched && (
          <div className="mt-7 animate-[rise_380ms_cubic-bezier(0.16,1,0.3,1)]">
            <div className="flex items-baseline justify-between gap-3 mb-4">
              <h2 className="m-0 font-display text-[17px] font-semibold tracking-tight">Ranked candidates</h2>
              <span className="text-[12.5px] tabular-nums text-[var(--color-text-secondary)]">
                threshold {matches.min_confidence.toFixed(2)}
              </span>
            </div>

            <ol className="m-0 p-0 list-none flex flex-col gap-3.5">
              {matches.results.map((r, i) => {
                const band = confidenceBand(r.similarity, matches.min_confidence);
                return (
                  <li key={r.person_id}>
                    <div className="flex items-baseline gap-2">
                      <span className="w-4 shrink-0 text-[13px] tabular-nums text-[var(--color-text-secondary)]">
                        {i + 1}
                      </span>
                      <span className={`truncate ${i === 0 ? "text-[17px] font-semibold" : "text-[15px]"}`}>
                        {r.name}
                      </span>
                      <span className="ml-auto text-[12.5px]" style={{ color: band.color }}>
                        {band.label}
                      </span>
                      <span className="w-[46px] text-right text-[13px] tabular-nums text-[var(--color-text-secondary)]">
                        {r.similarity.toFixed(3)}
                      </span>
                    </div>
                    <div className="relative mt-1.5 ml-6 h-1.5 rounded-full bg-[var(--color-surface)] overflow-hidden">
                      <div
                        className="h-full rounded-full origin-left animate-[sweep_620ms_cubic-bezier(0.16,1,0.3,1)]"
                        style={{ width: `${Math.max(r.similarity, 0) * 100}%`, background: band.color }}
                      />
                      {/* Notched in the card colour so the threshold stays readable across a filled bar. */}
                      <span
                        aria-hidden="true"
                        className="absolute top-0 h-full w-0.5 bg-[var(--color-card)]"
                        style={{ left: `${matches.min_confidence * 100}%` }}
                      />
                    </div>
                  </li>
                );
              })}
            </ol>
          </div>
        )}

        {enrolled && (
          <button onClick={resetEnrollment} className={`mt-6 ${ghostButton}`}>
            Enroll another person
          </button>
        )}

        <section
          className={`${mode === "identify" && showResults ? "" : "hidden "}mt-9 pt-8 border-t border-[var(--color-border)]`}
        >
          <h2 className="m-0 mb-5 font-display text-[17px] font-semibold tracking-tight">What was measured</h2>

          <div className="grid grid-cols-3 gap-x-3 gap-y-5">
            {[
              [stats.count, "keystrokes"],
              [stats.duration, "duration"],
              [stats.wpm, "wpm"],
              [stats.dwell, "avg dwell"],
              [stats.flight, "avg flight"],
            ].map(([value, label]) => (
              <div key={label} className="flex flex-col">
                <span className="font-display text-[22px] font-semibold tracking-tight tabular-nums">{value}</span>
                <span className="mt-0.5 text-xs text-[var(--color-text-secondary)]">{label}</span>
              </div>
            ))}
          </div>

          <div className="mt-7 h-14 flex items-end gap-0.5" role="img" aria-label="Dwell time per keystroke">
            {dwellValues.map((d, i) => (
              <div
                key={i}
                className="flex-1 min-w-[2px] bg-[var(--color-accent-deep)] rounded-t-sm"
                style={{ height: `${Math.max((d / maxDwell) * 100, 6)}%` }}
              />
            ))}
          </div>

          <div className="mt-5 flex gap-4">
            <button
              onClick={() => setShowRaw((prev) => !prev)}
              className="border-0 bg-transparent text-[var(--color-accent)] font-sans text-[13px] font-medium cursor-pointer p-0 hover:underline"
            >
              {showRaw ? "Hide raw event log" : "Show raw event log"}
            </button>
            <button
              onClick={() => {
                navigator.clipboard.writeText(rawText);
                setCopyLabel("Copied");
                setTimeout(() => setCopyLabel("Copy JSON"), 1200);
              }}
              className={`${showRaw ? "" : "hidden "}border-0 bg-transparent text-[var(--color-accent)] font-sans text-[13px] font-medium cursor-pointer p-0 hover:underline`}
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
              setStatus(null);
              newSentence(false);
            }}
            className={`mt-6 ${ghostButton}`}
          >
            Try another sentence
          </button>
        </section>
      </main>

      <footer className="w-full max-w-[560px] text-[12.5px] leading-relaxed text-[var(--color-text-secondary)]">
        Open-set 1:N gallery search on typing rhythm. New people enroll without retraining the model.
      </footer>
    </div>
  );
}
