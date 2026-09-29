import { useEffect, useRef, useState } from "react";
import { pickSentence } from "./sentences.js";
import { transcriptionMatch } from "./transcription.js";

const API = "/api"; // proxied to the backend by vite.config.js
// Prompts per operation. Thresholds exist for exactly these counts (backend/app/thresholds.py);
// more prompts give a steadier profile, and eval/calibrate_auth.py measured how much that buys.
const PROMPT_COUNTS = [1, 5, 10];
const DEFAULT_PROMPTS = { enroll: 5, identify: 1, authenticate: 1 };
const SHORTLIST = 5; // identify returns the top 10; ranks past this are shown greyed out
// Backend rejects fewer than 26 paired keystrokes (features.extract.MIN_KEYSTROKES + 1); leave margin.
const MIN_KEYSTROKES = 30;
// Below this the typing isn't transcription any more (see TODO item 1): someone typing a phrase
// they know by heart would match on memorised motor habit instead of rhythm.
const MIN_MATCH = 0.9;

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

function PromptProgress({ done, total }) {
  return (
    <div className="mt-2 flex items-center gap-1.5">
      {Array.from({ length: total }, (_, i) => (
        <span
          key={i}
          className={`h-1 rounded-full transition-colors duration-300 ${total > 5 ? "w-3.5" : "w-7"}`}
          style={{
            background:
              i < done ? "var(--color-accent)" : i === done ? "var(--color-accent-deep)" : "var(--color-border)",
          }}
        />
      ))}
      <span className="ml-2 text-[13px] text-[var(--color-text-secondary)]">
        {done + 1} of {total}
      </span>
    </div>
  );
}

export default function App() {
  const inputRef = useRef(null);
  const eventsRef = useRef([]);
  const keystrokeCountRef = useRef(0);
  const sessionsRef = useRef([]); // prompts typed so far in the current enrol / identify / verify
  const usedSentencesRef = useRef([]); // enrollment never repeats a sentence

  const [mode, setMode] = useState("enroll");
  const [name, setName] = useState("");
  const [nameLocked, setNameLocked] = useState(false);
  const [sessionCount, setSessionCount] = useState(0);
  const [promptCounts, setPromptCounts] = useState(DEFAULT_PROMPTS);
  const promptCount = promptCounts[mode];
  const [identifyRates, setIdentifyRates] = useState(null); // measured outcomes per prompt count
  const [enrolled, setEnrolled] = useState(false);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState(null); // { kind: success | error | declined, text }
  const [matches, setMatches] = useState(null);
  const [pendingDiscard, setPendingDiscard] = useState(null); // { run } — awaiting confirmation
  const [collision, setCollision] = useState(null); // { sessions } — name already enrolled
  const [people, setPeople] = useState([]); // enrolled people you can claim to be
  const [claimId, setClaimId] = useState(""); // the identity being claimed in authenticate mode
  const [auth, setAuth] = useState(null); // AuthenticateResponse

  const [sentence, setSentence] = useState("");
  const [progressText, setProgressText] = useState("0 keystrokes");
  const [recordDisabled, setRecordDisabled] = useState(true);
  const [showResults, setShowResults] = useState(false);
  const [showRaw, setShowRaw] = useState(false);
  const [copyLabel, setCopyLabel] = useState("Copy JSON");
  const [stats, setStats] = useState({ count: 0, duration: "0.0s", wpm: 0, dwell: "0ms", flight: "0ms" });
  const [dwellValues, setDwellValues] = useState([]);
  const [rawText, setRawText] = useState("");
  const [typed, setTyped] = useState(""); // mirrors the typing box, for the transcription check

  // The typing box only exists once we know whose typing it is: the enrolling person, the
  // identify query, or — for authenticate — the identity being claimed.
  const capturing =
    mode === "identify" || (mode === "authenticate" && !!claimId) || (nameLocked && !enrolled);

  function resetCapture() {
    eventsRef.current = [];
    keystrokeCountRef.current = 0;
    if (inputRef.current) inputRef.current.value = "";
    setTyped("");
    setRecordDisabled(true);
    setProgressText("0 keystrokes");
    setShowResults(false);
    setShowRaw(false);
    setCopyLabel("Copy JSON");
    inputRef.current?.focus();
  }

  // `forMode`: switchMode calls this before the setMode update lands, so it passes the new mode.
  function newSentence(accumulate, forMode = mode) {
    const next = pickSentence(forMode, usedSentencesRef.current);
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
    sessionsRef.current = [];
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
    if (sessionsRef.current.length > 0) setPendingDiscard({ run });
    else run();
  }

  function changePromptCount(n) {
    if (n === promptCount) return;
    guardDiscard(() => {
      setPromptCounts((counts) => ({ ...counts, [mode]: n }));
      setSessionCount(0);
      resetCapture();
    });
  }

  function switchMode(next) {
    if (next === mode) return;
    guardDiscard(() => {
      setMode(next);
      setMatches(null);
      setAuth(null);
      setClaimId("");
      resetEnrollment();
      if (next === "identify") newSentence(false, next);
      if (next === "authenticate") {
        fetch(`${API}/people`)
          .then((r) => r.json())
          .then(setPeople)
          .catch(() => setStatus({ kind: "error", text: "Couldn't load the enrolled people." }));
      }
    });
  }

  function claim(personId) {
    setClaimId(personId);
    setAuth(null);
    setStatus(null);
    if (personId) newSentence(false);
  }

  // After a verdict: clear it and start a fresh round on a new prompt.
  function nextRound() {
    setMatches(null);
    setAuth(null);
    setStatus(null);
    newSentence(false);
  }

  function onKeyDown(e) {
    // Typing at a finished verdict means "go again": start a fresh round on a new prompt. The key
    // itself is swallowed, since it was aimed at the old prompt.
    if (showingResult) {
      e.preventDefault();
      nextRound();
      return;
    }
    // Enter submits (same as the button) and is never part of the typing sample.
    if (e.key === "Enter") {
      e.preventDefault();
      if (canSubmit && !busy) submit();
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
        remaining > 0 ? `${count} keystrokes · ${remaining} more to record` : `${count} keystrokes`
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
      sessionsRef.current = [];
      setEnrolled(true);
      setStatus({
        kind: "success",
        text: `${name} is enrolled from ${sessions.length} typing ${sessions.length === 1 ? "sample" : "samples"}.`,
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
    const sessions = [...sessionsRef.current, { sentence, events: eventsRef.current }];
    if (sessions.length < promptCount) {
      sessionsRef.current = sessions;
      setSessionCount(sessions.length);
      setStatus(null);
      newSentence(true);
      return;
    }
    if (mode === "enroll") {
      await sendEnrollment(sessions, false);
      return;
    }

    // "What was measured" shows the last prompt; the decision uses all of them.
    inputRef.current?.blur();
    renderResults();
    sessionsRef.current = [];
    setSessionCount(0);
    setBusy(true);
    try {
      if (mode === "authenticate") {
        setAuth(await postJson("/authenticate", { person_id: claimId, sessions }));
        setStatus(null);
      } else {
        const res = await postJson("/identify", { sessions });
        setMatches(res);
        setStatus(
          res.matched
            ? null
            : {
                kind: "declined",
                text: `No confident match. Nobody reached the ${res.min_confidence.toFixed(
                  2
                )} similarity threshold, so TypeID declined to name anyone. The closest candidates are below.`,
              }
        );
      }
    } catch (err) {
      setStatus({ kind: "error", text: err.message });
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    if (capturing) inputRef.current?.focus();
  }, [capturing, sentence]);

  // Measured error per prompt count, from the backend's calibration table rather than a copy here.
  useEffect(() => {
    fetch(`${API}/identify/rates`)
      .then((r) => r.json())
      .then(setIdentifyRates)
      .catch(() => {}); // the hint is optional; identify works without it
  }, []);

  const match = transcriptionMatch(sentence, typed);
  const matchPercent = Math.round(match.accuracy * 100);
  // Once a verdict is showing, the typed prompts are spent: the next round starts on a new prompt,
  // so last round's keystrokes can't be resubmitted as the first prompt of a new one.
  const showingResult = mode !== "enroll" && !!(matches || auth);
  const canSubmit = !recordDisabled && !showingResult && match.accuracy >= MIN_MATCH;
  const maxDwell = Math.max(...dwellValues, 1);
  const lastPrompt = sessionCount === promptCount - 1;
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
        <div className="flex flex-wrap gap-2 mb-7">
          {["enroll", "identify", "authenticate"].map((m) => (
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

        {!enrolled && (
          <fieldset className="m-0 p-0 border-0 mb-7">
            <legend className="mb-2.5 p-0 text-[13px] font-medium text-[var(--color-text-secondary)]">
              Prompts to type · more prompts, steadier profile, fewer mistakes
            </legend>
            <div className="flex gap-2">
              {PROMPT_COUNTS.map((n) => (
                <label
                  key={n}
                  className={`grid place-items-center min-w-11 h-11 px-3 cursor-pointer rounded-full text-[15px] font-semibold tabular-nums border transition-colors duration-150 has-[:focus-visible]:outline has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-3 has-[:focus-visible]:outline-[var(--color-accent)] ${
                    promptCount === n
                      ? "border-transparent bg-[var(--color-accent)] text-[var(--color-card)]"
                      : "border-[var(--color-border)] bg-[var(--color-surface)] text-[var(--color-text)] hover:border-[var(--color-accent-deep)]"
                  }`}
                >
                  <input
                    type="radio"
                    name="prompt-count"
                    value={n}
                    checked={promptCount === n}
                    onChange={() => changePromptCount(n)}
                    className="sr-only"
                  />
                  {n}
                </label>
              ))}
            </div>
            {/* The cost of a snappy one-prompt identify, stated up front rather than discovered. */}
            {mode === "identify" && identifyRates && (
              <p className="m-0 mt-3 text-[12.5px] leading-snug text-[var(--color-text-secondary)]">
                Finds an enrolled person{" "}
                {PROMPT_COUNTS.map((n, i) => (
                  <span key={n}>
                    {i > 0 && " · "}
                    <span className={`whitespace-nowrap ${n === promptCount ? "font-semibold text-[var(--color-text)]" : ""}`}>
                      {Math.round(identifyRates.found_percent[n])}% with {n}
                    </span>
                  </span>
                ))}
                . Names someone who isn't enrolled about{" "}
                {Math.round(identifyRates.stranger_named_percent[promptCount])}% of the time.
              </p>
            )}
          </fieldset>
        )}

        {mode === "enroll" && !nameLocked && (
          <>
            <p className="m-0 mb-6 text-[15px] leading-relaxed text-[var(--color-text-secondary)]">
              Enrolling records your rhythm across {promptCount === 1 ? "one short sentence" : `${promptCount} short sentences`}. Nothing you type is
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
              onKeyDown={(e) => {
                if (e.key !== "Enter") return;
                // lockName moves focus into the typing box mid-keypress; without this, the browser
                // inserts this Enter's line break there and every character after it misaligns.
                e.preventDefault();
                lockName();
              }}
              placeholder="e.g. Andrei"
              className="w-full mb-7 border-0 border-b border-[var(--color-border)] bg-transparent text-[var(--color-text)] placeholder:text-[var(--color-text-secondary)] text-[17px] py-2 outline-none focus:border-b-[var(--color-accent)]"
            />
            <button onClick={lockName} disabled={!name.trim()} className={primaryButton}>
              Start enrollment
            </button>
          </>
        )}

        {mode === "authenticate" && (
          <>
            <p className="m-0 mb-6 text-[15px] leading-relaxed text-[var(--color-text-secondary)]">
              Verification is a yes/no question, not a search: claim an identity, type the sentence,
              and TypeID compares your rhythm against that one person's stored profile.
            </p>
            {/* Real radios behind styled labels: a native select popup is drawn by the OS, which
                ignores this palette and rendered light text on a white list. This keeps the
                arrow-key behaviour and semantics a pick-one control should have. */}
            <fieldset className="m-0 p-0 border-0 mb-7">
              <legend className="mb-3 p-0 text-[13px] font-medium text-[var(--color-text-secondary)]">
                Who do you claim to be?
              </legend>
              {people.length === 0 ? (
                <p className="m-0 text-[14px] text-[var(--color-text-secondary)]">
                  Nobody is enrolled yet — enroll someone first, then come back to verify them.
                </p>
              ) : (
                <div className="flex flex-wrap gap-2">
                  {people.map((p) => (
                    <label
                      key={p.person_id}
                      className={`cursor-pointer rounded-full px-4 py-2.5 text-[15px] font-medium border transition-colors duration-150 has-[:focus-visible]:outline has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-3 has-[:focus-visible]:outline-[var(--color-accent)] ${
                        claimId === p.person_id
                          ? "border-transparent bg-[var(--color-accent)] text-[var(--color-card)]"
                          : "border-[var(--color-border)] bg-[var(--color-surface)] text-[var(--color-text)] hover:border-[var(--color-accent-deep)]"
                      }`}
                    >
                      <input
                        type="radio"
                        name="claim"
                        value={p.person_id}
                        checked={claimId === p.person_id}
                        onChange={() => claim(p.person_id)}
                        className="sr-only"
                      />
                      {p.name}
                    </label>
                  ))}
                </div>
              )}
            </fieldset>
          </>
        )}

        {mode === "enroll" && nameLocked && !enrolled && (
          <div className="mb-7 flex items-center justify-between gap-4">
            <div className="min-w-0">
              <div className="text-[17px] font-medium truncate">Enrolling {name}</div>
              <PromptProgress done={sessionCount} total={promptCount} />
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
                Type {promptCount === 1 ? "the sentence below" : `${promptCount} sentences`} and TypeID will rank
                your rhythm against everyone enrolled.
              </p>
            )}
            {mode === "authenticate" && (
              <p className="m-0 mb-6 text-[15px] leading-relaxed text-[var(--color-text-secondary)]">
                Claiming to be{" "}
                <span className="font-semibold text-[var(--color-text)]">
                  {people.find((p) => p.person_id === claimId)?.name ?? claimId}
                </span>
                . Type {promptCount === 1 ? "the sentence below" : `${promptCount} sentences`} to prove it.
              </p>
            )}

            {mode !== "enroll" && promptCount > 1 && !showingResult && (
              <div className="mb-5">
                <PromptProgress done={sessionCount} total={promptCount} />
              </div>
            )}

            <div className="flex items-start gap-3 mb-7">
              <p className="flex-1 m-0 font-display text-[22px] sm:text-2xl leading-snug font-medium tracking-tight">
                <span className="sr-only">{sentence}</span>
                <span aria-hidden="true">
                  {[...sentence].map((ch, i) => (
                    <span
                      key={i}
                      className={
                        match.chars[i] === "correct"
                          ? "text-[var(--color-text)]"
                          : match.chars[i] === "wrong"
                            ? "rounded-sm text-[var(--color-danger)] bg-[color-mix(in_srgb,var(--color-danger)_20%,transparent)]"
                            : `text-[var(--color-text-secondary)]${
                                i === typed.length ? " underline decoration-2 underline-offset-4 decoration-[var(--color-accent)]" : ""
                              }`
                      }
                    >
                      {ch}
                    </span>
                  ))}
                  {match.extra > 0 && <span className="text-[var(--color-danger)]"> +{match.extra}</span>}
                </span>
              </p>
              <button
                onClick={() => newSentence(true)}
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
              onInput={(e) => setTyped(e.target.value)}
              onPaste={(e) => e.preventDefault()}
              readOnly={showingResult}
              onFocus={() => showingResult && nextRound()} // clicking back in to type starts the next round
              className="w-full resize-none border-0 border-b border-[var(--color-border)] bg-transparent text-[var(--color-text)] placeholder:text-[var(--color-text-secondary)] font-sans text-[17px] leading-relaxed py-2 outline-none transition-colors duration-150 focus:border-b-[var(--color-accent)]"
              rows="3"
              placeholder="Start typing… (pasting is disabled — the rhythm is the point)"
              autoComplete="off"
              autoCorrect="off"
              autoCapitalize="off"
              spellCheck="false"
            />

            <div className="mt-2.5 text-[13px] tabular-nums text-[var(--color-text-secondary)]">
              {progressText} ·{" "}
              <span className={typed && match.accuracy < MIN_MATCH ? "text-[var(--color-danger)]" : ""}>
                {matchPercent}% match{match.accuracy < MIN_MATCH && `, ${Math.round(MIN_MATCH * 100)}% needed`}
              </span>
            </div>

            {showingResult ? (
              // Takes the submit button's place and focus, so Enter moves straight on to the next round. The
              // distinct keys make React mount a new element; reusing the old one would skip autoFocus.
              <button
                key="next"
                onClick={nextRound}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.preventDefault(); // same focus hand-off as the name field: keep the newline out of the box
                    nextRound();
                  } else if (e.key.length === 1 && e.key !== " ") nextRound(); // just start typing
                }}
                autoFocus
                className={`mt-7 ${primaryButton}`}
              >
                Try another sentence
              </button>
            ) : (
              <button key="submit" onClick={submit} disabled={!canSubmit || busy} className={`mt-7 ${primaryButton}`}>
                <span className="inline-flex items-center justify-center gap-2.5">
                  {busy && <ScanRings size={17} active />}
                  {busy
                    ? mode === "identify"
                      ? "Comparing against the gallery…"
                      : mode === "authenticate"
                        ? "Checking against the claim…"
                        : "Building your profile…"
                    : !lastPrompt
                      ? "Next sentence"
                      : mode === "identify"
                        ? "Identify me"
                        : mode === "authenticate"
                          ? "Verify me"
                          : "Finish enrollment"}
                </span>
              </button>
            )}
          </>
        )}

        {pendingDiscard && (
          <Choice
            question={`Discard ${sessionsRef.current.length} recorded ${
              sessionsRef.current.length === 1 ? "sample" : "samples"
            }? They can't be recovered.`}
            actions={[
              { label: "Keep recording", onClick: () => setPendingDiscard(null) },
              {
                label: "Discard",
                destructive: true,
                onClick: () => {
                  const { run } = pendingDiscard;
                  setPendingDiscard(null);
                  sessionsRef.current = [];
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

        {mode === "identify" && matches?.results.length > 0 && (
          <div className="mt-7 animate-[rise_380ms_cubic-bezier(0.16,1,0.3,1)]">
            <div className="flex items-baseline justify-between gap-3 mb-4">
              <h2 className="m-0 font-display text-[17px] font-semibold tracking-tight">
                {matches.matched ? "Ranked candidates" : "Closest candidates"}
              </h2>
              <span className="text-[12.5px] tabular-nums text-[var(--color-text-secondary)]">
                threshold {matches.min_confidence.toFixed(2)} · {matches.query_prompts}{" "}
                {matches.query_prompts === 1 ? "prompt" : "prompts"} · finds {Math.round(matches.found_percent)}%
              </span>
            </div>

            <ol className="m-0 p-0 list-none flex flex-col gap-3.5">
              {matches.results.map((r, i) => {
                const band = confidenceBand(r.similarity, matches.min_confidence);
                return (
                  // Ranks 6-10 are context (how crowded it is just below the shortlist), so they recede.
                  <li key={r.person_id} className={i >= SHORTLIST ? "opacity-45" : ""}>
                    <div className="flex items-baseline gap-2">
                      <span className="w-5 shrink-0 text-[13px] tabular-nums text-[var(--color-text-secondary)]">
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
                    <div className="relative mt-1.5 ml-7 h-1.5 rounded-full bg-[var(--color-surface)] overflow-hidden">
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

        {mode === "authenticate" && auth && (
          <div className="mt-7 animate-[rise_380ms_cubic-bezier(0.16,1,0.3,1)]">
            <div
              className="flex items-start gap-3"
              style={{ color: auth.accepted ? "var(--color-accent)" : "var(--color-danger)" }}
            >
              <Icon path={auth.accepted ? icons.check : icons.declined} size={22} className="mt-1 shrink-0" />
              <div className="min-w-0">
                <h2 className="m-0 font-display text-[22px] font-semibold tracking-tight">
                  {auth.accepted ? `Verified as ${auth.name}` : "Not verified"}
                </h2>
                <p className="m-0 mt-1.5 text-[14.5px] leading-snug text-[var(--color-text-secondary)]">
                  {auth.accepted
                    ? `This typing rhythm matches ${auth.name}'s stored profile closely enough to accept.`
                    : `This typing rhythm isn't close enough to ${auth.name}'s stored profile to accept the claim.`}
                </p>
              </div>
            </div>

            {/* The decision is one number against one line — show both rather than just the verdict. */}
            <div className="mt-5">
              <div className="flex items-baseline justify-between text-[12.5px] tabular-nums text-[var(--color-text-secondary)]">
                <span>similarity {auth.similarity.toFixed(3)}</span>
                <span>
                  threshold {auth.threshold.toFixed(2)} · {auth.operating_point} {auth.eer_percent}%
                </span>
              </div>
              <div className="relative mt-1.5 h-2 rounded-full bg-[var(--color-surface)] overflow-hidden">
                <div
                  className="h-full rounded-full origin-left animate-[sweep_620ms_cubic-bezier(0.16,1,0.3,1)]"
                  style={{
                    width: `${Math.max(auth.similarity, 0) * 100}%`,
                    background: auth.accepted ? "var(--color-accent)" : "var(--color-danger)",
                  }}
                />
                <span
                  aria-hidden="true"
                  className="absolute top-0 h-full w-0.5 bg-[var(--color-card)]"
                  style={{ left: `${auth.threshold * 100}%` }}
                />
              </div>
              <p className="m-0 mt-3 text-[12.5px] leading-snug text-[var(--color-text-secondary)]">
                {auth.name} enrolled from {auth.enroll_prompts}, verified with {auth.query_prompts}.{" "}
                {auth.extrapolated
                  ? `No measured threshold for ${auth.enroll_prompts} + ${auth.query_prompts} (Aalto has at most 15 prompts per person), so this borrows the 10 + 5 one.`
                  : `At this setting, ${auth.eer_percent}% of decisions are wrong either way.`}
              </p>
            </div>
          </div>
        )}

        {enrolled && (
          // Testing yourself is the point of enrolling, so it's the primary action and takes focus
          // (Enter continues); enrolling someone else is the quieter alternative.
          <div className="mt-6 flex flex-col gap-3">
            <button onClick={() => switchMode("identify")} autoFocus className={primaryButton}>
              Identify yourself now
            </button>
            <button onClick={resetEnrollment} className={ghostButton}>
              Enroll another person
            </button>
          </div>
        )}

        <section
          className={`${mode !== "enroll" && showResults ? "" : "hidden "}mt-9 pt-8 border-t border-[var(--color-border)]`}
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
        </section>
      </main>

      <footer className="w-full max-w-[560px] text-[12.5px] leading-relaxed text-[var(--color-text-secondary)]">
        Open-set 1:N gallery search on typing rhythm. New people enroll without retraining the model.
      </footer>
    </div>
  );
}
