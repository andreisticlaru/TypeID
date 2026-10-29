import { useEffect, useRef, useState } from "react";
import { pickSentence } from "./sentences.js";
import { transcriptionMatch } from "./transcription.js";
import { Badge, Button, Card, FILL, GhostButton, Kbd, Label } from "./ui.jsx";

const API = "/api"; // proxied to the backend by vite.config.js
// Prompts per operation. Thresholds exist for exactly these counts (backend/app/thresholds.py);
// more prompts give a steadier profile, and eval/calibrate_auth.py measured how much that buys.
const PROMPT_COUNTS = [1, 5, 10];
const DEFAULT_PROMPTS = { enroll: 5, identify: 1, authenticate: 1 };
const SHORTLIST = 5; // identify returns the top 10; ranks past this are shown faded
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
  if (similarity >= threshold + 0.15) return "strong match";
  if (similarity >= threshold) return "match";
  return "below threshold";
}

const MODES = {
  enroll: { label: "Enroll", fill: "mint", icon: "user" },
  identify: { label: "Identify", fill: "sun", icon: "search" },
  authenticate: { label: "Authenticate", fill: "lilac", icon: "shield" },
};

// Square caps and mitred joins: the icons share the borders' hard edges.
const stroke = {
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 2,
  strokeLinecap: "square",
  strokeLinejoin: "miter",
};

function Icon({ path, size = 16, className = "" }) {
  return (
    <svg width={size} height={size} viewBox="0 0 16 16" aria-hidden="true" className={`shrink-0 ${className}`} {...stroke}>
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
  arrow: <path d="M3 8h10M9 4l4 4-4 4" />,
  user: <path d="M8 8a2.6 2.6 0 1 0 0-5.2A2.6 2.6 0 0 0 8 8zM3 14c.6-2.6 2.6-4 5-4s4.4 1.4 5 4" />,
  search: <path d="M7 12A5 5 0 1 0 7 2a5 5 0 0 0 0 10zM10.6 10.6L14 14" />,
  shield: <path d="M8 1.8l5 2v4c0 3-2.2 5.2-5 6.4C5.2 13 3 10.8 3 7.8v-4z" />,
  map: <path d="M1.8 3.8l4-1.6 4.4 1.6 4-1.6v10l-4 1.6-4.4-1.6-4 1.6zM5.8 2.2v10M10.2 3.8v10" />,
};

// The mark's two broken rings, reused as the working indicator: the logo scans while the model does.
function ScanRings({ size = 18 }) {
  const spin = { transformBox: "fill-box", transformOrigin: "center" };
  return (
    <svg width={size} height={size} viewBox="0 0 48 48" aria-hidden="true" className="shrink-0 motion-reduce:hidden">
      <circle cx="24" cy="24" r="19" fill="none" stroke="currentColor" strokeWidth="5" strokeDasharray="15 4.9"
        style={{ ...spin, animation: "scan-cw 2.4s linear infinite" }} />
      <circle cx="24" cy="24" r="10.5" fill="none" stroke="currentColor" strokeWidth="4" strokeDasharray="9 4.2"
        style={{ ...spin, animation: "scan-ccw 1.8s linear infinite" }} />
    </svg>
  );
}

// A square icon tile, like the library's avatar: used for mode marks and verdict icons.
function Tile({ children, size = "size-10", fill = "white", className = "" }) {
  return (
    <span className={`${size} grid place-items-center shrink-0 rounded-base border-2 border-ink ${FILL[fill]} ${className}`}>
      {children}
    </span>
  );
}

// The headline of every result: a filled card with an icon, a title, a line of detail and a stamp.
// With `focusOnMount`, the verdict takes focus when it appears: on a phone the submit button sat far below it,
// so it was off-screen, and focusing the heading is also what makes a screen reader read it out.
function Verdict({ fill, icon, title, detail, aside, stamp, focusOnMount = false, onKeyDown }) {
  const cardRef = useRef(null);
  const headingRef = useRef(null);
  useEffect(() => {
    if (!focusOnMount) return;
    cardRef.current.scrollIntoView({ block: "start" });
    headingRef.current.focus({ preventScroll: true });
  }, [focusOnMount]);
  return (
    <div ref={cardRef} className={`flex flex-wrap items-center gap-4 rounded-base border-2 border-ink p-4 sm:p-5 shadow-brutal scroll-mt-4 animate-[rise_320ms_cubic-bezier(0.16,1,0.3,1)] ${fill}`}>
      <Tile size="size-14" className="text-2xl font-display font-bold">{icon}</Tile>
      <div className="min-w-0 flex-1">
        <h2
          ref={headingRef}
          tabIndex={focusOnMount ? -1 : undefined}
          onKeyDown={onKeyDown}
          className="m-0 font-display text-[26px] sm:text-[30px] font-bold leading-tight tracking-tight break-words"
        >
          {title}
        </h2>
        {detail && <p className="m-0 mt-1 text-[15px] leading-snug">{detail}</p>}
      </div>
      {(aside || stamp) && (
        <div className="flex sm:flex-col items-center sm:items-end gap-3 sm:gap-2">
          {aside}
          {stamp && (
            <span className="-rotate-3 rounded-base border-2 border-ink bg-white px-2.5 py-1 text-[12.5px] font-bold uppercase tracking-[0.1em] shadow-brutal-sm">
              {stamp}
            </span>
          )}
        </div>
      )}
    </div>
  );
}

// A filled bar inside an ink track with the decision threshold as a dashed line through it.
function ThresholdBar({ value, threshold, fill, height = "h-[18px]", line = "border-dashed" }) {
  return (
    <div className={`relative ${height} rounded-[3px] border-2 border-ink bg-wash`}>
      <div
        className={`h-full border-r-2 border-ink origin-left animate-[sweep_620ms_cubic-bezier(0.16,1,0.3,1)] ${fill}`}
        style={{ width: `${Math.min(Math.max(value, 0), 1) * 100}%` }}
      />
      <span
        aria-hidden="true"
        className={`absolute -top-1.5 -bottom-1.5 border-l-2 border-ink ${line}`}
        style={{ left: `${threshold * 100}%` }}
      />
    </div>
  );
}

function PromptProgress({ done, total }) {
  return (
    <div className="flex items-center gap-3">
      <div className="flex gap-1.5" aria-hidden="true">
        {Array.from({ length: total }, (_, i) => (
          <span
            key={i}
            className={`h-3.5 rounded-[3px] border-2 border-ink ${total > 5 ? "w-4" : "w-8"} ${i < done ? "bg-mint" : "bg-white"}`}
            style={
              i === done
                ? { background: "repeating-linear-gradient(-45deg, var(--color-mint) 0 4px, #fff 4px 8px)" }
                : undefined
            }
          />
        ))}
      </div>
      <span className="text-[13.5px] font-bold">
        {Math.min(done + 1, total)} of {total}
      </span>
    </div>
  );
}

// Destructive steps ask first, in-world, rather than through a browser confirm(). Escape and the
// backdrop pick the safe option. A native modal <dialog> keeps Tab inside and makes the page behind inert.
function Dialog({ title, children, actions, onDismiss }) {
  const ref = useRef(null);
  useEffect(() => {
    const opener = document.activeElement;
    ref.current.showModal();
    return () => opener?.focus();
  }, []);
  return (
    <dialog
      ref={ref}
      role="alertdialog"
      aria-labelledby="dialog-title"
      onCancel={(e) => {
        e.preventDefault(); // Escape: let React unmount it, so the opener gets focus back
        onDismiss();
      }}
      onClick={(e) => e.target === e.currentTarget && onDismiss()} // only the backdrop targets the <dialog> itself
      className="m-auto w-[calc(100%-2rem)] max-w-[460px] p-0 rounded-base border-2 border-ink bg-white text-ink shadow-brutal-xl backdrop:bg-ink/55 animate-[rise_220ms_cubic-bezier(0.16,1,0.3,1)]"
    >
      <div>
        <div className="flex items-center gap-2.5 rounded-t-[3px] border-b-2 border-ink bg-coral px-4.5 py-3.5">
          <Icon path={icons.alert} size={18} />
          <h2 id="dialog-title" className="m-0 font-display text-lg font-bold">{title}</h2>
        </div>
        <div className="p-4.5">
          <p className="m-0 text-[15px] leading-relaxed">{children}</p>
          <div className="mt-5 flex flex-wrap justify-end gap-3">{actions}</div>
        </div>
      </div>
    </dialog>
  );
}

// Errors only: every success is a verdict card of its own.
function ErrorNote({ error }) {
  return (
    <div aria-live="polite">
      {error && (
        <p className="m-0 mt-6 flex items-start gap-2.5 rounded-base border-2 border-ink bg-coral-soft px-4 py-3 text-[14.5px] leading-snug shadow-brutal animate-[rise_320ms_cubic-bezier(0.16,1,0.3,1)]">
          <Icon path={icons.alert} className="mt-[3px]" />
          <span>{error}</span>
        </p>
      )}
    </div>
  );
}

const HOW_IT_WORKS = {
  enroll: [
    "Pick a name for your profile.",
    "Type each sentence as it appears. Only the timing between keys is kept, never the text.",
    "Your prompts are averaged into one 128-number profile in the gallery.",
  ],
  identify: [
    "Type the sentence. No name needed.",
    "TypeID ranks your rhythm against everyone enrolled, the 500 Aalto background people included.",
    "If nobody clears the threshold, it says so instead of guessing.",
  ],
  authenticate: [
    "Claim an enrolled identity.",
    "Type the sentence to prove it.",
    "One score against one threshold: accept or reject.",
  ],
};

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
  const [error, setError] = useState(null); // message of the last failed request
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
  const [stats, setStats] = useState({ count: 0, duration: "0.0s", wpm: 0, dwell: "0ms", flight: "0ms", backspaces: 0 });
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
    setError(null);
    setPendingDiscard(null);
    setCollision(null);
  }

  // Recorded samples are unrecoverable once dropped, so anything that would drop them asks first.
  function guardDiscard(run, reason) {
    if (sessionsRef.current.length > 0) setPendingDiscard({ run, reason });
    else run();
  }

  function changePromptCount(n) {
    if (n === promptCount) return;
    guardDiscard(() => {
      setPromptCounts((counts) => ({ ...counts, [mode]: n }));
      setSessionCount(0);
      resetCapture();
    }, `Changing to ${n} ${n === 1 ? "prompt" : "prompts"} starts over.`);
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
          .then((r) => (r.ok ? r.json() : Promise.reject()))
          .then(setPeople)
          .catch(() => setError("Couldn't load the enrolled people."));
      }
    }, `Switching to ${MODES[next].label} drops them.`);
  }

  function claim(personId) {
    setClaimId(personId);
    setAuth(null);
    setError(null);
    if (personId) newSentence(false);
  }

  // After a verdict: clear it and start a fresh round on a new prompt.
  function nextRound() {
    setMatches(null);
    setAuth(null);
    setError(null);
    newSentence(false);
  }

  // The no-match screen's way out: the same search with five prompts, which finds people far more often.
  function retryWithMorePrompts() {
    setPromptCounts((counts) => ({ ...counts, [mode]: 5 }));
    setSessionCount(0);
    nextRound();
  }

  function onKeyDown(e) {
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
    // The event's own timestamp: when the browser created it, on the performance.now() clock. Reading the
    // clock here instead would add however long the main thread was busy (measured p90 9 ms, 40 ms on a
    // throttled CPU), straight into the hold and flight times the model reads.
    eventsRef.current.push({ code: e.code, key: e.key, type: e.type, t: e.nativeEvent.timeStamp });

    if (e.type === "keydown") {
      keystrokeCountRef.current += 1;
      const count = keystrokeCountRef.current;
      const remaining = MIN_KEYSTROKES - count;
      setProgressText(remaining > 0 ? `${count} keystrokes · ${remaining} more to go` : `${count} keystrokes`);
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
      backspaces: records.filter((r) => r.key === "Backspace").length,
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
      setError(null);
    } catch (err) {
      if (err.status === 409) {
        // Hold the samples so choosing "replace" doesn't cost the user five sentences of retyping.
        setCollision({ sessions });
        setError(null);
      } else {
        setError(`${err.message} Retype this sentence to try again.`);
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
      setError(null);
      newSentence(true);
      return;
    }
    if (mode === "enroll") {
      await sendEnrollment(sessions, false);
      return;
    }

    // "What was measured" shows the last prompt; the decision uses all of them. Blur first, so keys
    // pressed while the request is out don't land in the event log the panel is built from.
    inputRef.current?.blur();
    renderResults();
    sessionsRef.current = [];
    setSessionCount(0);
    setBusy(true);
    try {
      if (mode === "authenticate") {
        setAuth(await postJson("/authenticate", { person_id: claimId, sessions }));
      } else {
        setMatches(await postJson("/identify", { sessions }));
      }
      setError(null);
    } catch (err) {
      setError(err.message);
      setShowResults(false);
    } finally {
      setBusy(false);
    }
  }

  // Measured error per prompt count, from the backend's calibration table rather than a copy here.
  useEffect(() => {
    fetch(`${API}/identify/rates`)
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then(setIdentifyRates)
      .catch(() => {}); // the hint is optional; identify works without it
  }, []);

  const match = transcriptionMatch(sentence, typed);
  // Rounded down, so 89.7% never shows as the "90% needed" it falls short of.
  const matchPercent = Math.floor(match.accuracy * 100);
  // Once a verdict is showing, the typed prompts are spent: the next round starts on a new prompt,
  // so last round's keystrokes can't be resubmitted as the first prompt of a new one.
  const showingResult = mode !== "enroll" && !!(matches || auth);
  const canSubmit = !recordDisabled && !showingResult && match.accuracy >= MIN_MATCH;
  const maxDwell = Math.max(...dwellValues, 1);
  const lastPrompt = sessionCount === promptCount - 1;
  const modeFill = MODES[mode].fill;
  const claimedName = people.find((p) => p.person_id === claimId)?.name ?? claimId;
  // The typing area gives way to whatever has to be answered first: a verdict or a name collision.
  const typingVisible = capturing && !showingResult && !collision;

  // Declared after showingResult: the typing box remounts when a verdict clears, and needs focus then.
  useEffect(() => {
    if (typingVisible) inputRef.current?.focus();
  }, [typingVisible, sentence]);

  const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;

  // Typing at a finished verdict means "go again": with focus on the verdict (or the next-round button),
  // Enter or any printable key starts a fresh round, swallowing the key that was aimed at the old prompt.
  function nextRoundKeys(e) {
    if (e.key === "Enter") {
      e.preventDefault(); // keep the newline out of the next typing box
      nextRound();
    } else if (e.key.length === 1 && e.key !== " ") {
      e.preventDefault();
      nextRound();
    }
  }

  let title;
  let headAside = null;
  if (mode === "enroll") {
    title = nameLocked && !enrolled ? `Enrolling ${name}` : "Enroll";
    headAside =
      nameLocked && !enrolled ? (
        <GhostButton onClick={() => guardDiscard(resetEnrollment, "Changing the name starts over.")}>
          <Icon path={icons.back} size={14} />
          Change name
        </GhostButton>
      ) : (
        <Badge>{enrolled ? "Done" : "Step 1 of 2"}</Badge>
      );
  } else if (mode === "identify") {
    title = "Who is typing?";
    headAside = (
      <Badge>
        {plural(promptCount, "prompt")}
        {matches && ` · threshold ${matches.min_confidence.toFixed(2)}`}
      </Badge>
    );
  } else {
    title = "Prove who you are";
    headAside = <Badge>{claimId ? `Claiming ${claimedName}` : "Step 1 of 2"}</Badge>;
  }

  return (
    <div className="min-h-screen flex flex-col">
      <header className="bg-white border-b-2 border-ink">
        <div className="mx-auto max-w-[1160px] px-4 py-3 sm:px-6 sm:py-3.5 flex flex-wrap items-center gap-x-6 gap-y-3">
          <div className="flex items-center gap-3 min-w-0">
            <img src="/favicon.svg" alt="" width="48" height="48" className="shrink-0 drop-shadow-[3px_3px_0_var(--color-ink)]" />
            <div className="min-w-0">
              <h1 className="m-0 font-display text-2xl font-bold leading-none tracking-tight">TypeID</h1>
              <p className="m-0 mt-1 text-[13px] leading-snug text-ink-2 hidden sm:block">
                Recognizes you by <b className="text-ink">how</b> you type, not what you type.
              </p>
            </div>
          </div>

          <nav
            aria-label="Mode"
            className="order-3 w-full md:order-none md:w-auto md:ml-auto flex gap-1.5 rounded-base border-2 border-ink bg-page p-1.5"
          >
            {Object.entries(MODES).map(([m, { label, fill, icon }]) => (
              <button
                key={m}
                onClick={() => switchMode(m)}
                aria-pressed={mode === m}
                className={`flex-1 md:flex-none inline-flex h-11 md:h-9 items-center justify-center gap-2 rounded-base border-2 px-3 sm:px-4 font-sans text-[14.5px] font-bold cursor-pointer transition-colors ${
                  mode === m
                    ? `${FILL[fill]} border-ink text-ink shadow-brutal-sm`
                    : "border-transparent bg-transparent text-ink-2 hover:text-ink hover:border-ink"
                }`}
              >
                <Icon path={icons[icon]} size={15} className="hidden sm:block" />
                {label}
              </button>
            ))}
          </nav>

          <a
            href={`${API}/map`}
            target="_blank"
            rel="noreferrer"
            className="ml-auto md:ml-0 inline-flex h-11 items-center gap-2 rounded-base border-2 border-ink bg-white px-4 font-bold text-ink no-underline shadow-brutal transition-all duration-100 hover:translate-x-1 hover:translate-y-1 hover:shadow-none"
          >
            <Icon path={icons.map} />
            Map
            <Icon path={icons.external} size={14} />
          </a>
        </div>
      </header>

      <div className="mx-auto w-full max-w-[1160px] flex-1 px-4 py-6 sm:px-6 sm:py-8 grid gap-6 lg:gap-7 lg:grid-cols-[minmax(0,1fr)_360px] items-start">
        <main className="min-w-0 rounded-base border-2 border-ink bg-white shadow-brutal-lg">
          <div className={`flex flex-wrap items-center gap-x-3.5 gap-y-2.5 rounded-t-[3px] border-b-2 border-ink px-4 py-3.5 sm:px-6 sm:py-4 ${FILL[modeFill]}`}>
            <Tile>
              <Icon path={icons[MODES[mode].icon]} size={18} />
            </Tile>
            <h2 className="m-0 min-w-[11ch] flex-1 font-display text-xl sm:text-2xl font-bold tracking-tight break-words">{title}</h2>
            <div className="shrink-0">{headAside}</div>
          </div>

          <div className="px-4 py-5 sm:px-6 sm:py-7">
            {mode === "enroll" && !nameLocked && (
              <div className="flex flex-col gap-5">
                <p className="m-0 text-[17px] leading-relaxed">
                  Enrolling records your rhythm across{" "}
                  <b>{promptCount === 1 ? "one short sentence" : `${promptCount} short sentences`}</b>. Nothing you type is
                  stored as text, only the timing between keys.
                </p>
                <div className="flex flex-col gap-2">
                  <Label as="label" htmlFor="person-name">Your name</Label>
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
                    className="h-13 w-full rounded-base border-2 border-ink bg-white px-3.5 font-sans text-[18px] text-ink placeholder:text-muted"
                  />
                </div>
                <Button size="lg" onClick={lockName} disabled={!name.trim()}>
                  Start enrollment
                  <Icon path={icons.arrow} />
                </Button>
                <p className="m-0 flex items-center gap-2 text-[13px] text-ink-2">
                  Press <Kbd>Enter ↵</Kbd> to start
                </p>
              </div>
            )}

            {mode === "authenticate" && !showingResult && (
              <>
                <p className="m-0 mb-5 text-[17px] leading-relaxed">
                  Verification is a yes/no question, not a search: claim an identity, type the sentence, and TypeID
                  compares your rhythm with that one person's stored profile.
                </p>
                {/* Real radios behind styled labels, so arrow keys and screen readers treat this as a
                    pick-one control, which a grid of buttons wouldn't be. */}
                <fieldset className="m-0 mb-7 p-0 border-0">
                  <Label as="legend" className="mb-3 p-0">Who do you claim to be?</Label>
                  {people.length === 0 ? (
                    <p className="m-0 text-[14.5px] text-ink-2">
                      Nobody is enrolled yet. Enroll someone first, then come back to verify them.
                    </p>
                  ) : (
                    <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                      {people.map((p) => (
                        <label
                          key={p.person_id}
                          className={`flex items-center gap-2.5 min-w-0 cursor-pointer rounded-base border-2 border-ink px-3 py-2.5 transition-all duration-100 has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-3 has-[:focus-visible]:outline-ink ${
                            claimId === p.person_id ? "bg-lilac shadow-brutal" : "bg-white hover:shadow-brutal-sm"
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
                          <Tile size="size-9" className="font-display font-bold text-[15px]">
                            {p.name.charAt(0).toUpperCase()}
                          </Tile>
                          <span className="min-w-0">
                            <span className="block truncate font-bold">{p.name}</span>
                            <span className="block text-xs text-ink-2">{plural(p.enroll_prompts, "prompt")}</span>
                          </span>
                        </label>
                      ))}
                    </div>
                  )}
                </fieldset>
              </>
            )}

            {mode === "enroll" && nameLocked && !enrolled && (
              <div className="mb-6">
                <PromptProgress done={collision ? promptCount : sessionCount} total={promptCount} />
              </div>
            )}

            {typingVisible && (
              <>
                {mode === "identify" && (
                  <p className="m-0 mb-5 text-[17px] leading-relaxed">
                    Type {promptCount === 1 ? "the sentence below" : `${promptCount} sentences`} and TypeID ranks your
                    rhythm against <b>everyone enrolled</b>.
                  </p>
                )}
                {mode !== "enroll" && promptCount > 1 && (
                  <div className="mb-5">
                    <PromptProgress done={sessionCount} total={promptCount} />
                  </div>
                )}

                <div className="relative rounded-base border-2 border-ink bg-soft p-4 pr-16 sm:p-5 sm:pr-18">
                  <Label className="block mb-2">Type this</Label>
                  <p className="m-0 font-display text-[22px] sm:text-[27px] leading-[1.42] font-medium tracking-tight">
                    <span className="sr-only">{sentence}</span>
                    <span aria-hidden="true">
                      {[...sentence].map((ch, i) => (
                        <span
                          key={i}
                          className={
                            match.chars[i] === "correct"
                              ? "text-ink bg-mint/40"
                              : match.chars[i] === "wrong"
                                ? "rounded-[2px] bg-coral text-ink shadow-[inset_0_-2px_0_var(--color-ink)]"
                                : `text-ink-2${i === typed.length ? " shadow-[inset_0_-4px_0_var(--color-mint-deep)]" : ""}`
                          }
                        >
                          {ch}
                        </span>
                      ))}
                      {match.extra > 0 && <span className="text-coral"> +{match.extra}</span>}
                    </span>
                  </p>
                  <Button
                    fill="white"
                    size="icon"
                    onClick={() => newSentence(true)}
                    className="absolute top-3 right-3"
                    title="Swap in a different sentence"
                    aria-label="Swap in a different sentence"
                  >
                    <Icon path={icons.refresh} />
                  </Button>
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
                  className="mt-4 block w-full resize-none rounded-base border-2 border-ink bg-white px-3.5 py-3 font-sans text-[18px] leading-relaxed text-ink placeholder:text-muted focus:outline-none focus-visible:outline-none focus:shadow-[0_0_0_3px_#fff,0_0_0_5px_var(--color-ink)]"
                  rows="3"
                  placeholder="Start typing… pasting is off, the rhythm is the point"
                  autoComplete="off"
                  autoCorrect="off"
                  autoCapitalize="off"
                  spellCheck="false"
                />

                <div className="mt-3 flex flex-wrap items-center gap-2 tabular-nums">
                  <Badge>{progressText}</Badge>
                  <Badge fill={typed && match.accuracy < MIN_MATCH ? "coral" : typed ? "mint" : "white"}>
                    {matchPercent}% match
                  </Badge>
                  <span className="ml-auto text-[13px] text-ink-2">{Math.round(MIN_MATCH * 100)}% match needed</span>
                </div>

                <Button
                  key="submit"
                  size="lg"
                  fill={modeFill}
                  onClick={submit}
                  disabled={!canSubmit || busy}
                  className="mt-6"
                >
                  {busy && <ScanRings size={18} />}
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
                  {!busy && <Icon path={icons.arrow} />}
                </Button>
                <p className="m-0 mt-3.5 flex items-center gap-2 text-[13px] text-ink-2">
                  <Kbd>Enter ↵</Kbd> submits · Enter is never part of the sample
                </p>
              </>
            )}

            {collision && (
              <div className="grid grid-cols-[18px_1fr] gap-x-3 gap-y-1 rounded-base border-2 border-ink bg-coral-soft px-4 py-3.5 shadow-brutal animate-[rise_280ms_cubic-bezier(0.16,1,0.3,1)]">
                <Icon path={icons.alert} size={18} className="mt-0.5" />
                <p className="m-0 font-bold">Someone is already enrolled as {name}</p>
                <p className="m-0 col-start-2 leading-snug">
                  Replacing overwrites their stored typing profile for good. Your {plural(collision.sessions.length, "sample")}{" "}
                  are kept until you choose.
                </p>
                <div className="col-start-2 mt-3 flex flex-wrap gap-3">
                  <Button fill="white" onClick={resetEnrollment}>Use a different name</Button>
                  <Button fill="coral" onClick={() => sendEnrollment(collision.sessions, true)} disabled={busy}>
                    {busy && <ScanRings size={16} />}
                    Replace {name}
                  </Button>
                </div>
              </div>
            )}

            {enrolled && (
              // Testing yourself is the point of enrolling, so it's the primary action and takes focus
              // (Enter continues); enrolling someone else is the quieter alternative.
              <div className="flex flex-col gap-6">
                <Verdict
                  fill="bg-mint"
                  icon={<Icon path={icons.check} size={28} />}
                  title={`${name} is enrolled`}
                  detail={`Profile built from ${plural(promptCount, "typing sample")}.`}
                  stamp="Saved"
                />
                <p className="m-0 text-ink-2">
                  Testing yourself is the point of enrolling: identify yourself now, without saying who you are.
                </p>
                <div className="flex flex-col gap-3">
                  <Button size="lg" fill="sun" onClick={() => switchMode("identify")} autoFocus>
                    <Icon path={icons.search} />
                    Identify yourself now
                  </Button>
                  <Button size="lg" fill="white" onClick={resetEnrollment}>
                    Enroll another person
                  </Button>
                </div>
              </div>
            )}

            {mode === "identify" && matches && (
              <div className="flex flex-col gap-6">
                {matches.matched ? (
                  <Verdict
                    fill="bg-sun"
                    icon={matches.results[0].name.charAt(0).toUpperCase()}
                    title={matches.results[0].name}
                    detail={`Best match, above the ${matches.min_confidence.toFixed(2)} threshold.`}
                    focusOnMount
                    onKeyDown={nextRoundKeys}
                    aside={
                      <span className="font-display text-[30px] font-bold leading-none tabular-nums">
                        {matches.results[0].similarity.toFixed(3)}
                      </span>
                    }
                    stamp={confidenceBand(matches.results[0].similarity, matches.min_confidence)}
                  />
                ) : (
                  <Verdict
                    fill="bg-sun-soft"
                    icon={<Icon path={icons.declined} size={26} />}
                    title="No confident match"
                    focusOnMount
                    onKeyDown={nextRoundKeys}
                    detail={`Nobody reached the ${matches.min_confidence.toFixed(
                      2
                    )} threshold, so TypeID declined to name anyone. That's the right outcome for someone who isn't enrolled.`}
                  />
                )}

                {matches.results.length > 0 && (
                  <div>
                    <div className="mb-2 flex flex-wrap items-baseline justify-between gap-x-3">
                      <h3 className="m-0 font-display text-lg font-bold">
                        {matches.matched ? "Ranked candidates" : "Closest candidates"}
                      </h3>
                      <span className="text-[13px] text-ink-2">
                        finds {Math.round(matches.found_percent)}% with {plural(matches.query_prompts, "prompt")}
                      </span>
                    </div>
                    <ol className="m-0 p-0 list-none flex flex-col gap-2.5">
                      {matches.results.map((r, i) => (
                        // Ranks past the shortlist are context (how crowded it is just below), so they recede: dashed
                        // border, ink-2 text and a faded bar. Fading the whole row would drop its text under 4.5:1.
                        <li
                          key={r.person_id}
                          className={`grid grid-cols-[34px_minmax(0,1fr)_64px] sm:grid-cols-[34px_minmax(0,150px)_1fr_74px] items-center gap-x-3.5 gap-y-2 rounded-base border-2 border-ink px-3 py-2.5 ${
                            i === 0 && matches.matched ? "bg-sun-soft shadow-brutal" : "bg-white"
                          } ${i >= SHORTLIST ? "border-dashed text-ink-2" : ""}`}
                        >
                          <Tile size="size-[34px]" fill={i === 0 && matches.matched ? "sun" : "white"} className="font-display font-bold text-[15px]">
                            {i + 1}
                          </Tile>
                          <span className="truncate font-bold">{r.name}</span>
                          <div className={`col-span-3 row-start-2 sm:col-span-1 sm:col-start-3 sm:row-start-1 ${i >= SHORTLIST ? "opacity-45" : ""}`}>
                            <ThresholdBar
                              value={r.similarity}
                              threshold={matches.min_confidence}
                              fill={r.similarity >= matches.min_confidence ? (i === 0 ? "bg-sun" : "bg-mint") : "bg-below"}
                            />
                          </div>
                          <span className="col-start-3 row-start-1 sm:col-start-4 text-right font-display text-[15px] font-bold tabular-nums">
                            {r.similarity.toFixed(3)}
                          </span>
                        </li>
                      ))}
                    </ol>
                    <p className="m-0 mt-2.5 text-[12.5px] text-ink-2">
                      Dashed line: the {matches.min_confidence.toFixed(2)} threshold for naming someone.
                    </p>
                  </div>
                )}

                <div className="flex flex-col sm:flex-row gap-3">
                  <Button size="lg" fill="sun" onClick={nextRound} onKeyDown={nextRoundKeys} className="sm:flex-1">
                    Try another sentence
                    <Icon path={icons.arrow} />
                  </Button>
                  {!matches.matched && promptCount < 5 && (
                    <Button size="lg" fill="white" onClick={retryWithMorePrompts} className="sm:flex-1">
                      Use 5 prompts
                    </Button>
                  )}
                </div>
              </div>
            )}

            {mode === "authenticate" && auth && (
              <div className="flex flex-col gap-6">
                <Verdict
                  fill={auth.accepted ? "bg-mint" : "bg-coral"}
                  icon={<Icon path={auth.accepted ? icons.check : icons.declined} size={28} />}
                  title={auth.accepted ? `Verified as ${auth.name}` : "Not verified"}
                  focusOnMount
                  onKeyDown={nextRoundKeys}
                  detail={
                    auth.accepted
                      ? `This rhythm matches ${auth.name}'s stored profile closely enough to accept.`
                      : `This rhythm isn't close enough to ${auth.name}'s stored profile to accept the claim.`
                  }
                  stamp={auth.accepted ? "Accepted" : "Rejected"}
                />

                {/* The decision is one number against one line: show both rather than just the verdict. */}
                <div>
                  <div className="mb-2 flex flex-wrap items-baseline justify-between gap-x-3 tabular-nums">
                    <span className="font-bold">Similarity {auth.similarity.toFixed(3)}</span>
                    <span className="text-[13px] text-ink-2">
                      threshold {auth.threshold.toFixed(2)} · {auth.operating_point} {auth.eer_percent}%
                    </span>
                  </div>
                  <ThresholdBar
                    value={auth.similarity}
                    threshold={auth.threshold}
                    fill={auth.accepted ? "bg-mint" : "bg-coral"}
                    height="h-[30px]"
                    line="border-solid border-l-[3px]"
                  />
                  <p className="m-0 mt-3 text-[13px] leading-snug text-ink-2">
                    {auth.name} enrolled from {auth.enroll_prompts}, verified with {auth.query_prompts}.{" "}
                    {auth.extrapolated
                      ? `No measured threshold for ${auth.enroll_prompts} + ${auth.query_prompts} (Aalto has at most 15 prompts per person), so this borrows the 10 + 5 one.`
                      : `At this setting, ${auth.eer_percent}% of decisions are wrong either way.`}
                  </p>
                </div>

                <div className="flex flex-col sm:flex-row gap-3">
                  <Button size="lg" fill="lilac" onClick={nextRound} onKeyDown={nextRoundKeys} className="sm:flex-1">
                    Try another sentence
                  </Button>
                  <Button size="lg" fill="white" onClick={() => claim("")} className="sm:flex-1">
                    Claim someone else
                  </Button>
                </div>
              </div>
            )}

            <ErrorNote error={error} />
          </div>
        </main>

        <aside className="flex flex-col gap-6 min-w-0">
          {mode !== "enroll" && showResults && (
            <Card className="p-4.5 pb-5">
              <h3 className="m-0 mb-3 font-display text-[17px] font-bold">What was measured</h3>
              <div className="grid grid-cols-3 gap-2">
                {[
                  [stats.count, "keystrokes"],
                  [stats.duration, "duration"],
                  [stats.wpm, "wpm"],
                  [stats.dwell, "avg dwell"],
                  [stats.flight, "avg flight"],
                  [stats.backspaces, "backspaces"],
                ].map(([value, label]) => (
                  <div key={label} className="rounded-base border-2 border-ink px-2.5 py-2">
                    <span className="block font-display text-xl font-bold leading-tight tabular-nums">{value}</span>
                    <span className="text-xs text-ink-2">{label}</span>
                  </div>
                ))}
              </div>
              <div
                className="mt-3.5 h-16 flex items-end gap-0.5 rounded-base border-2 border-ink bg-soft p-1"
                role="img"
                aria-label="Dwell time per keystroke"
              >
                {dwellValues.map((d, i) => (
                  <div
                    key={i}
                    className="flex-1 min-w-[2px] rounded-t-[2px] border-[1.5px] border-b-0 border-ink bg-mint"
                    style={{ height: `${Math.max((d / maxDwell) * 100, 6)}%` }}
                  />
                ))}
              </div>
              <p className="m-0 mt-1.5 text-[13px] text-ink-2">Dwell time per keystroke, last prompt</p>
              <div className="mt-3 flex gap-4">
                <GhostButton onClick={() => setShowRaw((prev) => !prev)} className="underline">
                  {showRaw ? "Hide raw event log" : "Show raw event log"}
                </GhostButton>
                {showRaw && (
                  <GhostButton
                    onClick={() => {
                      navigator.clipboard.writeText(rawText);
                      setCopyLabel("Copied");
                      setTimeout(() => setCopyLabel("Copy JSON"), 1200);
                    }}
                    className="underline"
                  >
                    {copyLabel}
                  </GhostButton>
                )}
              </div>
              {showRaw && (
                <pre className="m-0 mt-3 max-h-[220px] overflow-y-auto overflow-x-hidden whitespace-pre-wrap break-words rounded-base border-2 border-ink bg-soft p-3 font-mono text-xs leading-relaxed">
                  {rawText}
                </pre>
              )}
            </Card>
          )}

          {!enrolled && (
            <Card className="p-4.5 pb-5">
              <fieldset className="m-0 p-0 border-0">
                <legend className="m-0 mb-3 p-0 font-display text-[17px] font-bold">Prompts to type</legend>
                <div className="flex gap-2">
                  {PROMPT_COUNTS.map((n) => (
                    <label
                      key={n}
                      className={`grid place-items-center min-w-13 h-11 px-3.5 cursor-pointer rounded-base border-2 border-ink text-base font-bold tabular-nums transition-all duration-100 has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-3 has-[:focus-visible]:outline-ink ${
                        promptCount === n ? `${FILL[modeFill]} shadow-brutal` : "bg-white hover:shadow-brutal-sm"
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
              </fieldset>
              <p className="m-0 mt-2.5 text-[13px] text-ink-2">More prompts, steadier profile, fewer mistakes.</p>
              {/* The cost of a snappy one-prompt identify, stated up front rather than discovered. */}
              {mode === "identify" && identifyRates && (
                <>
                  <div className="mt-3.5 grid grid-cols-3 gap-2">
                    {PROMPT_COUNTS.map((n) => (
                      <div
                        key={n}
                        className={`rounded-base border-2 border-ink px-2.5 py-2 ${n === promptCount ? "bg-sun shadow-brutal-sm" : "bg-white"}`}
                      >
                        <span className="block font-display text-xl font-bold tabular-nums">
                          {Math.round(identifyRates.found_percent[n])}%
                        </span>
                        <span className={`text-xs ${n === promptCount ? "text-ink" : "text-ink-2"}`}>
                          found · {plural(n, "prompt")}
                        </span>
                      </div>
                    ))}
                  </div>
                  <p className="m-0 mt-2.5 text-[13px] leading-snug text-ink-2">
                    With {plural(promptCount, "prompt")}, someone who isn't enrolled gets named about{" "}
                    {Math.round(identifyRates.stranger_named_percent[promptCount])}% of the time.
                  </p>
                </>
              )}
            </Card>
          )}

          {!(mode !== "enroll" && showResults) && (
            <Card className="p-4.5 pb-5">
              <h3 className="m-0 mb-3 font-display text-[17px] font-bold">How it works</h3>
              <ol className="m-0 p-0 list-none flex flex-col gap-3">
                {HOW_IT_WORKS[mode].map((step, i) => (
                  <li key={i} className="grid grid-cols-[28px_1fr] gap-3 text-sm leading-snug">
                    <Tile size="size-7" fill={modeFill} className="text-[13px] font-bold">
                      {i + 1}
                    </Tile>
                    <span className="pt-1">{step}</span>
                  </li>
                ))}
              </ol>
            </Card>
          )}
        </aside>
      </div>

      <footer className="mx-auto w-full max-w-[1160px] px-4 pb-8 sm:px-6 text-[13px] leading-relaxed text-ink-2">
        Open-set 1:N gallery search on typing rhythm. New people enroll without retraining the model.
      </footer>

      {pendingDiscard && (
        <Dialog
          title={`Discard ${plural(sessionsRef.current.length, "recorded sample")}?`}
          onDismiss={() => setPendingDiscard(null)}
          actions={
            <>
              <Button fill="white" onClick={() => setPendingDiscard(null)} autoFocus>
                Keep recording
              </Button>
              <Button
                fill="coral"
                onClick={() => {
                  const { run } = pendingDiscard;
                  setPendingDiscard(null);
                  sessionsRef.current = [];
                  run();
                }}
              >
                Discard
              </Button>
            </>
          }
        >
          {pendingDiscard.reason} The samples you've typed can't be recovered.
        </Dialog>
      )}
    </div>
  );
}
