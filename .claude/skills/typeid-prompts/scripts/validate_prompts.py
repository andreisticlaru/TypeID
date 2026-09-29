"""Check TypeID transcription prompts against the Aalto-matched mechanics.

Usage:
  python validate_prompts.py candidates.txt [--existing frontend/src/sentences.js] [--json]
  (one prompt per line; blank lines and lines starting with # are ignored; "-" reads stdin)

Per prompt (a failure drops that prompt):
  - 30-48 characters: clears the 25-keystroke floor, fits one 50-keystroke window
  - starts with a capital, ends with . ? or !: Aalto sentences always do, so Shift and end
    punctuation are part of the rhythm the model learned
  - exactly one sentence: a mid-prompt ". X" adds a boundary pause + Shift that Aalto almost never has
  - plain keyboard characters only: curly quotes, dashes and ellipses can't be typed as shown
  - no double quotes (they'd break the JS string list), not already in the existing pools

Per batch (a failure means rebalance, not drop):
  - commas in 7-16% of prompts (Aalto: 11%)
  - 1.2-1.9 Shift characters per prompt (Aalto: 1.69)
  - no single first word opening more than 35% of prompts (Aalto openings vary widely)

With --count N, surplus prompts are trimmed toward the batch targets (comma-carrying prompts
while commas run high, the most common opening while one dominates) so a slightly unbalanced
draft still yields a valid set without a rewrite.

Exit code 0 when the delivered set passes every batch check.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

MIN_CHARS, MAX_CHARS = 30, 48
COMMA_BAND = (0.07, 0.16)
SHIFT_BAND = (1.2, 1.9)
MAX_OPENING_SHARE = 0.35

TYPEABLE = re.compile(r"^[A-Za-z0-9 ,.?!'\-:;()&$%/]+$")
SECOND_SENTENCE = re.compile(r"[.?!]\s+\S")  # a sentence end with more text after it


def shift_chars(s: str) -> int:
    return sum(c.isupper() or c in '?!:()&$%' for c in s)


def existing_prompts(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return set(re.findall(r'^\s*"(.*)",\s*$', path.read_text(encoding="utf-8"), re.M))


def problems(s: str, seen: set[str], existing: set[str]) -> list[str]:
    out = []
    if not MIN_CHARS <= len(s) <= MAX_CHARS:
        out.append(f"{len(s)} chars (need {MIN_CHARS}-{MAX_CHARS})")
    if not s[:1].isupper():
        out.append("must start with a capital")
    if s[-1:] not in ".?!":
        out.append("must end with . ? or !")
    if SECOND_SENTENCE.search(s):
        out.append("two sentences (join them with a comma or 'but')")
    if '"' in s:
        out.append("contains a double quote")
    elif not TYPEABLE.match(s):
        bad = sorted({c for c in s if not TYPEABLE.match(c)})
        out.append(f"untypeable characters {bad} (use plain ' and -)")
    if s in existing:
        out.append("already in sentences.js")
    elif s in seen:
        out.append("duplicate in this batch")
    return out


def opener(s: str) -> str:
    return s.split()[0].strip(",").lower()


def trim(good: list[str], count: int) -> list[str]:
    """Drop surplus prompts from whichever side is over target, keeping the rest in order."""
    good = list(good)
    target_commas = sum(COMMA_BAND) / 2
    while len(good) > count:
        commas = [s for s in good if "," in s]
        top, n = Counter(map(opener, good)).most_common(1)[0]
        if len(commas) / len(good) > target_commas:
            good.remove(commas[-1])
        elif n / len(good) > MAX_OPENING_SHARE:
            good.remove([s for s in good if opener(s) == top][-1])
        else:
            good.pop()
    return good


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates", help="file with one prompt per line, or - for stdin")
    ap.add_argument("--existing", default="frontend/src/sentences.js")
    ap.add_argument("--json", action="store_true", help="print the passing prompts as a JSON array")
    ap.add_argument("--count", type=int, help="deliver exactly this many, trimming surplus toward the targets")
    args = ap.parse_args()

    text = sys.stdin.read() if args.candidates == "-" else Path(args.candidates).read_text(encoding="utf-8")
    lines = [l.strip() for l in text.splitlines() if l.strip() and not l.strip().startswith("#")]
    existing = existing_prompts(Path(args.existing))

    good, seen = [], set()
    for s in lines:
        issues = problems(s, seen, existing)
        seen.add(s)
        if issues:
            print(f"DROP  {s!r}: {'; '.join(issues)}")
        else:
            good.append(s)

    dropped = len(lines) - len(good)
    if args.count:
        if len(good) < args.count:
            print(f"only {len(good)} valid prompts for --count {args.count}: generate more")
            return 1
        good = trim(good, args.count)
    ok = True
    if good:
        commas = sum("," in s for s in good) / len(good)
        shift = sum(map(shift_chars, good)) / len(good)
        top, count = Counter(map(opener, good)).most_common(1)[0]
        checks = [
            (COMMA_BAND[0] <= commas <= COMMA_BAND[1], f"commas in {100 * commas:.0f}% (want {100 * COMMA_BAND[0]:.0f}-{100 * COMMA_BAND[1]:.0f}%)"),
            (SHIFT_BAND[0] <= shift <= SHIFT_BAND[1], f"{shift:.2f} Shift chars per prompt (want {SHIFT_BAND[0]}-{SHIFT_BAND[1]})"),
            (count / len(good) <= MAX_OPENING_SHARE, f"'{top}' opens {100 * count / len(good):.0f}% (max {100 * MAX_OPENING_SHARE:.0f}%)"),
        ]
        for passed, msg in checks:
            print(f"{'ok  ' if passed else 'FIX '} batch: {msg}")
            ok &= passed
    print(f"{len(good)} prompts delivered ({dropped} of {len(lines)} dropped)" + ("" if ok else " -- rebalance or regenerate"))
    if args.json:
        print(json.dumps(good, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
