---
name: typeid-prompts
description: Generate fresh TypeID transcription prompts - bland email-style "known samples" for enrollment/authentication, or petty-guilt "confessions" for identification - that match the typing mechanics of the Aalto sentences the model was trained on, then validate them with the bundled script. Use this whenever someone wants new, more, or replacement prompts/sentences for TypeID's sentences.js, a fresh prompt set for an identification or authentication round, prompts for a demo, or asks to expand, refresh or rebalance the prompt pools - even if they just say "give me some new sentences" in this repo.
---

# TypeID prompt generator

TypeID identifies people by typing rhythm. Visitors transcribe a prompt, and a model trained on the
Aalto 136M Keystrokes dataset embeds the timing. The model never sees the words, only when keys go
down and up. So the words are for the visitor, but their **shape** decides which keys get pressed
and when, and that must match what the model learned.

## Two kinds of prompt

**Known samples** (enrollment and authentication). Bland, work-email or everyday-admin register,
the kind of thing anyone might write. They build the reference profile.
- "Please send the revised figures by Friday."
- "Also, the shop only accepts cash on weekends."
- "Kevin sent over the latest estimates."

**Confessions** (identification). The "questioned document": something the typist would rather
not own, so being named by their rhythm lands. Petty, relatable guilt, told in the first person.
- "I told HR it was Mark but it was me."
- "Don't tell anyone I took the good stapler."
- "Nobody saw me take the umbrella, right?"

Keep confessions at the petty level: office theft, fudged expenses, white lies, snooping, fake sick
days, ghosting, regifting. No violence, threats, drugs, sex, self-harm, anything involving minors,
slurs, or real people, companies or brands. People type these on a portfolio demo and recruiters
read them; mild discomfort is the point, being put off isn't. Use common fictional first names.

## Mechanics (identical for both kinds)

Only the content should differ between the two kinds. If the mechanics differ, enrollment and
identification differ systematically and matching gets worse for everyone.

| Rule | Why |
|---|---|
| 30-48 characters | Clears the 25-keystroke floor and fits one 50-keystroke window |
| Capital first letter, ends in `.` `?` or `!` | Aalto always does; Shift and its overlap with the next key are signal |
| Exactly one sentence | Aalto has two sentences 0.1% of the time; a mid-prompt ". X" adds a pause the model rarely saw |
| Commas in about 11% of a batch | Aalto's rate; a comma is a keystroke plus a pause |
| About 1.4-1.7 Shift characters per prompt | Capitals from "I", names, and sentence starts; Aalto averages 1.69 |
| Varied openings | Aalto openings vary; don't start everything with "I" or "The" |
| Plain keyboard characters | Curly quotes, em dashes and ellipses can't be typed as shown |
| No double quotes | The prompts live in a JS string list |

To keep a punchline in one sentence, use a comma or a joining word: "I said the dog ate it, but we
have no dog." For confessions, vary the opening with "My", "Nobody", "Don't", "Tell", "We",
"Honestly,", or a name, so "I" doesn't open everything.

## Workflow (fast path)

1. Write about 20% more prompts than asked for, one per line, straight into a scratch file. Draft
   them in a single pass. The validator catches mechanical slips faster than deliberating does.
2. Validate from the repo root, passing the number you actually need:
   `python .claude/skills/typeid-prompts/scripts/validate_prompts.py <file> --count <N> --json`
   It drops prompts that break a per-prompt rule, checks for duplicates against
   `frontend/src/sentences.js`, then trims the surplus toward the batch targets (comma rate,
   repeated openings) and prints the final set.
3. Exit code 0 means done: use the printed set as is. Rewrite only if it reports too few valid
   prompts or a batch check still fails, and then only the flagged side (e.g. a few more prompts
   without commas, or different openings), not the whole batch.
4. Deliver in the format asked for:
   - **Pool update:** append to `KNOWN_SAMPLES` or `CONFESSIONS` in `frontend/src/sentences.js` as
     `"...",` lines, then rerun the validator on the whole pool. After a pool change, run
     `npx vite build` in `frontend/`.
   - **A round's fresh set** (e.g. for a server that pre-generates prompts): the JSON array that
     `--json` prints.

Report the validator's summary line and batch checks with the result, so the reader can see the
set was checked rather than assumed.
