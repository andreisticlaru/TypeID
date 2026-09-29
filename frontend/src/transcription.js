// Compares what's in the typing box against the prompt, position by position. It reads the text
// only, never the key events, so it can't disturb the timing capture that the model depends on.
// Case-sensitive on purpose: capitals are what make people press Shift, and Shift timing is part
// of what the model learned from Aalto.
export function transcriptionMatch(prompt, typed) {
  const chars = [...prompt].map((ch, i) =>
    i >= typed.length ? "untyped" : typed[i] === ch ? "correct" : "wrong"
  );
  const correct = chars.filter((s) => s === "correct").length;
  const extra = Math.max(0, typed.length - prompt.length);
  // Divide by the longer of the two so trailing junk can't pad a full prompt up to 100%.
  const accuracy = correct / Math.max(prompt.length, typed.length, 1);
  return { chars, extra, accuracy };
}
