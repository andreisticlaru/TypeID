// Neobrutalist primitives, ported from neobrutalism.dev (ekmas/neobrutalism-components, MIT).
// The library builds on shadcn/Radix; these keep its class recipes (2px ink border, hard shadow that
// the element "presses into" on hover) as plain elements, so the app takes on no new dependency.

// Fills as literal class names: Tailwind only generates classes it can find spelled out in source.
export const FILL = {
  mint: "bg-mint",
  sun: "bg-sun",
  lilac: "bg-lilac",
  coral: "bg-coral",
  white: "bg-white",
};

const buttonBase =
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-base border-2 border-ink font-sans font-bold text-ink cursor-pointer transition-all duration-100 disabled:cursor-not-allowed disabled:border-dashed disabled:bg-off disabled:text-ink-2 disabled:shadow-none";
// Pressing into the shadow is the library's hover: the button moves by exactly the shadow offset.
const pressable = "shadow-brutal enabled:hover:translate-x-1 enabled:hover:translate-y-1 enabled:hover:shadow-none";
const buttonSizes = {
  default: "h-11 px-5 text-[15px]",
  lg: "h-14 w-full px-6 text-[17px]",
  icon: "size-11 shrink-0",
};

export function Button({ fill = "mint", size = "default", className = "", ...props }) {
  return <button className={`${buttonBase} ${pressable} ${FILL[fill]} ${buttonSizes[size]} ${className}`} {...props} />;
}

// A quiet text action (e.g. "Change name"): no border or shadow. Ink, not ink-2: it sits on mode fills, where
// ink-2 drops under 4.5:1. min-h-11 gives a 44px hit area without drawing a bigger box.
export function GhostButton({ className = "", ...props }) {
  return (
    <button
      className={`inline-flex min-h-11 items-center gap-1.5 border-0 bg-transparent p-0 font-sans text-[13.5px] font-bold text-ink cursor-pointer hover:underline underline-offset-4 decoration-2 ${className}`}
      {...props}
    />
  );
}

export function Badge({ fill = "white", className = "", ...props }) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-base border-2 border-ink px-2.5 py-0.5 text-[12.5px] font-bold whitespace-nowrap ${FILL[fill]} ${className}`}
      {...props}
    />
  );
}

export function Kbd({ children }) {
  return (
    <kbd className="inline-grid place-items-center whitespace-nowrap min-w-[22px] h-[22px] px-1.5 rounded-base border-2 border-ink bg-white font-sans text-[11.5px] font-bold shadow-brutal-sm">
      {children}
    </kbd>
  );
}

export function Card({ as: Tag = "section", className = "", ...props }) {
  return <Tag className={`rounded-base border-2 border-ink bg-white shadow-brutal ${className}`} {...props} />;
}

export function Label({ as: Tag = "span", className = "", ...props }) {
  return <Tag className={`text-xs font-bold uppercase tracking-[0.08em] text-ink-2 ${className}`} {...props} />;
}
