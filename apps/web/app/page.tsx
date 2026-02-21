import Link from "next/link";
import { VideoUpload, MatchDashboard } from "@/components/dynamic-sections";

export default function Home() {
  return (
    <main className="min-h-screen relative overflow-x-hidden bg-background text-foreground">

      {/* ── Navigation ──────────────────────────────────── */}
      <nav className="flex items-center justify-between py-6 px-8 max-w-[1440px] mx-auto">
        <Link href="/" className="flex items-center gap-3 transition-opacity duration-200 hover:opacity-80 focus:outline-none focus-visible:ring-2 focus-visible:ring-primary rounded-sm" aria-label="Go to homepage">
          <div
            className="size-8 flex items-center justify-center shrink-0 bg-primary"
          >
            <span className="font-display text-[#07080F] text-[22px] leading-none">M</span>
          </div>
          <div className="flex items-baseline gap-1">
            <span className="font-display tracking-[0.12em] text-[20px] text-foreground">MATCHA</span>
            <span className="font-display tracking-[0.12em] ml-1 text-[20px] text-primary">AI</span>
          </div>
          <div className="w-px h-4 mx-2 bg-border" />
          <span className="font-mono text-[9px] text-muted-foreground uppercase tracking-[0.14em]">DTU EDITION</span>
        </Link>

        <div className="flex items-center gap-5">
          <div className="flex items-center gap-2">
            <span className="size-1.5 rounded-full animate-blink bg-destructive" />
            <span className="font-mono text-[9px] text-destructive uppercase tracking-[0.14em]">LIVE</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="size-1.5 rounded-full bg-primary" />
            <span className="font-mono text-[9px] text-muted-foreground uppercase tracking-[0.14em]">SYS NOMINAL</span>
          </div>
        </div>
      </nav>

      <div className="relative z-10 max-w-[1440px] mx-auto px-8">

        {/* ── Hero band ─────────────────────────────────── */}
        <div className="py-12 pb-8 border-b border-border">
          <div className="flex flex-col lg:flex-row items-end justify-between gap-8 flex-wrap">
            <div>
              <div className="flex items-center gap-3 mb-5">
                <span className="chip chip-green">CH&nbsp;01 ▸ ANALYSIS PIPELINE</span>
              </div>
              <h1
                className="font-display leading-none text-[clamp(52px,7vw,92px)] text-foreground"
              >
                MATCH<br />
                <span className="text-primary">INTELLIGENCE</span>
              </h1>
            </div>

            {/* Stat counters */}
            <div className="flex gap-8 pb-1 shrink-0">
              {[
                { n: "09", label: "EVENTS / MATCH" },
                { n: "04", label: "HIGHLIGHT REELS" },
                { n: "∞",  label: "HOURS SAVED" },
              ].map((s) => (
                <div key={s.label} className="text-right pr-8 border-r last:border-0 last:pr-0 border-border">
                  <div className="font-display leading-none text-[46px] text-primary">{s.n}</div>
                  <div className="font-mono mt-1 text-[9px] text-muted-foreground uppercase tracking-[0.12em]">{s.label}</div>
                </div>
              ))}
            </div>
          </div>

          <p className="font-heading mt-5 max-w-xl leading-relaxed text-[15px] font-normal text-muted-foreground">
            Upload raw match footage. The AI pipeline detects goals, fouls, saves &amp; tackles —
            scores each moment — and builds highlight reels with live neural commentary.
          </p>
        </div>

        {/* ── Main grid ─────────────────────────────────── */}
        <div
          className="grid gap-0 grid-cols-1 lg:grid-cols-[minmax(0,500px)_1fr]"
        >
          {/* LEFT — Upload */}
          <div className="lg:border-r border-b lg:border-b-0 py-10 lg:pr-10 border-border">
            <div className="flex items-center gap-3 mb-6">
              <span className="font-mono text-[10px] text-muted-foreground uppercase tracking-[0.14em]">FOOTAGE INPUT</span>
              <div className="flex-1 h-px bg-border" />
            </div>
            <VideoUpload />

            {/* Capability grid */}
            <div className="mt-8 grid grid-cols-2 gap-px bg-border">
              {[
                "⚽  GOAL DETECTION",
                "🟡  FOUL DETECTION",
                "🧤  SAVE DETECTION",
                "⚡  MOTION ANALYSIS",
                "🎬  HIGHLIGHT REELS",
                "🎙  NEURAL COMMENTARY",
              ].map((f) => (
                <div
                  key={f}
                  className="font-mono px-4 py-3 transition-colors text-[9px] bg-card text-muted-foreground uppercase tracking-[0.10em]"
                >
                  {f}
                </div>
              ))}
            </div>
          </div>

          {/* RIGHT — Feed */}
          <div className="py-10 lg:pl-10">
            <div className="flex items-center gap-3 mb-6">
              <span className="font-mono text-[10px] text-muted-foreground uppercase tracking-[0.14em]">ANALYSIS FEED</span>
              <div className="flex-1 h-px bg-border" />
            </div>
            <MatchDashboard />
          </div>
        </div>
      </div>

      {/* Bottom rule */}
      <div className="fixed bottom-0 left-0 right-0 h-px z-50 bg-border" />
    </main>
  );
}

