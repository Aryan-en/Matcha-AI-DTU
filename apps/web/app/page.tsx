import { VideoUpload } from "@/components/video-upload";
import { MatchDashboard } from "@/components/match-dashboard";

export default function Home() {
  return (
    <main className="min-h-screen relative overflow-x-hidden" style={{ background: "var(--bg)", color: "var(--text)" }}>

      {/* ── Navigation ──────────────────────────────────── */}
      <nav className="flex items-center justify-between py-6 px-8 max-w-[1440px] mx-auto">
        <div className="flex items-center gap-3">
          <div
            className="w-8 h-8 flex items-center justify-center shrink-0"
            style={{ background: "var(--green)" }}
          >
            <span className="font-display text-[#07080F]" style={{ fontSize: "22px", lineHeight: 1 }}>M</span>
          </div>
          <div className="flex items-baseline gap-1">
            <span className="font-display tracking-[0.12em]" style={{ fontSize: "20px", color: "var(--text)" }}>MATCHA</span>
            <span className="font-display tracking-[0.12em] ml-1" style={{ fontSize: "20px", color: "var(--green)" }}>AI</span>
          </div>
          <div className="w-px h-4 mx-2" style={{ background: "var(--border)" }} />
          <span className="font-mono" style={{ fontSize: "9px", color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.14em" }}>DTU EDITION</span>
        </div>

        <div className="flex items-center gap-5">
          <div className="flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full animate-blink" style={{ background: "var(--red)" }} />
            <span className="font-mono" style={{ fontSize: "9px", color: "var(--red)", textTransform: "uppercase", letterSpacing: "0.14em" }}>LIVE</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full" style={{ background: "var(--green)" }} />
            <span className="font-mono" style={{ fontSize: "9px", color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.14em" }}>SYS NOMINAL</span>
          </div>
        </div>
      </nav>

      <div className="relative z-10 max-w-[1440px] mx-auto px-8">

        {/* ── Hero band ─────────────────────────────────── */}
        <div className="py-12 pb-8 border-b" style={{ borderColor: "var(--border)" }}>
          <div className="flex items-end justify-between gap-8 flex-wrap">
            <div>
              <div className="flex items-center gap-3 mb-5">
                <span className="chip chip-green">CH&nbsp;01 ▸ ANALYSIS PIPELINE</span>
              </div>
              <h1
                className="font-display leading-none"
                style={{ fontSize: "clamp(52px, 7vw, 92px)", color: "var(--text)" }}
              >
                MATCH<br />
                <span style={{ color: "var(--green)" }}>INTELLIGENCE</span>
              </h1>
            </div>

            {/* Stat counters */}
            <div className="flex gap-8 pb-1 shrink-0">
              {[
                { n: "09", label: "EVENTS / MATCH" },
                { n: "04", label: "HIGHLIGHT REELS" },
                { n: "∞",  label: "HOURS SAVED" },
              ].map((s) => (
                <div key={s.label} className="text-right pr-8 border-r last:border-0 last:pr-0" style={{ borderColor: "var(--border)" }}>
                  <div className="font-display leading-none" style={{ fontSize: "46px", color: "var(--green)" }}>{s.n}</div>
                  <div className="font-mono mt-1" style={{ fontSize: "9px", color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.12em" }}>{s.label}</div>
                </div>
              ))}
            </div>
          </div>

          <p className="font-heading mt-5 max-w-xl leading-relaxed" style={{ fontSize: "15px", fontWeight: 400, color: "var(--text-mid)" }}>
            Upload raw match footage. The AI pipeline detects goals, fouls, saves &amp; tackles —
            scores each moment — and builds highlight reels with live neural commentary.
          </p>
        </div>

        {/* ── Main grid ─────────────────────────────────── */}
        <div
          className="grid gap-0"
          style={{ gridTemplateColumns: "minmax(0, 500px) 1fr" }}
        >
          {/* LEFT — Upload */}
          <div className="border-r py-10 pr-10" style={{ borderColor: "var(--border)" }}>
            <div className="flex items-center gap-3 mb-6">
              <span className="font-mono" style={{ fontSize: "10px", color: "var(--text-mid)", textTransform: "uppercase", letterSpacing: "0.14em" }}>FOOTAGE INPUT</span>
              <div className="flex-1 h-px" style={{ background: "var(--border)" }} />
            </div>
            <VideoUpload />

            {/* Capability grid */}
            <div className="mt-8 grid grid-cols-2 gap-px" style={{ background: "var(--border)" }}>
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
                  className="font-mono px-4 py-3 transition-colors"
                  style={{ fontSize: "9px", background: "var(--surface-1)", color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.10em" }}
                >
                  {f}
                </div>
              ))}
            </div>
          </div>

          {/* RIGHT — Feed */}
          <div className="py-10 pl-10">
            <div className="flex items-center gap-3 mb-6">
              <span className="font-mono" style={{ fontSize: "10px", color: "var(--text-mid)", textTransform: "uppercase", letterSpacing: "0.14em" }}>ANALYSIS FEED</span>
              <div className="flex-1 h-px" style={{ background: "var(--border)" }} />
            </div>
            <MatchDashboard />
          </div>
        </div>
      </div>

      {/* Bottom rule */}
      <div className="fixed bottom-0 left-0 right-0 h-px z-50" style={{ background: "var(--border)" }} />
    </main>
  );
}

