"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { VideoUpload, MatchDashboard } from "@/components/dynamic-sections";
import { useAuth } from "@/contexts/AuthContext";

export default function Home() {
  const { user, loading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!loading && !user) {
      router.push("/login");
    }
  }, [user, loading, router]);

  if (loading || !user) {
    return (
      <div className="flex-1 flex items-center justify-center p-4 bg-background min-h-[calc(100vh-80px)]">
        <div className="size-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="relative w-full overflow-hidden">
      {/* ── Hero band ─────────────────────────────────── */}
      <div className="relative min-h-[calc(100vh-80px)] flex flex-col items-center justify-center border-b border-border py-16">
        {/* Full-width Background Image Layer */}
        <div 
          className="absolute inset-0 z-0 bg-cover bg-center"
          style={{ backgroundImage: 'url("/favicons/image.png")' }}
        />
        {/* Blue Tint Overlay Layer */}
        <div className="absolute inset-0 z-0 bg-background/85 backdrop-blur-[2px]" />

        {/* Hero Content Layer */}
        <div className="relative z-10 max-w-[1440px] mx-auto px-4 sm:px-8 w-full flex flex-col items-center text-center gap-8 md:gap-10">
          
          <div className="flex flex-col items-center opacity-0" style={{ animation: "fadeup 0.8s ease-out forwards" }}>
            <h1 className="font-display flex flex-col items-center leading-[0.8] text-[clamp(42px,11vw,120px)] text-foreground shadow-sm">
              <span>MATCH</span>
              <span 
                className="text-primary drop-shadow-[0_0_20px_rgba(var(--color-primary),0.6)] italic tracking-normal normal-case text-[clamp(32px,8vw,90px)] -mt-2 sm:-mt-4"
                style={{ fontFamily: "Georgia, 'Times New Roman', Times, serif" }}
              >
                Intelligence
              </span>
            </h1>
          </div>

          <p 
            className="font-heading max-w-2xl leading-relaxed text-[15px] sm:text-[19px] font-medium text-foreground/90 drop-shadow-md opacity-0"
            style={{ animation: "fadeup 0.8s ease-out 0.2s forwards" }}
          >
            Upload raw match footage. The AI pipeline detects goals, fouls, saves &amp; tackles —
            scores each moment — and builds highlight reels with live neural commentary.
          </p>

          {/* Stat counters */}
          <div 
            className="flex w-full md:w-auto items-center justify-center gap-4 sm:gap-16 mt-4 backdrop-blur-md bg-background/50 px-6 sm:px-12 py-4 sm:py-6 rounded-xl border border-border/60 shadow-2xl relative overflow-hidden group opacity-0"
            style={{ animation: "fadeup 0.8s ease-out 0.4s forwards" }}
          >
            <div className="absolute inset-0 bg-gradient-to-r from-primary/5 via-transparent to-primary/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
            {[
              { n: "09", label: "EVENTS" },
              { n: "04", label: "REELS" },
              { n: "∞",  label: "SAVED" },
            ].map((s) => (
              <div key={s.label} className="text-center pr-4 sm:pr-16 last:pr-0 border-r last:border-r-0 border-border/60 relative z-10 flex-1 sm:flex-none">
                <div className="font-display leading-none text-[26px] sm:text-[54px] text-primary drop-shadow-[0_0_12px_rgba(var(--color-primary),0.5)]">{s.n}</div>
                <div className="font-mono mt-1 sm:mt-2 text-[8px] sm:text-[10.5px] text-muted-foreground uppercase tracking-[0.16em] font-bold">{s.label}</div>
              </div>
            ))}
          </div>

        </div>
      </div>

      {/* ── Main grid ─────────────────────────────────── */}
      <div className="relative z-10 max-w-[1440px] mx-auto px-8 w-full">
        <div className="grid gap-0 grid-cols-1 lg:grid-cols-[minmax(0,500px)_1fr]">
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
    </div>
  );
}
