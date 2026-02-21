"use client";

import React, { useState, useCallback } from "react";
import Link from "next/link";
import { CheckCircle2, Loader2, Upload, XCircle } from "lucide-react";
import { useMatches } from "@/hooks/useMatches";
import { motion, AnimatePresence } from "framer-motion";

const STATUS_CONFIG: Record<string, { label: string; color: string; icon: React.ReactNode }> = {
  COMPLETED:  { label: "Completed",  color: "text-emerald-400 bg-emerald-500/10 border-emerald-500/40", icon: <CheckCircle2 className="w-3 h-3" /> },
  PROCESSING: { label: "Processing", color: "text-blue-400 bg-blue-500/10 border-blue-500/40",         icon: <Loader2 className="w-3 h-3 animate-spin" /> },
  UPLOADED:   { label: "Queued",     color: "text-amber-400 bg-amber-500/10 border-amber-500/40",      icon: <Upload className="w-3 h-3" /> },
  FAILED:     { label: "Failed",     color: "text-red-400 bg-red-500/10 border-red-500/40",            icon: <XCircle className="w-3 h-3" /> },
};

function formatDuration(secs: number | null) {
  if (!secs) return "—";
  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

function timeAgo(iso: string) {
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return `${Math.floor(hrs / 24)}d ago`;
}

const FILTER_OPTIONS = ["ALL", "COMPLETED", "PROCESSING", "UPLOADED", "FAILED"] as const;
type FilterOption = typeof FILTER_OPTIONS[number];

export const MatchDashboard = React.memo(function MatchDashboardContent() {
  const { matches, loading, progressMap, deleteMatch } = useMatches();
  const [filter, setFilter] = useState<FilterOption>("ALL");
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [confirmId, setConfirmId] = useState<string | null>(null);

  const handleDelete = useCallback(async (id: string) => {
    setDeletingId(id);
    await deleteMatch(id);
    setDeletingId(null);
    setConfirmId(null);
  }, [deleteMatch]);

  const visible = filter === "ALL" ? matches : matches.filter((m) => m.status === filter);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-48">
        <span className="font-mono text-[10px] text-muted-foreground uppercase tracking-[0.14em] animate-blink">
          LOADING FEED...
        </span>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {matches.length > 0 && (
        <div className="flex items-center gap-1 flex-wrap">
          <span className="font-mono text-[9px] text-muted-foreground uppercase tracking-[0.10em] mr-2">FILTER</span>
          {FILTER_OPTIONS.map((f) => {
            const count = f === "ALL" ? matches.length : matches.filter((m) => m.status === f).length;
            const active = filter === f;
            return (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={`font-mono text-[9px] uppercase tracking-[0.10em] px-3 py-1 border transition-colors duration-200 cursor-pointer focus:outline-none focus-visible:ring-2 focus-visible:ring-accent ${
                  active
                    ? "border-accent bg-accent/10 text-accent"
                    : "border-border hover:bg-muted text-muted-foreground"
                }`}
                aria-pressed={active}
                aria-label={`Filter by ${f}`}
              >
                {f === "ALL" ? "ALL" : STATUS_CONFIG[f]?.label ?? f}
                {count > 0 && <span className="ml-1 opacity-60">{count}</span>}
              </button>
            );
          })}
        </div>
      )}

      {!visible.length && (
        <div className="flex flex-col items-center justify-center h-48 border border-dashed border-border">
          <svg className="size-7 text-muted-foreground mb-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1">
            <rect x="2" y="2" width="20" height="20" /><path d="M8 10l4-4 4 4M12 6v9" /><path d="M6 18h12" />
          </svg>
          <p className="font-mono text-[9px] text-muted-foreground uppercase tracking-[0.12em]">
            {matches.length ? "NO MATCHES FOR THIS FILTER" : "NO MATCHES YET — UPLOAD TO START"}
          </p>
        </div>
      )}

      <div className="space-y-px">
        <AnimatePresence mode="popLayout">
          {visible.map((m) => {
            const cfg = STATUS_CONFIG[m.status] ?? STATUS_CONFIG.UPLOADED;
            const isConfirming = confirmId === m.id;
            const isDeleting = deletingId === m.id;
            const progress = progressMap[m.id] ?? 0;
            const isProcessing = m.status === "PROCESSING" || m.status === "UPLOADED";
            const accentColor =
              m.status === "COMPLETED" ? "var(--green)"
              : m.status === "PROCESSING" ? "var(--cyan)"
              : m.status === "UPLOADED" ? "var(--yellow)"
              : "var(--red)";
            const chipClass =
              m.status === "COMPLETED" ? "chip chip-green"
              : m.status === "PROCESSING" ? "chip chip-cyan"
              : m.status === "FAILED" ? "chip chip-red"
              : "chip chip-ghost";

            return (
              <motion.div 
                key={m.id} 
                layout
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.98, transition: { duration: 0.2 } }}
                className="card relative group overflow-hidden bg-card transition-colors hover:border-border-2" 
                style={{ borderLeft: `2px solid ${accentColor}` }}
              >
                {isProcessing && progress > 0 && (
                  <div className="h-[1px] bg-border absolute top-0 left-0 right-0 overflow-hidden">
                    <div className="h-full bg-blue-400 transition-[width] duration-500 ease-out" style={{ width: `${progress}%` }} />
                  </div>
                )}
                
                <Link 
                  href={`/matches/${m.id}`} 
                  className="block p-4 focus:outline-none focus-visible:bg-muted/50 rounded-lg relative z-0"
                  onKeyDown={(e) => {
                    if (e.key === "Delete" || e.key === "Backspace") {
                      e.preventDefault();
                      setConfirmId(m.id);
                    }
                  }}
                  aria-label={`View Match ${m.id}. Press Delete to remove.`}
                >
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 sm:gap-4">
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2 mb-1 sm:mb-2">
                        <span className={chipClass}>
                          {cfg.label}{isProcessing && progress > 0 && <span className="ml-1">{progress}%</span>}
                        </span>
                        <span className="font-mono text-[9px] text-muted-foreground uppercase tracking-[0.08em]">
                          {timeAgo(m.createdAt)}
                        </span>
                      </div>
                      <p className="font-mono text-[11px] text-muted-foreground truncate">
                        {m.id}
                      </p>
                    </div>

                    <div className="text-left sm:text-right shrink-0 space-y-1">
                      <div className="font-mono flex items-center justify-start sm:justify-end gap-1 text-[11px] text-muted-foreground">
                        <svg className="size-2.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <circle cx="12" cy="12" r="10" /><polyline points="12 6 12 12 16 14" />
                        </svg>
                        {formatDuration(m.duration)}
                      </div>
                      {m.status === "COMPLETED" && (
                        <div className="font-mono flex items-center justify-start sm:justify-end gap-1 text-[10px] text-muted-foreground">
                          <span><span className="text-accent">{m._count.events}</span> ev</span>
                          <span>&middot;</span>
                          <span><span className="text-accent">{m._count.highlights}</span> cl</span>
                        </div>
                      )}
                    </div>
                  </div>
                </Link>

                <div className="absolute top-3 right-3 flex items-center gap-1">
                  {!isConfirming ? (
                    <button
                      onClick={(e) => { e.preventDefault(); setConfirmId(m.id); }}
                      className="opacity-0 group-hover:opacity-100 transition-opacity duration-200 p-1.5 focus:outline-none focus-visible:opacity-100 focus-visible:ring-1 focus-visible:ring-border cursor-pointer text-muted-foreground"
                      title="Delete"
                      aria-label={`Delete Match ${m.id}`}
                    >
                      <svg className="size-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <polyline points="3 6 5 6 21 6" /><path d="M19 6l-1 14H6L5 6" />
                        <path d="M10 11v6M14 11v6M9 6V4h6v2" />
                      </svg>
                    </button>
                  ) : (
                    <div className="flex items-center gap-1 px-2 py-1 border bg-muted border-[var(--border-2)]">
                      <span className="font-mono text-[9px] text-muted-foreground uppercase">DEL?</span>
                      <button
                        onClick={(e) => { e.preventDefault(); handleDelete(m.id); }}
                        disabled={isDeleting}
                        className="font-mono px-1.5 text-[9px] text-destructive uppercase"
                      >
                        {isDeleting ? "..." : "YES"}
                      </button>
                      <button
                        onClick={(e) => { e.preventDefault(); setConfirmId(null); }}
                        className="font-mono px-1.5 text-[9px] text-muted-foreground uppercase"
                      >
                        NO
                      </button>
                    </div>
                  )}
                </div>
              </motion.div>
            );
          })}
        </AnimatePresence>
      </div>
    </div>
  );
});
