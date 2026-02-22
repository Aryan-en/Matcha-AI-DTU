"use client";

import React, { useState, useCallback } from "react";
import Link from "next/link";
import { CheckCircle2, Loader2, Upload, XCircle, LayoutGrid, Clock, AlertTriangle } from "lucide-react";
import { useMatches } from "@/hooks/useMatches";
import { motion, AnimatePresence } from "framer-motion";
import { STATUS_CONFIG as SHARED_STATUS_CONFIG, formatTime, timeAgo } from "@matcha/shared";

const THEME_MAP: Record<string, { color: string }> = {
  success: { color: "text-emerald-400 bg-emerald-500/15 border-emerald-500/40" },
  info:    { color: "text-blue-400 bg-blue-500/15 border-blue-500/40"       },
  warning: { color: "text-amber-400 bg-amber-500/15 border-amber-500/40"    },
  error:   { color: "text-red-400 bg-red-500/15 border-red-500/40"          },
};

const STATUS_CONFIG: Record<string, { label: string; color: string; icon: React.ReactNode }> = {
  COMPLETED:  { ...SHARED_STATUS_CONFIG.COMPLETED,  ...THEME_MAP[SHARED_STATUS_CONFIG.COMPLETED.theme],  icon: <CheckCircle2 className="w-3 h-3" /> },
  PROCESSING: { ...SHARED_STATUS_CONFIG.PROCESSING, ...THEME_MAP[SHARED_STATUS_CONFIG.PROCESSING.theme], icon: <Loader2 className="w-3 h-3 animate-spin" /> },
  UPLOADED:   { ...SHARED_STATUS_CONFIG.UPLOADED,   ...THEME_MAP[SHARED_STATUS_CONFIG.UPLOADED.theme],   icon: <Upload className="w-3 h-3" /> },
  FAILED:     { ...SHARED_STATUS_CONFIG.FAILED,     ...THEME_MAP[SHARED_STATUS_CONFIG.FAILED.theme],     icon: <XCircle className="w-3 h-3" /> },
};

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
      <div className="flex flex-col items-center justify-center h-64 border border-dashed border-border/50 bg-card/30">
        <Loader2 className="size-6 text-accent animate-spin mb-4" />
        <span className="font-mono text-[10px] text-muted-foreground uppercase tracking-[0.2em] animate-blink">
          ESTABLISHING BROADCAST UPLINK...
        </span>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Broadcast Tab Navigation */}
      {matches.length > 0 && (
        <div className="flex gap-1 border-b border-border overflow-x-auto pb-px hide-scrollbar">
          <div className="flex items-center px-3 sm:px-4 py-2 border-r border-border bg-muted/20 shrink-0">
            <LayoutGrid className="size-3 sm:size-3.5 text-muted-foreground mr-1.5 sm:mr-2" />
            <span className="font-mono text-[9px] sm:text-[10px] text-muted-foreground uppercase tracking-[0.15em] whitespace-nowrap">
              DATA FEED
            </span>
          </div>
          {FILTER_OPTIONS.map((f) => {
            const count = f === "ALL" ? matches.length : matches.filter((m) => m.status === f).length;
            const active = filter === f;
            return (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={`group relative flex items-center justify-center gap-1.5 sm:gap-2 px-3 sm:px-5 py-2 sm:py-2.5 transition-all duration-300 shrink-0 ${
                  active ? "bg-accent/10" : "hover:bg-muted/40"
                }`}
              >
                <div className={`font-mono text-[9px] sm:text-[10px] uppercase tracking-[0.1em] transition-colors whitespace-nowrap ${
                  active ? "text-accent font-semibold" : "text-muted-foreground group-hover:text-foreground"
                }`}>
                  {f === "ALL" ? "MASTER FEED" : STATUS_CONFIG[f]?.label ?? f}
                </div>
                {count > 0 && (
                  <div className={`px-1.5 py-0.5 font-mono text-[8.5px] sm:text-[9px] rounded-sm ${
                    active ? "bg-accent/20 text-accent" : "bg-muted text-muted-foreground group-hover:bg-border group-hover:text-foreground"
                  }`}>
                    {count}
                  </div>
                )}
                {/* Active Indicator Line */}
                {active && (
                  <div className="absolute bottom-0 left-0 right-0 h-[2px] bg-accent shadow-[0_-2px_8px_rgba(var(--color-accent),0.5)]" />
                )}
              </button>
            );
          })}
        </div>
      )}

      {/* Empty State */}
      {!visible.length && (
        <div className="flex flex-col items-center justify-center py-20 border border-dashed border-border/60 bg-[radial-gradient(ellipse_at_center,_var(--surface-2)_0%,_transparent_100%)]">
          <AlertTriangle className="size-8 text-muted-foreground/50 mb-4" />
          <h3 className="font-display text-xl tracking-widest text-muted-foreground uppercase opacity-80">
            {matches.length ? "NO FEEDS FOUND" : "TRANSMISSION DEAD"}
          </h3>
          <p className="font-mono text-[10px] text-muted-foreground uppercase tracking-[0.15em] mt-2 opacity-60">
            {matches.length ? "ADJUST FILTER PARAMETERS" : "AWAITING SECURE UPLOAD INITIATION"}
          </p>
        </div>
      )}

      {/* Match Grid / List */}
      <div className="grid grid-cols-1 gap-3">
        <AnimatePresence mode="popLayout">
          {visible.map((m) => {
            const cfg = STATUS_CONFIG[m.status] ?? STATUS_CONFIG.UPLOADED;
            const isConfirming = confirmId === m.id;
            const isDeleting = deletingId === m.id;
            const progress = progressMap[m.id] ?? 0;
            const isProcessing = m.status === "PROCESSING" || m.status === "UPLOADED";
            
            // Re-map colors to match our theme strictly
            const accentColor = m.status === "COMPLETED" ? "var(--green)" 
                              : m.status === "PROCESSING" ? "oklch(60% 0.15 250)" // Blue
                              : m.status === "FAILED" ? "var(--red)" 
                              : "var(--amber-dim)";

            return (
              <motion.div 
                key={m.id} 
                layout
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.98, filter: "brightness(0.5)", transition: { duration: 0.2 } }}
                className="card relative group bg-card transition-all duration-300 hover:-translate-y-0.5 hover:shadow-[0_8px_24px_-8px_rgba(0,0,0,0.5)]" 
              >
                {/* Left Status Bar */}
                <div 
                  className="absolute left-0 top-0 bottom-0 w-1.5 transition-all duration-300 group-hover:w-2" 
                  style={{ backgroundColor: accentColor, opacity: 0.8, boxShadow: `0 0 12px ${accentColor}` }}
                />

                {/* Processing Progress Overlay */}
                {isProcessing && progress > 0 && (
                  <div className="absolute left-1.5 top-0 bottom-0 bg-blue-500/5 transition-[width] duration-300 ease-out z-0" style={{ width: `${progress}%` }} />
                )}

                <Link 
                  href={`/matches/${m.id}`} 
                  className="block p-4 sm:p-5 pl-5 sm:pl-7 relative z-10 focus:outline-none focus-visible:bg-white/5"
                  onKeyDown={(e) => {
                    if (e.key === "Delete" || e.key === "Backspace") {
                      e.preventDefault();
                      setConfirmId(m.id);
                    }
                  }}
                >
                  <div className="flex flex-col md:flex-row md:items-center justify-between gap-2 sm:gap-4">
                    
                    {/* Left side: Hero ID & Status */}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 sm:gap-3 mb-1.5 sm:mb-2 flex-wrap">
                        <div className={`flex items-center gap-1.5 px-2.5 py-1 border font-mono text-[9px] uppercase tracking-widest ${cfg.color}`}>
                          {cfg.icon}
                          {cfg.label}
                          {isProcessing && progress > 0 && <span className="ml-1 font-bold">{progress}%</span>}
                        </div>
                        <div className="flex items-center text-muted-foreground">
                          <Clock className="size-3 mr-1.5 opacity-50" />
                          <span className="font-mono text-[10px] uppercase tracking-wider">{timeAgo(m.createdAt)}</span>
                        </div>
                      </div>
                      <h4 className="font-display text-lg sm:text-2xl tracking-wide text-foreground group-hover:text-white transition-colors truncate">
                        {m.id}
                      </h4>
                    </div>

                    {/* Right side: Broadcast Metrics */}
                    <div className="flex items-center justify-start md:justify-end gap-3 sm:gap-4 text-left md:text-right shrink-0 mt-2 md:mt-0">
                      
                      {/* Duration */}
                      {m.duration != null && m.duration > 0 && (
                        <div className="flex flex-col items-start md:items-end">
                          <span className="font-mono text-[8px] text-muted-foreground uppercase tracking-widest mb-0.5 sm:mb-1">RUNTIME</span>
                          <span className="font-mono text-xs sm:text-sm text-foreground/90">{formatTime(m.duration ?? 0)}</span>
                        </div>
                      )}

                      {/* Stat Pills */}
                      {m.status === "COMPLETED" && (
                        <>
                          <div className="w-px h-6 sm:h-8 bg-border/50 mx-1 hidden md:block" />
                          <div className="flex flex-col items-start md:items-end">
                            <span className="font-mono text-[8px] text-muted-foreground uppercase tracking-widest mb-0.5 sm:mb-1">EVENTS</span>
                            <span className="font-mono text-xs sm:text-sm px-1.5 sm:px-2 py-0.5 bg-accent/10 text-accent border border-accent/20 rounded-sm">
                              {m._count.events.toString().padStart(2, "0")}
                            </span>
                          </div>
                          <div className="flex flex-col items-start md:items-end">
                            <span className="font-mono text-[8px] text-muted-foreground uppercase tracking-widest mb-0.5 sm:mb-1">CLIPS</span>
                            <span className="font-mono text-xs sm:text-sm px-1.5 sm:px-2 py-0.5 bg-primary/10 text-primary border border-primary/20 rounded-sm">
                              {m._count.highlights.toString().padStart(2, "0")}
                            </span>
                          </div>
                        </>
                      )}
                    </div>

                  </div>
                </Link>

                {/* Highly Visible Delete Action */}
                <div className="absolute right-4 top-1/2 -translate-y-1/2 z-20">
                  {!isConfirming ? (
                    <button
                      onClick={(e) => { e.preventDefault(); e.stopPropagation(); setConfirmId(m.id); }}
                      className="opacity-0 group-hover:opacity-100 transition-all duration-200 p-2.5 bg-background/50 hover:bg-destructive/10 text-muted-foreground hover:text-destructive border border-transparent hover:border-destructive/30 backdrop-blur-sm cursor-pointer"
                      title="Terminate Feed"
                    >
                      <XCircle className="size-4" />
                    </button>
                  ) : (
                    <div className="flex items-center bg-card border border-destructive/50 shadow-lg overflow-hidden animate-in fade-in slide-in-from-right-2">
                      <div className="bg-destructive/10 px-3 py-2 flex items-center justify-center border-r border-destructive/30">
                        <AlertTriangle className="size-3.5 text-destructive animate-pulse" />
                        <span className="font-mono text-[9px] text-destructive ml-2 uppercase tracking-widest font-bold">TERMINATE?</span>
                      </div>
                      <button
                        onClick={(e) => { e.preventDefault(); e.stopPropagation(); handleDelete(m.id); }}
                        disabled={isDeleting}
                        className="font-mono px-4 py-2 text-[10px] text-foreground hover:bg-destructive hover:text-white transition-colors uppercase tracking-widest font-bold"
                      >
                        {isDeleting ? "..." : "YES"}
                      </button>
                      <div className="w-px bg-border h-full" />
                      <button
                        onClick={(e) => { e.preventDefault(); e.stopPropagation(); setConfirmId(null); }}
                        className="font-mono px-4 py-2 text-[10px] text-muted-foreground hover:bg-muted transition-colors uppercase tracking-widest"
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
