"use client";

import React, { useState, useCallback } from "react";
import Link from "next/link";
import { CheckCircle2, Loader2, Upload, XCircle, LayoutGrid, Clock, AlertTriangle, PlayCircle, BarChart3, Scissors, RefreshCw } from "lucide-react";
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

  const handleReanalyze = useCallback(async (id: string) => {
    // In a real app, this would call an API endpoint like POST /matches/:id/reanalyze
    // For now, we'll simulate it by logging
    console.log("Reanalyzing match:", id);
  }, []);

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

      {/* Header for Stats - Aligned to Card Grid - Hidden on Mobile */}
      {visible.length > 0 && (
        <div className="hidden lg:grid grid-cols-[120px,1fr,80px,80px,80px,120px,120px] gap-0 px-0 border-b border-border/20 bg-muted/5">
          <div className="flex items-center justify-center font-mono text-[9px] uppercase tracking-[0.2em] text-muted-foreground/30 border-r border-border/10">PREV</div>
          <div className="flex items-center px-5 font-mono text-[9px] uppercase tracking-[0.2em] text-muted-foreground/30">INTELLIGENCE FEED</div>
          <div className="flex items-center justify-center text-muted-foreground/30 border-l border-border/10">
            <PlayCircle className="size-3" />
          </div>
          <div className="flex items-center justify-center text-muted-foreground/30 border-l border-border/10">
            <BarChart3 className="size-3" />
          </div>
          <div className="flex items-center justify-center text-muted-foreground/30 border-l border-border/10">
            <Scissors className="size-3" />
          </div>
          <div className="flex items-center justify-center font-mono text-[9px] uppercase tracking-widest text-muted-foreground/30 border-l border-border/10">SHORTCUTS</div>
          <div className="flex items-center justify-center font-mono text-[9px] uppercase tracking-widest text-muted-foreground/30 border-l border-border/10">CONTROL</div>
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
      <motion.div 
        className="grid grid-cols-1 gap-4"
        initial="hidden"
        animate="show"
        variants={{
          hidden: { opacity: 0 },
          show: {
            opacity: 1,
            transition: {
              staggerChildren: 0.05
            }
          }
        }}
      >
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

            const formattedDate = new Date(m.createdAt).toLocaleDateString('en-US', {
              month: 'short', day: 'numeric', year: 'numeric'
            });

            const formattedTime = new Date(m.createdAt).toLocaleTimeString('en-US', {
              hour: '2-digit', minute: '2-digit'
            });

            return (
              <motion.div 
                key={m.id} 
                layout
                variants={{
                  hidden: { opacity: 0, y: 20 },
                  show: { opacity: 1, y: 0 }
                }}
                exit={{ opacity: 0, scale: 0.98, filter: "brightness(0.5)", transition: { duration: 0.2 } }}
                className="card relative group bg-card/40 backdrop-blur-md border border-white/5 transition-all duration-300 hover:bg-card/60 hover:border-white/10" 
              >
                {/* Left Status Bar */}
                <div 
                  className="absolute left-0 top-0 bottom-0 w-1 transition-all duration-500 group-hover:w-1.5" 
                  style={{ 
                    backgroundColor: accentColor, 
                    opacity: 0.8, 
                    boxShadow: `4px 0 20px -4px ${accentColor}` 
                  }}
                />

                <div className="flex flex-col lg:flex-row lg:h-[72px] items-stretch relative overflow-hidden">
                  
                  {/* Top Bar (Mobile Only) - Status indicator */}
                  <div className="lg:hidden flex items-center justify-between px-4 py-2 border-b border-white/5 bg-white/[0.02]">
                    <div className={`px-1.5 py-0.5 border font-mono text-[7px] uppercase tracking-[0.1em] font-bold ${cfg.color}`}>
                      {cfg.label}
                    </div>
                    <span className="font-mono text-[8px] text-muted-foreground/60 uppercase tracking-widest">
                      {timeAgo(m.createdAt)}
                    </span>
                  </div>

                  <div className="flex flex-1 items-stretch min-h-[80px] lg:min-h-0">
                    {/* Thumbnail Section */}
                    <Link 
                      href={`/matches/${m.id}`}
                      className="w-[80px] sm:w-[100px] lg:w-[120px] h-full shrink-0 relative overflow-hidden group/thumb border-r border-white/10 bg-black/40"
                    >
                      {m.thumbnailUrl || m.heatmapUrl ? (
                        <img 
                          src={m.thumbnailUrl || m.heatmapUrl || ""} 
                          alt="Preview" 
                          className="w-full h-full object-cover opacity-60 group-hover/thumb:opacity-100 transition-all duration-700 scale-110 group-hover/thumb:scale-100 saturate-50 group-hover/thumb:saturate-100" 
                        />
                      ) : (
                        <div className="w-full h-full flex items-center justify-center bg-muted/10">
                          <PlayCircle className="size-5 text-muted-foreground/20" />
                        </div>
                      )}
                      <div className="absolute inset-0 bg-gradient-to-r from-black/60 via-transparent to-transparent opacity-60" />
                    </Link>

                    {/* Main Identity Area */}
                    <Link 
                      href={`/matches/${m.id}`} 
                      className="flex-1 flex flex-col justify-center px-4 sm:px-6 min-w-0 group/id focus:outline-none"
                    >
                      <div className="flex items-center gap-2 mb-1">
                        <div className={`hidden lg:block px-1.5 py-0.5 border font-mono text-[7px] uppercase tracking-[0.1em] font-bold shrink-0 ${cfg.color} ${m.status === 'PROCESSING' ? 'animate-pulse' : ''}`}>
                          {cfg.label}
                        </div>
                        <h4 className="font-display text-sm sm:text-base tracking-[0.05em] text-foreground group-hover/id:text-white transition-colors truncate">
                          {formattedDate} — Analysis
                        </h4>
                      </div>
                      <p className="font-mono text-[8px] sm:text-[9px] text-muted-foreground/40 uppercase tracking-widest truncate">
                         {formattedTime} • ID: {m.id.split('-')[0]}
                      </p>
                    </Link>
                  </div>

                  {/* Stats & Controls Wrapper */}
                  <div className="flex flex-col sm:flex-row lg:flex-row items-stretch border-t lg:border-t-0 lg:border-l border-white/5">
                    
                    {/* Stats Columns - Re-flow on mobile */}
                    <div className="flex flex-1 items-stretch divide-x divide-white/5 border-b sm:border-b-0 sm:border-r border-white/5 lg:border-r-0">
                      <div className="flex-1 lg:w-[70px] xl:w-[80px] flex items-center justify-center py-3 lg:py-0 bg-white/[0.01]">
                        <div className="flex flex-col items-center">
                          <span className="lg:hidden font-mono text-[7px] text-muted-foreground/40 uppercase mb-1">Duration</span>
                          <span className="font-mono text-[10px] text-white/70 tabular-nums">{m.duration ? formatTime(m.duration) : "--:--"}</span>
                        </div>
                      </div>
                      <div className="flex-1 lg:w-[70px] xl:w-[80px] flex items-center justify-center py-3 lg:py-0">
                        <div className="flex flex-col items-center">
                          <span className="lg:hidden font-mono text-[7px] text-muted-foreground/40 uppercase mb-1">Events</span>
                          <span className="font-display text-[14px] text-accent drop-shadow-[0_0_8px_rgba(var(--color-accent),0.4)]">
                            {m.status === "COMPLETED" ? m._count.events.toString().padStart(2, "0") : "--"}
                          </span>
                        </div>
                      </div>
                      <div className="flex-1 lg:w-[70px] xl:w-[80px] flex items-center justify-center py-3 lg:py-0 bg-white/[0.01]">
                         <div className="flex flex-col items-center">
                          <span className="lg:hidden font-mono text-[7px] text-muted-foreground/40 uppercase mb-1">Reels</span>
                          <span className="font-display text-[14px] text-primary drop-shadow-[0_0_8px_rgba(var(--color-primary),0.4)]">
                            {m.status === "COMPLETED" ? m._count.highlights.toString().padStart(2, "0") : "--"}
                          </span>
                        </div>
                      </div>
                    </div>

                    {/* Actions Area */}
                    <div className="flex items-stretch divide-x divide-white/5 lg:divide-x-0">
                      {/* Highlight Shortcut */}
                      <div className="flex-1 sm:flex-none lg:w-[120px] flex items-center justify-center px-4 py-3 lg:py-0 border-r lg:border-r-0 lg:border-l border-white/5">
                         <Link
                            href={`/matches/${m.id}#highlights`}
                            className="w-full sm:w-auto flex items-center justify-center gap-1.5 px-3 py-1.5 bg-accent/5 hover:bg-accent/15 border border-accent/20 hover:border-accent/40 text-accent transition-all group/high rounded-sm"
                          >
                             <Scissors className="size-3 transition-transform" />
                             <span className="font-mono text-[8px] uppercase tracking-widest font-bold">Highlights</span>
                          </Link>
                      </div>

                      {/* Control Panel */}
                      <div className="flex-1 sm:flex-none lg:w-[120px] flex items-center justify-center px-4 py-3 lg:py-0 border-l border-white/10 bg-white/[0.02]">
                        {!isConfirming ? (
                          <div className="flex items-center gap-3 lg:gap-2">
                            <button
                              onClick={(e) => { e.preventDefault(); e.stopPropagation(); handleReanalyze(m.id); }}
                              className="flex items-center justify-center size-9 lg:size-8 bg-white/5 hover:bg-accent/10 text-muted-foreground hover:text-accent border border-white/5 hover:border-accent/30 transition-all rounded-full"
                              title="Reanalyze"
                            >
                              <RefreshCw className="size-4 lg:size-3.5" />
                            </button>
                            <button
                              onClick={(e) => { e.preventDefault(); e.stopPropagation(); setConfirmId(m.id); }}
                              className="flex items-center justify-center size-9 lg:size-8 bg-white/5 hover:bg-destructive/10 text-muted-foreground hover:text-destructive border border-white/5 hover:border-destructive/30 transition-all rounded-full"
                              title="Delete"
                            >
                              <XCircle className="size-4 lg:size-3.5" />
                            </button>
                          </div>
                        ) : (
                          <div className="flex items-center bg-card shadow-2xl scale-95 lg:scale-90 border border-destructive/20 overflow-hidden">
                            <button
                              onClick={(e) => { e.preventDefault(); e.stopPropagation(); handleDelete(m.id); }}
                              disabled={isDeleting}
                              className="font-mono px-4 lg:px-3 py-2.5 lg:py-2 text-[9px] lg:text-[8px] bg-destructive text-white uppercase tracking-widest font-bold hover:brightness-110"
                            >
                              {isDeleting ? "..." : "DEL"}
                            </button>
                            <button
                              onClick={(e) => { e.preventDefault(); e.stopPropagation(); setConfirmId(null); }}
                              className="font-mono px-4 lg:px-3 py-2.5 lg:py-2 text-[9px] lg:text-[8px] text-muted-foreground hover:bg-white/5 uppercase tracking-widest border-l border-white/10"
                            >
                              X
                            </button>
                          </div>
                        )}
                      </div>
                    </div>

                  </div>
                </div>

              </motion.div>
            );
          })}
        </AnimatePresence>
      </motion.div>
    </div>
  );
});
