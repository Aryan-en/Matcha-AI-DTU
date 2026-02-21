"use client";

import { useEffect, useState, useRef } from "react";
import Link from "next/link";
import { Clock, CheckCircle2, Loader2, Upload, XCircle, Film, Trash2, Filter } from "lucide-react";
import { io, Socket } from "socket.io-client";

interface MatchSummary {
  id: string;
  status: string;
  progress: number; // 0-100 stored in DB
  duration: number | null;
  createdAt: string;
  uploadUrl: string;
  _count: { events: number; highlights: number };
}

interface ProgressMap {
  [matchId: string]: number;
}

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

export function MatchDashboard() {
  const [matches, setMatches] = useState<MatchSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<FilterOption>("ALL");
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [confirmId, setConfirmId] = useState<string | null>(null);
  const [progressMap, setProgressMap] = useState<ProgressMap>({});
  const socketRef = useRef<Socket | null>(null);

  const fetchMatches = async () => {
    try {
      const res = await fetch("http://localhost:4000/matches");
      if (res.ok) {
        const data: MatchSummary[] = await res.json();
        setMatches(data);
        // Initialize progress from stored DB values
        const initialProgress: ProgressMap = {};
        data.forEach(m => {
          if (m.progress > 0 && m.status !== "COMPLETED") {
            initialProgress[m.id] = m.progress;
          }
        });
        setProgressMap(prev => ({ ...initialProgress, ...prev }));
      }
    } catch { /* orchestrator offline */ }
    finally { setLoading(false); }
  };

  // WebSocket for real-time progress
  useEffect(() => {
    socketRef.current = io("http://localhost:4000");
    
    socketRef.current.on("progress", (data: { matchId: string; progress: number }) => {
      setProgressMap(prev => ({ ...prev, [data.matchId]: data.progress }));
      // If progress is 100, refetch to update status
      if (data.progress >= 100) {
        setTimeout(fetchMatches, 500);
      }
    });

    socketRef.current.on("complete", (data: { matchId: string }) => {
      setProgressMap(prev => ({ ...prev, [data.matchId]: 100 }));
      fetchMatches();
    });

    return () => {
      socketRef.current?.disconnect();
    };
  }, []);

  // Join rooms for all processing matches
  useEffect(() => {
    if (!socketRef.current) return;
    const processingMatches = matches.filter(m => m.status === "PROCESSING" || m.status === "UPLOADED");
    processingMatches.forEach(m => {
      socketRef.current?.emit("joinMatch", m.id);
    });
  }, [matches]);

  useEffect(() => {
    fetchMatches();
    const interval = setInterval(fetchMatches, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleDelete = async (id: string) => {
    setDeletingId(id);
    try {
      await fetch(`http://localhost:4000/matches/${id}`, { method: "DELETE" });
      setMatches((prev) => prev.filter((m) => m.id !== id));
    } catch { /* ignore */ }
    finally {
      setDeletingId(null);
      setConfirmId(null);
    }
  };

  const visible = filter === "ALL" ? matches : matches.filter((m) => m.status === filter);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-48 text-zinc-500">
        <Loader2 className="w-6 h-6 animate-spin mr-2" /> Loading matches...
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Filter bar */}
      {matches.length > 0 && (
        <div className="flex items-center gap-2 flex-wrap">
          <Filter className="w-3.5 h-3.5 text-zinc-500 shrink-0" />
          {FILTER_OPTIONS.map((f) => {
            const count = f === "ALL" ? matches.length : matches.filter((m) => m.status === f).length;
            return (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={`text-xs px-4 py-1.5 border transition-all font-medium uppercase tracking-wide ${
                  filter === f
                    ? "bg-emerald-500/20 border-emerald-500 text-emerald-400"
                    : "bg-zinc-900 border-zinc-700 text-zinc-500 hover:border-zinc-500 hover:text-zinc-300"
                }`}
              >
                {f === "ALL" ? "All" : STATUS_CONFIG[f]?.label ?? f}
                {count > 0 && <span className="ml-1.5 text-[10px] opacity-70">{count}</span>}
              </button>
            );
          })}
        </div>
      )}

      {!visible.length && (
        <div className="border border-dashed border-zinc-700 p-12 text-center text-zinc-500 bg-zinc-900/50">
          <Film className="w-10 h-10 mx-auto mb-3 opacity-40" />
          <p className="text-sm uppercase tracking-wide">{matches.length ? "No matches match this filter." : "No matches yet — upload one to get started."}</p>
        </div>
      )}

      {visible.map((m) => {
        const cfg = STATUS_CONFIG[m.status] ?? STATUS_CONFIG.UPLOADED;
        const isConfirming = confirmId === m.id;
        const isDeleting = deletingId === m.id;
        const progress = progressMap[m.id] ?? 0;
        const isProcessing = m.status === "PROCESSING" || m.status === "UPLOADED";

        return (
          <div key={m.id} className="bg-[#141414] border border-zinc-800 hover:border-emerald-500/50 transition-all group relative overflow-hidden card-flat">
            {/* Progress bar at top */}
            {isProcessing && (
              <div className="absolute top-0 left-0 right-0 h-0.5 bg-zinc-800">
                <div 
                  className="h-full bg-emerald-500 transition-all duration-300 ease-out"
                  style={{ width: `${progress}%` }}
                />
              </div>
            )}
            <Link href={`/matches/${m.id}`} className="block p-4">
              <div className="flex items-start justify-between gap-4">
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2 mb-1.5">
                    <span className={`inline-flex items-center gap-1 text-xs font-medium px-2.5 py-1 border ${cfg.color} uppercase tracking-wide`}>
                      {cfg.icon} {cfg.label}
                      {isProcessing && progress > 0 && (
                        <span className="ml-1 tabular-nums">{progress}%</span>
                      )}
                    </span>
                    <span className="text-xs text-zinc-600 uppercase">{timeAgo(m.createdAt)}</span>
                  </div>
                  <p className="text-sm text-zinc-500 font-mono truncate group-hover:text-zinc-300 transition-colors">
                    {m.id}
                  </p>
                </div>
                <div className="text-right shrink-0 space-y-1">
                  <div className="flex items-center gap-1 justify-end text-xs text-zinc-500">
                    <Clock className="w-3 h-3" />{formatDuration(m.duration)}
                  </div>
                  {m.status === "COMPLETED" && (
                    <div className="text-xs text-zinc-600">
                      <span className="text-emerald-400 font-semibold">{m._count.events}</span> events ·{" "}
                      <span className="text-emerald-400 font-semibold">{m._count.highlights}</span> clips
                    </div>
                  )}
                </div>
              </div>
            </Link>

            {/* Delete button */}
            <div className="absolute top-3 right-3 flex items-center gap-2">
              {!isConfirming ? (
                <button
                  onClick={(e) => { e.preventDefault(); setConfirmId(m.id); }}
                  className="opacity-0 group-hover:opacity-100 p-1.5 text-zinc-600 hover:text-red-400 hover:bg-red-500/10 transition-all border border-transparent hover:border-red-500/30"
                  title="Delete analysis"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              ) : (
                <div className="flex items-center gap-1.5 bg-zinc-800 border border-zinc-600 px-2 py-1">
                  <span className="text-xs text-zinc-400 uppercase">Delete?</span>
                  <button
                    onClick={(e) => { e.preventDefault(); handleDelete(m.id); }}
                    disabled={isDeleting}
                    className="text-xs text-red-400 hover:text-red-300 font-medium px-2 py-0.5 hover:bg-red-500/10 transition-all uppercase"
                  >
                    {isDeleting ? <Loader2 className="w-3 h-3 animate-spin" /> : "Yes"}
                  </button>
                  <button
                    onClick={(e) => { e.preventDefault(); setConfirmId(null); }}
                    className="text-xs text-zinc-500 hover:text-zinc-300 px-2 py-0.5 transition-colors uppercase"
                  >
                    No
                  </button>
                </div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

