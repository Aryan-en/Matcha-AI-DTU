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
      <div className="flex items-center justify-center h-48">
        <span className="font-mono animate-blink" style={{ fontSize: "10px", color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.14em" }}>
          LOADING FEED...
        </span>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Filter strip */}
      {matches.length > 0 && (
        <div className="flex items-center gap-1 flex-wrap">
          <span className="font-mono mr-2" style={{ fontSize: "9px", color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.10em" }}>FILTER</span>
          {FILTER_OPTIONS.map((f) => {
            const count = f === "ALL" ? matches.length : matches.filter((m) => m.status === f).length;
            const active = filter === f;
            return (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className="font-mono transition-colors"
                style={{
                  fontSize: "9px",
                  textTransform: "uppercase",
                  letterSpacing: "0.10em",
                  padding: "4px 12px",
                  border: `1px solid ${active ? "var(--amber)" : "var(--border)"}`,
                  background: active ? "var(--amber-glow)" : "transparent",
                  color: active ? "var(--amber)" : "var(--text-dim)",
                }}
              >
                {f === "ALL" ? "ALL" : STATUS_CONFIG[f]?.label ?? f}
                {count > 0 && <span style={{ marginLeft: "4px", opacity: 0.6 }}>{count}</span>}
              </button>
            );
          })}
        </div>
      )}

      {/* Empty state */}
      {!visible.length && (
        <div className="flex flex-col items-center justify-center h-48 border border-dashed" style={{ borderColor: "var(--border)" }}>
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" style={{ color: "var(--text-dim)", marginBottom: "12px" }}>
            <rect x="2" y="2" width="20" height="20" /><path d="M8 10l4-4 4 4M12 6v9" /><path d="M6 18h12" />
          </svg>
          <p className="font-mono" style={{ fontSize: "9px", color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.12em" }}>
            {matches.length ? "NO MATCHES FOR THIS FILTER" : "NO MATCHES YET — UPLOAD TO START"}
          </p>
        </div>
      )}

      {/* Match rows */}
      <div className="space-y-px">
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
          <div key={m.id} className="card card-amber relative group overflow-hidden" style={{ borderLeft: `2px solid ${accentColor}` }}>
            {/* Top progress line */}
            {isProcessing && progress > 0 && (
              <div className="progress-track absolute top-0 left-0 right-0" style={{ height: "1px" }}>
                <div className="progress-fill progress-fill-cyan" style={{ width: `${progress}%` }} />
              </div>
            )}
            <Link href={`/matches/${m.id}`} className="block p-4">
              <div className="flex items-center justify-between gap-4">
                <div className="min-w-0 flex-1">
                  {/* Status + time */}
                  <div className="flex items-center gap-2 mb-2">
                    <span className={chipClass}>
                      {cfg.label}{isProcessing && progress > 0 && <span style={{ marginLeft: "4px" }}>{progress}%</span>}
                    </span>
                    <span className="font-mono" style={{ fontSize: "9px", color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.08em" }}>
                      {timeAgo(m.createdAt)}
                    </span>
                  </div>
                  {/* ID */}
                  <p className="font-mono truncate" style={{ fontSize: "11px", color: "var(--text-dim)" }}>
                    {m.id}
                  </p>
                </div>\n\n                {/* Right: duration + counts */}
                <div className="text-right shrink-0 space-y-1">
                  <div className="font-mono flex items-center justify-end gap-1" style={{ fontSize: "11px", color: "var(--text-dim)" }}>
                    <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <circle cx="12" cy="12" r="10" /><polyline points="12 6 12 12 16 14" />
                    </svg>
                    {formatDuration(m.duration)}
                  </div>
                  {m.status === "COMPLETED" && (
                    <div className="font-mono" style={{ fontSize: "10px", color: "var(--text-dim)" }}>
                      <span style={{ color: "var(--amber)" }}>{m._count.events}</span> ev &middot;{" "}
                      <span style={{ color: "var(--amber)" }}>{m._count.highlights}</span> cl
                    </div>
                  )}
                </div>
              </div>
            </Link>

            {/* Delete */}
            <div className="absolute top-3 right-3 flex items-center gap-1">
              {!isConfirming ? (
                <button
                  onClick={(e) => { e.preventDefault(); setConfirmId(m.id); }}
                  className="opacity-0 group-hover:opacity-100 transition-opacity p-1.5"
                  style={{ color: "var(--text-dim)" }}
                  title="Delete"
                >
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <polyline points="3 6 5 6 21 6" /><path d="M19 6l-1 14H6L5 6" />
                    <path d="M10 11v6M14 11v6M9 6V4h6v2" />
                  </svg>
                </button>
              ) : (
                <div className="flex items-center gap-1 px-2 py-1 border" style={{ background: "var(--surface-2)", borderColor: "var(--border-2)" }}>
                  <span className="font-mono" style={{ fontSize: "9px", color: "var(--text-dim)", textTransform: "uppercase" }}>DEL?</span>
                  <button
                    onClick={(e) => { e.preventDefault(); handleDelete(m.id); }}
                    disabled={isDeleting}
                    className="font-mono px-1.5"
                    style={{ fontSize: "9px", color: "var(--red)", textTransform: "uppercase" }}
                  >
                    {isDeleting ? "..." : "YES"}
                  </button>
                  <button
                    onClick={(e) => { e.preventDefault(); setConfirmId(null); }}
                    className="font-mono px-1.5"
                    style={{ fontSize: "9px", color: "var(--text-dim)", textTransform: "uppercase" }}
                  >
                    NO
                  </button>
                </div>
              )}
            </div>
          </div>
        );
        })}
      </div>
    </div>
  );
}

