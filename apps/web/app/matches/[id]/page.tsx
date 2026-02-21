"use client";

import { useEffect, useState, useRef, useMemo, useCallback } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import {
  ArrowLeft, Play, Clock, Target, Shield, AlertTriangle,
  Zap, Star, BarChart3, TrendingUp, Film, Loader2,
  Trash2, Copy, Check, Trophy, Cpu, Radio
} from "lucide-react";
import { io, Socket } from "socket.io-client";
import VideoPlayer from "@/components/VideoPlayer";


interface MatchEvent {
  id: string; timestamp: number; type: string; confidence: number;
  finalScore: number; commentary: string | null;
}
interface Highlight {
  id: string; startTime: number; endTime: number;
  score: number; eventType: string | null; commentary: string | null; videoUrl: string | null;
}
interface EmotionScore {
  timestamp: number; audioScore: number; motionScore: number; finalScore: number;
}
// Tracking frame: {t: seconds, b: [[nx,ny,nw,nh,conf],...], p: [[nx,ny,nw,nh,tid,team],...]}
type TrackFrame = { t: number; b: number[][]; p: number[][] };

interface MatchDetail {
  id: string; status: string; duration: number | null;
  uploadUrl: string; createdAt: string;
  summary: string | null;
  highlightReelUrl: string | null;
  trackingData: TrackFrame[] | null;
  teamColors: number[][] | null;
  events: MatchEvent[];
  highlights: Highlight[];
  emotionScores: EmotionScore[];
}


const EVENT_CONFIG: Record<string, { label: string; color: string; bg: string; border: string; icon: React.ReactNode }> = {
  GOAL:      { label: "Goal",    color: "text-emerald-400", bg: "bg-emerald-400/15", border: "border-emerald-400/40", icon: <Target className="w-3.5 h-3.5" /> },
  TACKLE:    { label: "Tackle",  color: "text-amber-400",   bg: "bg-amber-400/15",   border: "border-amber-400/40",   icon: <Zap className="w-3.5 h-3.5" /> },
  FOUL:      { label: "Foul",    color: "text-red-400",     bg: "bg-red-400/15",     border: "border-red-400/40",     icon: <AlertTriangle className="w-3.5 h-3.5" /> },
  SAVE:      { label: "Save",    color: "text-blue-400",    bg: "bg-blue-400/15",    border: "border-blue-400/40",    icon: <Shield className="w-3.5 h-3.5" /> },
  Celebrate: { label: "Celeb",   color: "text-purple-400",  bg: "bg-purple-400/15",  border: "border-purple-400/40",  icon: <Star className="w-3.5 h-3.5" /> },
};
const DEFAULT_EVT = { label: "Event", color: "text-zinc-400", bg: "bg-zinc-400/15", border: "border-zinc-400/40", icon: <Star className="w-3.5 h-3.5" /> };

function fmt(secs: number) {
  const m = Math.floor(secs / 60), s = Math.floor(secs % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

function ScoreBadge({ score }: { score: number }) {
  const color = score >= 7.5 ? "text-emerald-400" : score >= 5 ? "text-amber-400" : "text-zinc-500";
  return (
    <span className={`font-mono text-sm font-bold ${color}`}>
      {score.toFixed(1)}
    </span>
  );
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  const copy = async (e: React.MouseEvent) => {
    e.stopPropagation();
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };
  return (
    <button
      onClick={copy}
      className="p-1 rounded text-zinc-600 transition-all duration-200 hover:text-zinc-300 hover:bg-zinc-700 cursor-pointer focus:outline-none focus-visible:ring-2 focus-visible:ring-zinc-400"
      title="Copy commentary"
      aria-label="Copy commentary to clipboard"
    >
      {copied ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
    </button>
  );
}

// â”€â”€â”€ Intensity sparkline â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function IntensityChart({ scores, duration }: { scores: EmotionScore[]; duration: number }) {
  if (!scores.length || !duration) return null;
  const W = 600, H = 60;
  const pts = scores.map(s => {
    const x = (s.timestamp / duration) * W;
    const y = H - s.motionScore * H;
    return `${x},${y}`;
  });

  return (
    <div className="w-full overflow-hidden bg-card border border-border p-4">
      <div className="flex items-center gap-2 mb-3">
        <TrendingUp className="size-4 text-emerald-500" />
        <span className="text-xs font-semibold text-foreground uppercase tracking-wide">Match Intensity</span>
        <span className="ml-auto text-xs text-muted-foreground">{scores.length} data points</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-14" preserveAspectRatio="none">
        <defs>
          <linearGradient id="intensityGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#10b981" stopOpacity="0.35" />
            <stop offset="100%" stopColor="#10b981" stopOpacity="0.0" />
          </linearGradient>
        </defs>
        {pts.length > 1 && (
          <>
            <polyline
              points={[`0,${H}`, ...pts, `${W},${H}`].join(" ")}
              fill="url(#intensityGrad)" stroke="none"
            />
            <polyline
              points={pts.join(" ")}
              fill="none" stroke="#10b981" strokeWidth="1.5"
              strokeLinejoin="round" strokeLinecap="round"
            />
          </>
        )}
      </svg>
      <div className="flex justify-between text-[10px] text-muted-foreground mt-1">
        <span>0:00</span><span>{fmt(duration / 2)}</span><span>{fmt(duration)}</span>
      </div>
    </div>
  );
}

// â”€â”€â”€ Events Timeline â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function EventsTimeline({ events, duration, onSeek }: { events: MatchEvent[]; duration: number; onSeek: (t: number) => void }) {
  if (!duration) return null;
  return (
    <div className="bg-card border border-border p-4">
      <div className="flex items-center gap-2 mb-3">
        <BarChart3 className="size-4 text-emerald-500" />
        <span className="text-sm font-semibold text-foreground uppercase tracking-wide">Events Timeline</span>
        <span className="ml-auto text-xs text-muted-foreground">{events.length} events</span>
      </div>
      <div className="relative h-8 bg-muted overflow-visible">
        {events.map((ev) => {
          const cfg = EVENT_CONFIG[ev.type] ?? DEFAULT_EVT;
          const pct = (ev.timestamp / duration) * 100;
          return (
            <button
              key={ev.id}
              title={`${cfg.label} @ ${fmt(ev.timestamp)} (score: ${ev.finalScore.toFixed(1)})`}
              onClick={() => onSeek(ev.timestamp)}
              className={`absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-2.5 h-2.5 rounded-full border-2 border-black cursor-pointer hover:scale-150 transition-transform z-10 ${cfg.color.replace("text-", "bg-")}`}
              style={{ left: `${pct}%` }}
            />
          );
        })}
      </div>
      <div className="flex justify-between text-[10px] text-muted-foreground mt-1.5">
        <span>0:00</span><span>{fmt(duration / 2)}</span><span>{fmt(duration)}</span>
      </div>
      <div className="flex flex-wrap gap-3 mt-3">
        {Object.entries(EVENT_CONFIG).map(([k, v]) => (
          <div key={k} className="flex items-center gap-1 text-xs text-muted-foreground">
            <div className={`size-2 rounded-full ${v.color.replace("text-", "bg-")}`} />
            {v.label}
          </div>
        ))}
      </div>
    </div>
  );
}

// â”€â”€â”€ Delete Modal â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function DeleteModal({ onConfirm, onCancel, loading }: { onConfirm: () => void; onCancel: () => void; loading: boolean }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm">
      <div className="bg-card border border-border rounded-2xl p-6 max-w-sm w-full mx-4 shadow-2xl">
        <div className="flex items-center gap-3 mb-4">
          <div className="size-10 rounded-full bg-destructive/15 border border-destructive/30 flex items-center justify-center">
            <Trash2 className="size-5 text-destructive" />
          </div>
          <div>
            <p className="font-semibold text-foreground">Delete Analysis</p>
            <p className="text-xs text-muted-foreground">This cannot be undone</p>
          </div>
        </div>
        <p className="text-sm text-foreground/80 mb-6 leading-relaxed">
          All events, highlights, emotion scores, and commentary for this match will be permanently removed.
        </p>
        <div className="flex gap-3">
          <button
            onClick={onCancel}
            className="flex-1 py-2 rounded-lg border border-border text-foreground text-sm hover:bg-muted transition-colors cursor-pointer"
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            disabled={loading}
            className="flex-1 py-2 rounded-lg bg-destructive hover:bg-destructive/90 disabled:opacity-50 text-destructive-foreground text-sm font-medium transition-colors flex items-center justify-center gap-2 cursor-pointer focus:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-destructive"
          >
            {loading ? <Loader2 className="size-4 animate-spin" /> : <Trash2 className="size-4" />}
            Delete
          </button>
        </div>
      </div>
    </div>
  );
}

// â”€â”€â”€ Main page â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
export default function MatchDetailPage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const [match, setMatch] = useState<MatchDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeHighlight, setActiveHighlight] = useState<Highlight | null>(null);
  const [activeTab, setActiveTab] = useState<"highlights" | "events">("highlights");
  const [eventTypeFilter, setEventTypeFilter] = useState<string>("ALL");
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [reanalyzing, setReanalyzing] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [showOverlay, setShowOverlay] = useState(true);
  const [liveEvents, setLiveEvents] = useState<MatchEvent[]>([]);
  // seekFnRef: VideoPlayer injects its internal seekTo so parent buttons can seek
  const videoSeekRef = useRef<(t: number) => void>(() => {});
  const socketRef = useRef<Socket | null>(null);
  // Throttle currentTime updates — only update state every 500ms to avoid 60fps re-renders
  const lastTimeUpdateRef = useRef(0);
  const handleTimeUpdate = useCallback((t: number) => {
    const now = Date.now();
    if (now - lastTimeUpdateRef.current > 500) {
      lastTimeUpdateRef.current = now;
      setCurrentTime(t);
    }
  }, []);

  useEffect(() => {
    if (!id) return;

    const load = async () => {
      try {
        const res = await fetch(`http://localhost:4000/matches/${id}`);
        if (res.ok) setMatch(await res.json());
      } catch { /* ignore */ }
      finally { setLoading(false); }
    };
    load();
    const iv = setInterval(load, 5000);

    const socket: Socket = io("http://localhost:4000", { transports: ["websocket"] });
    socketRef.current = socket;
    socket.emit("joinMatch", id);

    socket.on("matchEvent", (payload: { matchId: string; event: MatchEvent }) => {
      if (payload.matchId !== id) return;
      setLiveEvents(prev => {
        // de-duplicate by timestamp + type
        const exists = prev.some(
          e => e.timestamp === payload.event.timestamp && e.type === payload.event.type
        );
        if (exists) return prev;
        return [...prev, payload.event];
      });
    });

    return () => {
      clearInterval(iv);
      socket.disconnect();
    };
  }, [id]);

  const seekTo = useCallback((t: number) => { videoSeekRef.current(t); }, []);

  const playHighlight = useCallback((h: Highlight) => {
    setActiveHighlight(h);
    setTimeout(() => videoSeekRef.current(h.startTime), 100);
  }, []);

  const handleDelete = useCallback(async () => {
    setDeleting(true);
    try {
      await fetch(`http://localhost:4000/matches/${id}`, { method: "DELETE" });
      router.push("/");
    } catch { setDeleting(false); }
  }, [id, router]);

  const handleReanalyze = useCallback(async () => {
    setReanalyzing(true);
    try {
      await fetch(`http://localhost:4000/matches/${id}/reanalyze`, { method: "POST" });
      setMatch(prev => prev ? { ...prev, status: "PROCESSING", trackingData: null, teamColors: null } : prev);
    } catch { /* ignore */ } finally {
      setReanalyzing(false);
    }
  }, [id]);

  // useMemo calls must be above early returns — Rules of Hooks.
  // null-safe defaults ensure they always run unconditionally.
  const events = match?.events ?? [];
  const emotionScores = match?.emotionScores ?? [];

  const byType = useMemo(() => events.reduce<Record<string, number>>((acc, e) => {
    acc[e.type] = (acc[e.type] ?? 0) + 1; return acc;
  }, {}), [events]);

  const topScore = useMemo(() =>
    events.length ? Math.max(...events.map(e => e.finalScore)) : 0
  , [events]);

  const avgConf = useMemo(() =>
    events.length ? events.reduce((s, e) => s + e.confidence, 0) / events.length : 0
  , [events]);

  const top5Moments = useMemo(() =>
    [...events].sort((a, b) => b.finalScore - a.finalScore).slice(0, 5)
  , [events]);

  const liveIntensity = useMemo(() => {
    if (!emotionScores.length) return 0;
    const nearest = emotionScores.reduce((prev, cur) =>
      Math.abs(cur.timestamp - currentTime) < Math.abs(prev.timestamp - currentTime) ? cur : prev
    );
    return nearest.motionScore;
  }, [emotionScores, currentTime]);

  const allEventTypes = useMemo(() =>
    Array.from(new Set(events.map(e => e.type)))
  , [events]);

  const filteredEvents = useMemo(() =>
    eventTypeFilter === "ALL" ? events : events.filter(e => e.type === eventTypeFilter)
  , [events, eventTypeFilter]);

  const sortedLive = useMemo(() =>
    [...liveEvents].sort((a, b) => b.timestamp - a.timestamp)
  , [liveEvents]);
  // ──────────────────────────────────────────────────────────────────────────

  if (loading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center text-muted-foreground">
        <Loader2 className="size-6 animate-spin mr-2" /> Loading match…
      </div>
    );
  }
  if (!match) {
    return (
      <div className="min-h-screen bg-black flex flex-col items-center justify-center text-zinc-400 gap-4">
        <Film className="w-12 h-12 opacity-30" />
        <p>Match not found.</p>
        <Link href="/" className="text-emerald-400 hover:underline text-sm">â† Back to dashboard</Link>
      </div>
    );
  }

  const duration = match.duration ?? 0;
  const goalCount = byType["GOAL"] ?? 0;
  const saveCount = byType["SAVE"] ?? 0;

  return (
    <div className="min-h-screen bg-background text-foreground">
      {showDeleteModal && (
        <DeleteModal
          onConfirm={handleDelete}
          onCancel={() => setShowDeleteModal(false)}
          loading={deleting}
        />
      )}

      <nav className="border-b border-border px-4 sm:px-6 py-4 flex flex-wrap items-center gap-3 sm:gap-4 bg-muted/30">
        <Link href="/" className="flex items-center gap-2 text-muted-foreground hover:text-foreground transition-opacity duration-200 hover:opacity-80 text-sm uppercase tracking-wide focus:outline-none focus-visible:ring-2 focus-visible:ring-border rounded-sm">
          <ArrowLeft className="size-4" /> Back
        </Link>
        <div className="hidden sm:block h-4 w-px bg-border" />
        <span className="font-mono text-xs sm:text-sm text-muted-foreground truncate max-w-[150px] sm:max-w-xs">{match.id}</span>
        <span className={`ml-auto inline-flex items-center gap-1 text-[10px] sm:text-xs px-2 sm:px-3 py-1 sm:py-1.5 border font-medium uppercase tracking-wide
          ${match.status === "COMPLETED" ? "text-emerald-400 bg-emerald-500/10 border-emerald-500/40"
          : match.status === "PROCESSING" ? "text-blue-400 bg-blue-500/10 border-blue-500/40"
          : "text-zinc-400 bg-zinc-500/10 border-zinc-500/20"}`}>
          {match.status === "PROCESSING" && <Loader2 className="w-3 h-3 animate-spin" />}
          {match.status}
        </span>
        <button
          onClick={handleReanalyze}
          disabled={reanalyzing || match.status === "PROCESSING"}
          className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-primary border border-border hover:border-primary/50 px-4 py-1.5 transition-all duration-200 disabled:opacity-40 uppercase tracking-wide cursor-pointer focus:outline-none focus-visible:ring-2 focus-visible:ring-primary disabled:cursor-not-allowed"
          title="Re-run AI analysis (updates tracking, events, highlights)"
          aria-label="Re-analyze match"
        >
          {reanalyzing
            ? <Loader2 className="size-3.5 animate-spin" />
            : <Cpu className="size-3.5" />
          }
          Re-analyze
        </button>
        <button
          onClick={() => setShowDeleteModal(true)}
          className="flex items-center gap-1.5 text-[10px] sm:text-xs text-zinc-500 hover:text-red-400 border border-zinc-700 hover:border-red-500/50 px-3 sm:px-4 py-1.5 transition-all duration-200 uppercase tracking-wide cursor-pointer focus:outline-none focus-visible:ring-2 focus-visible:ring-red-400"
          aria-label="Delete match analysis"
        >
          <Trash2 className="w-3.5 h-3.5" /> <span className="hidden sm:inline">Delete</span>
        </button>
      </nav>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-6 sm:py-8 space-y-6">

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          {[
            { label: "Duration",    value: fmt(duration),                       sub: "match length" },
            { label: "Events",      value: match.events.length.toString(),      sub: `${(avgConf * 100).toFixed(0)}% avg conf` },
            { label: "Highlights",  value: match.highlights.length.toString(),  sub: "key moments" },
            { label: "Top Score",   value: topScore.toFixed(1),                 sub: "out of 10" },
          ].map((stat) => (
            <div key={stat.label} className="bg-card border border-border p-4 hover:border-primary/40 transition-all card-flat">
              <p className="text-xs text-muted-foreground mb-1 uppercase tracking-wide">{stat.label}</p>
              <p className="text-3xl font-bold text-foreground font-heading">{stat.value}</p>
              <p className="text-xs text-muted-foreground/80 mt-0.5">{stat.sub}</p>
            </div>
          ))}
        </div>

        {match.summary && (
          <div className="bg-card border border-border p-5">
            <div className="flex items-center gap-2 mb-3">
              <Cpu className="size-4 text-emerald-400" />
              <span className="text-sm font-semibold text-foreground font-heading uppercase tracking-wide">AI Match Analysis</span>
              <span className="ml-auto text-[10px] text-muted-foreground border border-border-2 px-2 py-0.5 uppercase tracking-wider">Gemini 2.0 Flash</span>
            </div>
            <p className="text-sm text-muted-foreground leading-relaxed whitespace-pre-line">{match.summary}</p>
          </div>
        )}

        {top5Moments.length > 0 && (
          <div>
            <div className="flex items-center gap-2 mb-3">
              <Trophy className="size-4 text-amber-400" />
              <span className="text-sm font-semibold text-foreground font-heading uppercase tracking-wide">Top 5 Moments</span>
              <span className="text-xs text-muted-foreground ml-1">by context score · click to seek</span>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-5 gap-3">
              {top5Moments.map((ev, i) => {
                const cfg = EVENT_CONFIG[ev.type] ?? DEFAULT_EVT;
                const rank = i + 1;
                const rankStyle = rank === 1 ? "border-amber-400/50 bg-amber-400/5 hover:border-amber-400"
                                : rank === 2 ? "border-muted-foreground/50 bg-muted-foreground/5 hover:border-muted-foreground"
                                : rank === 3 ? "border-amber-700/50 bg-amber-700/5 hover:border-amber-600"
                                : "border-border bg-muted/50 hover:border-border-2";
                const rankColor = rank === 1 ? "text-amber-400" : rank === 2 ? "text-muted-foreground" : rank === 3 ? "text-amber-600" : "text-muted-foreground/50";
                return (
                  <button
                    key={ev.id}
                    onClick={() => seekTo(ev.timestamp)}
                    className={`relative text-left border p-3 transition-all hover:translate-y-[-2px] group cursor-pointer focus:outline-none focus-visible:ring-2 focus-visible:ring-primary ${rankStyle}`}
                  >
                    <div className={`absolute -top-2 -left-2 size-5 bg-background border flex items-center justify-center text-[10px] font-black ${rankColor} border-current`}>
                      {rank}
                    </div>
                    <span className={`inline-flex items-center gap-1 text-xs px-1.5 py-0.5 border mb-2 ${cfg.bg} ${cfg.border} ${cfg.color}`}>
                      {cfg.icon} {cfg.label}
                    </span>
                    <p className="font-mono text-xs text-muted-foreground mb-1">{fmt(ev.timestamp)}</p>
                    <div className="flex items-center justify-between mb-1.5">
                      <ScoreBadge score={ev.finalScore} />
                      <div className="h-1 w-12 bg-border overflow-hidden">
                        <div className="h-full bg-emerald-500" style={{ width: `${(ev.finalScore / 10) * 100}%` }} />
                      </div>
                    </div>
                    {ev.commentary && (
                      <p className="text-[10px] text-muted-foreground/80 line-clamp-2 group-hover:text-muted-foreground transition-colors">{ev.commentary}</p>
                    )}
                  </button>
                );
              })}
            </div>
          </div>
        )}

        {Object.keys(byType).length > 0 && (
          <div className="flex flex-wrap gap-2">
            {Object.entries(byType).map(([type, count]) => {
              const cfg = EVENT_CONFIG[type] ?? DEFAULT_EVT;
              return (
                <div key={type} className={`flex items-center gap-2 px-3 py-1.5 border text-sm ${cfg.bg} ${cfg.border}`}>
                  <span className={cfg.color}>{cfg.icon}</span>
                  <span className={`font-medium ${cfg.color} uppercase tracking-wide text-xs`}>{cfg.label}</span>
                  <span className="text-muted-foreground text-xs font-mono">{count}</span>
                </div>
              );
            })}
          </div>
        )}

        <div className="grid lg:grid-cols-5 gap-6">
          {/* â”€â”€ Left: Video + intensity â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
          <div className="lg:col-span-3 space-y-4">
            {match.uploadUrl && (
              <div className="space-y-3">
                <VideoPlayer
                  src={match.uploadUrl}
                  events={match.events}
                  highlights={match.highlights}
                  initialTeamColors={match.teamColors}
                  seekFnRef={videoSeekRef}
                  onTimeUpdate={handleTimeUpdate}
                />

                <div className="bg-muted/30 border border-border px-3 sm:px-4 py-3 flex flex-col sm:flex-row sm:items-center gap-4 sm:gap-6">
                  <div className="flex items-center justify-between sm:justify-start gap-3 w-full sm:w-auto">
                    <div className="flex items-center gap-1.5">
                      <div className={`size-2 ${
                        match.status === "COMPLETED" ? "bg-emerald-400" :
                        match.status === "PROCESSING" ? "bg-blue-400 animate-pulse" : "bg-muted-foreground"
                      }`} />
                      <span className="text-[10px] sm:text-xs font-bold tracking-widest text-muted-foreground uppercase">
                        {match.status === "PROCESSING" ? "Live" : "Full Time"}
                      </span>
                    </div>
                    <span className="font-mono text-xs text-muted-foreground sm:hidden">{fmt(currentTime)}</span>
                  </div>
                  <div className="flex items-center gap-4 flex-1 w-full justify-between sm:justify-start">
                    <div className="text-center">
                      <p className="text-[9px] text-muted-foreground uppercase tracking-wide">Goals</p>
                      <p className="text-xl font-black text-foreground leading-none">{goalCount}</p>
                    </div>
                    <div className="text-muted-foreground/50 text-sm">•</div>
                    <div className="text-center">
                      <p className="text-[9px] text-muted-foreground uppercase tracking-wide">Saves</p>
                      <p className="text-xl font-black text-foreground leading-none">{saveCount}</p>
                    </div>
                    <div className="flex-1 space-y-1 ml-4">
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-[9px] text-muted-foreground uppercase tracking-wide">Intensity</span>
                        <span className="text-[9px] font-mono text-emerald-400">{(liveIntensity * 100).toFixed(0)}%</span>
                      </div>
                      <div className="h-1.5 bg-border overflow-hidden">
                        <div
                          className="h-full bg-emerald-500 transition-all duration-500"
                          style={{ width: `${liveIntensity * 100}%` }}
                        />
                      </div>
                    </div>
                  </div>
                  <span className="font-mono text-xs text-muted-foreground hidden sm:block">{fmt(currentTime)}</span>
                </div>
              </div>
            )}
            {match.emotionScores.length > 0 && (
              <IntensityChart scores={match.emotionScores} duration={duration} />
            )}
            {match.events.length > 0 && (
              <EventsTimeline events={match.events} duration={duration} onSeek={seekTo} />
            )}
          </div>

          {/* â”€â”€ Right: Highlights + Events â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
          <div className="lg:col-span-2 space-y-4">
            {/* ── Live Detection Feed ─────────────────────────────────────
                 Shown during processing. Events appear in real-time via WS.    */}
            {(match.status === "PROCESSING" || (match.status === "UPLOADED" && liveEvents.length > 0)) && (
              <div className="bg-card border border-blue-500/30 overflow-hidden">
                <div className="flex items-center gap-2 px-4 py-2.5 border-b border-border bg-blue-500/5">
                  <Radio className="size-3.5 text-blue-400 animate-pulse" />
                  <span className="text-xs font-bold text-blue-300 tracking-widest uppercase">Live Detection</span>
                  <span className="ml-auto text-[10px] text-muted-foreground">{liveEvents.length} events found</span>
                </div>
                <div className="max-h-52 overflow-y-auto space-y-px">
                  {liveEvents.length === 0 && (
                    <div className="flex items-center gap-2 px-4 py-3 text-xs text-muted-foreground/80">
                      <Loader2 className="size-3.5 animate-spin" /> Scanning frames…
                    </div>
                  )}
                  {sortedLive.map((ev, i) => {
                    const cfg = EVENT_CONFIG[ev.type] ?? DEFAULT_EVT;
                    return (
                      <button
                        key={i}
                        onClick={() => seekTo(ev.timestamp)}
                        className={`w-full flex items-center gap-3 px-4 py-2 text-left hover:bg-muted transition-colors cursor-pointer focus:outline-none focus:bg-muted ${i === 0 ? 'animate-pulse' : ''}`}
                      >
                        <span className={`inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 border shrink-0 ${cfg.bg} ${cfg.border} ${cfg.color}`}>
                          {cfg.icon} {cfg.label}
                        </span>
                        <span className="font-mono text-[10px] text-muted-foreground">{fmt(ev.timestamp)}</span>
                        <span className={`ml-auto font-bold text-[10px] font-mono ${
                          ev.finalScore >= 7.5 ? 'text-emerald-400' : ev.finalScore >= 5 ? 'text-amber-400' : 'text-muted-foreground'
                        }`}>{ev.finalScore?.toFixed(1)}</span>
                      </button>
                    );
                  })}
                </div>
              </div>
            )}

            <div className="flex bg-muted/30 border border-border p-1 gap-1">
              {(["highlights", "events"] as const).map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`flex-1 text-sm py-2 font-medium transition-all uppercase tracking-wide cursor-pointer focus:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-inset
                    ${activeTab === tab
                      ? "bg-primary text-[#07080F]"
                      : "text-muted-foreground hover:text-foreground hover:bg-muted"}`}
                >
                  {tab} {tab === "highlights"
                    ? `(${match.highlights.length})`
                    : match.status === "PROCESSING"
                      ? `(${liveEvents.length} live)`
                      : `(${match.events.length})`
                  }
                </button>
              ))}
            </div>

            {activeTab === "highlights" && (
              <div className="space-y-3 max-h-[600px] overflow-y-auto pr-1">
                {match.highlightReelUrl && (
                  <div className="mb-4 p-4 bg-primary/20 border border-primary/30">
                    <h3 className="text-sm font-bold text-primary mb-2 flex items-center gap-2 font-heading uppercase tracking-wide">
                      <Film className="size-4" /> Full Highlight Reel
                    </h3>
                    <p className="text-xs text-muted-foreground mb-3">
                      A combined broadcast-style reel with AI commentary, crowd noise, and music.
                    </p>
                    <a
                      href={match.highlightReelUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-2 text-sm font-semibold text-background bg-primary hover:bg-primary/90 px-4 py-2 transition-all uppercase tracking-wide"
                    >
                      <Play className="size-4" /> Watch Reel
                    </a>
                  </div>
                )}
                {!match.highlights.length && (
                  <div className="text-center text-muted-foreground/80 text-sm py-12 border border-dashed border-border-2 bg-muted/50">
                    <Film className="size-8 mx-auto mb-2 opacity-30" />
                    No highlights yet — re-upload to generate
                  </div>
                )}
                {match.highlights.map((h, i) => {
                  const cfg = EVENT_CONFIG[h.eventType ?? ""] ?? DEFAULT_EVT;
                  const isActive = activeHighlight?.id === h.id;
                  return (
                    <div key={h.id} className={`border p-4 transition-all ${isActive ? "border-primary/50 bg-primary/5" : "border-border bg-card hover:border-border-2"}`}>
                      <div className="flex items-start justify-between gap-2 mb-2">
                        <div className="flex items-center gap-2">
                          <span className="size-6 bg-muted flex items-center justify-center text-xs text-muted-foreground font-bold shrink-0">{i + 1}</span>
                          {h.eventType && (
                            <span className={`inline-flex items-center gap-1 text-xs px-2 py-0.5 border ${cfg.bg} ${cfg.border} ${cfg.color}`}>
                              {cfg.icon} {cfg.label}
                            </span>
                          )}
                        </div>
                        <ScoreBadge score={h.score} />
                      </div>

                      <div className="h-1 bg-border mb-3 overflow-hidden">
                        <div
                          className="h-full bg-emerald-500 transition-all"
                          style={{ width: `${(h.score / 10) * 100}%` }}
                        />
                      </div>

                      <div className="flex items-center gap-3 text-xs text-muted-foreground mb-2">
                        <Clock className="size-3" />
                        <span className="font-mono">{fmt(h.startTime)} → {fmt(h.endTime)}</span>
                        <span className="text-muted-foreground/50">·</span>
                        <span>{Math.round(h.endTime - h.startTime)}s</span>
                      </div>

                      {h.commentary && (
                        <div className="flex items-start gap-1.5 mb-3">
                          <p className="text-xs text-muted-foreground italic leading-relaxed flex-1">
                            "{h.commentary}"
                          </p>
                          <CopyButton text={h.commentary} />
                        </div>
                      )}

                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => playHighlight(h)}
                          className="flex items-center gap-1.5 text-xs font-medium text-primary hover:text-primary/80 border border-primary/30 hover:border-primary px-3 py-1.5 transition-all bg-primary/5 hover:bg-primary/10 uppercase tracking-wide cursor-pointer focus:outline-none focus:ring-2 focus:ring-primary focus:ring-inset"
                        >
                          <Play className="size-3.5" /> Play Clip
                        </button>
                        {h.videoUrl && (
                          <a
                            href={h.videoUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex items-center gap-1.5 text-xs font-medium text-blue-400 hover:text-blue-300 border border-blue-500/30 hover:border-blue-500 px-3 py-1.5 transition-all bg-blue-500/5 hover:bg-blue-500/10 uppercase tracking-wide cursor-pointer focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-inset"
                          >
                            <Film className="size-3.5" /> View Generated Clip
                          </a>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            {activeTab === "events" && (
              <div className="space-y-3">
                {allEventTypes.length > 0 && (
                  <div className="flex flex-wrap gap-1.5">
                    <button
                      onClick={() => setEventTypeFilter("ALL")}
                      className={`text-xs px-3 py-1.5 border transition-all uppercase tracking-wide cursor-pointer focus:outline-none focus:ring-2 focus:ring-primary focus:ring-inset ${
                        eventTypeFilter === "ALL"
                          ? "bg-muted-foreground/20 border-border-2 text-foreground"
                          : "bg-transparent border-border text-muted-foreground hover:border-border-2 hover:text-foreground"
                      }`}
                    >
                      All ({match.events.length})
                    </button>
                    {allEventTypes.map((type) => {
                      const cfg = EVENT_CONFIG[type] ?? DEFAULT_EVT;
                      return (
                        <button
                          key={type}
                          onClick={() => setEventTypeFilter(type)}
                          className={`inline-flex items-center gap-1 text-xs px-3 py-1.5 border transition-all uppercase tracking-wide cursor-pointer focus:outline-none focus:ring-2 focus:ring-primary focus:ring-inset ${
                            eventTypeFilter === type
                              ? `${cfg.bg} ${cfg.border} ${cfg.color}`
                              : "bg-transparent border-border text-muted-foreground hover:border-border-2 hover:text-foreground"
                          }`}
                        >
                          {cfg.icon} {cfg.label} ({byType[type] ?? 0})
                        </button>
                      );
                    })}
                  </div>
                )}

                <div className="space-y-1.5 max-h-[520px] overflow-y-auto pr-1">
                  {!filteredEvents.length && (
                    <div className="text-center text-muted-foreground/80 text-sm py-12 border border-dashed border-border-2 bg-muted/50">
                      No events detected
                    </div>
                  )}
                  {filteredEvents.map((ev) => {
                    const cfg = EVENT_CONFIG[ev.type] ?? DEFAULT_EVT;
                    return (
                      <button
                        key={ev.id}
                        onClick={() => seekTo(ev.timestamp)}
                        className="w-full text-left border border-border hover:border-border-2 bg-card hover:bg-muted p-3 transition-all group cursor-pointer focus:outline-none focus:ring-2 focus:ring-primary focus:ring-inset"
                      >
                        <div className="flex items-center gap-3">
                          <span className={`inline-flex items-center gap-1 text-xs px-2 py-0.5 border shrink-0 ${cfg.bg} ${cfg.border} ${cfg.color}`}>
                            {cfg.icon} {cfg.label}
                          </span>
                          <span className="text-xs text-muted-foreground font-mono shrink-0">{fmt(ev.timestamp)}</span>
                          <div className="flex-1 min-w-0">
                            {ev.commentary && (
                              <p className="text-xs text-muted-foreground/80 truncate group-hover:text-muted-foreground transition-colors">
                                {ev.commentary}
                              </p>
                            )}
                          </div>
                          <div className="shrink-0 flex items-center gap-2">
                            {ev.commentary && (
                              <CopyButton text={ev.commentary} />
                            )}
                            <span className="text-xs text-muted-foreground/50">{(ev.confidence * 100).toFixed(0)}%</span>
                            <ScoreBadge score={ev.finalScore} />
                          </div>
                        </div>
                      </button>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

