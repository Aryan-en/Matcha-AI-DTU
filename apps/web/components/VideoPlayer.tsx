"use client";

/**
 * VideoPlayer — custom HTML5 player with REAL-TIME in-browser AI tracking.
 *
 * Tracking is computed LIVE from video frames using COCO-SSD (TF.js) running
 * entirely in the browser — no database, no backend call, works for every video.
 *
 *  1. COCO-SSD  → person + sports-ball bounding boxes per frame (~10 fps async)
 *  2. Jersey-pixel sampling via 1-pixel offscreen canvas → RGB per player
 *  3. Incremental K-means (2 clusters) → team assignment & colour swatches
 *  4. Canvas drawn at 60 fps (smooth) — detection runs async in parallel
 */

import {
  useRef, useEffect, useState, useCallback, type MouseEvent as RMouseEvent,
} from "react";
import {
  Play, Pause, Volume2, VolumeX, Maximize, Minimize,
  SkipForward, SkipBack, Gauge, Film, Volume1, MessageSquare, Cpu,
} from "lucide-react";

// ── Types ─────────────────────────────────────────────────────────────────────
export interface VPEvent {
  id: string; timestamp: number; type: string;
  finalScore: number; commentary: string | null;
}
export interface VPHighlight {
  id: string; startTime: number; endTime: number;
  score: number; eventType: string | null; commentary: string | null; videoUrl: string | null;
}
// kept for page.tsx compat — VideoPlayer no longer reads this
export type TrackFrame = { t: number; b: number[][]; p: number[][] };

interface Props {
  src: string;
  events:        VPEvent[];
  highlights:    VPHighlight[];
  onTimeUpdate?: (t: number) => void;
  /** Parent can call seekTo from outside (e.g. Top 5 Moments buttons) */
  seekFnRef?:         React.MutableRefObject<(t: number) => void>;
  /** DB-stored team colours used as initial seed; overridden by live detection */
  initialTeamColors?: number[][] | null;
}

// ── In-browser K-means (2 clusters) ─────────────────────────────────────────
function colorDist(a: number[], b: number[]): number {
  return Math.sqrt(a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0));
}
function kMeans2(samples: number[][]): [number[], number[]] {
  if (samples.length < 4) return [[220, 50, 50], [50, 100, 220]];
  let c1 = [...samples[0]];
  let c2 = [...samples.reduce((best, c) =>
    colorDist(c, c1) > colorDist(best, c1) ? c : best, samples[1])];
  for (let iter = 0; iter < 12; iter++) {
    const g1: number[][] = [], g2: number[][] = [];
    for (const c of samples) (colorDist(c, c1) <= colorDist(c, c2) ? g1 : g2).push(c);
    if (!g1.length || !g2.length) break;
    const mean = (g: number[][]) =>
      g[0].map((_, i) => Math.round(g.reduce((s, c) => s + c[i], 0) / g.length));
    c1 = mean(g1); c2 = mean(g2);
  }
  return [c1, c2];
}

// ── Jersey colour sampler (1-pixel offscreen canvas) ─────────────────────────
let _offscreen: HTMLCanvasElement | null = null;
function sampleJersey(
  video: HTMLVideoElement, bx: number, by: number, bw: number, bh: number,
): [number, number, number] | null {
  try {
    if (!_offscreen) { _offscreen = document.createElement("canvas"); _offscreen.width = 1; _offscreen.height = 1; }
    const ctx = _offscreen.getContext("2d", { willReadFrequently: true })!;
    const sw = bw * 0.60, sh = bh * 0.35;
    if (sw < 2 || sh < 2) return null;
    ctx.drawImage(video, bx + bw * 0.20, by + bh * 0.25, sw, sh, 0, 0, 1, 1);
    const d = ctx.getImageData(0, 0, 1, 1).data;
    return [d[0], d[1], d[2]];
  } catch { return null; }
}

// ── Misc helpers ──────────────────────────────────────────────────────────────
const fmt = (s: number) =>
  `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

const EVT_COLORS: Record<string, string> = {
  GOAL: "#10b981", TACKLE: "#f59e0b", FOUL: "#ef4444",
  SAVE: "#3b82f6", Celebrate: "#a855f7",
};

function toRgba(rgb: number[], alpha = 0.85) {
  return `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${alpha})`;
}

function speak(text: string) {
  if (typeof window === "undefined" || !window.speechSynthesis) return;
  window.speechSynthesis.cancel();
  const utt = new SpeechSynthesisUtterance(text);
  utt.rate = 1.1; utt.pitch = 1.05;
  const pref = window.speechSynthesis.getVoices().find(v => /en.*(US|GB|AU)/i.test(v.lang));
  if (pref) utt.voice = pref;
  window.speechSynthesis.speak(utt);
}
const stopSpeaking = () =>
  typeof window !== "undefined" && window.speechSynthesis?.cancel();

// COCO-SSD detection shape
interface Detection { bbox: [number,number,number,number]; class: string; score: number; }

// ── Component ─────────────────────────────────────────────────────────────────
export default function VideoPlayer({
  src, events, highlights, onTimeUpdate, seekFnRef, initialTeamColors,
}: Props) {
  const wrapRef    = useRef<HTMLDivElement>(null);
  const vidRef     = useRef<HTMLVideoElement>(null);
  const canvasRef  = useRef<HTMLCanvasElement>(null);
  const rafRef     = useRef<number>(0);
  const modelRef   = useRef<any>(null);
  const predsRef   = useRef<Detection[]>([]);
  const detectingRef = useRef(false);
  const frameIdx   = useRef(0);
  const jerseyBuf  = useRef<number[][]>([]);
  const teamCols   = useRef<[number[], number[]]>(
    initialTeamColors?.length === 2
      ? [initialTeamColors[0], initialTeamColors[1]]
      : [[220, 50, 50], [50, 100, 220]]
  );
  const seenEvents = useRef<Set<string>>(new Set());
  const toastTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const seekRef    = useRef<HTMLDivElement>(null);

  const [playing,      setPlaying]      = useState(false);
  const [current,      setCurrent]      = useState(0);
  const [duration,     setDuration]     = useState(0);
  const [volume,       setVolume]       = useState(1);
  const [muted,        setMuted]        = useState(false);
  const [speed,        setSpeed]        = useState(1);
  const [fullscreen,   setFullscreen]   = useState(false);
  const [showTracking, setShowTracking] = useState(true);
  const [showSpeed,    setShowSpeed]    = useState(false);
  const [toast,        setToast]        = useState<string | null>(null);
  const [modelState,   setModelState]   = useState<"loading"|"ready"|"error">("loading");
  const [sampleCount,  setSampleCount]  = useState(0); // re-render trigger for team swatches

  // ── Load COCO-SSD (lazy import, browser only) ─────────────────────────────
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        // eslint-disable-next-line @typescript-eslint/ban-ts-comment
        // @ts-ignore – TF.js hoisted to workspace root; types resolve at runtime
        await import("@tensorflow/tfjs");
        // @ts-ignore
        const cocoSsd = await import("@tensorflow-models/coco-ssd");
        const m = await (cocoSsd as any).load({ base: "lite_mobilenet_v2" });
        if (!cancelled) { modelRef.current = m; setModelState("ready"); }
      } catch { if (!cancelled) setModelState("error"); }
    })();
    return () => { cancelled = true; };
  }, []);

  // ── Async detection (non-blocking, called from rAF) ───────────────────────
  const runDetection = useCallback(async () => {
    const video = vidRef.current;
    const model = modelRef.current;
    if (!video || !model || detectingRef.current) return;
    if (video.readyState < 2 || video.paused || video.ended) return;
    detectingRef.current = true;
    try {
      const preds: Detection[] = await model.detect(video);
      predsRef.current = preds;
      // Accumulate jersey colours → re-cluster every 40 new samples
      let added = 0;
      for (const p of preds) {
        if (p.class !== "person" || p.score < 0.45) continue;
        const [bx, by, bw, bh] = p.bbox;
        const col = sampleJersey(video, bx, by, bw, bh);
        if (col) { jerseyBuf.current.push(col); added++; }
      }
      if (added && jerseyBuf.current.length >= 40) {
        if (jerseyBuf.current.length > 300) jerseyBuf.current.splice(0, jerseyBuf.current.length - 300);
        teamCols.current = kMeans2(jerseyBuf.current);
        setSampleCount(jerseyBuf.current.length); // trigger swatch re-render
      }
    } finally { detectingRef.current = false; }
  }, []);

  // ── rAF draw loop (60 fps canvas; detection async at ~10 fps) ────────────
  const drawLoop = useCallback(() => {
    const canvas = canvasRef.current;
    const video  = vidRef.current;
    if (!canvas || !video) { rafRef.current = requestAnimationFrame(drawLoop); return; }
    const ctx = canvas.getContext("2d");
    if (!ctx)  { rafRef.current = requestAnimationFrame(drawLoop); return; }

    const rect = video.getBoundingClientRect();
    if (canvas.width !== Math.round(rect.width) || canvas.height !== Math.round(rect.height)) {
      canvas.width = rect.width; canvas.height = rect.height;
    }
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    frameIdx.current++;
    if (frameIdx.current % 6 === 0 && showTracking) runDetection();

    if (showTracking && video.readyState >= 2) {
      // letterbox correction
      const nW = video.videoWidth  || rect.width;
      const nH = video.videoHeight || rect.height;
      const nR = nW / nH, eR = rect.width / rect.height;
      let dW: number, dH: number, oX: number, oY: number;
      if (nR > eR) { dW = rect.width;  dH = dW / nR; oX = 0; oY = (rect.height - dH) / 2; }
      else         { dH = rect.height; dW = dH * nR; oY = 0; oX = (rect.width  - dW) / 2; }
      const sX = dW / nW, sY = dH / nH;

      for (const p of predsRef.current) {
        const [bx, by, bw, bh] = p.bbox;
        const cx = oX + bx * sX, cy = oY + by * sY;
        const cw = bw * sX,      ch = bh * sY;

        if (p.class === "person" && p.score >= 0.45) {
          const col = sampleJersey(video, bx, by, bw, bh);
          const [c0, c1] = teamCols.current;
          const team = col ? (colorDist(col, c0) <= colorDist(col, c1) ? 0 : 1) : 0;
          const stroke = toRgba(teamCols.current[team]);
          ctx.strokeStyle = stroke; ctx.lineWidth = 2;
          ctx.beginPath();
          if (ctx.roundRect) ctx.roundRect(cx, cy, cw, ch, 3); else ctx.rect(cx, cy, cw, ch);
          ctx.stroke();
          ctx.fillStyle = stroke;
          ctx.beginPath(); ctx.arc(cx + 5, cy + 5, 4, 0, Math.PI * 2); ctx.fill();
        }

        if (p.class === "sports ball" && p.score >= 0.30) {
          const bCx = oX + (bx + bw / 2) * sX, bCy = oY + (by + bh / 2) * sY;
          const r = Math.max(7, (bw * sX) / 2);
          const grd = ctx.createRadialGradient(bCx, bCy, 0, bCx, bCy, r * 3);
          grd.addColorStop(0, "rgba(16,185,129,0.55)"); grd.addColorStop(1, "rgba(16,185,129,0)");
          ctx.fillStyle = grd; ctx.beginPath(); ctx.arc(bCx, bCy, r * 3, 0, Math.PI * 2); ctx.fill();
          ctx.fillStyle = "rgba(16,185,129,0.95)"; ctx.strokeStyle = "rgba(255,255,255,0.9)"; ctx.lineWidth = 1.5;
          ctx.beginPath(); ctx.arc(bCx, bCy, r, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
          ctx.fillStyle = "rgba(255,255,255,0.85)";
          ctx.font = `bold ${Math.max(8, r * 0.8)}px monospace`;
          ctx.textAlign = "center";
          ctx.fillText(`${Math.round(p.score * 100)}%`, bCx, bCy - r - 3);
          ctx.textAlign = "left";
        }
      }
    }
    rafRef.current = requestAnimationFrame(drawLoop);
  }, [showTracking, runDetection]);

  useEffect(() => {
    rafRef.current = requestAnimationFrame(drawLoop);
    return () => cancelAnimationFrame(rafRef.current);
  }, [drawLoop]);

  // ── Event toast + time sync ────────────────────────────────────────────────
  const handleTimeUpdate = useCallback(() => {
    const v = vidRef.current;
    if (!v) return;
    const t = v.currentTime;
    setCurrent(t);
    onTimeUpdate?.(t);
    for (const ev of events) {
      if (!seenEvents.current.has(ev.id) &&
          ev.timestamp >= t - 0.8 && ev.timestamp <= t + 0.8) {
        seenEvents.current.add(ev.id);
        setToast(ev.commentary ?? `${ev.type} @ ${fmt(ev.timestamp)}`);
        if (toastTimer.current) clearTimeout(toastTimer.current);
        toastTimer.current = setTimeout(() => setToast(null), 3500);
        if (ev.commentary) speak(ev.commentary);
      }
    }
  }, [events, onTimeUpdate]);

  // ── Keyboard shortcuts ──────────────────────────────────────────────────────
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const v = vidRef.current;
      if (!v || (e.target as HTMLElement).tagName === "INPUT") return;
      if (e.key === " ")        { e.preventDefault(); togglePlay(); }
      if (e.key === "ArrowRight") v.currentTime += 5;
      if (e.key === "ArrowLeft")  v.currentTime -= 5;
      if (e.key === "m")        { toggleMute(); }
      if (e.key === "f")        { toggleFullscreen(); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  // ── Fullscreen change listener ───────────────────────────────────────────────
  useEffect(() => {
    const onFS = () => setFullscreen(!!document.fullscreenElement);
    document.addEventListener("fullscreenchange", onFS);
    return () => document.removeEventListener("fullscreenchange", onFS);
  }, []);

  // ── Controls ─────────────────────────────────────────────────────────────────
  const togglePlay = () => {
    const v = vidRef.current;
    if (!v) return;
    v.paused ? v.play() : v.pause();
  };
  const toggleMute = () => {
    const v = vidRef.current;
    if (!v) return;
    v.muted = !v.muted;
    setMuted(v.muted);
  };
  const toggleFullscreen = () => {
    const el = wrapRef.current;
    if (!el) return;
    if (!document.fullscreenElement) el.requestFullscreen?.();
    else document.exitFullscreen?.();
  };

  const onVolumeChange = (val: number) => {
    const v = vidRef.current;
    if (!v) return;
    v.volume = val;
    setVolume(val);
    if (val > 0 && v.muted) { v.muted = false; setMuted(false); }
  };

  const setPlaybackRate = (r: number) => {
    const v = vidRef.current;
    if (!v) return;
    v.playbackRate = r;
    setSpeed(r);
    setShowSpeed(false);
  };

  // ── Seek bar click / drag ────────────────────────────────────────────────────
  const seekTo = useCallback((t: number) => {
    const v = vidRef.current;
    if (!v) return;
    v.currentTime = Math.max(0, Math.min(t, duration));
  }, [duration]);

  // Expose seekTo to parent via ref
  useEffect(() => {
    if (seekFnRef) seekFnRef.current = seekTo;
  }, [seekTo, seekFnRef]);

  const onSeekClick = (e: RMouseEvent<HTMLDivElement>) => {
    const rect = (e.currentTarget as HTMLDivElement).getBoundingClientRect();
    const frac = (e.clientX - rect.left) / rect.width;
    seekTo(frac * duration);
  };

  // ── Play highlight (seek + speak commentary) ─────────────────────────────────
  const playHighlight = (h: VPHighlight) => {
    stopSpeaking();
    seekTo(h.startTime);
    vidRef.current?.play();
    if (h.commentary) {
      // Slight delay so speech doesn't overlap with seek sound
      setTimeout(() => speak(h.commentary!), 400);
    }
  };

  const progressPct = duration > 0 ? (current / duration) * 100 : 0;

  return (
    <div
      ref={wrapRef}
      className="flex flex-col bg-black rounded-xl overflow-hidden select-none focus:outline-none"
      tabIndex={0}
    >
      {/* ── Video + Canvas Overlay ─────────────────────────────────────────── */}
      <div className="relative group bg-black">
        <video
          ref={vidRef}
          src={src}
          crossOrigin="anonymous"
          className="w-full aspect-video cursor-pointer"
          onClick={togglePlay}
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onLoadedMetadata={() => setDuration(vidRef.current?.duration ?? 0)}
          onTimeUpdate={handleTimeUpdate}
          preload="metadata"
        />

        {/* Canvas overlay for tracking */}
        <canvas
          ref={canvasRef}
          className="absolute inset-0 pointer-events-none"
          style={{ width: "100%", height: "100%" }}
        />

        {/* Big play/pause overlay (fades when playing) */}
        <div
          onClick={togglePlay}
          className={`absolute inset-0 flex items-center justify-center transition-opacity duration-300 cursor-pointer
            ${playing ? "opacity-0 group-hover:opacity-0" : "opacity-100"}`}
        >
          {!playing && (
            <div className="w-16 h-16 rounded-full bg-black/60 backdrop-blur flex items-center justify-center border border-white/20">
              <Play className="w-7 h-7 text-white ml-1" />
            </div>
          )}
        </div>

        {/* Event toast */}
        {toast && (
          <div className="absolute top-3 left-1/2 -translate-x-1/2 z-30 max-w-xs
                          bg-black/80 backdrop-blur border border-emerald-500/40 text-white
                          text-xs px-3 py-2 rounded-xl shadow-xl text-center animate-in fade-in">
            💬 {toast}
          </div>
        )}

        {/* AI tracking badge + toggle */}
        <div className="absolute top-2 right-2 z-20 flex flex-col items-end gap-1.5">
          <div className={`flex items-center gap-1 text-[10px] px-2 py-1 rounded-lg backdrop-blur-sm
            ${modelState === "ready"   ? "bg-emerald-900/70 border border-emerald-500/40 text-emerald-300"
            : modelState === "loading" ? "bg-black/60 border border-white/10 text-zinc-400 animate-pulse"
            : "bg-red-900/70 border border-red-500/40 text-red-300"}`}
          >
            <Cpu className="w-3 h-3" />
            {modelState === "ready" ? "AI Live" : modelState === "loading" ? "Loading AI…" : "AI Error"}
          </div>
          <button
            onClick={() => setShowTracking(v => !v)}
            className="bg-black/60 hover:bg-black/80 border border-white/10 text-[10px] px-2 py-1
                       rounded-lg transition-all backdrop-blur-sm flex items-center gap-1 text-white/70"
          >
            <span className={`w-1.5 h-1.5 rounded-full ${showTracking ? "bg-emerald-400" : "bg-zinc-600"}`} />
            {showTracking ? "Hide Tracking" : "Show Tracking"}
          </button>
          {sampleCount >= 40 && (
            <div className="flex gap-1.5 items-center bg-black/60 border border-white/10 px-2 py-1 rounded-lg backdrop-blur-sm">
              {teamCols.current.map((col, i) => (
                <span
                  key={i}
                  title={`Team ${i + 1}: rgb(${col.join(",")})`}
                  style={{ background: `rgb(${col.join(",")})` }}
                  className="w-3 h-3 rounded-full border border-white/30"
                />
              ))}
            </div>
          )}
        </div>
      </div>

      {/* ── Custom Controls ────────────────────────────────────────────────── */}
      <div className="bg-zinc-950 border-t border-zinc-800 px-3 pt-2 pb-2.5 space-y-2">

        {/* ── Seek bar + event markers ───────────────────────────────────────*/}
        <div
          ref={seekRef}
          className="relative h-7 flex items-center cursor-pointer group/seek"
          onClick={onSeekClick}
        >
          {/* Track */}
          <div className="absolute inset-x-0 top-1/2 -translate-y-1/2 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-emerald-600 to-emerald-400 rounded-full transition-none"
              style={{ width: `${progressPct}%` }}
            />
          </div>

          {/* Thumb */}
          <div
            className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-3.5 h-3.5 rounded-full
                       bg-white shadow-lg transition-transform group-hover/seek:scale-125 z-10"
            style={{ left: `${progressPct}%` }}
          />

          {/* Event markers */}
          {duration > 0 && events.map((ev) => {
            const pct = (ev.timestamp / duration) * 100;
            const col = EVT_COLORS[ev.type] ?? "#71717a";
            return (
              <button
                key={ev.id}
                title={`${ev.type} ${fmt(ev.timestamp)} — score ${ev.finalScore.toFixed(1)}`}
                onClick={(e) => { e.stopPropagation(); seekTo(ev.timestamp); }}
                className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 z-20
                           w-2 h-2 rounded-full border border-black hover:scale-150 transition-transform"
                style={{ left: `${pct}%`, background: col }}
              />
            );
          })}

          {/* Highlight regions */}
          {duration > 0 && highlights.map((h) => (
            <div
              key={h.id}
              className="absolute top-1/2 -translate-y-1/2 h-1.5 bg-amber-400/35 rounded-sm pointer-events-none z-5"
              style={{
                left:  `${(h.startTime / duration) * 100}%`,
                width: `${((h.endTime - h.startTime) / duration) * 100}%`,
              }}
            />
          ))}
        </div>

        {/* ── Time + Controls row ────────────────────────────────────────────*/}
        <div className="flex items-center gap-2">
          {/* Skip back 10s */}
          <button
            onClick={() => seekTo(current - 10)}
            className="text-zinc-400 hover:text-white transition-colors"
            title="Back 10s (←)"
          >
            <SkipBack className="w-4 h-4" />
          </button>

          {/* Play/Pause */}
          <button
            onClick={togglePlay}
            className="w-8 h-8 rounded-full bg-white text-black flex items-center justify-center
                       hover:bg-zinc-200 transition-colors shrink-0"
          >
            {playing
              ? <Pause className="w-4 h-4" />
              : <Play  className="w-4 h-4 ml-0.5" />
            }
          </button>

          {/* Skip forward 10s */}
          <button
            onClick={() => seekTo(current + 10)}
            className="text-zinc-400 hover:text-white transition-colors"
            title="Forward 10s (→)"
          >
            <SkipForward className="w-4 h-4" />
          </button>

          {/* Time display */}
          <span className="text-xs font-mono text-zinc-400 whitespace-nowrap">
            {fmt(current)} / {fmt(duration)}
          </span>

          <div className="flex-1" />

          {/* Volume */}
          <button onClick={toggleMute} className="text-zinc-400 hover:text-white transition-colors">
            {muted || volume === 0
              ? <VolumeX className="w-4 h-4" />
              : volume < 0.5
              ? <Volume1 className="w-4 h-4" />
              : <Volume2 className="w-4 h-4" />
            }
          </button>
          <input
            type="range" min={0} max={1} step={0.05} value={muted ? 0 : volume}
            onChange={(e) => onVolumeChange(Number(e.target.value))}
            className="w-16 h-1 accent-emerald-500 cursor-pointer"
          />

          {/* Speed popup */}
          <div className="relative">
            <button
              onClick={() => setShowSpeed(v => !v)}
              className="flex items-center gap-0.5 text-xs text-zinc-400 hover:text-white
                         border border-zinc-800 hover:border-zinc-600 px-2 py-0.5 rounded transition-all"
              title="Playback speed"
            >
              <Gauge className="w-3 h-3" /> {speed}x
            </button>
            {showSpeed && (
              <div className="absolute bottom-8 right-0 bg-zinc-900 border border-zinc-700
                              rounded-lg overflow-hidden shadow-xl z-30 min-w-[70px]">
                {[0.5, 0.75, 1, 1.25, 1.5, 2].map((r) => (
                  <button
                    key={r}
                    onClick={() => setPlaybackRate(r)}
                    className={`w-full text-left px-3 py-1.5 text-xs transition-colors
                      ${speed === r
                        ? "bg-emerald-500/20 text-emerald-400"
                        : "text-zinc-400 hover:bg-zinc-800 hover:text-white"}`}
                  >
                    {r}x
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Fullscreen */}
          <button
            onClick={toggleFullscreen}
            className="text-zinc-400 hover:text-white transition-colors"
            title="Fullscreen (F)"
          >
            {fullscreen
              ? <Minimize className="w-4 h-4" />
              : <Maximize className="w-4 h-4" />
            }
          </button>
        </div>
      </div>

      {/* ── Highlight Clips Strip ──────────────────────────────────────────── */}
      {highlights.length > 0 && (
        <div className="bg-zinc-950 border-t border-zinc-800 px-3 py-2">
          <div className="flex items-center gap-1.5 mb-2">
            <Film className="w-3.5 h-3.5 text-amber-400" />
            <span className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wide">
              Highlights
            </span>
          </div>
          <div className="flex gap-2 overflow-x-auto pb-1 scrollbar-thin">
            {highlights.map((h, i) => {
              const col = EVT_COLORS[h.eventType ?? ""] ?? "#71717a";
              return (
                <button
                  key={h.id}
                  onClick={() => playHighlight(h)}
                  title={h.commentary ?? h.eventType ?? "Highlight"}
                  className="shrink-0 flex flex-col gap-0.5 bg-zinc-900 hover:bg-zinc-800
                             border border-zinc-800 hover:border-zinc-600 rounded-lg
                             px-3 py-2 text-left transition-all group/hl min-w-[90px]"
                >
                  <div className="flex items-center gap-1.5">
                    <span
                      className="w-2 h-2 rounded-full shrink-0"
                      style={{ background: col }}
                    />
                    <span className="text-[10px] font-bold text-zinc-300">
                      #{i + 1}
                    </span>
                  </div>
                  <span className="font-mono text-[10px] text-zinc-500">
                    {fmt(h.startTime)} – {fmt(h.endTime)}
                  </span>
                  <div className="flex items-center gap-1 mt-0.5">
                    <Play className="w-2.5 h-2.5 text-emerald-400" />
                    {h.commentary && (
                      <span title="Has commentary (will be spoken)">
                        <MessageSquare className="w-2.5 h-2.5 text-blue-400" />
                      </span>
                    )}
                    <span className="ml-auto text-[9px] font-mono text-zinc-600">
                      {h.score.toFixed(1)}
                    </span>
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
