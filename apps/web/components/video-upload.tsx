"use client";

import React, { useState, useCallback, useEffect } from "react";
import { useRouter } from "next/navigation";
import { useDropzone } from "react-dropzone";
import { Upload, FileVideo, X, CheckCircle2, AlertCircle, Loader2, ArrowRight } from "lucide-react";
import { cn } from "@/lib/utils";
import { io, Socket } from "socket.io-client";
import { createApiClient, WsEvents, isYoutubeUrl, extractYoutubeId } from "@matcha/shared";

const ORCHESTRATOR_URL = process.env.NEXT_PUBLIC_ORCHESTRATOR_URL ?? "http://localhost:4000";
const api = createApiClient(ORCHESTRATOR_URL);

export const VideoUpload = React.memo(function VideoUploadContent() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [youtubeUrl, setYoutubeUrl] = useState("");
  const [startTime, setStartTime] = useState("");
  const [endTime, setEndTime] = useState("");
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [processingProgress, setProcessingProgress] = useState(0);
  const [status, setStatus] = useState<"idle" | "uploading" | "processing" | "success" | "error">("idle");
  const [matchId, setMatchId] = useState<string | null>(null);
  const [socket, setSocket] = useState<Socket | null>(null);

  const parseTimeToSeconds = (timeStr: string): number | undefined => {
    if (!timeStr || !timeStr.trim()) return undefined;
    const parts = timeStr.trim().split(":").map(part => Number(part.trim()));
    if (parts.some(isNaN)) return undefined;

    if (parts.length === 3) {
      return parts[0] * 3600 + parts[1] * 60 + parts[2];
    } else if (parts.length === 2) {
      return parts[0] * 60 + parts[1];
    } else if (parts.length === 1) {
      return parts[0];
    }
    return undefined;
  };

  // Reset YouTube range when URL is cleared
  useEffect(() => {
    if (!youtubeUrl) {
      setStartTime("");
      setEndTime("");
    }
  }, [youtubeUrl]);

  useEffect(() => {
    const newSocket = io(ORCHESTRATOR_URL);
    setSocket(newSocket);

    return () => {
      newSocket.disconnect();
    };
  }, []);

  // Socket.IO live progress
  useEffect(() => {
    if (!socket || !matchId) return;

    socket.emit(WsEvents.JOIN_MATCH, matchId);

    const onProgress = (data: { progress: number }) => {
      if (data.progress === -1) {
        setStatus("error");
        return;
      }
      setProcessingProgress(Math.min(data.progress, 99));
      if (data.progress >= 100) {
        setProcessingProgress(100);
        setStatus("success");
      }
    };

    const onComplete = () => {
      setProcessingProgress(100);
      setStatus("success");
      setTimeout(() => {
        if (matchId) router.push(`/matches/${matchId}`);
      }, 1500);
    };

    socket.on(WsEvents.PROGRESS, onProgress);
    socket.on(WsEvents.COMPLETE, onComplete);

    return () => {
      socket.off(WsEvents.PROGRESS, onProgress);
      socket.off(WsEvents.COMPLETE, onComplete);
    };
  }, [socket, matchId, router]);

  // HTTP polling fallback — runs alongside the socket so progress never stays
  // stuck at 0% if the socket is slow to connect or temporarily disconnected.
  useEffect(() => {
    if (!matchId || status === "success" || status === "error" || status === "idle" || status === "uploading") return;

    const poll = async () => {
      try {
        const m = await api.getMatch(matchId);
        if (!m) return;
        if (m.status === "COMPLETED") {
          setProcessingProgress(100);
          setStatus("success");
          setTimeout(() => router.push(`/matches/${matchId}`), 1500);
        } else if (m.status === "FAILED") {
          setStatus("error");
        } else if (typeof m.progress === "number" && m.progress > 0) {
          // Only update from HTTP if we haven't already gotten a higher value via socket
          setProcessingProgress((prev) => Math.max(prev, Math.min(m.progress ?? 0, 99)));
        }
      } catch {
        // silently ignore — socket will also try
      }
    };

    poll(); // immediate first check
    const iv = setInterval(poll, 3000);
    return () => clearInterval(iv);
  }, [matchId, status, router]);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles?.length) {
      setFile(acceptedFiles[0]);
      setStatus("idle");
      setUploadProgress(0);
      setProcessingProgress(0);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "video/*": [".mp4", ".mov", ".avi", ".mkv"],
    },
    maxFiles: 1,
    multiple: false,
  });

  const uploadFile = async () => {
    if (!file) return;

    setUploading(true);

    const fileToUpload = file;

    try {
      // Skip client-side compression - FFmpeg WASM is too slow for large files
      // Server-side processing handles videos efficiently

      setStatus("uploading");
      setUploadProgress(0);

      try {
        const data = await api.uploadVideo(fileToUpload, (pct) => {
          setUploadProgress(pct);
        });

        setUploadProgress(100);
        setMatchId(data.id);
        setStatus("processing");
        // Trigger dashboard refresh
        window.dispatchEvent(new CustomEvent("matcha:refresh"));
      } catch (err) {
        throw err;
      }

    } catch (error) {
      console.error(error);
      setStatus("error");
      setUploadProgress(0);
    } finally {
      setUploading(false);
    }
  };

  const uploadYoutube = async () => {
    if (!youtubeUrl) return;

    setUploading(true);
    setStatus("uploading");
    setUploadProgress(100); // No real upload progress for URL submission

    try {
      const startSec = parseTimeToSeconds(startTime);
      const endSec = parseTimeToSeconds(endTime);

      const data = await api.uploadYoutube(youtubeUrl, startSec, endSec);
      setMatchId(data.id);
      setStatus("processing");
      // Trigger dashboard refresh
      window.dispatchEvent(new CustomEvent("matcha:refresh"));
    } catch (error) {
      console.error(error);
      setStatus("error");
      setUploadProgress(0);
    } finally {
      setUploading(false);
    }
  };

  const removeFile = (e: React.MouseEvent) => {
    e.stopPropagation();
    setFile(null);
    setYoutubeUrl("");
    setStartTime("");
    setEndTime("");
    setStatus("idle");
    setUploadProgress(0);
    setProcessingProgress(0);
  };

  const isYoutube = isYoutubeUrl(youtubeUrl);

  return (
    <div className="w-full flex flex-col gap-6">
      {/* YouTube URL Input Area */}
      <div className="flex flex-col gap-3">
        <div className="flex gap-2">
          <input
            type="text"
            placeholder="PASTE YOUTUBE URL HERE..."
            value={youtubeUrl}
            onChange={(e) => setYoutubeUrl(e.target.value)}
            disabled={status !== "idle" || file !== null}
            className="flex-1 bg-background border border-border px-4 py-3 font-mono text-[11px] uppercase tracking-widest text-foreground focus:outline-none focus:border-primary transition-colors disabled:opacity-50"
          />
          <button
            onClick={uploadYoutube}
            disabled={!isYoutube || status !== "idle" || file !== null || uploading}
            className="font-mono text-[10px] uppercase tracking-widest px-6 py-3 transition-all duration-200 hover:opacity-90 focus:outline-none focus-visible:ring-2 focus-visible:ring-primary cursor-pointer bg-primary text-[#07080F] font-medium disabled:opacity-50 disabled:cursor-not-allowed whitespace-nowrap flex items-center gap-2"
          >
            {uploading ? (
              <>
                <Loader2 className="size-3 animate-spin" />
                ANALYSE URL
              </>
            ) : (
              "ANALYSE URL"
            )}
          </button>
        </div>

        {isYoutube && status === "idle" && (
          <div className="flex gap-4 items-center animate-fade-in">
            <div className="flex-1 flex gap-2">
              <input
                type="text"
                placeholder="START (eg. 30:00)"
                value={startTime}
                onChange={(e) => setStartTime(e.target.value)}
                className="w-1/2 bg-background border border-border px-4 py-2 font-mono text-[10px] uppercase tracking-widest text-foreground focus:outline-none focus:border-primary transition-colors"
              />
              <input
                type="text"
                placeholder="END (eg. 45:00)"
                value={endTime}
                onChange={(e) => setEndTime(e.target.value)}
                className="w-1/2 bg-background border border-border px-4 py-2 font-mono text-[10px] uppercase tracking-widest text-foreground focus:outline-none focus:border-primary transition-colors"
              />
            </div>
            <span className="font-mono text-[8px] text-muted-foreground uppercase tracking-widest whitespace-nowrap">
              LEAVE BLANK FOR FULL VIDEO (MAX 3H)
            </span>
          </div>
        )}
      </div>

      <div className="flex items-center gap-4">
        <div className="flex-1 h-px bg-border/50" />
        <span className="font-mono text-[9px] text-muted-foreground uppercase tracking-widest">OR</span>
        <div className="flex-1 h-px bg-border/50" />
      </div>

      <div
        {...getRootProps()}
        className={[
          "drop-zone bracket relative p-10 transition-colors duration-200  focus:outline-none focus-visible:ring-2 focus-visible:ring-primary",
          status !== "idle" || !!youtubeUrl ? "opacity-50 cursor-not-allowed" : "cursor-pointer",
          isDragActive ? "active" : "",
          file ? "has-file" : "",
        ].join(" ")}
        aria-label="Upload tactical Match video"
      >
        <input {...getInputProps()} />

        <div className="flex flex-col items-center text-center gap-5">

          {/* Icon */}
          <div
            className={`size-14 flex items-center justify-center border transition-colors ${file ? 'border-primary bg-primary/10' : 'border-border-2 bg-muted'}`}
          >
            {status === "processing" ? (
              <svg className="animate-spin size-5.5 text-primary" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M21 12a9 9 0 11-6.219-8.56" />
              </svg>
            ) : status === "success" ? (
              <svg className="size-5.5 text-primary" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <polyline points="20 6 9 17 4 12" />
              </svg>
            ) : status === "error" ? (
              <svg className="size-5.5 text-destructive" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <circle cx="12" cy="12" r="10" /><line x1="15" y1="9" x2="9" y2="15" /><line x1="9" y1="9" x2="15" y2="15" />
              </svg>
            ) : file ? (
              <svg className="size-5.5 text-primary" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <rect x="2" y="2" width="20" height="20" /><path d="M8 10l4-4 4 4M12 6v9" /><path d="M6 18h12" />
              </svg>
            ) : (
              <svg className="size-5.5 text-muted-foreground" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
            )}
          </div>

          {/* Label */}
          <div>
            <p className={`font-display text-[28px] tracking-[0.05em] ${file ? 'text-primary' : 'text-foreground'}`}>
              {isDragActive ? "DROP TO ANALYSE" : file ? file.name.toUpperCase() : "DROP FOOTAGE HERE"}
            </p>
            <p className="font-mono mt-1 text-[9px] text-muted-foreground uppercase tracking-[0.12em]">
              {file
                ? `${(file.size / (1024 * 1024)).toFixed(2)} MB · MP4 / MOV / AVI / MKV`
                : "Click to select · MP4 · MOV · AVI · MKV"}
            </p>
          </div>

          {/* Action */}
          {file && status === "idle" && (
            <div className="flex gap-2">
              <button
                onClick={(e) => { e.stopPropagation(); uploadFile(); }}
                disabled={uploading}
                className="font-mono text-[10px] uppercase tracking-widest px-7 py-2.5 transition-all duration-200 hover:opacity-90 focus:outline-none focus-visible:ring-2 focus-visible:ring-primary cursor-pointer bg-primary text-[#07080F] font-medium flex items-center gap-2"
                aria-label="Analyze Match"
              >
                {uploading ? (
                  <>
                    <Loader2 className="size-3 animate-spin" />
                    ANALYSING...
                  </>
                ) : (
                  "▸ ANALYSE MATCH"
                )}
              </button>
              <button
                onClick={removeFile}
                className="font-mono text-[10px] uppercase tracking-widest px-3 py-2.5 border border-border-2 text-muted-foreground transition-colors duration-200 hover:bg-muted focus:outline-none focus-visible:ring-2 focus-visible:ring-border-2 cursor-pointer"
                aria-label="Remove File"
              >
                ✕
              </button>
            </div>
          )}

          {/* Upload progress */}
          {status === "uploading" && (
            <div className="w-full max-w-75">
              <div className="flex justify-between mb-2">
                <span className="font-mono text-[9px] text-primary uppercase tracking-[0.12em]">UPLOADING</span>
                <span className="font-mono text-[9px] text-primary">{uploadProgress}%</span>
              </div>
              <div className="h-0.5 bg-border overflow-hidden">
                <div className="h-full bg-primary transition-[width] duration-300 ease-out" style={{ width: `${uploadProgress}%` }} />
              </div>
            </div>
          )}

          {/* Processing progress */}
          {status === "processing" && (
            <div className="w-full max-w-75">
              <div className="flex justify-between mb-2">
                <span className="font-mono text-[9px] text-primary uppercase tracking-[0.12em] animate-blink">ANALYSING MATCH</span>
                <span className="font-mono text-[9px] text-primary">{processingProgress}%</span>
              </div>
              <div className="h-0.5 bg-border overflow-hidden">
                <div className="h-full bg-primary transition-[width] duration-300 ease-out" style={{ width: `${processingProgress}%` }} />
              </div>
            </div>
          )}

          {/* Success */}
          {status === "success" && (
            <div className="flex flex-col items-center gap-3">
              <p className="font-display text-[28px] text-primary tracking-[0.05em]">ANALYSIS COMPLETE</p>
              {matchId && (
                <button
                  onClick={() => router.push(`/matches/${matchId}`)}
                  className="font-mono text-[10px] uppercase tracking-widest px-7 py-2.5 transition-opacity hover:opacity-75 bg-primary text-[#07080F] font-medium"
                >
                  VIEW RESULTS ▸
                </button>
              )}
            </div>
          )}

          {/* Error */}
          {status === "error" && (
            <p className="font-mono text-[9px] text-destructive uppercase tracking-[0.12em]">
              ✕ UPLOAD FAILED — TRY AGAIN
            </p>
          )}
        </div>
      </div>
    </div>
  );
});
