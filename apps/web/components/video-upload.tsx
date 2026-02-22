"use client";

import React, { useState, useCallback, useEffect } from "react";
import { useRouter } from "next/navigation";
import { useDropzone } from "react-dropzone";
import { Upload, FileVideo, X, CheckCircle2, AlertCircle, Loader2, ArrowRight } from "lucide-react";
import { cn } from "@/lib/utils";
import { io, Socket } from "socket.io-client";
import { createApiClient, WsEvents, isYoutubeUrl, extractYoutubeId } from "@matcha/shared";

const api = createApiClient("http://localhost:4000/api/v1");

export const VideoUpload = React.memo(function VideoUploadContent() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [youtubeUrl, setYoutubeUrl] = useState("");
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [processingProgress, setProcessingProgress] = useState(0);
  const [status, setStatus] = useState<"idle" | "uploading" | "processing" | "success" | "error">("idle");
  const [matchId, setMatchId] = useState<string | null>(null);
  const [socket, setSocket] = useState<Socket | null>(null);

  useEffect(() => {
    const newSocket = io("http://localhost:4000");
    setSocket(newSocket);

    return () => {
      newSocket.disconnect();
    };
  }, []);

  useEffect(() => {
    if (socket && matchId) {
      socket.emit(WsEvents.JOIN_MATCH, matchId);

      socket.on(WsEvents.PROGRESS, (data: { progress: number }) => {
        if (data.progress === -1) {
          setStatus("error");
          return;
        }
        setProcessingProgress(Math.min(data.progress, 99)); // hold at 99 until complete event
        if (data.progress >= 100) {
          setProcessingProgress(100);
          setStatus("success");
        }
      });

      socket.on(WsEvents.COMPLETE, () => {
        setProcessingProgress(100);
        setStatus("success");
        // Auto-navigate to match detail after a short pause
        setTimeout(() => {
          if (matchId) router.push(`/matches/${matchId}`);
        }, 1500);
      });

      return () => {
        socket.off("progress");
        socket.off("complete");
      };
    }
  }, [socket, matchId]);

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
      const data = await api.uploadYoutube(youtubeUrl);
      setMatchId(data.id);
      setStatus("processing");
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
    setStatus("idle");
    setUploadProgress(0);
    setProcessingProgress(0);
  };

  return (
    <div className="w-full flex flex-col gap-6">
      {/* YouTube URL Input Area */}
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
          disabled={!isYoutubeUrl(youtubeUrl) || status !== "idle" || file !== null || uploading}
          className="font-mono text-[10px] uppercase tracking-widest px-6 py-3 transition-colors duration-200 hover:opacity-90 focus:outline-none focus-visible:ring-2 focus-visible:ring-primary cursor-pointer bg-primary text-[#07080F] font-medium disabled:opacity-50 disabled:cursor-not-allowed whitespace-nowrap"
        >
          ANALYSE URL
        </button>
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
                className="font-mono text-[10px] uppercase tracking-widest px-7 py-2.5 transition-colors duration-200 hover:opacity-90 focus:outline-none focus-visible:ring-2 focus-visible:ring-primary cursor-pointer bg-primary text-[#07080F] font-medium"
                aria-label="Analyze Match"
              >
                ▸ ANALYSE MATCH
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
            <div className="w-full max-w-[300px]">
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
            <div className="w-full max-w-[300px]">
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
