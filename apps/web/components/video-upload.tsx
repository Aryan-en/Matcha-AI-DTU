"use client";

import { useState, useCallback, useEffect } from "react";
import { useRouter } from "next/navigation";
import { useDropzone } from "react-dropzone";
import { Upload, FileVideo, X, CheckCircle2, AlertCircle, Loader2, ArrowRight } from "lucide-react";
import { cn } from "@/lib/utils";
import { io, Socket } from "socket.io-client";

export function VideoUpload() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
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
      socket.emit("joinMatch", matchId);

      socket.on("progress", (data: { progress: number }) => {
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

      socket.on("complete", () => {
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

      const formData = new FormData();
      formData.append("file", fileToUpload);

      // Simulate upload progress
      const progressInterval = setInterval(() => {
        setUploadProgress((prev) => {
          if (prev >= 90) {
            return 90;
          }
          return prev + 10;
        });
      }, 200);

      const response = await fetch("http://localhost:4000/matches/upload", {
        method: "POST",
        body: formData,
      });

      clearInterval(progressInterval);

      if (!response.ok) {
        throw new Error("Upload failed");
      }

      const data = await response.json();
      setUploadProgress(100);
      setMatchId(data.id);
      setStatus("processing"); // Switch to processing state
      
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
    <div className="w-full">
      <div
        {...getRootProps()}
        className={[
          "drop-zone bracket relative p-10 transition-all duration-200",
          isDragActive ? "active" : "",
          file ? "has-file" : "",
        ].join(" ")}
      >
        <input {...getInputProps()} />

        <div className="flex flex-col items-center text-center gap-5">

          {/* Icon */}
          <div
            className="w-14 h-14 flex items-center justify-center border transition-colors"
            style={{
              borderColor: file ? "var(--green)" : "var(--border-2)",
              background: file ? "rgba(10,232,124,0.10)" : "var(--surface-2)",
            }}
          >
            {status === "processing" ? (
              <svg className="animate-spin" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="var(--green)" strokeWidth="1.5">
                <path d="M21 12a9 9 0 11-6.219-8.56" />
              </svg>
            ) : status === "success" ? (
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="var(--green)" strokeWidth="1.5">
                <polyline points="20 6 9 17 4 12" />
              </svg>
            ) : status === "error" ? (
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="var(--red)" strokeWidth="1.5">
                <circle cx="12" cy="12" r="10" /><line x1="15" y1="9" x2="9" y2="15" /><line x1="9" y1="9" x2="15" y2="15" />
              </svg>
            ) : file ? (
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="var(--green)" strokeWidth="1.5">
                <rect x="2" y="2" width="20" height="20" /><path d="M8 10l4-4 4 4M12 6v9" /><path d="M6 18h12" />
              </svg>
            ) : (
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" style={{ color: "var(--text-dim)" }}>
                <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
            )}
          </div>

          {/* Label */}
          <div>
            <p
              className="font-display"
              style={{ fontSize: "28px", color: file ? "var(--green)" : "var(--text)", letterSpacing: "0.05em" }}
            >
              {isDragActive ? "DROP TO ANALYSE" : file ? file.name.toUpperCase() : "DROP FOOTAGE HERE"}
            </p>
            <p className="font-mono mt-1" style={{ fontSize: "9px", color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.12em" }}>
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
                className="font-mono text-[10px] uppercase tracking-widest px-7 py-2.5 transition-opacity hover:opacity-75"
                style={{ background: "var(--green)", color: "#07080F", fontWeight: 500, letterSpacing: "0.12em" }}
              >
                ▸ ANALYSE MATCH
              </button>
              <button
                onClick={removeFile}
                className="font-mono text-[10px] uppercase tracking-widest px-3 py-2.5 border transition-colors hover:opacity-75"
                style={{ borderColor: "var(--border-2)", color: "var(--text-dim)" }}
              >
                ✕
              </button>
            </div>
          )}

          {/* Upload progress */}
          {status === "uploading" && (
            <div style={{ width: "100%", maxWidth: "300px" }}>
              <div className="flex justify-between mb-2">
                <span className="font-mono" style={{ fontSize: "9px", color: "var(--green)", textTransform: "uppercase", letterSpacing: "0.12em" }}>UPLOADING</span>
                <span className="font-mono" style={{ fontSize: "9px", color: "var(--green)" }}>{uploadProgress}%</span>
              </div>
              <div className="progress-track">
                <div className="progress-fill" style={{ width: `${uploadProgress}%` }} />
              </div>
            </div>
          )}

          {/* Processing progress */}
          {status === "processing" && (
            <div style={{ width: "100%", maxWidth: "300px" }}>
              <div className="flex justify-between mb-2">
                <span className="font-mono animate-blink" style={{ fontSize: "9px", color: "var(--green)", textTransform: "uppercase", letterSpacing: "0.12em" }}>ANALYSING MATCH</span>
                <span className="font-mono" style={{ fontSize: "9px", color: "var(--green)" }}>{processingProgress}%</span>
              </div>
              <div className="progress-track">
                <div className="progress-fill progress-fill-green" style={{ width: `${processingProgress}%` }} />
              </div>
            </div>
          )}

          {/* Success */}
          {status === "success" && (
            <div className="flex flex-col items-center gap-3">
              <p className="font-display" style={{ fontSize: "28px", color: "var(--green)", letterSpacing: "0.05em" }}>ANALYSIS COMPLETE</p>
              {matchId && (
                <button
                  onClick={() => router.push(`/matches/${matchId}`)}
                  className="font-mono text-[10px] uppercase tracking-widest px-7 py-2.5 transition-opacity hover:opacity-75"
                  style={{ background: "var(--green)", color: "#07080F", fontWeight: 500, letterSpacing: "0.12em" }}
                >
                  VIEW RESULTS ▸
                </button>
              )}
            </div>
          )}

          {/* Error */}
          {status === "error" && (
            <p className="font-mono" style={{ fontSize: "9px", color: "var(--red)", textTransform: "uppercase", letterSpacing: "0.12em" }}>
              ✕ UPLOAD FAILED — TRY AGAIN
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
