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
    <div className="w-full max-w-xl mx-auto">
      <div
        {...getRootProps()}
        className={cn(
          "relative group border-2 border-dashed p-12 transition-all duration-200 cursor-pointer bg-[#0f0f0f]",
          isDragActive
            ? "border-emerald-500 bg-emerald-500/5"
            : "border-zinc-700 hover:border-emerald-500/50 hover:bg-zinc-900/50",
          file && "border-solid border-emerald-500/30 bg-emerald-500/5"
        )}
      >
        <input {...getInputProps()} />
        
        <div className="flex flex-col items-center justify-center text-center space-y-4">
          <div className="relative">
            <div className={cn(
              "p-4 bg-zinc-800 transition-transform duration-200",
              isDragActive && "bg-emerald-900/30"
            )}>
              {file ? (
                <FileVideo className="w-8 h-8 text-emerald-400" />
              ) : (
                <Upload className="w-8 h-8 text-zinc-500 group-hover:text-emerald-500 transition-colors" />
              )}
            </div>
            {status === "success" && (
              <div className="absolute -right-1 -top-1 bg-emerald-500 p-1">
                <CheckCircle2 className="w-4 h-4 text-black" />
              </div>
            )}
            {status === "error" && (
              <div className="absolute -right-1 -top-1 bg-red-500 p-1">
                <AlertCircle className="w-4 h-4 text-black" />
              </div>
            )}
             {status === "processing" && (
              <div className="absolute -right-1 -top-1 bg-blue-500 p-1">
                <Loader2 className="w-4 h-4 text-black animate-spin" />
              </div>
            )}
          </div>

          <div className="space-y-2">
            <h3 className="text-lg font-semibold text-zinc-100 font-heading">
              {file ? file.name : "Upload Match Footage"}
            </h3>
            <p className="text-sm text-zinc-500 max-w-xs mx-auto">
              {file
                ? `${(file.size / (1024 * 1024)).toFixed(2)} MB`
                : "Drag & drop your video here, or click to select"}
            </p>
          </div>

          {file && status === 'idle' && (
            <div className="flex gap-3 pt-4">
              <button
                onClick={(e) => { e.stopPropagation(); uploadFile(); }}
                className="px-6 py-2.5 bg-emerald-500 hover:bg-emerald-400 text-black font-semibold transition-all uppercase tracking-wide text-sm"
              >
                Start Upload
              </button>
              <button
                onClick={removeFile}
                className="p-2.5 bg-zinc-800 hover:bg-red-500/20 text-zinc-400 hover:text-red-400 transition-colors border border-zinc-700 hover:border-red-500/50"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
          )}

          {status === "uploading" && (
            <div className="w-full max-w-xs space-y-2 pt-4">
              <div className="h-1 bg-zinc-800 overflow-hidden">
                <div
                  className="h-full bg-emerald-500 transition-all duration-500 ease-out"
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
              <p className="text-xs text-zinc-400 animate-pulse uppercase tracking-wide">
                Uploading... {uploadProgress}%
              </p>
            </div>
          )}

          {status === "processing" && (
             <div className="w-full max-w-xs space-y-2 pt-4">
             <div className="h-1 bg-zinc-800 overflow-hidden">
               <div
                 className="h-full bg-blue-500 transition-all duration-300 ease-out"
                 style={{ width: `${processingProgress}%` }}
               />
             </div>
             <p className="text-xs text-blue-400 animate-pulse font-medium uppercase tracking-wide">
               Analyzing Match... {processingProgress}%
             </p>
           </div>
          )}

          {status === "success" && (
            <div className="pt-4 flex flex-col items-center gap-4">
              <p className="text-emerald-400 font-semibold text-lg font-heading uppercase tracking-wide">Analysis Complete!</p>
              {matchId && (
                <button
                  onClick={() => router.push(`/matches/${matchId}`)}
                  className="flex items-center gap-2 px-6 py-2.5 bg-emerald-500 hover:bg-emerald-400 text-black font-semibold transition-all text-sm uppercase tracking-wide"
                >
                  View Results <ArrowRight className="w-4 h-4" />
                </button>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
