"use client";

import { useState, useCallback, useEffect } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, FileVideo, X, CheckCircle2, AlertCircle, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { io, Socket } from "socket.io-client";

export function VideoUpload() {
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
        setProcessingProgress(data.progress);
        if (data.progress >= 100) {
          setStatus("success");
        }
      });

      return () => {
        socket.off("progress");
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
    setStatus("uploading");
    setUploadProgress(0);

    const formData = new FormData();
    formData.append("file", file);

    try {
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
    <div className="w-full max-w-xl mx-auto p-6">
      <div
        {...getRootProps()}
        className={cn(
          "relative group border-2 border-dashed rounded-2xl p-12 transition-all duration-300 ease-in-out cursor-pointer",
          isDragActive
            ? "border-emerald-500 bg-emerald-500/5 scale-[1.02]"
            : "border-slate-300 dark:border-slate-700 hover:border-emerald-500/50 hover:bg-slate-50 dark:hover:bg-slate-900/50",
          file && "border-solid border-emerald-500/20 bg-emerald-500/5"
        )}
      >
        <input {...getInputProps()} />
        
        <div className="flex flex-col items-center justify-center text-center space-y-4">
          <div className="relative">
            <div className={cn(
              "p-4 rounded-full bg-slate-100 dark:bg-slate-800 transition-transform duration-300 group-hover:scale-110",
              isDragActive && "bg-emerald-100 dark:bg-emerald-900/30"
            )}>
              {file ? (
                <FileVideo className="w-8 h-8 text-emerald-600 dark:text-emerald-400" />
              ) : (
                <Upload className="w-8 h-8 text-slate-400 group-hover:text-emerald-500 transition-colors" />
              )}
            </div>
            {status === "success" && (
              <div className="absolute -right-2 -top-2 bg-emerald-500 rounded-full p-1 shadow-lg animate-in zoom-in">
                <CheckCircle2 className="w-4 h-4 text-white" />
              </div>
            )}
            {status === "error" && (
              <div className="absolute -right-2 -top-2 bg-red-500 rounded-full p-1 shadow-lg animate-in zoom-in">
                <AlertCircle className="w-4 h-4 text-white" />
              </div>
            )}
             {status === "processing" && (
              <div className="absolute -right-2 -top-2 bg-blue-500 rounded-full p-1 shadow-lg animate-in zoom-in">
                <Loader2 className="w-4 h-4 text-white animate-spin" />
              </div>
            )}
          </div>

          <div className="space-y-2">
            <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
              {file ? file.name : "Upload Match Footage"}
            </h3>
            <p className="text-sm text-slate-500 dark:text-slate-400 max-w-xs mx-auto">
              {file
                ? `${(file.size / (1024 * 1024)).toFixed(2)} MB`
                : "Drag & drop your video here, or click to select"}
            </p>
          </div>

          {file && status === 'idle' && (
            <div className="flex gap-3 pt-2">
              <button
                onClick={uploadFile}
                className="px-6 py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg font-medium transition-colors shadow-lg shadow-emerald-500/20"
              >
                Start Upload
              </button>
              <button
                onClick={removeFile}
                className="p-2 hover:bg-red-100 dark:hover:bg-red-900/30 text-slate-400 hover:text-red-500 rounded-lg transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
          )}

          {status === "uploading" && (
            <div className="w-full max-w-xs space-y-2 pt-2">
              <div className="h-2 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-emerald-500 transition-all duration-500 ease-out"
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
              <p className="text-xs text-slate-400 animate-pulse">
                Uploading... {uploadProgress}%
              </p>
            </div>
          )}

          {status === "processing" && (
             <div className="w-full max-w-xs space-y-2 pt-2">
             <div className="h-2 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
               <div
                 className="h-full bg-blue-500 transition-all duration-300 ease-out"
                 style={{ width: `${processingProgress}%` }}
               />
             </div>
             <p className="text-xs text-blue-500 animate-pulse font-medium">
               Analyzing Match... {processingProgress}%
             </p>
           </div>
          )}

          {status === "success" && (
            <div className="pt-2 animate-in fade-in slide-in-from-bottom-2">
              <p className="text-emerald-600 dark:text-emerald-400 font-medium">
                Analysis Complete!
              </p>
              {matchId && <p className="text-xs text-slate-400 mt-1">ID: {matchId}</p>}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
