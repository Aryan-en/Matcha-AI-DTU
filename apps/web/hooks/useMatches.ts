import { useState, useEffect, useRef, useCallback } from "react";
import { io, Socket } from "socket.io-client";

export interface MatchSummary {
  id: string;
  status: string;
  progress: number;
  duration: number | null;
  createdAt: string;
  uploadUrl: string;
  _count: { events: number; highlights: number };
}

export interface ProgressMap {
  [matchId: string]: number;
}

export function useMatches(url: string = "http://localhost:4000") {
  const [matches, setMatches] = useState<MatchSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [progressMap, setProgressMap] = useState<ProgressMap>({});
  
  const socketRef = useRef<Socket | null>(null);

  const fetchMatches = useCallback(async () => {
    try {
      const res = await fetch(`${url}/matches`);
      if (res.ok) {
        const data: MatchSummary[] = await res.json();
        setMatches(data);
        const initialProgress: ProgressMap = {};
        data.forEach(m => {
          if (m.progress > 0 && m.status !== "COMPLETED") {
            initialProgress[m.id] = m.progress;
          }
        });
        setProgressMap(prev => ({ ...initialProgress, ...prev }));
      }
    } catch { 
      console.warn("Match Orchestrator offline"); 
    } finally { 
      setLoading(false); 
    }
  }, [url]);

  useEffect(() => {
    socketRef.current = io(url);
    
    socketRef.current.on("progress", (data: { matchId: string; progress: number }) => {
      setProgressMap(prev => ({ ...prev, [data.matchId]: data.progress }));
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
  }, [url, fetchMatches]);

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
  }, [url, fetchMatches]);

  const deleteMatch = async (id: string) => {
    try {
      await fetch(`${url}/matches/${id}`, { method: "DELETE" });
      setMatches(prev => prev.filter(m => m.id !== id));
      return true;
    } catch {
      return false;
    }
  };

  return { matches, loading, progressMap, deleteMatch, refetch: fetchMatches };
}
