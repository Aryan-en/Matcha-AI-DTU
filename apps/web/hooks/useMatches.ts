import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { io, Socket } from "socket.io-client";
import type { MatchSummary, ProgressMap } from "@matcha/shared";
import { createApiClient } from "@matcha/shared";

export function useMatches(baseUrl: string = "http://localhost:4000") {
  const [matches, setMatches] = useState<MatchSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [progressMap, setProgressMap] = useState<ProgressMap>({});
  
  const socketRef = useRef<Socket | null>(null);
  const client = useMemo(() => createApiClient(baseUrl), [baseUrl]);

  const fetchMatches = useCallback(async () => {
    try {
      const data = await client.getMatches();
      setMatches(data);
      const initialProgress: ProgressMap = {};
      data.forEach(m => {
        if (m.progress > 0 && m.status !== "COMPLETED") {
          initialProgress[m.id] = m.progress;
        }
      });
      setProgressMap((prev: ProgressMap) => ({ ...initialProgress, ...prev }));
    } catch { 
      console.warn("Match Orchestrator offline"); 
    } finally { 
      setLoading(false); 
    }
  }, [client]);

  useEffect(() => {
    socketRef.current = io(baseUrl);
    
    socketRef.current.on("progress", (data: { matchId: string; progress: number }) => {
      setProgressMap((prev: ProgressMap) => ({ ...prev, [data.matchId]: data.progress }));
      if (data.progress >= 100) {
        setTimeout(fetchMatches, 500);
      }
    });

    socketRef.current.on("complete", (data: { matchId: string }) => {
      setProgressMap((prev: ProgressMap) => ({ ...prev, [data.matchId]: 100 }));
      fetchMatches();
    });

    return () => {
      socketRef.current?.disconnect();
    };
  }, [baseUrl, fetchMatches]);

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
  }, [baseUrl, fetchMatches]);

  const deleteMatch = async (id: string) => {
    try {
      await client.deleteMatch(id);
      setMatches(prev => prev.filter(m => m.id !== id));
      return true;
    } catch {
      return false;
    }
  };

  return { matches, loading, progressMap, deleteMatch, refetch: fetchMatches };
}
