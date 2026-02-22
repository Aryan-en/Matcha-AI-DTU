"use client";

import { useEffect, useState, useRef } from "react";
import { io, Socket } from "socket.io-client";
import { MatchEvent } from "@matcha/shared";

interface UseMatchSocketOptions {
  matchId: string;
  url: string;
  enabled?: boolean;
}

export function useMatchSocket({ matchId, url, enabled = true }: UseMatchSocketOptions) {
  const [liveEvents, setLiveEvents] = useState<MatchEvent[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [liveProgress, setLiveProgress] = useState<number | null>(null);
  const socketRef = useRef<Socket | null>(null);

  useEffect(() => {
    if (!enabled || !matchId) return;

    const socket: Socket = io(url, { 
      transports: ["websocket"],
      reconnectionAttempts: 5,
    });
    
    socketRef.current = socket;

    socket.on("connect", () => {
      setIsConnected(true);
      socket.emit("joinMatch", matchId);
    });

    socket.on("disconnect", () => {
      setIsConnected(false);
    });

    socket.on("matchEvent", (payload: { matchId: string; event: MatchEvent }) => {
      if (payload.matchId !== matchId) return;
      
      setLiveEvents(prev => {
        const exists = prev.some(
          e => e.id === payload.event.id || 
               (e.timestamp === payload.event.timestamp && e.type === payload.event.type)
        );
        if (exists) return prev;
        return [...prev, payload.event];
      });
    });

    socket.on("progress", (payload: { matchId: string; progress: number }) => {
      if (payload.matchId !== matchId) return;
      setLiveProgress(payload.progress);
    });

    socket.on("complete", (payload: { matchId: string }) => {
      if (payload.matchId !== matchId) return;
      setLiveProgress(100);
    });

    return () => {
      socket.disconnect();
      socketRef.current = null;
    };
  }, [matchId, url, enabled]);

  return {
    liveEvents,
    liveProgress,
    isConnected,
    socket: socketRef.current,
  };
}
