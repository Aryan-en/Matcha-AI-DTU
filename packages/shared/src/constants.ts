import type { MatchStatus } from "./types";

/** Status display config — same colours used in web and mobile status chips. */
export const STATUS_CONFIG: Record<MatchStatus, { label: string; color: string }> = {
  COMPLETED: { label: "Completed", color: "#34d399" },
  PROCESSING: { label: "Processing", color: "#60a5fa" },
  UPLOADED:   { label: "Uploaded",   color: "#fbbf24" },
  FAILED:     { label: "Failed",     color: "#f87171" },
};

/** Event type display config — colours, labels used in event lists and timeline. */
export const EVENT_CONFIG: Record<string, { label: string; color: string }> = {
  GOAL:      { label: "Goal",    color: "#34d399" },
  TACKLE:    { label: "Tackle",  color: "#fbbf24" },
  FOUL:      { label: "Foul",    color: "#f87171" },
  SAVE:      { label: "Save",    color: "#60a5fa" },
  Celebrate: { label: "Celeb",   color: "#c084fc" },
};

export const DEFAULT_EVENT_CONFIG = { label: "Event", color: "#71717a" };

/** Format seconds → "m:ss" */
export function formatTime(secs: number): string {
  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}
