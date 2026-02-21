import type { MatchEvent, EmotionScore } from "./types";

/** Sort events by finalScore descending, return top N. */
export function getTop5Moments(events: MatchEvent[], n = 5): MatchEvent[] {
  return [...events].sort((a, b) => b.finalScore - a.finalScore).slice(0, n);
}

/** Count events by type — { GOAL: 2, TACKLE: 5, ... } */
export function countEventsByType(events: MatchEvent[]): Record<string, number> {
  return events.reduce<Record<string, number>>((acc, e) => {
    acc[e.type] = (acc[e.type] ?? 0) + 1;
    return acc;
  }, {});
}

/** Filter events by type string, or return all if type is "ALL". */
export function filterEventsByType(events: MatchEvent[], type: string): MatchEvent[] {
  return type === "ALL" ? events : events.filter((e) => e.type === type);
}

/** Find the motion score nearest to currentTime in the emotion score array. */
export function getLiveIntensity(scores: EmotionScore[], currentTime: number): number {
  if (!scores.length) return 0;
  const nearest = scores.reduce((prev, cur) =>
    Math.abs(cur.timestamp - currentTime) < Math.abs(prev.timestamp - currentTime) ? cur : prev
  );
  return nearest.motionScore;
}

/** Average confidence of all events, 0 if none. */
export function avgConfidence(events: MatchEvent[]): number {
  if (!events.length) return 0;
  return events.reduce((s, e) => s + e.confidence, 0) / events.length;
}

/** Highest finalScore across all events, 0 if none. */
export function maxScore(events: MatchEvent[]): number {
  if (!events.length) return 0;
  return Math.max(...events.map((e) => e.finalScore));
}
