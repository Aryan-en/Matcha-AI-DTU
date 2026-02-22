var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
/** Sort events by finalScore descending, return top N. */
export function getTop5Moments(events, n = 5) {
    return [...events].sort((a, b) => b.finalScore - a.finalScore).slice(0, n);
}
/** Count events by type — { GOAL: 2, TACKLE: 5, ... } */
export function countEventsByType(events) {
    return events.reduce((acc, e) => {
        var _a;
        acc[e.type] = ((_a = acc[e.type]) !== null && _a !== void 0 ? _a : 0) + 1;
        return acc;
    }, {});
}
/** Filter events by type string, or return all if type is "ALL". */
export function filterEventsByType(events, type) {
    return type === "ALL" ? events : events.filter((e) => e.type === type);
}
/** Find the motion score nearest to currentTime in the emotion score array. */
export function getLiveIntensity(scores, currentTime) {
    if (!scores.length)
        return 0;
    const nearest = scores.reduce((prev, cur) => Math.abs(cur.timestamp - currentTime) < Math.abs(prev.timestamp - currentTime) ? cur : prev);
    return nearest.motionScore;
}
/** Average confidence of all events, 0 if none. */
export function avgConfidence(events) {
    if (!events.length)
        return 0;
    return events.reduce((s, e) => s + e.confidence, 0) / events.length;
}
/** Highest finalScore across all events, 0 if none. */
export function maxScore(events) {
    if (!events.length)
        return 0;
    return Math.max(...events.map((e) => e.finalScore));
}
/** Format seconds → "m:ss" */
export function formatTime(secs) {
    if (secs === null || secs === undefined)
        return "—";
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return `${m}:${s.toString().padStart(2, "0")}`;
}
/** Relative time formatter: "5m ago", "just now", etc. */
export function timeAgo(iso) {
    if (!iso)
        return "";
    const diff = Date.now() - new Date(iso).getTime();
    const mins = Math.floor(diff / 60000);
    if (mins < 1)
        return "just now";
    if (mins < 60)
        return `${mins}m ago`;
    const hrs = Math.floor(mins / 60);
    if (hrs < 24)
        return `${hrs}h ago`;
    return `${Math.floor(hrs / 24)}d ago`;
}
/** Regex for all standard YouTube URL formats */
export const YOUTUBE_REGEX = /^(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})/;
/** Validate if a string is a valid YouTube URL */
export function isYoutubeUrl(url) {
    return YOUTUBE_REGEX.test(url);
}
/** Extract video ID from any YouTube URL */
export function extractYoutubeId(url) {
    const match = url.match(YOUTUBE_REGEX);
    return match ? match[1] : null;
}
/** Fetch with automatic retries and exponential backoff */
export function fetchWithRetry(url_1) {
    return __awaiter(this, arguments, void 0, function* (url, options = {}, retries = 3, backoff = 300) {
        try {
            const res = yield fetch(url, options);
            // Only retry on network errors or 5xx server errors
            if (!res.ok && res.status >= 500 && retries > 0) {
                throw new Error(`Server error: ${res.status}`);
            }
            return res;
        }
        catch (err) {
            if (retries <= 0)
                throw err;
            yield new Promise((resolve) => setTimeout(resolve, backoff));
            return fetchWithRetry(url, options, retries - 1, backoff * 2);
        }
    });
}
