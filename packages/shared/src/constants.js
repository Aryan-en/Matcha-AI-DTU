export const STATUS_CONFIG = {
    COMPLETED: { label: "Completed", theme: "success" },
    PROCESSING: { label: "Processing", theme: "info" },
    UPLOADED: { label: "Uploaded", theme: "warning" },
    FAILED: { label: "Failed", theme: "error" },
};
export const EVENT_CONFIG = {
    GOAL: { label: "Goal", theme: "success" },
    TACKLE: { label: "Tackle", theme: "warning" },
    FOUL: { label: "Foul", theme: "error" },
    SAVE: { label: "Save", theme: "info" },
    Celebrate: { label: "Celeb", theme: "accent" },
};
export const DEFAULT_EVENT_CONFIG = { label: "Event", theme: "neutral" };
/** WebSocket Event Names */
export var WsEvents;
(function (WsEvents) {
    WsEvents["JOIN_MATCH"] = "joinMatch";
    WsEvents["MATCH_EVENT"] = "matchEvent";
    WsEvents["PROGRESS"] = "progress";
    WsEvents["COMPLETE"] = "complete";
})(WsEvents || (WsEvents = {}));
