import { io } from "socket.io-client";
import { WsEvents } from "./constants";
/**
 * Connects to the orchestrator, joins a match room, and subscribes to real-time events.
 * Returns a cleanup function — call it on unmount to disconnect.
 */
export function createMatchSocket(url, matchId, handlers) {
    const socket = io(url, { transports: ["websocket"] });
    socket.emit(WsEvents.JOIN_MATCH, matchId);
    socket.on(WsEvents.MATCH_EVENT, (payload) => {
        var _a;
        if (payload.matchId !== matchId)
            return;
        (_a = handlers.onEvent) === null || _a === void 0 ? void 0 : _a.call(handlers, payload.event);
    });
    socket.on(WsEvents.PROGRESS, (data) => {
        var _a;
        (_a = handlers.onProgress) === null || _a === void 0 ? void 0 : _a.call(handlers, data.matchId, data.progress);
    });
    socket.on(WsEvents.COMPLETE, (data) => {
        var _a;
        (_a = handlers.onComplete) === null || _a === void 0 ? void 0 : _a.call(handlers, data.matchId);
    });
    return () => socket.disconnect();
}
