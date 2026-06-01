import { useEffect, useRef, useCallback } from "react";
import { io } from "socket.io-client";

const EMOTION_URL =
    process.env.REACT_APP_EMOTION_SOCKET_URL ||
    process.env.REACT_APP_SERVER_URL ||
    "http://localhost:8000";

const VALID_EMOTIONS = new Set([
    "angry", "fearful", "disgust", "happy", "sad", "neutral/calm", "neutral",
]);

export default function useEmotionSocket({ setEmotionsMap, updateParticipantMediaState }) {
    const poolRef = useRef(new Map());
    const setEmotionsMapRef = useRef(setEmotionsMap);
    const serverCapsRef = useRef({ targetFps: 5, suggestedFps: null, modalityStaleSec: 3 });

    useEffect(() => {
        setEmotionsMapRef.current = setEmotionsMap;
    }, [setEmotionsMap]);

    const handleEmotion = useCallback((payload) => {
        try {
            const participantId =
                payload?.participantId ||
                payload?.participant_id ||
                payload?.from ||
                payload?.userId;
            if (!participantId) return;

            const result = payload?.result;
            if (!result) return;

            const labelRaw = result?.emotion || result?.label || result?.top;
            if (!labelRaw) return;

            const label = String(labelRaw).toLowerCase().trim();
            if (!VALID_EMOTIONS.has(label)) return;

            const scoreRaw = result?.confidence ?? result?.score ?? result?.probability;
            const score = typeof scoreRaw === "number" ? scoreRaw : Number(scoreRaw) || 0;
            if (score < 0.05) return;

            setEmotionsMapRef.current((prev) => {
                const existing = prev[participantId] || [];
                return {
                    ...prev,
                    [participantId]: [
                        ...existing,
                        {
                            label,
                            score,
                            ts: Date.now(),
                            modality: result?.modality ?? null,
                            anomaly: result?.anomaly ?? false,
                        },
                    ].slice(-20),
                };
            });
        } catch (err) {
            console.error("[EmotionSocket] parse error:", err);
        }
    }, []);

    const _createSocket = useCallback((participantId) => {
        const socket = io(EMOTION_URL, {
            path: "/socket.io",
            transports: ["websocket"],
            timeout: 20000,
            reconnection: true,
            reconnectionAttempts: 5,
            reconnectionDelay: 2000,
            reconnectionDelayMax: 10000,
            auth: { participantId },
        });

        socket.on("connect", () => {
            const entry = poolRef.current.get(participantId);
            if (entry) entry.connected = true;
        });

        socket.on("connect_error", (err) => {
            const entry = poolRef.current.get(participantId);
            if (entry) entry.connected = false;
            console.warn(`[EmotionSocket] connect_error pid=${participantId}: ${err.message}`);
        });

        socket.on("disconnect", (reason) => {
            const entry = poolRef.current.get(participantId);
            if (entry) entry.connected = false;
        });

        socket.on("server.status", (payload) => {
            try {
                const fps = Number(payload?.targetFps);
                const staleSec = Number(payload?.modalityStaleSec);
                if (fps > 0) {
                    serverCapsRef.current.targetFps = fps;
                    serverCapsRef.current.suggestedFps = null;
                }
                if (staleSec > 0) serverCapsRef.current.modalityStaleSec = staleSec;
            } catch { }
        });

        socket.on("backpressure", (payload) => {
            try {
                const suggested = Number(payload?.suggestedFps);
                if (suggested > 0) serverCapsRef.current.suggestedFps = suggested;
            } catch { }
        });

        socket.on("emotion.error", (payload) => {
            console.warn(`[EmotionSocket] emotion.error pid=${participantId}:`, payload?.code);
        });

        socket.on("emotion.result", handleEmotion);

        return socket;
    }, [handleEmotion]);

    const ensureSocket = useCallback((participantId) => {
        if (!participantId) return null;
        if (poolRef.current.has(participantId)) {
            return poolRef.current.get(participantId).socket;
        }
        const socket = _createSocket(participantId);
        poolRef.current.set(participantId, { socket, connected: false });
        return socket;
    }, [_createSocket]);

    const getSocketForParticipant = useCallback((participantId) => {
        return poolRef.current.get(participantId)?.socket ?? null;
    }, []);

    const releaseSocket = useCallback((participantId) => {
        const entry = poolRef.current.get(participantId);
        if (!entry) return;
        try {
            entry.socket.off("emotion.result", handleEmotion);
            entry.socket.disconnect();
        } catch { }
        poolRef.current.delete(participantId);
    }, [handleEmotion]);

    const notifyMediaState = useCallback((participantId, { micEnabled, cameraEnabled }) => {
        if (!participantId) return;

        let socket = poolRef.current.get(participantId)?.socket ?? null;

        if (!socket?.connected) {
            for (const [, entry] of poolRef.current) {
                if (entry?.socket?.connected) {
                    socket = entry.socket;
                    break;
                }
            }
        }

        if (!socket?.connected) return;

        socket.emit("participant.media_state", {
            participantId,
            micEnabled: Boolean(micEnabled),
            cameraEnabled: Boolean(cameraEnabled),
        });

        updateParticipantMediaState?.(participantId, { micEnabled, cameraEnabled });
    }, [updateParticipantMediaState]);

    useEffect(() => {
        const pool = poolRef.current;
        return () => {
            for (const [pid] of pool) {
                releaseSocket(pid);
            }
        };
    }, [releaseSocket]);

    return {
        ensureSocket,
        getSocketForParticipant,
        releaseSocket,
        notifyMediaState,
        serverCapsRef,
        poolRef,
    };
}