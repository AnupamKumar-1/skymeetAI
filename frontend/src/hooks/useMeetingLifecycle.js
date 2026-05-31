import { useEffect, useRef, useCallback } from "react";
import io from "socket.io-client";
import {
  initMediaController,
  setExternalCleaners,
  resetMediaController,
} from "../utils/mediaController";
import { notifyLocalStreamReady } from "./useWebRTC";

const _activeRooms = new Set();

export default function useMeetingLifecycle({
  roomId,
  navigate,
  socketRef,
  localStreamRef,
  localVideoRef,
  prevLocalStreamRef,
  pcsRef,
  participantsMeta,
  isHost,
  addToUserHistory,
  TRANSCRIPTS_ENABLED,
  TRANSCRIPT_ENDPOINT,
  API_BASE,
  createAnalyzerForStream,
  removeAnalyzer,
  startRecordingForStream,
  stopAllRecorders,
  uploadRecordingsAndStoreTranscript,
  stopPeriodicEmotionCapture,
  setConnecting,
  setParticipantsMeta,
  setRemoteStreams,
  recordersRef,
  makingOfferRef,
  politeRef,
  pendingCandidatesRef,
  ignoreOfferRef,
  isSettingRemoteAnswerPending,
  SOCKET_SERVER_URL,
  onSocketReady,
  onHostConfirmed,
}) {
  const participantsMetaRef = useRef(participantsMeta);
  const isHostRef = useRef(false);
  const instanceKey = useRef(`${roomId}:${Date.now()}:${Math.random()}`);

  useEffect(() => {
    participantsMetaRef.current = participantsMeta;
  }, [participantsMeta]);

  const persistHistorySnapshot = useCallback(async () => {
    if (!isHostRef.current || typeof addToUserHistory !== "function") return;
    try {
      const participantList = (participantsMetaRef.current || []).map(
        (p) =>
          p?.meta?.name ||
          p?.meta?.displayName ||
          p?.meta?.username ||
          p?.id ||
          "Guest"
      );
      await addToUserHistory({
        meetingCode: (roomId || "").toUpperCase(),
        hostName: localStorage.getItem("displayName") || "Host",
        participants: participantList,
        createdAt: new Date().toISOString(),
        link: `${window.location.origin}/room/${(roomId || "").toUpperCase()}`,
      });
    } catch { }
  }, [roomId, addToUserHistory]);

  const cleanupAll = useCallback(async () => {
    _activeRooms.delete(instanceKey.current);

    try {
      socketRef.current?.removeAllListeners?.();
      socketRef.current?.disconnect();
    } catch { }
    if (socketRef && "current" in socketRef) socketRef.current = null;

    try { window.myId = null; } catch { }

    try { localStreamRef.current?.getTracks?.().forEach((t) => t.stop()); } catch { }
    try { prevLocalStreamRef.current?.getTracks?.().forEach((t) => t.stop()); } catch { }
    try {
      const prev = window.__previousLocalStreamForToggle;
      if (prev?.getTracks) prev.getTracks().forEach((t) => t.stop());
      window.__previousLocalStreamForToggle = null;
    } catch { }

    if (localVideoRef.current) {
      try {
        localVideoRef.current.pause();
        localVideoRef.current.srcObject = null;
      } catch { }
    }

    localStreamRef.current = null;
    prevLocalStreamRef.current = null;

    try {
      Object.values(pcsRef.current || {}).forEach((pc) => {
        try {
          pc.ontrack = null;
          pc.onicecandidate = null;
          pc.onnegotiationneeded = null;
          pc.close();
        } catch { }
      });
    } catch { }

    pcsRef.current = {};

    const resetRef = (ref) => {
      if (!ref || typeof ref !== "object" || !("current" in ref)) return;
      ref.current = {};
    };

    resetRef(makingOfferRef);
    resetRef(politeRef);
    resetRef(pendingCandidatesRef);
    resetRef(ignoreOfferRef);
    resetRef(isSettingRemoteAnswerPending);
    resetRef(recordersRef);

    setRemoteStreams({});
    setParticipantsMeta([]);

    try { stopPeriodicEmotionCapture(); } catch { }

    resetMediaController();
  }, [
    socketRef,
    localStreamRef,
    prevLocalStreamRef,
    localVideoRef,
    pcsRef,
    setRemoteStreams,
    setParticipantsMeta,
    stopPeriodicEmotionCapture,
  ]);

  async function start(key) {
    if (_activeRooms.has(key)) return;
    _activeRooms.add(key);

    try {
      setConnecting(true);
      pcsRef.current = {};

      const resetRef = (ref) => {
        if (!ref || typeof ref !== "object" || !("current" in ref)) return;
        ref.current = {};
      };

      resetRef(makingOfferRef);
      resetRef(politeRef);
      resetRef(pendingCandidatesRef);
      resetRef(ignoreOfferRef);
      resetRef(isSettingRemoteAnswerPending);

      if (!_activeRooms.has(key)) return;

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: true,
        video: {
          width: 640,
          height: 360,
          frameRate: 15,
        }
      });

      if (!_activeRooms.has(key)) {
        stream.getTracks().forEach((t) => t.stop());
        return;
      }

      localStreamRef.current = stream;

      if (localVideoRef.current) {
        localVideoRef.current.autoplay = true;
        localVideoRef.current.playsInline = true;
        localVideoRef.current.muted = true;
        localVideoRef.current.srcObject = stream;
        try { await localVideoRef.current.play(); } catch { }
      }

      if (socketRef.current) {
        try {
          socketRef.current.removeAllListeners();
          socketRef.current.disconnect();
        } catch { }
        socketRef.current = null;
      }

      const socket = io(SOCKET_SERVER_URL, { autoConnect: false });
      socketRef.current = socket;

      await new Promise((resolve, reject) => {
        const TO = setTimeout(() => reject(new Error("socket timeout")), 8000);
        const cleanup = () => {
          clearTimeout(TO);
          socket.off("connect", onConnect);
          socket.off("connect_error", onError);
        };
        const onConnect = () => { cleanup(); resolve(); };
        const onError = () => { cleanup(); reject(new Error("socket connect_error")); };
        socket.on("connect", onConnect);
        socket.on("connect_error", onError);
        socket.connect();
      });

      if (!_activeRooms.has(key)) {
        socket.disconnect();
        stream.getTracks().forEach((t) => t.stop());
        return;
      }

      initMediaController(stream, socketRef.current, pcsRef.current, localVideoRef.current);

      setExternalCleaners({
        recordersRef,
        removeAnalyzerFn: removeAnalyzer,
        prevLocalStreamRef,
      });

      createAnalyzerForStream("local", stream);
      startRecordingForStream("local", stream);

      notifyLocalStreamReady(stream);

      onSocketReady?.();

      const token =
        localStorage.getItem("token") || localStorage.getItem("accessToken");

      let uid = null;
      if (token) {
        try {
          const payload = JSON.parse(atob(token.split(".")[1]));
          uid = payload._id || payload.sub;
        } catch { }
      }

      if (!uid) {
        uid = localStorage.getItem("userId");
        if (!uid) {
          uid = crypto.randomUUID();
          localStorage.setItem("userId", uid);
        }
      }

      const displayName = localStorage.getItem("displayName") || "Guest";
      const code = (roomId || "").toUpperCase();

      socketRef.current.emit("join-call", roomId, { name: displayName, userId: uid });

      const hostDataRaw = localStorage.getItem(`host:${code}`);
      const hostData = hostDataRaw ? JSON.parse(hostDataRaw) : null;

      if (hostData?.hostSecret) {
        socketRef.current.once("assigned-role", () => {
          if (socketRef.current?.connected) {
            socketRef.current.emit("declare-host", code, hostData.hostSecret, (ack) => {
              if (ack?.ok) {
                isHostRef.current = true;
                onHostConfirmed?.();
              } else {
                console.error("[declare-host] rejected by server:", ack?.reason);
                if (API_BASE) {
                  const token = localStorage.getItem("token");
                  fetch(`${API_BASE}/users/meetings/${code}/participants`, {
                    method: "POST",
                    headers: {
                      "Content-Type": "application/json",
                      ...(token ? { Authorization: `Bearer ${token}` } : {}),
                    },
                    body: JSON.stringify({ name: displayName }),
                  }).catch(() => { });
                }
              }
            });
          }
        });
      } else {
  
        if (API_BASE) {
          const token = localStorage.getItem("token");
          fetch(`${API_BASE}/users/meetings/${code}/participants`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              ...(token ? { Authorization: `Bearer ${token}` } : {}),
            },
            body: JSON.stringify({ name: displayName }),
          }).catch(() => { });
        }
      }
    } catch (err) {
      console.error(err);
      _activeRooms.delete(key);
      alert("Unable to access camera/mic or connect.");
    } finally {
      setConnecting(false);
    }
  }

  async function leaveCall() {
    try {
      if (socketRef.current?.connected) {
        socketRef.current.emit("leave-call", roomId);
        await new Promise((r) => setTimeout(r, 80));
      }
    } catch { }
    await cleanupAll();
    navigate("/home");
  }

  async function uploadTranscriptWithRetry(code, hostSecret, recordersSnapshot, participantsSnapshot, maxRetries = 3) {
    console.log("[transcript] participantsSnapshot:", JSON.stringify(participantsSnapshot));
    console.log("[transcript] recorderSnapshot keys:", Object.keys(recordersSnapshot));
    const speakerMap = {};
    const currentMeta = participantsSnapshot || [];

    currentMeta.forEach((p) => {
      const name =
        p?.meta?.name ||
        p?.meta?.displayName ||
        p?.name ||
        `Guest-${(p.id || "").slice(0, 6)}`;
      if (p.id) speakerMap[p.id] = name;
    });

    speakerMap["local"] = localStorage.getItem("displayName") || "Host";

    const fd = new FormData();
    fd.append("meeting_code", code);
    fd.append("speaker_map", JSON.stringify(speakerMap));

    let fileCount = 0;

    for (const [id, rec] of Object.entries(recordersSnapshot)) {
      const chunks = rec?.chunks;
      if (!chunks?.length) continue;

      const blob = new Blob(chunks, { type: "audio/webm" });
      fd.append("audio_files", blob, `${id}.webm`);
      fileCount++;
    }

    if (fileCount === 0) return;

    const token = localStorage.getItem("token");

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        const resp = await fetch(TRANSCRIPT_ENDPOINT, {
          method: "POST",
          headers: {
            "x-host-secret": hostSecret,
            ...(token ? { "x-user-token": token } : {}),
          },
          body: fd,
        });

        if (resp.ok) {
          const data = await resp.json();
          if (data?.success) {
            const text =
              data?.transcriptText ||
              data?.transcript ||
              data?.metadata?.transcriptText ||
              "";

            if (text.trim()) {
              try {
                const existingRaw = localStorage.getItem(`host:${code}`);
                const existing = existingRaw ? JSON.parse(existingRaw) : {};

                localStorage.setItem(
                  `host:${code}`,
                  JSON.stringify({
                    ...existing,
                    meetingCode: code,
                    lastTranscriptAt: new Date().toISOString(),
                  })
                );
              } catch { }
            }
          }
          return;
        }

        if (resp.status >= 400 && resp.status < 500) {
          console.error(`[transcript] upload failed with client error ${resp.status} – not retrying`);
          return;
        }

        console.warn(`[transcript] upload failed with status ${resp.status}, attempt ${attempt}/${maxRetries}`);
      } catch (err) {
        console.warn(`[transcript] upload network error on attempt ${attempt}/${maxRetries}: ${err.message}`);
      }

      if (attempt < maxRetries) {
        const delay = Math.pow(2, attempt - 1) * 1000;
        await new Promise((r) => setTimeout(r, delay));
      }
    }

    alert("Failed to upload meeting transcript after multiple attempts. Your audio recording may be lost.");
  }

  async function endMeeting(emotionsMap = {}) {
    if (!isHostRef.current) {
      await leaveCall();
      return;
    }

    const code = (roomId || "").toUpperCase();
    const hostDataRaw = localStorage.getItem(`host:${code}`);
    const hostData = hostDataRaw ? JSON.parse(hostDataRaw) : null;

    try {
      if (emotionsMap && Object.keys(emotionsMap).length > 0) {
        const emotionSnapshot = {};
        for (const [pid, history] of Object.entries(emotionsMap)) {
          if (Array.isArray(history) && history.length > 0) {
            emotionSnapshot[pid] = history;
          }
        }
        if (Object.keys(emotionSnapshot).length > 0) {
          localStorage.setItem(`emotions:${code}`, JSON.stringify(emotionSnapshot));
        }
      }
      const participantNames = {};
      (participantsMetaRef.current || []).forEach((p) => {
        const userId = p.meta?.userId || p.id;
        const name = p.meta?.name || p.meta?.displayName || p.meta?.username || userId;
        if (userId) participantNames[userId] = name;
      });
      participantNames["local"] = localStorage.getItem("displayName") || "Host";
      localStorage.setItem(`emotionNames:${code}`, JSON.stringify(participantNames));
    } catch { }

    const participantsSnapshot = JSON.parse(JSON.stringify(participantsMetaRef.current || []));

    try { await stopAllRecorders(); } catch { }

    const recordersSnapshot = {};
    for (const [id, rec] of Object.entries(recordersRef.current || {})) {
      recordersSnapshot[id] = {
        chunks: Array.isArray(rec?.chunks) ? [...rec.chunks] : [],
        gateState: rec?.gateState ? { ...rec.gateState } : null,
      };
    }

    try {
      if (socketRef.current?.connected) socketRef.current.emit("leave-call", roomId);
    } catch { }

    persistHistorySnapshot().catch(() => { });

    await cleanupAll();
    navigate("/home", { state: { meetingEnded: true, meetingCode: code } });

    if (TRANSCRIPTS_ENABLED && hostData?.hostSecret && TRANSCRIPT_ENDPOINT) {
      uploadTranscriptWithRetry(code, hostData.hostSecret, recordersSnapshot, participantsSnapshot).catch(() => { });
    }
  }

  useEffect(() => {
    const name =
      localStorage.getItem("displayName") ||
      prompt("Enter your display name", "Guest") ||
      "Guest";
    localStorage.setItem("displayName", name);

    const key = instanceKey.current;
    start(key);

    const onBeforeUnload = () => {
      try {
        if (isHostRef.current) persistHistorySnapshot();
        socketRef.current?.emit("leave-call", roomId);
      } catch { }
    };

    window.addEventListener("beforeunload", onBeforeUnload);

    return () => {
      cleanupAll();
      window.removeEventListener("beforeunload", onBeforeUnload);
    };
  }, [roomId]);

  return {
    start,
    leaveCall,
    endMeeting,
    cleanupAll,
    persistHistorySnapshot,
  };
}