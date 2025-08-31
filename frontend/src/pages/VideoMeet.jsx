// VideoMeet.jsx
import React, { useEffect, useRef, useState, useContext } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { AuthContext } from "../contexts/AuthContext";
import io from "socket.io-client";
import styles from "../styles/videoComponent.module.css";
import {
  FaMicrophone,
  FaMicrophoneSlash,
  FaVideo,
  FaVideoSlash,
  FaDesktop,
  FaPhoneSlash,
  FaComments,
  FaUserAlt,
  FaRegComments,
} from "react-icons/fa";
import { motion, AnimatePresence } from "framer-motion";
import {
  initMediaController,
  toggleAudio as mediaToggleAudio,
  toggleVideo as mediaToggleVideo,
  setLocalStream,
  setPeerConnections,
  setVideoElement,
  setSocketRef,
  setExternalCleaners,
  //stopAndRemoveTracks,
  //stopTransceivers,

} from "../utils/mediaController";


const SOCKET_SERVER_URL =
  process.env.REACT_APP_SIGNALING_URL || "http://localhost:8000";

// Transcript (AI) service — default: http://localhost:5001/process_meeting
const TRANSCRIPT_ENDPOINT = (() => {
  const env = process.env.REACT_APP_TRANSCRIPT_URL || process.env.REACT_APP_AI_URL;
  if (!env) return "http://localhost:5001/process_meeting";
  const trimmed = env.replace(/\/+$/, "");
  return trimmed.endsWith("/process_meeting") ? trimmed : `${trimmed}/process_meeting`;
})();

const EMOTION_ENDPOINT = (() => {
  const env = process.env.REACT_APP_EMOTION_URL;
  if (!env) return "http://localhost:5002/analyze";
  const trimmed = env.replace(/\/+$/, "");
  return trimmed.endsWith("/analyze") ? trimmed : `${trimmed}/analyze`;
})();

const API_BASE = process.env.REACT_APP_API_URL || "http://localhost:8000/api/v1";

const ICE_CONFIG = {
  iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
};

export default function VideoMeet() {
  const { roomId } = useParams();
  const navigate = useNavigate();

  const DEBUG_SHOW_EMOTION_FOR_EVERYONE = false;
  const localVideoRef = useRef(null);
  const socketRef = useRef(null);
  const localStreamRef = useRef(null);
  const prevLocalStreamRef = useRef(null);
  const pcsRef = useRef({});
  const makingOfferRef = useRef({});
  const ignoreOfferRef = useRef({});
  const politeRef = useRef({});
  const pendingCandidatesRef = useRef({});
  const settingRemoteRef = useRef({});

  const [remoteStreams, setRemoteStreams] = useState({});
  const remoteStreamsRef = useRef(remoteStreams);
  const [connecting, setConnecting] = useState(true);
  const [muted, setMuted] = useState(false);
  const [videoOff, setVideoOff] = useState(false);
  const [chatOpen, setChatOpen] = useState(false);
const [chatMessages, setChatMessages] = useState([]);
const seenMsgIdsRef = useRef(new Set());
  const [participantsMeta, setParticipantsMeta] = useState([]);
  const [myId, setMyId] = useState(null);
const { userData } = useContext(AuthContext);

  const recordersRef = useRef({});

const mutedRef = useRef(muted);
useEffect(() => { mutedRef.current = muted; }, [muted]);

const videoOffRef = useRef(videoOff);
useEffect(() => { videoOffRef.current = videoOff; }, [videoOff]);


  // Active speaker detection
  const [activeSpeakerId, setActiveSpeakerId] = useState(null);
  const activeSpeakerIdRef = useRef(activeSpeakerId);
 useEffect(() => { activeSpeakerIdRef.current = activeSpeakerId; }, [activeSpeakerId]);
  const audioContextRef = useRef(null);
  const analyzersRef = useRef({});
  const rafRef = useRef(null);

  const chatEndRef = useRef(null);
  const chatContainerRef = useRef(null);
  const recoveryRef = useRef({});
  const lastOfferOriginRef = useRef({});

  const isHost = !!localStorage.getItem(`host:${(roomId || "").toUpperCase()}`);

  const [shareEmotion, setShareEmotion] = useState(false);
  const [emotionsMap, setEmotionsMap] = useState({});

  useEffect(() => {
    remoteStreamsRef.current = remoteStreams;
  }, [remoteStreams]);

  useEffect(() => {
    const name =
      localStorage.getItem("displayName") ||
      prompt("Enter display name", "Guest") ||
      "Guest";
    localStorage.setItem("displayName", name);
    start();

    const onBeforeUnload = () => {
      try {
        socketRef.current?.emit("leave-call", roomId);
      } catch {}
    };
    window.addEventListener("beforeunload", onBeforeUnload);

    return () => {
      cleanupAll();
      window.removeEventListener("beforeunload", onBeforeUnload);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [roomId]);

  function isInitiatorFor(peerId) {
    try {
      const me = socketRef.current?.id;
      if (!me || !peerId) return false;
      return String(me) < String(peerId);
    } catch {
      return false;
    }
  }

  useEffect(() => {

  try {

    window.setupVolumeAnalyzer = (stream) => {
      try {
        if (stream) {
          localStreamRef.current = stream;
        }
        createAnalyzerForStream("local", stream || localStreamRef.current);
      } catch (e) { console.warn("setupVolumeAnalyzer wrapper failed", e); }
    };
    window.stopVolumeAnalyzer = () => {
      try {
        removeAnalyzer("local");
      } catch (e) { console.warn("stopVolumeAnalyzer wrapper failed", e); }
    };
    window.startTranscription = (stream) => {
      try { startRecordingForStream("local", stream || localStreamRef.current); } catch (e) { console.warn("startTranscription wrapper failed", e); }
    };
    window.stopTranscription = () => {
      try {
        const rec = recordersRef.current && recordersRef.current["local"];
        if (rec && rec.recorder && rec.recorder.state !== "inactive") rec.recorder.stop();
        if (recordersRef.current) delete recordersRef.current["local"];
      } catch (e) { console.warn("stopTranscription wrapper failed", e); }
    };

    window.startRecording = (kind, stream) => {
      try { startRecordingForStream("local", stream || localStreamRef.current); } catch (e) { console.warn("startRecording wrapper failed", e); }
    };
    window.stopRecording = (kind) => {
      try {
        if (typeof kind !== "undefined" && kind !== null) {
          const rec = recordersRef.current && recordersRef.current["local"];
          if (rec && rec.recorder && rec.recorder.state !== "inactive") rec.recorder.stop();
          if (recordersRef.current) delete recordersRef.current["local"];
        } else {
          stopAllRecorders();
        }
      } catch (e) { console.warn("stopRecording wrapper failed", e); }
    };

    window.startPeriodicEmotionCapture = (...args) => {
      try { startPeriodicEmotionCapture(...(args && args.length ? args[0] : {})); } catch (e) { console.warn("startPeriodicEmotionCapture wrapper failed", e); }
    };
    window.stopPeriodicEmotionCapture = () => {
      try { stopPeriodicEmotionCapture(); } catch (e) { console.warn("stopPeriodicEmotionCapture wrapper failed", e); }
    };

  } catch (e) {
    console.warn("Failed to install mediaController globals:", e);
  }

  return () => {
    try {
      delete window.setupVolumeAnalyzer;
      delete window.stopVolumeAnalyzer;
      delete window.startTranscription;
      delete window.stopTranscription;
      delete window.startRecording;
      delete window.stopRecording;
      delete window.startPeriodicEmotionCapture;
      delete window.stopPeriodicEmotionCapture;
    } catch (e) {}
  };
}, []);



  function extractSdpOrigin(sdp) {
  if (!sdp || typeof sdp !== "string") return null;
  try {
    const m = sdp.match(/^o=.*$/m);
    return m && m[0] ? m[0] : null;
  } catch (e) {
    return null;
  }
}

  function ensureAudioContext() {
    if (!audioContextRef.current) {
      try {
        audioContextRef.current = new (window.AudioContext ||
          window.webkitAudioContext)();
      } catch (err) {
        console.warn("AudioContext unavailable:", err);
      }
    }
    return audioContextRef.current;
  }

  function streamHasAudio(stream) {
  try {
    return !!(
      stream &&
      typeof stream.getAudioTracks === "function" &&
      stream.getAudioTracks().length > 0
    );
  } catch (e) {
    return false;
  }
}


  function createAnalyzerForStream(id, stream) {
  // defensive: require a stream with at least one audio track
  if (!stream || !streamHasAudio(stream)) {
    // if an analyzer exists for this id, remove it
    if (analyzersRef.current[id]) {
      try { removeAnalyzer(id); } catch (e) {}
    }
    return;
  }

  const audioCtx = ensureAudioContext();
  if (!audioCtx) return;

  const existing = analyzersRef.current[id];
  if (existing && existing.stream === stream) return;

  if (existing) {
    try { existing.source.disconnect(); } catch {}
    try { existing.analyser.disconnect(); } catch {}
    delete analyzersRef.current[id];
  }

  try {
    // still wrap this in try/catch (some browsers/contexts may reject)
    const source = audioCtx.createMediaStreamSource(stream);
    const analyser = audioCtx.createAnalyser();
    analyser.fftSize = 2048;
    source.connect(analyser);

    const data = new Float32Array(analyser.fftSize);
    analyzersRef.current[id] = {
      id,
      stream,
      source,
      analyser,
      data,
      lastSpokeAt: 0,
    };

    if (!rafRef.current) startAnalyzersLoop();
  } catch (err) {
    console.warn(`[createAnalyzerForStream:${id}]`, err);
    // ensure there's no partial analyzer left
    try { delete analyzersRef.current[id]; } catch {}
  }
}


  function removeAnalyzer(id) {
    const entry = analyzersRef.current[id];
    if (!entry) return;
    try {
      entry.source.disconnect();
    } catch {}
    try {
      entry.analyser.disconnect();
    } catch {}
    delete analyzersRef.current[id];

    if (Object.keys(analyzersRef.current).length === 0) {
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    }
  }

  function computeRMS(float32Array) {
    let sum = 0;
    for (let i = 0; i < float32Array.length; i++) {
      const v = float32Array[i];
      sum += v * v;
    }
    return Math.sqrt(sum / float32Array.length);
  }

  function startAnalyzersLoop() {
  const baseThreshold = 0.01;
  const spikeFactor = 2.0;
  const holdMs = 2000;        // keep highlight ~2s after last speech
  const silenceGrace = 2500;  // only clear after ~2.5s of silence

  const baselines = {};
  let candidateSpeaker = null;
  let candidateSince = 0;

  const step = () => {
    const now = Date.now();
    let mostRecentId = null;
    let mostRecentTs = 0;

    for (const [id, entry] of Object.entries(analyzersRef.current || {})) {
      if (!entry || !entry.analyser || !entry.data) continue;
      try {
        entry.analyser.getFloatTimeDomainData(entry.data);
        const rms = computeRMS(entry.data);

        // smooth baseline RMS
        baselines[id] = baselines[id]
          ? baselines[id] * 0.9 + rms * 0.1
          : rms;

        const baseline = Math.max(baselines[id], baseThreshold);
        const threshold = baseline * spikeFactor;

        if (rms >= threshold) {
          entry.lastSpokeAt = now;
        }
      } catch {}
      const last = entry.lastSpokeAt || 0;
      if (last > mostRecentTs) {
        mostRecentTs = last;
        mostRecentId = id;
      }
    }

    // if local is muted, never mark as speaker
    if (mostRecentId === "local" && mutedRef?.current) {
      mostRecentId = null;
      mostRecentTs = 0;
    }

    const currentActive = activeSpeakerIdRef.current;

    if (mostRecentId && now - mostRecentTs <= holdMs) {
      // debounce speaker change
      if (mostRecentId !== currentActive) {
        if (candidateSpeaker !== mostRecentId) {
          candidateSpeaker = mostRecentId;
          candidateSince = now;
        } else if (now - candidateSince >= 500) {
          // require ~0.5s stability before switch
          setActiveSpeakerId(mostRecentId);
        }
      }
    } else {
      // silence case
      if (currentActive && now - mostRecentTs > silenceGrace) {
        setActiveSpeakerId(null);
      }
    }

    rafRef.current = requestAnimationFrame(step);
  };

  if (!rafRef.current) rafRef.current = requestAnimationFrame(step);
}



  function startRecordingForStream(id, stream) {
  if (!isHost) return;
  if (!stream) return;

  if (recordersRef.current[id]) return;

  try {
    const audioTracks = stream.getAudioTracks();
    if (!audioTracks || audioTracks.length === 0) {
      console.warn(`[recorder] no audio tracks for ${id}`);
      return;
    }

    const audioStream = new MediaStream([audioTracks[0]]);

    // pick the best supported mime type
    let mimeType = "audio/webm;codecs=opus";
    if (!MediaRecorder.isTypeSupported(mimeType)) {
      mimeType = "audio/webm";
      if (!MediaRecorder.isTypeSupported(mimeType)) {
        mimeType = "";
      }
    }
    const options = mimeType ? { mimeType } : undefined;

    const recorder = new MediaRecorder(audioStream, options);
    const chunks = [];

    recorder.ondataavailable = (ev) => {
      if (ev.data && ev.data.size > 0) chunks.push(ev.data);
    };

    recorder.onstop = () => {
      console.log(`[recorder] stopped for ${id} (${chunks.length} chunks)`);
    };

    recorder.start(1000); // collect data every 1s
    recordersRef.current[id] = { recorder, chunks };

    console.log(`[recorder] started for ${id}`);
  } catch (err) {
    console.warn(`[recorder] failed to start for ${id}:`, err);
  }
}


  function stopAllRecorders() {
    Object.entries(recordersRef.current).forEach(([id, rec]) => {
      try {
        if (rec && rec.recorder && rec.recorder.state !== "inactive") {
          rec.recorder.stop();
        }
      } catch (err) {}
    });
  }

  async function uploadRecordingsAndStoreTranscript() {
  if (!isHost) return null;
  try {

    stopAllRecorders();
    await new Promise((r) => setTimeout(r, 1200));

    const fd = new FormData();
    fd.append("meeting_code", (roomId || "").toUpperCase());

    const speakerMap = {};
    try {
      participantsMeta?.forEach((p) => {
        const display =
          p?.meta?.name ||
          p?.meta?.displayName ||
          `Guest-${p.id.slice(0, 6)}`;
        speakerMap[p.id] = display;
      });
      speakerMap["local"] =
        localStorage.getItem("displayName") || "Host";
    } catch (e) {
      console.warn("speaker map build failed", e);
    }
    fd.append("speaker_map", JSON.stringify(speakerMap));

    // attach audio files
    let fileCount = 0;
    for (const [id, { chunks }] of Object.entries(
      recordersRef.current || {}
    )) {
      if (!chunks || chunks.length === 0) continue;
      const blob = new Blob(chunks, { type: "audio/webm" });
      fd.append("audio_files", blob, `${id}.webm`);
      fileCount++;
    }

    if (fileCount === 0) {
      console.warn(
        "[transcript] No audio files recorded — skipping transcript upload"
      );
      return null;
    }

    // send to Flask transcript service
    const resp = await fetch(TRANSCRIPT_ENDPOINT, {
      method: "POST",
      body: fd,
    });

    if (!resp.ok) {
      console.error(
        "upload recordings failed",
        await resp.text()
      );
      return null;
    }

    const data = await resp.json();
    if (data?.success) {
      const payload = {
        meeting_code: (roomId || "").toUpperCase(),
        transcript: data.transcript_text || "",
        createdAt: new Date().toISOString(),
        txt_filename: data.txt_filename,
        // build download URL from TRANSCRIPT_ENDPOINT base
        downloadUrlFlask: `${TRANSCRIPT_ENDPOINT.replace(
          "/process_meeting",
          ""
        ).replace(/\/$/, "")}/outputs/${data.txt_filename}`,
      };

      // guard: skip Node persist if transcript missing/empty
      if (!payload.transcript || payload.transcript.trim() === "") {
        console.warn(
          "[transcript] Transcript missing/empty, skipping Node persistence"
        );
        return data;
      }

      // save in localStorage
      localStorage.setItem(
        `transcript:${payload.meeting_code}`,
        JSON.stringify({
          meeting_code: payload.meeting_code,
          transcript: payload.transcript,
          createdAt: payload.createdAt,
          downloadUrlFlask: payload.downloadUrlFlask,
        })
      );

      // persist to Node backend
      try {
        const backendResp = await fetch(`${API_BASE}/transcript`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            meetingCode: payload.meeting_code,
            transcriptText: payload.transcript,
            fileName: payload.txt_filename,
          }),
        });
        if (!backendResp.ok) {
          console.warn(
            "persist transcript to node failed",
            await backendResp.text()
          );
        } else {
          const backendData = await backendResp.json();
          console.log("Persisted transcript to backend:", backendData);
        }
      } catch (err) {
        console.warn("Persist transcript network error:", err);
      }

      return data;
    } else {
      console.warn("AI service returned no success:", data);
      return null;
    }
  } catch (err) {
    console.error("uploadRecordingsAndStoreTranscript err:", err);
    return null;
  } finally {
    recordersRef.current = {};
  }
}


  const EMO_CONFIG = {
    clipDurationMs: 1500,
    captureIntervalMs: 3000,
    eventName: "emotion.frame",
    preferVideoMime: ["video/webm;codecs=vp9","video/webm;codecs=vp8","video/mp4"],
    preferAudioMime: ["audio/webm","audio/wav","audio/ogg"],
  };

  function chooseSupportedMime(preferredList) {
    if (typeof MediaRecorder === "undefined" || !MediaRecorder.isTypeSupported) {
      return "";
    }
    for (const m of preferredList) {
      try {
        if (MediaRecorder.isTypeSupported(m)) return m;
      } catch (e) {}
    }
    return "";
  }

  function getTypeForStream(stream) {
    if (!stream) return null;
    const audioCount = (stream.getAudioTracks && stream.getAudioTracks().length) || 0;
    const videoCount = (stream.getVideoTracks && stream.getVideoTracks().length) || 0;
    if (videoCount > 0) return "video";
    if (audioCount > 0) return "audio";
    return null;
  }

  const recordingState = useRef(new Map());
  const emoIntervalHandleRef = useRef(null);

  // Utility: convert Blob -> dataURL (base64)
  function blobToDataURL(blob) {
    return new Promise((resolve, reject) => {
      try {
        const fr = new FileReader();
        fr.onload = () => resolve(fr.result);
        fr.onerror = (e) => reject(e);
        fr.readAsDataURL(blob);
      } catch (e) {
        reject(e);
      }
    });
  }

  function emitWithAckTimeout(event, payload, timeoutMs = 8000) {
    return new Promise((resolve) => {
      let done = false;
      try {
        socketRef.current?.emit(event, payload, (ack) => {
          if (done) return;
          done = true;
          resolve({ ok: true, ack });
        });
      } catch (e) {
        if (!done) {
          done = true;
          resolve({ ok: false, reason: e });
        }
      }
      setTimeout(() => {
        if (done) return;
        done = true;
        resolve({ ok: false, reason: "timeout" });
      }, timeoutMs);
    });
  }

  // Build analyze URL (use EMOTION_ENDPOINT directly)
  function buildAnalyzeUrl() {
    return EMOTION_ENDPOINT;
  }

  async function recordAndSendClip({ stream, meetingId, participantId, durationMs }) {
    const socket = socketRef.current;
    if (!stream || !socket || !socket.connected) {
      console.debug("[emotion] skipping capture — no socket or disconnected", { participantId, connected: !!socket?.connected });
      return;
    }
    if (!participantId) return;
    if (participantId === myId) return;

    if (recordingState.current.get(participantId)) return;
    recordingState.current.set(participantId, true);

    const type = getTypeForStream(stream);
    if (!type) {
      recordingState.current.delete(participantId);
      return;
    }

    let mime = "";
    if (type === "video") {
      mime = chooseSupportedMime(EMO_CONFIG.preferVideoMime) || "";
    } else {
      mime = chooseSupportedMime(EMO_CONFIG.preferAudioMime) || "";
    }

    let recorder;
    try {
      recorder = mime ? new MediaRecorder(stream, { mimeType: mime }) : new MediaRecorder(stream);
      mime = recorder.mimeType || mime;
    } catch (err) {
      try {
        recorder = new MediaRecorder(stream);
        mime = recorder.mimeType || "";
      } catch (err2) {
        console.error("MediaRecorder creation failed for", participantId, err2);
        recordingState.current.delete(participantId);
        return;
      }
    }

    const chunks = [];
    let stopped = false;

    const onData = (ev) => {
      if (ev.data && ev.data.size) chunks.push(ev.data);
    };
    recorder.ondataavailable = onData;
    recorder.onerror = (err) => {
      console.error("MediaRecorder error for", participantId, err);
    };

    recorder.onstop = async () => {
      if (stopped) return;
      stopped = true;
      if (!chunks.length) {
        recordingState.current.delete(participantId);
        return;
      }
      try {
        const blob = new Blob(chunks, { type: chunks[0].type || mime });
        let arrayBuffer;
        try {
          arrayBuffer = await blob.arrayBuffer();
        } catch (e) {
          console.warn("[emotion] blob.arrayBuffer() failed, will try base64 fallback", e);
          arrayBuffer = null;
        }

        // extension derived from mime
        let ext = "webm";
        try {
          ext = (blob.type && blob.type.split("/")[1].split(";")[0]) || (type === "video" ? "webm" : "webm");
          ext = ext.replace(/[^a-z0-9]/gi, "");
        } catch (e) {}

        const filename = `${participantId}.${ext}`;

        // build payload (binary-first)
        const payload = {
          meetingId: (meetingId || "").toUpperCase(),
          participantId,
          type,
          buffer: arrayBuffer,
          mime: blob.type || mime,
          filename,
          timestamp: Date.now(),
        };

        console.debug("[emotion] prepared clip for", participantId, { mime: payload.mime, size: arrayBuffer ? arrayBuffer.byteLength : "unknown" });

        // Attempt 1: try sending binary payload with ACK (use larger timeout)
        let sentViaSocket = false;
        if (arrayBuffer) {
          const resp = await emitWithAckTimeout(EMO_CONFIG.eventName, payload, 8000);
          if (resp.ok) {
            console.debug("[emotion] sent binary clip (ack)", participantId, resp.ack);
            recordingState.current.delete(participantId);
            sentViaSocket = true;
            return;
          } else {
            console.warn("[emotion] binary emit failed or timed out for", participantId, resp.reason);
          }
        } else {
          console.warn("[emotion] no arrayBuffer available to send for", participantId);
        }

        // Fallback: send base64 Data URL (slower but more compatible)
        if (!sentViaSocket) {
          try {
            const dataUrl = await blobToDataURL(blob);
            const payloadBase64 = {
              meetingId: (meetingId || "").toUpperCase(),
              participantId,
              type,
              dataUrl, // data:image/...;base64,...  OR data:audio/...
              mime: blob.type || mime,
              filename,
              timestamp: Date.now(),
            };
            // we use a different event name so server can detect fallback if needed
            const resp2 = await emitWithAckTimeout(`${EMO_CONFIG.eventName}.base64`, payloadBase64, 12000);
            if (resp2.ok) {
              console.debug("[emotion] sent base64 fallback clip (ack)", participantId);
              recordingState.current.delete(participantId);
              sentViaSocket = true;
              return;
            } else {
              console.warn("[emotion] base64 fallback emit failed for", participantId, resp2.reason);
            }
          } catch (fbErr) {
            console.error("[emotion] fallback base64 send failed for", participantId, fbErr);
          }
        }

        // If both socket attempts failed, try REST fallback to Emotion service /analyze
        if (!sentViaSocket) {
          try {
            const analyzeUrl = buildAnalyzeUrl();
            const fd = new FormData();
            // emotion_service app.py expects meeting_id and participant_id
            fd.append("meeting_id", (meetingId || "").toUpperCase());
            fd.append("participant_id", participantId);
            fd.append("type", type || "audio");
            fd.append("file", blob, filename);

            console.debug("[emotion] attempting REST fallback to", analyzeUrl, { participantId, filename });
            const r = await fetch(analyzeUrl, {
              method: "POST",
              body: fd,
            });
            if (r.ok) {
              console.debug("[emotion] REST fallback upload ok for", participantId);
            } else {
              console.warn("[emotion] REST fallback returned non-ok", participantId, await r.text());
            }
          } catch (restErr) {
            console.error("[emotion] REST fallback error for", participantId, restErr);
          }
        }
      } catch (e) {
        console.error("Failed to finalize/send clip for", participantId, e);
      } finally {
        recordingState.current.delete(participantId);
      }
    };

    try {
      recorder.start();
      setTimeout(() => {
        try {
          if (recorder && recorder.state !== "inactive") recorder.stop();
        } catch (e) {
          console.warn("Failed to stop recorder for", participantId, e);
          recordingState.current.delete(participantId);
        }
      }, durationMs);
    } catch (e) {
      console.error("Failed to start recorder for", participantId, e);
      recordingState.current.delete(participantId);
    }
  }

  function startPeriodicEmotionCapture({
  clipDurationMs = EMO_CONFIG.clipDurationMs,
  intervalMs = EMO_CONFIG.captureIntervalMs,
} = {}) {
  stopPeriodicEmotionCapture();

  // Only allow the host (or debug opt-in) to run periodic captures.
  if (!isHost && !DEBUG_SHOW_EMOTION_FOR_EVERYONE) {
    console.debug("[emotion] startPeriodicEmotionCapture called but this client is not host — ignoring");
    return;
  }

  const doCapturePass = async () => {
    try {
      const streamsMap = remoteStreamsRef.current || {};
      for (const [participantId, stream] of Object.entries(streamsMap)) {
        if (!participantId || participantId === myId) continue;
        if (!stream) continue;
        recordAndSendClip({
          stream,
          meetingId: roomId,
          participantId,
          durationMs: clipDurationMs,
        });
      }
    } catch (e) {
      console.error("Error in emotion capture pass", e);
    }
  };

  doCapturePass();
  emoIntervalHandleRef.current = setInterval(doCapturePass, intervalMs);
}


  function stopPeriodicEmotionCapture() {
    if (emoIntervalHandleRef.current) {
      clearInterval(emoIntervalHandleRef.current);
      emoIntervalHandleRef.current = null;
    }
    recordingState.current.clear();
  }

  useEffect(() => {
    return () => {
      stopPeriodicEmotionCapture();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
  // Only start periodic capture if the local client is the host.
  if (shareEmotion && isHost) {
    startPeriodicEmotionCapture({});
  } else {
    stopPeriodicEmotionCapture();
  }
  // eslint-disable-next-line react-hooks/exhaustive-deps
}, [shareEmotion, myId, isHost]);


 async function start() {
  try {
    setConnecting(true);

    const constraints = {
      audio: true,
      video: { width: { ideal: 1280 }, height: { ideal: 720 } },
    };
    const stream = await navigator.mediaDevices.getUserMedia(constraints);

    localStreamRef.current = stream;
    if (localVideoRef.current) localVideoRef.current.srcObject = stream;

    createAnalyzerForStream("local", stream);
    startRecordingForStream("local", stream);

    const socket = io(SOCKET_SERVER_URL, { autoConnect: false });
    socketRef.current = socket;
    attachSocketHandlers();

    await new Promise((resolve, reject) => {
      const CLEANUP = () => {
        socket.off("connect", onConnect);
        socket.off("connect_error", onError);
      };

      const onConnect = () => {
        CLEANUP();
        resolve();
      };

      const onError = (err) => {
        CLEANUP();
        reject(err || new Error("socket connect error"));
      };

      socket.on("connect", onConnect);
      socket.on("connect_error", onError);

      const TO = setTimeout(() => {
        CLEANUP();
        reject(new Error("socket connect timeout"));
      }, 5000);

      const origResolve = resolve;
      resolve = (...args) => { clearTimeout(TO); origResolve(...args); };
      const origReject = reject;
      reject = (...args) => { clearTimeout(TO); origReject(...args); };

      socket.connect();
    });

    initMediaController(
      localStreamRef.current,
      socketRef.current,
      pcsRef.current,
      localVideoRef.current
    );

    try {
      if (typeof setExternalCleaners === "function") {
        setExternalCleaners({
          recordersRef: typeof recordersRef !== "undefined" ? recordersRef : null,
          audioContextRef: typeof audioContextRef !== "undefined" ? audioContextRef : null,
          removeAnalyzerFn: typeof removeAnalyzer === "function" ? removeAnalyzer : null,
          prevLocalStreamRef: typeof prevLocalStreamRef !== "undefined" ? prevLocalStreamRef : null,
        });
      }
    } catch (e) {
      console.warn("registering external cleaners failed:", e);
    }

    setLocalStream(localStreamRef.current);
    setPeerConnections(pcsRef.current);
    setVideoElement(localVideoRef.current);
    setSocketRef(socketRef.current);


    let uid = localStorage.getItem("userId");
    if (!uid) {
      uid = crypto.randomUUID();
      localStorage.setItem("userId", uid);
    }

    socketRef.current.emit("join-call", roomId, {
      name: localStorage.getItem("displayName") || "Guest",
      userId: uid,
    });
  } catch (err) {
    console.error("start error:", err);
    alert("Unable to access camera/mic or connect to signaling server.");
  } finally {
    setConnecting(false);
  }
}


  function attachSocketHandlers() {
    const socket = socketRef.current;
    if (!socket) return;

    socket.on("connect", () => {
      setConnecting(true);
      setMyId(socket.id);
      try { window.myId = socket.id; } catch (e) {}
    });

    socket.once("chat-history", (history = []) => {
  const seen = seenMsgIdsRef.current;
  const unique = [];

  for (const m of history) {
    const id = m.id || `${m.userId}:${m.ts}`;
    if (seen.has(id)) continue;
    seen.add(id);
    unique.push(m);
  }

  setChatMessages(unique);
});


    socket.on("participants-updated", (participants) => {
      if (!Array.isArray(participants)) return;
      setParticipantsMeta(
        participants
          .filter((p) => p.id !== socket.id)
          .map((p) => ({
            id: p.id,
            meta: p.meta || {},
            polite: !!p.polite,
          }))
      );
    });

    socket.on("existing-participants", async (existing) => {
  const normalized = (Array.isArray(existing) ? existing : [])
    .map((item) => {
      if (!item) return null;
      return {
        id: item.id,
        polite: typeof item.polite === "boolean" ? item.polite : undefined,
        meta: item.meta || {},
      };
    })
    .filter(Boolean);

  setParticipantsMeta((prev) => {
    const map = {};
    prev.forEach((p) => (map[p.id] = p));
    normalized.forEach((p) => {
      map[p.id] = { id: p.id, meta: p.meta || {} };
    });
    return Object.values(map);
  });

  for (const p of normalized) {
    if (p.id === socket.id) continue;

    politeRef.current[p.id] =
      typeof p.polite === "boolean" ? p.polite : !isInitiatorFor(p.id);

    pendingCandidatesRef.current[p.id] = pendingCandidatesRef.current[p.id] || [];
    createPeerConnection(p.id);
  }

  for (const p of normalized.filter((p) => p.id !== socket.id)) {
    const backoff = Math.floor(Math.random() * 150) + 50;
    await new Promise((r) => setTimeout(r, backoff));

    if (isInitiatorFor(p.id)) {
      await safeNegotiateOffer(p.id);
    } else {
      console.debug("[existing-participants] deferring offer, waiting for remote to initiate:", p.id);
    }
  }

  setConnecting(false);
});


    socket.on("user-joined", async (peer) => {
  try {
    const peerId = peer?.id;
    if (!peerId || peerId === socket.id) return;

    setParticipantsMeta((prev) => {
      if (prev.some((p) => p.id === peerId)) return prev;
      return [...prev, { id: peerId, meta: peer.meta || {} }];
    });

    politeRef.current[peerId] =
      typeof peer?.polite === "boolean" ? peer.polite : !isInitiatorFor(peerId);

    pendingCandidatesRef.current[peerId] = pendingCandidatesRef.current[peerId] || [];
    createPeerConnection(peerId);

    const backoff = Math.floor(Math.random() * 150) + 50;
    await new Promise((r) => setTimeout(r, backoff));

    if (isInitiatorFor(peerId)) {
      await safeNegotiateOffer(peerId);
    } else {
      console.debug("[user-joined] deferring offer (polite), waiting for remote to initiate:", peerId);
    }
  } catch (err) {
    console.error("[user-joined] handler error:", err);
  }
});


    socket.on("user-left", (peerId) => {
      setParticipantsMeta((prev) => prev.filter((p) => p.id !== peerId));
      closePeer(peerId);
      removeAnalyzer(peerId);

      try {
        const rec = recordersRef.current[peerId];
        if (rec) {
          if (rec.recorder && rec.recorder.state !== "inactive") rec.recorder.stop();
        }
      } catch {}
      delete recordersRef.current[peerId];

      setEmotionsMap((prev) => {
        const copy = { ...prev };
        delete copy[peerId];
        return copy;
      });
    });

    socket.on("signal", async (fromId, messageStr) => {
      await handleSignal(fromId, messageStr).catch(console.error);
    });

    socket.on("chat-message", (m) => {
  const id = m.id || `${m.userId}:${m.ts}`;
  if (seenMsgIdsRef.current.has(id)) return;
  seenMsgIdsRef.current.add(id);

  setChatMessages((prev) => [...prev, m]);
});

      socket.on("chat-ack", (msg) => {
  seenMsgIdsRef.current.add(msg.id);

  setChatMessages((prev) =>
    prev.map((m) => (m.id === msg.id ? { ...m, ...msg, confirmed: true } : m))
  );
});



    socket.on("participant-meta-updated", ({ id, meta }) => {
      setParticipantsMeta((prev) =>
        prev.map((p) =>
          p.id === id ? { ...p, meta: { ...(p.meta || {}), ...meta } } : p
        )
      );

      if (id === socket.id) {
        if (typeof meta.muted !== "undefined") setMuted(meta.muted);
        if (typeof meta.video !== "undefined") setVideoOff(!meta.video);
      }
    });

    socket.on("end-meeting", async () => {
  cleanupAll();
  navigate("/home")

});

socket.on("disconnect", () => {
  cleanupAll();
  navigate("/home")

});


const emotionHandler = (payload) => {
  try {

    const participantId = payload.participant_id || payload.participantId || payload.from || payload.userId;
    const emotionData = payload.emotion || payload.scores || payload.result || payload || {};
    if (!participantId) return;


    const nameFromPayload =
      payload.name ||
      payload.display_name ||
      payload.displayName ||
      (payload.meta && (payload.meta.name || payload.meta.displayName));

    const merged = {
      ...emotionData,
      __name: nameFromPayload || undefined,
      __ts: String(payload.ts || payload.timestamp || Date.now()),
    };

    setEmotionsMap((prev) => ({ ...prev, [participantId]: merged }));
  } catch (e) {
    console.warn("[socket] emotion.update handler error", e);
  }
};


    socket.on("emotion.update", emotionHandler);
    socket.on("emotion-update", emotionHandler);
    socket.on("emotion.result", emotionHandler);
    socket.on("emotion", emotionHandler);

    socket.once("disconnect", () => {
      socket.off("emotion.update", emotionHandler);
      socket.off("emotion-update", emotionHandler);
      socket.off("emotion.result", emotionHandler);
      socket.off("emotion", emotionHandler);
    });
  }


// function startInboundVideoMonitor(peerId, pc) {
//   const CHECK_INTERVAL = 1000; // ms
//   const STALL_THRESHOLD = 3000; // ms of no new frames -> consider stalled
//   const lastFrameInfo = new Map(); // key: trackId -> { frames, ts }
//   // save interval id on pc so we can clear it
//   pc._statsInterval = setInterval(async () => {
//     try {
//       const receivers = pc.getReceivers ? pc.getReceivers() : [];
//       for (const r of receivers) {
//         if (!r.track || r.track.kind !== "video") continue;

//         // Try receiver.getStats first (works in modern browsers). Fallback to pc.getStats(r.track)
//         let stats = null;
//         try {
//           if (typeof r.getStats === "function") stats = await r.getStats();
//           else if (typeof pc.getStats === "function") stats = await pc.getStats(r.track);
//         } catch (e) {
//           // ignore stats failures
//           stats = null;
//         }

//         if (!stats) continue;

//         // extract frames counter from inbound-rtp report
//         let frames = null;
//         stats.forEach((report) => {
//           if (report && report.type && report.type.toLowerCase().includes("inbound-rtp")) {
//             // prefer framesDecoded, fallback to framesReceived
//             frames = report.framesDecoded ?? report.framesReceived ?? report.packetsReceived ?? frames;
//           }
//         });

//         if (frames == null) {
//           // couldn't read frames; skip
//           continue;
//         }

//         const prev = lastFrameInfo.get(r.track.id) || { frames, ts: Date.now() };
//         if (frames > prev.frames) {
//           // frames advanced — mark as healthy and store timestamp
//           lastFrameInfo.set(r.track.id, { frames, ts: Date.now() });

//           // If we previously replaced with empty stream because of stall, restore real stream
//           // pc._remoteStream should hold the current real stream (or we replaced it previously)
//           const currentStream = pc._remoteStream;
//           if (currentStream && currentStream.getVideoTracks().length > 0) {
//             // ensure app state has the real stream object (restore)
//             setRemoteStreams((s) => {
//               if (s && s[peerId] === currentStream) return s; // already set
//               return { ...s, [peerId]: currentStream };
//             });
//           }
//         } else {
//           // frames did NOT increase
//           const elapsed = Date.now() - prev.ts;
//           if (elapsed > STALL_THRESHOLD) {
//             // treat as stalled: replace app-level stream object with an *empty* MediaStream
//             // so ParticipantCard's ref effect will clear el.srcObject and stop the frozen frame
//             setRemoteStreams((s) => {
//               const cur = s && s[peerId];
//               // if already empty, skip
//               if (cur && cur.getTracks && cur.getTracks().length === 0) return s;
//               const empty = new MediaStream();
//               return { ...s, [peerId]: empty };
//             });
//             // keep prev record timestamp so we don't spam creating empty stream repeatedly
//             lastFrameInfo.set(r.track.id, { frames: prev.frames, ts: prev.ts });
//           }
//         }
//       }
//     } catch (err) {
//       // swallow errors so monitor keeps running
//       // console.debug("stats monitor error", err);
//     }
//   }, CHECK_INTERVAL);
// }


function createPeerConnection(peerId) {
  if (pcsRef.current[peerId]) return pcsRef.current[peerId];
  const pc = new RTCPeerConnection(ICE_CONFIG);
  pcsRef.current[peerId] = pc;
  setPeerConnections?.(pcsRef.current);
  makingOfferRef.current[peerId] = false;
  ignoreOfferRef.current[peerId] = false;
  pendingCandidatesRef.current[peerId] =
    pendingCandidatesRef.current[peerId] || [];

  const ls = localStreamRef.current;
  if (ls) {
    try {
      const audioTracks = ls.getAudioTracks().filter((t) => t.readyState === "live");
      const videoTracks = ls.getVideoTracks().filter((t) => t.readyState === "live");

      audioTracks.forEach((t) => pc.addTrack(t, ls));
      videoTracks.forEach((t) => pc.addTrack(t, ls));
    } catch (err) {
      console.warn("addTrack failed:", err);
    }
  }

  try {
    const existingKinds =
      (pc.getTransceivers && pc.getTransceivers().map((t) => t.kind)) || [];
    const hasAudio = ls && ls.getAudioTracks().some((t) => t.readyState === "live");
    const hasVideo = ls && ls.getVideoTracks().some((t) => t.readyState === "live");

    const desiredOrder = ["audio", "video"];
    for (const kind of desiredOrder) {
      if (!existingKinds.includes(kind)) {
        const hasKind = kind === "audio" ? hasAudio : hasVideo;
        if (!hasKind) {
          try {
            pc.addTransceiver(kind, { direction: "recvonly" });
          } catch (e) {}
        }
      }
    }
  } catch (e) {
    console.warn("transceiver fallback failed", e);
  }

  pc._remoteStream = new MediaStream();
  pc._trackListeners = [];
  pc._statsInterval = null;

  function startInboundVideoMonitor() {
    const CHECK_INTERVAL = 1000; // ms
    const STALL_THRESHOLD = 3000; // ms without new frames => treat as stalled
    const lastFrameInfo = new Map(); // trackId -> { frames, ts }

    // clear any previous interval just in case
    if (pc._statsInterval) {
      clearInterval(pc._statsInterval);
      pc._statsInterval = null;
    }

    pc._statsInterval = setInterval(async () => {
      try {
        const receivers = typeof pc.getReceivers === "function" ? pc.getReceivers() : [];
        for (const r of receivers) {
          if (!r.track || r.track.kind !== "video") continue;

          let stats = null;
          try {
            if (typeof r.getStats === "function") {
              stats = await r.getStats();
            } else if (typeof pc.getStats === "function") {
              stats = await pc.getStats(r.track);
            }
          } catch (e) {
            // ignore stats errors
            stats = null;
          }

          if (!stats) continue;

          // Pull frames counter from inbound-rtp report (framesDecoded preferred)
          let frames = null;
          try {
            // stats may be an RTCStatsReport (iterable with .forEach)
            stats.forEach((report) => {
              if (!report || !report.type) return;
              const t = report.type.toLowerCase();
              if (t.includes("inbound-rtp")) {
                frames = report.framesDecoded ?? report.framesReceived ?? report.packetsReceived ?? frames;
              }
            });
          } catch (e) {
            continue; // can't read stats
          }

          if (frames == null) continue;

          const prev = lastFrameInfo.get(r.track.id) || { frames, ts: Date.now() };

          if (frames > prev.frames) {
            // frames advanced => healthy
            lastFrameInfo.set(r.track.id, { frames, ts: Date.now() });

            // If we previously replaced the displayed stream with an empty stream,
            // restore the real stream object so the video element can reattach.
            const curAppStream = (remoteStreamsRef && remoteStreamsRef.current) ? remoteStreamsRef.current[peerId] : null;
            // If the current app-level stream is empty, restore
            if (pc._remoteStream && pc._remoteStream.getVideoTracks().length > 0 && (!curAppStream || curAppStream.getTracks().length === 0)) {
              setRemoteStreams((s) => ({ ...s, [peerId]: pc._remoteStream }));
              console.debug("[stats-monitor] restored stream for", peerId);
            }
          } else {
            // frames did NOT increase
            const elapsed = Date.now() - prev.ts;
            if (elapsed > STALL_THRESHOLD) {
              // treat as stall: replace the app-visible stream with an empty one
              setRemoteStreams((s) => {
                const current = s && s[peerId];
                if (current && current.getTracks && current.getTracks().length === 0) {
                  // already empty
                  return s;
                }
                console.debug("[stats-monitor] detected stall, clearing display for", peerId);
                const empty = new MediaStream(); // new identity -> ParticipantCard will clear srcObject
                return { ...s, [peerId]: empty };
              });
              // keep prev timestamp to avoid spamming
              lastFrameInfo.set(r.track.id, { frames: prev.frames, ts: prev.ts });
            }
          }
        }
      } catch (err) {
        // swallow to keep monitor running
        // console.debug("stats monitor error", err);
      }
    }, CHECK_INTERVAL);
  }

  // ontrack: add incoming tracks to pc._remoteStream and attach lifecycle listeners
  pc.ontrack = (ev) => {
    const incoming = ev.streams && ev.streams.length
      ? ev.streams.flatMap((s) => s.getTracks())
      : ev.track ? [ev.track] : [];

    incoming.forEach((tr) => {
      try {
        // add only once
        if (!pc._remoteStream.getTracks().some((t) => t.id === tr.id)) {
          pc._remoteStream.addTrack(tr);
        }
      } catch (e) {
        console.warn("failed to add incoming track", e);
      }

      // debug logs to help see what events fire
      try {
        if (typeof tr.addEventListener === "function") {
          tr.addEventListener("ended", () => console.debug("[pc] track ended", peerId, tr.id));
          tr.addEventListener("mute", () => console.debug("[pc] track mute", peerId, tr.id));
          tr.addEventListener("unmute", () => console.debug("[pc] track unmute", peerId, tr.id));
        }
      } catch (e) {}

      // handler when this track updates state (ended/mute/unmute)
      const handleTrackUpdate = () => {
        try {
          // remove ended tracks from the managed stream
          if (tr.readyState === "ended") {
            try { pc._remoteStream.removeTrack(tr); } catch (e) {}
          }
        } finally {
          // create a new MediaStream object (new identity) so React/video ref rebinds
          try {
            const cloned = new MediaStream(pc._remoteStream.getTracks().slice());
pc._remoteStream = cloned;
setRemoteStreams((prev) => {
  const next = { ...prev, [peerId]: cloned };
  try {
    if (streamHasAudio(cloned)) {
      createAnalyzerForStream(peerId, cloned);
      startRecordingForStream(peerId, cloned);
    } else {
      // remove any stale analyzer/recorder for this peer
      try { removeAnalyzer(peerId); } catch {}
      try {
        const rec = recordersRef.current && recordersRef.current[peerId];
        if (rec && rec.recorder && rec.recorder.state !== "inactive") rec.recorder.stop();
        delete recordersRef.current[peerId];
      } catch (e) {}
    }
  } catch (err) {}
  return next;
});

          } catch (err) {
            // fallback: set whatever we have
            setRemoteStreams((prev) => {
              const next = { ...prev, [peerId]: pc._remoteStream };
              try {
                createAnalyzerForStream(peerId, next[peerId]);
                startRecordingForStream(peerId, next[peerId]);
              } catch (err) {}
              return next;
            });
          }
        }
      };

      // attach listeners (use addEventListener for modern browsers)
      try {
        if (typeof tr.addEventListener === "function") {
          tr.addEventListener("ended", handleTrackUpdate);
          tr.addEventListener("mute", handleTrackUpdate);
          tr.addEventListener("unmute", handleTrackUpdate);
          pc._trackListeners.push({ track: tr, fn: handleTrackUpdate, useEvent: true });
        } else {
          tr.onended = handleTrackUpdate;
          tr.onmute = handleTrackUpdate;
          tr.onunmute = handleTrackUpdate;
          pc._trackListeners.push({ track: tr, fn: handleTrackUpdate, useEvent: false });
        }
      } catch (err) {
        console.warn("failed to attach track listeners", err);
      }
    });

    // publish current managed stream to app state so UI attaches
    setRemoteStreams((prev) => {
      const next = { ...prev, [peerId]: pc._remoteStream };
      try {
        createAnalyzerForStream(peerId, pc._remoteStream);
        startRecordingForStream(peerId, pc._remoteStream);
      } catch (err) {}
      return next;
    });

    // ensure the inbound monitor is running
    try {
      startInboundVideoMonitor();
    } catch (e) {
      console.warn("failed to start inbound monitor", e);
    }
  };

  pc.onicecandidate = (ev) => {
    if (ev.candidate) sendSignal(peerId, { candidate: ev.candidate });
  };

  pc.onconnectionstatechange = () => {
    // When connection closes or fails, do cleanup
    if (["failed", "disconnected", "closed"].includes(pc.connectionState)) {
      try {
        // detach any attached track listeners
        if (pc._trackListeners && pc._trackListeners.length) {
          pc._trackListeners.forEach(({ track, fn, useEvent }) => {
            try {
              if (useEvent && typeof track.removeEventListener === "function") {
                track.removeEventListener("ended", fn);
                track.removeEventListener("mute", fn);
                track.removeEventListener("unmute", fn);
              } else {
                track.onended = null;
                track.onmute = null;
                track.onunmute = null;
              }
            } catch (e) {}
          });
          pc._trackListeners = [];
        }
      } catch (e) {
        console.warn("error cleaning track listeners", e);
      }

      // stop stats monitor
      try {
        if (pc._statsInterval) {
          clearInterval(pc._statsInterval);
          pc._statsInterval = null;
        }
      } catch (e) {}

      // remove the remote stream from app state
      try {
        setRemoteStreams((prev) => {
          const copy = { ...prev };
          delete copy[peerId];
          return copy;
        });
      } catch (e) {}

      closePeer(peerId);
      removeAnalyzer(peerId);
    }
  };

  return pc;
}




async function negotiateCreateOffer(peerId) {
  const pc = createPeerConnection(peerId);
  if (pc.signalingState !== "stable" || makingOfferRef.current[peerId]) return;

  if (settingRemoteRef.current[peerId]) {
    await new Promise((r) => setTimeout(r, 50));
    if (settingRemoteRef.current[peerId]) return;
  }

  try {
    makingOfferRef.current[peerId] = true;
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    try {
      const sdp = pc.localDescription && pc.localDescription.sdp;
      const origin = extractSdpOrigin(sdp);
      if (origin) lastOfferOriginRef.current[peerId] = origin;
    } catch (e) {

    }

    sendSignal(peerId, { sdp: pc.localDescription });
  } finally {
    makingOfferRef.current[peerId] = false;
  }
}



  async function safeNegotiateOffer(peerId) {
    if (!localStreamRef.current) {
      for (let i = 0; i < 20; i++) {
        await new Promise((r) => setTimeout(r, 100));
        if (localStreamRef.current) break;
      }
      if (!localStreamRef.current) {
        console.warn(`[safeNegotiateOffer] No local stream for ${peerId}`);
        return;
      }
    }

    await negotiateCreateOffer(peerId).catch((err) =>
      console.error(`[safeNegotiateOffer] Error for ${peerId}:`, err)
    );
  }

  async function handleSignal(fromId, messageStr) {
  if (!socketRef.current || fromId === socketRef.current.id) return;

  let msg;
  try {
    msg = JSON.parse(messageStr);
  } catch {
    console.warn(`[handleSignal] Invalid JSON from ${fromId}`);
    return;
  }

  const pc = pcsRef.current[fromId] || createPeerConnection(fromId);
  const polite = !!politeRef.current[fromId];
  const isMakingOffer = !!makingOfferRef.current[fromId];

  if (msg.sdp) {
    const desc = msg.sdp;
    const isOffer = desc.type === "offer";

    const offerCollision = isOffer && (isMakingOffer || pc.signalingState !== "stable");
    ignoreOfferRef.current[fromId] = !polite && offerCollision;
    if (ignoreOfferRef.current[fromId]) {
      console.warn(`[handleSignal] Ignoring offer from ${fromId} (collision)`);
      return;
    }

    try {
      while (settingRemoteRef.current[fromId]) {
        await new Promise((r) => setTimeout(r, 25));
      }

      if (!isOffer && pc.signalingState === "stable") {

        try {
          const now = Date.now();
          const last = recoveryRef.current[fromId] || 0;
          const COOLDOWN_MS = 5000;

          if (now - last < COOLDOWN_MS) {
            console.warn(`[handleSignal] Recent recovery attempted for ${fromId} (${now - last}ms); ignoring answer.`);
            return;
          }

          recoveryRef.current[fromId] = now;
          console.warn(`[handleSignal] Received unexpected answer from ${fromId} while pc.signalingState is 'stable'. Attempting recovery (close/recreate/negotiate).`);

          try { pcsRef.current[fromId]?.close(); } catch (e) {}
          delete pcsRef.current[fromId];
          pendingCandidatesRef.current[fromId] = [];

          await new Promise((r) => setTimeout(r, 150 + Math.floor(Math.random() * 200)));

          createPeerConnection(fromId);
          await safeNegotiateOffer(fromId);
        } catch (recoveryErr) {
          console.error(`[handleSignal] Recovery attempt for ${fromId} failed:`, recoveryErr);
        }
        return;
      }

      settingRemoteRef.current[fromId] = true;
      await pc.setRemoteDescription(desc);
      settingRemoteRef.current[fromId] = false;

      const queued = pendingCandidatesRef.current[fromId] || [];
      for (const candidate of queued) {
        try {
          await pc.addIceCandidate(candidate);
        } catch (err) {
          console.error(`[handleSignal] Failed to add queued ICE candidate from ${fromId}:`, err);
        }
      }
      pendingCandidatesRef.current[fromId] = [];

      if (isOffer) {
        const answer = await pc.createAnswer();
        await pc.setLocalDescription(answer);
        sendSignal(fromId, { sdp: pc.localDescription });
      }
    } catch (err) {
      settingRemoteRef.current[fromId] = false;

      try {
        console.error(`[handleSignal] Error handling SDP from ${fromId}:`, err);

        try {
          const trans = (pc.getTransceivers && pc.getTransceivers().map(t => ({ kind: t.kind, mid: t.mid, direction: t.direction }))) || [];
          console.warn(`[handleSignal] pc.getTransceivers() for ${fromId}:`, trans);
        } catch (tErr) {
          console.warn("Failed to dump transceivers", tErr);
        }

        if (msg.sdp && msg.sdp.sdp) {
          const incoming = msg.sdp.sdp.match(/^m=.*$/gm) || [];
          console.warn("incoming m-lines:", incoming);
        }
        if (pc.localDescription && pc.localDescription.sdp) {
          const localM = pc.localDescription.sdp.match(/^m=.*$/gm) || [];
          console.warn("local m-lines:", localM);
        }
      } catch (dumpErr) {
        console.error("Failed to dump SDPs for debug:", dumpErr);
      }

      const message = (err && err.message) ? err.message.toLowerCase() : "";
      const isMLineError =
        message.includes("order of m-lines") ||
        message.includes("failed to set remote offer sdp") ||
        message.includes("failed to set remote description") ||
        message.includes("session error code");

      if (isMLineError) {
        console.warn(`[handleSignal] Detected m-line / SDP mismatch from ${fromId} — attempting recovery: close, recreate, re-negotiate.`);

        try {
          try { pcsRef.current[fromId]?.close(); } catch (e) {}
          delete pcsRef.current[fromId];
          pendingCandidatesRef.current[fromId] = [];

          await new Promise((r) => setTimeout(r, 150 + Math.floor(Math.random()*200)));

          createPeerConnection(fromId);

          await safeNegotiateOffer(fromId);
        } catch (recoveryErr) {
          console.error(`[handleSignal] Recovery attempt for ${fromId} failed:`, recoveryErr);
        }

        return;
      }

      console.error(`[handleSignal] Error handling SDP from ${fromId}:`, err);
    }
  } else if (msg.candidate) {
    try {
      if (pc.remoteDescription && pc.remoteDescription.type) {
        await pc.addIceCandidate(msg.candidate);
      } else {
        pendingCandidatesRef.current[fromId] =
          pendingCandidatesRef.current[fromId] || [];
        pendingCandidatesRef.current[fromId].push(msg.candidate);
      }
    } catch (err) {
      console.error(`[handleSignal] Error adding ICE candidate from ${fromId}:`, err);
    }
  }
}


  function sendSignal(targetId, payload) {
    socketRef.current?.emit("signal", targetId, JSON.stringify(payload));
  }

  async function toggleMute() {
  const expectedNext = !muted;

  try {
    // delegate to mediaController
    const newMuted = await mediaToggleAudio(muted);

    setMuted(newMuted);
    if (mutedRef) mutedRef.current = newMuted;

    if (newMuted) {
      // 🔇 Muting host mic
      try { window.stopTranscription?.(); } catch {}
      try {
        const rec = recordersRef.current?.["local"];
        if (rec?.recorder && rec.recorder.state !== "inactive") rec.recorder.stop();
        if (recordersRef.current) delete recordersRef.current["local"];
      } catch {}
      setActiveSpeakerId(prev => (prev === "local" ? null : prev));
    } else {
      // 🔊 Unmuting host mic
      try { window.startTranscription?.(localStreamRef.current); } catch {}
      try { window.startRecording?.("audio", localStreamRef.current); } catch {}
    }
  } catch (err) {
    console.error("toggleMute error:", err);
    if (mutedRef) mutedRef.current = muted; // rollback
  }
}

async function toggleVideo() {
  const expectedNext = !videoOff;

  try {
    // delegate to mediaController
    const newVideoOff = await mediaToggleVideo(videoOff);

    setVideoOff(newVideoOff);
    if (videoOffRef) videoOffRef.current = newVideoOff;

    if (newVideoOff) {
      // --- TURN CAMERA OFF ---
      try { stopPeriodicEmotionCapture(); } catch {}
      try { window.stopRecording?.("video"); } catch {}
      try { removeAnalyzer("local"); } catch {}
      try { createAnalyzerForStream("local", localStreamRef.current); } catch {}
      try { startRecordingForStream("local", localStreamRef.current); } catch {}
    } else {
      // --- TURN CAMERA ON ---
      try { removeAnalyzer("local"); } catch {}
      try { createAnalyzerForStream("local", localStreamRef.current); } catch {}
      try { startRecordingForStream("local", localStreamRef.current); } catch {}
      try { startPeriodicEmotionCapture(); } catch {}
    }
  } catch (err) {
    console.error("toggleVideo error:", err);
    if (videoOffRef) videoOffRef.current = videoOff; // rollback
  }
}




  async function startScreenShare() {
    try {
      const screenStream = await navigator.mediaDevices.getDisplayMedia({
        video: true,
        audio: false,
      });

      const screenTrack = screenStream.getVideoTracks()[0];

      prevLocalStreamRef.current = localStreamRef.current;

      localStreamRef.current = screenStream;
      setLocalStream(localStreamRef.current);
      if (localVideoRef.current) {
        localVideoRef.current.srcObject = screenStream;
        setVideoElement(localVideoRef.current);
      }

      Object.values(pcsRef.current).forEach((pc) => {
        const sender = pc
          .getSenders()
          .find((s) => s.track && s.track.kind === "video");
        if (sender) {
          try {
            sender.replaceTrack(screenTrack);
          } catch (err) {
            console.warn("replaceTrack(screen) failed:", err);
          }
        }
      });
      socketRef.current?.emit("update-participant-state", { screen: true });

      screenTrack.onended = async () => {
        try {
          let camStream = prevLocalStreamRef.current;

          if (!camStream) {
            try {
              camStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
            } catch (err) {
              console.warn("Could not reacquire camera after screen share:", err);
            }
          }

          if (camStream) {
            localStreamRef.current = camStream;
            setLocalStream(localStreamRef.current);

            const camTrack = camStream.getVideoTracks()[0];
            if (camTrack) {
              Object.values(pcsRef.current).forEach((pc) => {
                const sender = pc.getSenders().find((s) => s.track && s.track.kind === "video");
                if (sender) {
                  try { sender.replaceTrack(camTrack); } catch (err) { console.warn("replaceTrack(cam) failed:", err); }
                }
              });
            }

            if (localVideoRef.current) {
              localVideoRef.current.srcObject = localStreamRef.current;
              setVideoElement(localVideoRef.current);
            }

            createAnalyzerForStream("local", camStream);
            startRecordingForStream("local", camStream);
          } else {
            if (localVideoRef.current) localVideoRef.current.srcObject = null;
            localStreamRef.current = null;
            setLocalStream(null);
          }

          socketRef.current?.emit("update-participant-state", { screen: false });
        } finally {
          prevLocalStreamRef.current = null;
          try { screenStream.getTracks().forEach((t) => { if (t.readyState !== "ended") t.stop(); }); } catch {}
        }
      };
    } catch (err) {
      console.error("Screen share error:", err);
    }
  }

  function sendChatMessage(text) {
  if (!text || !socketRef.current) return;

  const userId = localStorage.getItem("userId");
  const name = localStorage.getItem("displayName") || "Guest";

  const id = crypto.randomUUID();
  const msg = {
    id,
    userId,
    fromSocketId: socketRef.current.id,
    name,
    text: String(text).slice(0, 2000),
    meta: { name, userId },
    ts: Date.now(),
    pending: true,
  };

  // 👌 Optimistically add
  seenMsgIdsRef.current.add(id);
  setChatMessages((prev) => [...prev, msg]);

  // Send to server
  socketRef.current.emit("chat-message", roomId, msg);
}




  function closePeer(peerId) {
    const pc = pcsRef.current[peerId];
    if (pc) {
      try {
        pc.close();
      } catch {}
      delete pcsRef.current[peerId];
    }
    setRemoteStreams((prev) => {
      const copy = { ...prev };
      delete copy[peerId];
      return copy;
    });
    delete makingOfferRef.current[peerId];
    delete ignoreOfferRef.current[peerId];
    delete politeRef.current[peerId];
    pendingCandidatesRef.current[peerId] = [];
  }

  async function cleanupAll() {
  // 🔌 Disconnect signaling
  try {
    socketRef.current?.removeAllListeners?.();
    socketRef.current?.disconnect();
  } catch {}

  try { window.myId = null; } catch {}

  // 🎙️ Stop all local media tracks
  try {
    localStreamRef.current?.getTracks()?.forEach(t => t.stop());
  } catch {}
  try {
    prevLocalStreamRef.current?.getTracks()?.forEach(t => t.stop());
  } catch {}
  try {
    const prev = window.__previousLocalStreamForToggle;
    if (prev && prev.getTracks) {
      prev.getTracks().forEach(t => t.stop());
    }
    window.__previousLocalStreamForToggle = null;
  } catch {}

  // 🖼️ Stop placeholder canvas stream if present
  try {
    if (localStreamRef.current && localStreamRef.current.__placeholderCanvas) {
      localStreamRef.current.getTracks().forEach(t => t.stop());
      localStreamRef.current.__placeholderCanvas = null;
    }
  } catch {}

  // 🖼️ Clear video element
  if (localVideoRef.current) localVideoRef.current.srcObject = null;
  localStreamRef.current = null;
  prevLocalStreamRef.current = null;

  // 🎚️ Cleanup analyzers & animation loop
  try {
    Object.keys(analyzersRef.current).forEach(id => removeAnalyzer(id));
  } catch {}
  if (rafRef.current) {
    cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
  }

  // ❌ Close all peer connections
  try {
    Object.keys(pcsRef.current).forEach(pid => {
      try { pcsRef.current[pid].close(); } catch {}
    });
  } catch {}
  pcsRef.current = {};
  setRemoteStreams({});
  setParticipantsMeta([]);

  // 🔇 Close AudioContext (mic lock)
  try {
    if (audioContextRef.current) {
      audioContextRef.current.close().catch(() => {});
      audioContextRef.current = null;
    }
  } catch {}

  // 🛑 Stop any background emotion capture loops
  stopPeriodicEmotionCapture();

  // 🛡️ Final safety: force release mic & camera
  try {
    const tmpStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: true });
    tmpStream.getTracks().forEach(track => track.stop());
  } catch (e) {
    console.warn("Force release failed:", e);
  }
}



async function leaveCall() {
  try {
    if (socketRef.current?.connected) {
      socketRef.current.emit("leave-call", roomId);
      await new Promise(r => setTimeout(r, 50));
    }
  } catch (e) {
    console.warn("leaveCall emit failed", e);
  }

  cleanupAll();
  navigate("/home");
}



async function endMeeting() {
  try {
    if (isHost) {
      // Host can still upload transcript if you want to keep that feature
      try {
        await uploadRecordingsAndStoreTranscript();
      } catch (e) {
        console.warn("uploadRecordingsAndStoreTranscript failed", e);
      }

      if (socketRef.current?.connected) {
        socketRef.current.emit("end-meeting", roomId);
        await new Promise((r) => setTimeout(r, 50));
      }
    } else {
      try {
        await leaveCall();
      } catch (e) {
        console.warn("leaveCall failed", e);
      }
    }
  } catch (err) {
    console.error("endMeeting error:", err);
  } finally {
    try {
      cleanupAll();
    } catch (e) {
      console.warn("cleanupAll failed", e);
    }
    navigate("/home");
  }
}


  const remoteEntries = Object.entries(remoteStreams).filter(
    ([peerId, stream]) =>
      peerId && peerId !== myId && stream && stream.getTracks().length
  );

  let spotlightId = null;
  let spotlightStream = null;
  if (activeSpeakerId && remoteStreams[activeSpeakerId]) {
    spotlightId = activeSpeakerId;
    spotlightStream = remoteStreams[activeSpeakerId];
  }

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [chatMessages, chatOpen]);

  useEffect(() => {
    const onKeyDown = (e) => {
      const tag = (e.target && e.target.tagName) || "";
      if (["INPUT", "TEXTAREA"].includes(tag)) return;
      if (e.key === "m" || e.key === "M") toggleMute();
      if (e.key === "v" || e.key === "V") toggleVideo();
      if (e.key === "c" || e.key === "C") setChatOpen((v) => !v);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [muted, videoOff, chatOpen]);

  useEffect(() => {
    Object.entries(remoteStreams).forEach(([peerId, stream]) => {
      if (!peerId) return;
      createAnalyzerForStream(peerId, stream);
    });

    if (localStreamRef.current) createAnalyzerForStream("local", localStreamRef.current);

    const present = new Set(Object.keys(remoteStreams).concat("local"));
    Object.keys(analyzersRef.current).forEach((id) => {
      if (!present.has(id)) removeAnalyzer(id);
    });
  }, [remoteStreams]);

  const isLocal = (peerId) => {
    if (!peerId) return false;
    if (typeof myId !== "undefined" && myId !== null) {
      return peerId === myId;
    }
    return false;
  };

function formatTopEmotion(emotion) {
  if (!emotion) return null;
  try {
    // 1) explicit top object { top: { label, score } }
    if (emotion.top && emotion.top.label) {
      return { label: String(emotion.top.label).toLowerCase(), score: Number(emotion.top.score) || 0 };
    }

    // 2) summary object (server might send that shape)
    if (emotion.summary && typeof emotion.summary === "object") {
      const entries = Object.entries(emotion.summary).filter(([k, v]) => typeof v === "number");
      if (entries.length) {
        entries.sort((a, b) => b[1] - a[1]);
        return { label: String(entries[0][0]).toLowerCase(), score: Number(entries[0][1]) || 0 };
      }
    }

    // 3) common nested probability shapes
    const nestedKeys = ["emotion_probs", "scores", "probs", "probabilities", "results"];
    for (const nk of nestedKeys) {
      const nested = emotion[nk];
      if (nested && typeof nested === "object") {
        const entries = Object.entries(nested).filter(([k, v]) => typeof v === "number");
        if (entries.length) {
          entries.sort((a, b) => b[1] - a[1]);
          return { label: String(entries[0][0]).toLowerCase(), score: Number(entries[0][1]) || 0 };
        }
      }
    }

    // 4) fallback: top-level numeric entries — but ignore meta/internal keys (very important)
    const isMetaKey = (k) =>
      k.startsWith("__") ||
      ["anomaly_score", "anomalyFlag", "anomaly_flag", "timestamp", "ts", "time", "frameTime"].includes(k);
    const entries = Object.entries(emotion).filter(([k, v]) => typeof v === "number" && !isMetaKey(k));
    if (entries.length) {
      entries.sort((a, b) => b[1] - a[1]);
      return { label: String(entries[0][0]).toLowerCase(), score: Number(entries[0][1]) || 0 };
    }

    // 5) emotion is a plain string label
    if (typeof emotion === "string") return { label: emotion.toLowerCase(), score: 0 };

    return null;
  } catch (err) {
    console.warn("formatTopEmotion error:", err);
    return null;
  }
}

const EMOJI_MAP = {
  angry: "😠",
  disgust: "🤢",
  fear: "😨",
  happy: "😄",
  neutral: "😐",
  sad: "😢",
  joy: "😄",
  surprise: "😮",
  contempt: "😒",
};

function getTopEmotionLabel(emotion) {
  const top = formatTopEmotion(emotion);
  return top ? String(top.label).toLowerCase() : null;
}

function renderEmojiLabelForEmotion(emotion) {
  const label = getTopEmotionLabel(emotion);
  if (!label) return null;
  const emoji = EMOJI_MAP[label] || "🫥";
  const labelCap = label.charAt(0).toUpperCase() + label.slice(1);
  return `${emoji} ${labelCap}`;
}

  function renderEmotionBadgeForId(participantId) {
  const em = emotionsMap[participantId];
  if (!em) return null;
  const display = renderEmojiLabelForEmotion(em);
  if (!display) return null;
  return (
    <div
      style={{
        position: "absolute",
        right: 8,
        top: 8,
        background: "rgba(0,0,0,0.6)",
        color: "white",
        padding: "4px 8px",
        borderRadius: 12,
        fontSize: 12,
        zIndex: 1000,
        pointerEvents: "none",
      }}
    >
      {display}
    </div>
  );
}


function ParticipantCard({ peerId, stream, compact = false, style = {} }) {
  const videoRef = useRef(null);
  const lastFrameTsRef = useRef(Date.now());
  const lastTimeRef = useRef(0);
  const [stalled, setStalled] = useState(false);

  const meta = participantsMeta.find((p) => p.id === peerId)?.meta || {};
  const em = emotionsMap[peerId];
  const name =
    (em && (em.__name || em.name || em.displayName || em.display_name)) ||
    meta.name ||
    (peerId ? peerId.slice(0, 6) : "Unknown");
  const isSpeaking = activeSpeakerId === peerId;

  const hasVideoTrack =
    stream &&
    typeof stream.getVideoTracks === "function" &&
    stream.getVideoTracks().some((t) => t.readyState === "live");

  // we consider "displaying video" only when there is a live track AND not stalled
  const showVideo = hasVideoTrack && !stalled;

  useEffect(() => {
    const el = videoRef.current;
    if (!el) return;

    // Helper: attach or clear srcObject depending on stalled state & stream
    const applyStreamToElement = () => {
      const live = hasVideoTrack;
      if (live && !stalled) {
        if (el.srcObject !== stream) {
          try {
            el.srcObject = stream;
            el.style.transform = "none";
            el.style.WebkitTransform = "none";
          } catch (err) {
            console.warn("failed to assign srcObject on participant video", err);
          }
        }
      } else {
        // If no live video or stalled, clear srcObject so browser stops showing last frame
        if (el.srcObject) {
          try {
            el.srcObject = null;
          } catch (err) {
            console.warn("failed to clear srcObject on participant video", err);
          }
        }
      }
    };

    // frame arrival callbacks
    let vframeHandle = null;
    let monitorInterval = null;
    let fallbackInterval = null;

    const STALL_MS = 2000; // adjust if you want faster/slower reaction
    const CHECK_MS = 300;

    // called when a new frame is rendered
    const onVideoFrame = () => {
      lastFrameTsRef.current = Date.now();
      // re-register the callback
      try {
        if (el.requestVideoFrameCallback) {
          vframeHandle = el.requestVideoFrameCallback(() => onVideoFrame());
        }
      } catch (err) {
        // ignore
      }
    };

    // start frame callback if supported
    try {
      if (typeof el.requestVideoFrameCallback === "function") {
        // prime it
        vframeHandle = el.requestVideoFrameCallback(() => onVideoFrame());
      } else {
        // fallback: monitor currentTime changes
        lastTimeRef.current = el.currentTime || 0;
        fallbackInterval = setInterval(() => {
          const ct = el.currentTime || 0;
          if (ct > lastTimeRef.current + 0.0001) {
            lastTimeRef.current = ct;
            lastFrameTsRef.current = Date.now();
          }
        }, 250);
      }
    } catch (err) {
      console.warn("video frame callback start failed", err);
    }

    // monitor to detect stall; toggles `stalled` state and applies stream attachments
    monitorInterval = setInterval(() => {
      const elapsed = Date.now() - lastFrameTsRef.current;
      if (elapsed > STALL_MS) {
        if (!stalled) {
          console.debug("[video-monitor] stalled for", peerId, elapsed, "ms - clearing display");
          setStalled(true);
        }
      } else {
        if (stalled) {
          console.debug("[video-monitor] frames resumed for", peerId);
          setStalled(false);
        }
      }
      // ensure srcObject applied/cleared according to new value
      applyStreamToElement();
    }, CHECK_MS);

    // Also apply immediately
    applyStreamToElement();

    return () => {
      // cleanup
      try {
        if (vframeHandle && typeof el.cancelVideoFrameCallback === "function") {
          el.cancelVideoFrameCallback(vframeHandle);
        }
      } catch (e) {}
      try {
        if (monitorInterval) clearInterval(monitorInterval);
        if (fallbackInterval) clearInterval(fallbackInterval);
      } catch (e) {}

      try {
        if (el && el.srcObject) el.srcObject = null;
      } catch (e) {}
    };
    // run when stream changes so we reattach; also re-run when hasVideoTrack changes
  }, [stream, peerId, hasVideoTrack, stalled]);

  // reset stalled when stream changes (so new stream starts as not-stalled)
  useEffect(() => {
    setStalled(false);
    lastFrameTsRef.current = Date.now();
  }, [stream]);

  return (
    <motion.div
      layout
      initial={{ opacity: 0, scale: 0.96 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.96 }}
      className={`${styles.participantCard} ${compact ? styles.participantCardCompact : ""} ${
        isSpeaking ? styles.participantCardSpeaking : ""
      }`}
      title={name}
      style={{
        width: compact ? 160 : "100%",
        height: compact ? 90 : "100%",
        aspectRatio: compact ? "16/9" : undefined,
        ...style,
      }}
    >
      {showVideo ? (
        <video
          autoPlay
          playsInline
          ref={videoRef}
          // muted attribute may be needed locally to prevent echo; keep current behavior
        />
      ) : (
        <div className={styles.cameraOffPlaceholder}>
          <div style={{ textAlign: "center" }}>
            <div style={{ fontSize: compact ? 18 : 36 }}>
              {name && name[0] ? name[0].toUpperCase() : <FaUserAlt />}
            </div>
            {!compact && (
              <div style={{ opacity: 0.9, marginTop: 6, fontSize: 13 }}>
                {name} • Camera off
              </div>
            )}
          </div>
        </div>
      )}

      <div className={styles.participantOverlay}>
        <div className={styles.namePill}>
          <FaUserAlt /> <span>{name}</span>
        </div>
        {isSpeaking && (
          <div
            aria-hidden
            style={{
              position: "absolute",
              right: 8,
              top: 8,
              background: "rgba(0,150,255,0.15)",
              color: "rgba(0,150,255,0.95)",
              padding: "4px 8px",
              borderRadius: 6,
              fontWeight: 600,
              fontSize: 12,
            }}
          >
            Speaking
          </div>
        )}
      </div>

      {(isHost || DEBUG_SHOW_EMOTION_FOR_EVERYONE) && renderEmotionBadgeForId(peerId)}
    </motion.div>
  );
}

// SpotlightCard with the same robust video-frame monitor
function SpotlightCard({ id, stream }) {
  const videoRef = useRef(null);
  const lastFrameTsRef = useRef(Date.now());
  const lastTimeRef = useRef(0);
  const [stalled, setStalled] = useState(false);

  const meta = participantsMeta.find((p) => p.id === id)?.meta || {};
  const em = emotionsMap[id];
  const name =
    (em && (em.__name || em.name || em.displayName || em.display_name)) ||
    meta.name ||
    (id ? id.slice(0, 6) : "Unknown");
  const isSpeaking = activeSpeakerId === id;

  const hasVideoTrack =
    stream &&
    typeof stream.getVideoTracks === "function" &&
    stream.getVideoTracks().some((t) => t.readyState === "live");

  const showVideo = hasVideoTrack && !stalled;

  useEffect(() => {
    const el = videoRef.current;
    if (!el) return;

    const applyStreamToElement = () => {
      const live = hasVideoTrack;
      if (live && !stalled) {
        if (el.srcObject !== stream) {
          try {
            el.srcObject = stream;
            el.style.transform = "none";
            el.style.WebkitTransform = "none";
          } catch (err) {
            console.warn("failed to assign srcObject on spotlight video", err);
          }
        }
      } else {
        if (el.srcObject) {
          try {
            el.srcObject = null;
          } catch (err) {
            console.warn("failed to clear srcObject on spotlight video", err);
          }
        }
      }
    };

    let vframeHandle = null;
    let monitorInterval = null;
    let fallbackInterval = null;

    const STALL_MS = 2000;
    const CHECK_MS = 300;

    const onVideoFrame = () => {
      lastFrameTsRef.current = Date.now();
      try {
        if (el.requestVideoFrameCallback) {
          vframeHandle = el.requestVideoFrameCallback(() => onVideoFrame());
        }
      } catch (err) {}
    };

    try {
      if (typeof el.requestVideoFrameCallback === "function") {
        vframeHandle = el.requestVideoFrameCallback(() => onVideoFrame());
      } else {
        lastTimeRef.current = el.currentTime || 0;
        fallbackInterval = setInterval(() => {
          const ct = el.currentTime || 0;
          if (ct > lastTimeRef.current + 0.0001) {
            lastTimeRef.current = ct;
            lastFrameTsRef.current = Date.now();
          }
        }, 250);
      }
    } catch (err) {
      console.warn("video frame callback start failed", err);
    }

    monitorInterval = setInterval(() => {
      const elapsed = Date.now() - lastFrameTsRef.current;
      if (elapsed > STALL_MS) {
        if (!stalled) {
          console.debug("[video-monitor] spotlight stalled for", id, elapsed, "ms - clearing display");
          setStalled(true);
        }
      } else {
        if (stalled) {
          console.debug("[video-monitor] spotlight frames resumed for", id);
          setStalled(false);
        }
      }
      applyStreamToElement();
    }, CHECK_MS);

    applyStreamToElement();

    return () => {
      try {
        if (vframeHandle && typeof el.cancelVideoFrameCallback === "function") {
          el.cancelVideoFrameCallback(vframeHandle);
        }
      } catch (e) {}
      try {
        if (monitorInterval) clearInterval(monitorInterval);
        if (fallbackInterval) clearInterval(fallbackInterval);
      } catch (e) {}
      try {
        if (el && el.srcObject) el.srcObject = null;
      } catch (e) {}
    };
  }, [stream, id, hasVideoTrack, stalled]);

  useEffect(() => {
    setStalled(false);
    lastFrameTsRef.current = Date.now();
  }, [stream]);

  return (
    <motion.div
      layout
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.98 }}
      className={`${styles.spotlight} ${isSpeaking ? styles.speaking : ""}`}
      style={{
        width: "100%",
        height: "100%",
        borderRadius: 12,
        position: "relative",
        display: "flex",
      }}
    >
      {showVideo ? (
        <video autoPlay playsInline ref={videoRef} style={{ width: "100%", height: "100%" }} />
      ) : (
        <div className={styles.cameraOffPlaceholder}>
          <div style={{ textAlign: "center" }}>
            <div style={{ fontSize: 56 }}>
              {name && name[0] ? name[0].toUpperCase() : <FaUserAlt />}
            </div>
            <div style={{ opacity: 0.9, marginTop: 8, fontSize: 16 }}>{name}</div>
            <div style={{ opacity: 0.75, marginTop: 6, fontSize: 13 }}>Camera off</div>
          </div>
        </div>
      )}

      <div style={{ position: "absolute", left: 12, bottom: 12 }}>
        <div
          style={{
            background: "rgba(0,0,0,0.45)",
            color: "white",
            padding: "6px 10px",
            borderRadius: 8,
            display: "flex",
            alignItems: "center",
            gap: 8,
            fontWeight: 700,
          }}
        >
          <FaUserAlt />
          <span>{name}</span>
        </div>
      </div>

      {isSpeaking && (
        <div
          style={{
            position: "absolute",
            right: 12,
            top: 12,
            background: "rgba(0,150,255,0.15)",
            color: "rgba(0,150,255,0.95)",
            padding: "6px 10px",
            borderRadius: 6,
            fontWeight: 700,
            fontSize: 13,
          }}
        >
          Speaking
        </div>
      )}

      {(isHost || DEBUG_SHOW_EMOTION_FOR_EVERYONE) && renderEmotionBadgeForId(id)}
    </motion.div>
  );
}



  function EmotionServicePanel() {
  if (!isHost && !DEBUG_SHOW_EMOTION_FOR_EVERYONE) return null;
  const rows = Object.entries(emotionsMap || {});

  function findTopInAnyShape(em) {
    try {
      const direct = renderEmojiLabelForEmotion(em);
      if (direct) return direct;
    } catch (e) {}

    const candidateKeys = ["emotion", "emotion_probs", "scores", "probs", "probabilities", "results", "summary"];
    for (const k of candidateKeys) {
      if (em && typeof em === "object" && em[k]) {
        try {
          const top = formatTopEmotion(em[k]);
          if (top && top.label) {
            const emoji = EMOJI_MAP[top.label] || "🫥";
            const labelCap = top.label.charAt(0).toUpperCase() + top.label.slice(1);
            return `${emoji} ${labelCap}`;
          }
        } catch (e) {}
      }
    }

    if (em && typeof em === "object") {
      for (const v of Object.values(em)) {
        if (v && typeof v === "object") {
          try {
            const top2 = formatTopEmotion(v);
            if (top2 && top2.label) {
              const emoji = EMOJI_MAP[top2.label] || "🫥";
              const labelCap = top2.label.charAt(0).toUpperCase() + top2.label.slice(1);
              return `${emoji} ${labelCap}`;
            }
          } catch (e) {}
        }
      }
    }
    return null;
  }

  return (
    <div className={styles.emotionPanel}>
      <div className={styles.emotionStreamHeader}>
        <FaRegComments /> <span>Emotion Stream</span>
      </div>
      {rows.length === 0 ? (
        <div className={styles.emotionEmpty}>No emotion updates yet</div>
      ) : (
        rows.map(([pid, em]) => {
          const nameFromEmotion = em && (em.__name || em.name || em.displayName || em.display_name);
          const nameFromMeta = (participantsMeta.find(p => p.id === pid)?.meta?.name) ||
                               (participantsMeta.find(p => p.id === pid)?.meta?.displayName);
          const displayName = nameFromEmotion || nameFromMeta || (pid ? pid.slice(0,6) : "Unknown");

          const emojiLabel = findTopInAnyShape(em) || "—";

          return (
            <div key={pid} className={styles.emotionRow}>
              <div className={styles.emotionName}>{displayName}</div>
              <div className={styles.emotionEmoji}>{emojiLabel}</div>
            </div>
          );
        })
      )}
    </div>
  );
}



  function ChatInput({ onSend }) {
    const [text, setText] = useState("");
    return (
      <div className={styles.chatInput || ""} style={{ display: "flex", gap: 8, padding: 8 }}>
        <input
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              const t = text.trim();
              if (t) {
                onSend(t);
                setText("");
              }
            }
          }}
          placeholder="Type a message..."
          style={{ flex: 1, padding: "8px 10px", borderRadius: 6, border: "1px solid rgba(255,255,255,0.06)", background: "rgba(255,255,255,0.02)", color: "#E6EEF9" }}
        />
        <button
          onClick={() => {
            const t = text.trim();
            if (t) {
              onSend(t);
              setText("");
            }
          }}
          style={{ padding: "8px 12px", borderRadius: 6, background: "rgba(0,150,255,0.12)", border: "none", color: "#E6EEF9" }}
        >
          Send
        </button>
      </div>
    );
  }

  return (
    <div className={`${styles.meetVideoContainer} ${chatOpen ? styles.chatOpen : ""}`}>
      <div className={styles.bgSparkles} aria-hidden />

      {connecting && <div className={styles.connecting}>Connecting...</div>}

      <div className={styles.conferenceWrap}>
        <div className={styles.conferenceView} aria-live="polite">
          <AnimatePresence>
            {spotlightStream ? (
              <motion.div
                key={spotlightId || "spotlight"}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                style={{
                  display: "flex",
                  gap: 12,
                  width: "100%",
                  height: "100%",
                  alignItems: "stretch",
                }}
              >
                {/* Left: Large Spotlight */}
                <div style={{ flex: 1, minHeight: 320 }}>
                  <SpotlightCard id={spotlightId} stream={spotlightStream} />
                </div>

                {/* Right: Vertical list of *other* remote participants (no local duplicate) */}
                <div
                  className={styles.rightColumn}
                  style={{
                    width: 320,
                    display: "flex",
                    flexDirection: "column",
                    gap: 10,
                    alignItems: "stretch",
                    overflowY: "auto",
                    paddingBottom: 6,
                    boxSizing: "border-box",
                  }}
                >
                  {/* Render other remotes (exclude spotlight and local/myId) */}
                  {remoteEntries
                    .filter(([peerId]) => peerId !== spotlightId && peerId !== myId)
                    .map(([peerId, stream]) => (
                      <div key={peerId} style={{ height: 140 }}>
                        <ParticipantCard peerId={peerId} stream={stream} compact />
                      </div>
                    ))}

                  {/* optional filler to keep spacing stable */}
                  <div style={{ flex: 1 }} />
                </div>
              </motion.div>
            ) : (

              <>
                {remoteEntries.length === 0 && !connecting && (
                  <motion.div
                    key="empty-state"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className={styles.emptyState}
                  >
                    <div style={{ textAlign: "center", maxWidth: 640 }}>
                      <h2 style={{ margin: "0 0 8px 0" }}>
                        You're the only one here
                      </h2>
                      <p style={{ margin: 0, color: "rgba(230,238,249,0.66)" }}>
                        Share the room URL with others to start a call — your
                        preview is on the bottom-right.
                      </p>
                    </div>
                  </motion.div>
                )}

                <div
                  style={{
                    display: "grid",
                    gap: 12,
                    gridTemplateColumns:
                      remoteEntries.length <= 1 ? "1fr" : "repeat(auto-fill, minmax(320px, 1fr))",
                    width: "100%",
                  }}
                >
                  {remoteEntries.map(([peerId, stream]) => (
                    <ParticipantCard key={peerId} peerId={peerId} stream={stream} />
                  ))}
                </div>
              </>
            )}
          </AnimatePresence>
        </div>
      </div>

      <motion.div
        className={styles.localPreview}
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        drag
        dragConstraints={{ left: 0, right: 0, top: 0, bottom: 0 }}
        dragMomentum={false}
        whileHover={{ scale: 1.02 }}
        aria-label="Local preview"
        title="Local preview (drag to reposition)"
        style={{
          width: 200,
          height: 112,
          borderRadius: 8,
          overflow: "hidden",
          position: "fixed",
          right: 20,
          bottom: 20,
          zIndex: 80,
        }}
      >
        <video
          ref={localVideoRef}
          autoPlay
          muted
          playsInline
          style={{ width: "100%", height: "100%", objectFit: "cover" }}
        />
        <div className={styles.youBadge} style={activeSpeakerId === "local" ? { boxShadow: "0 0 14px rgba(0,150,255,0.9)" } : {}}>
          You
        </div>
        {/* Removed aria-hidden on this wrapper to avoid hiding focusable children from AT */}
        <div className={styles.previewControls}>
          <button
            className={`${styles.iconButton} ${muted ? styles.active : ""}`}
            onClick={toggleMute}
            aria-label={muted ? "Unmute" : "Mute"}
            title={muted ? "Unmute" : "Mute"}
            style={{ minWidth: 40, minHeight: 40 }}
          >
            {muted ? <FaMicrophoneSlash /> : <FaMicrophone />}
          </button>

          <button
            className={`${styles.iconButton} ${videoOff ? styles.active : ""}`}
            onClick={toggleVideo}
            aria-label={videoOff ? "Turn camera on" : "Turn camera off"}
            title={videoOff ? "Turn camera on" : "Turn camera off"}
            style={{ minWidth: 40, minHeight: 40 }}
          >
            {videoOff ? <FaVideoSlash /> : <FaVideo />}
          </button>

          {/* Share emotion toggle button (captures remote tiles only) */}
{(isHost || DEBUG_SHOW_EMOTION_FOR_EVERYONE) && (
  <button
    className={`${styles.iconButton} ${shareEmotion ? styles.active : ""}`}
    onClick={() => {
      const next = !shareEmotion;
      setShareEmotion(next);
    }}
    aria-pressed={shareEmotion}
    aria-label={shareEmotion ? "Stop sending remote emotion clips" : "Send remote emotion clips to host"}
    title={shareEmotion ? "Stop sending remote emotion clips" : "Send remote emotion clips to host"}
    style={{ minWidth: 40, minHeight: 40, fontSize: 16 }}
  >
    😊
  </button>
)}

        </div>
      </motion.div>

      <AnimatePresence>
  {chatOpen && (
    <motion.aside
      className={styles.chatRoom}
      initial={{ opacity: 0, x: 80 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 80 }}
    >
      <div className={styles.chatHeader}>
        <FaRegComments />
        <strong>Chat</strong>
        <div
          style={{
            marginLeft: "auto",
            fontWeight: 600,
            color: "rgba(230,238,249,0.7)",
          }}
        >
          {participantsMeta.length} in call
        </div>
      </div>

      <div
        ref={chatContainerRef}
        className={styles.chatMessages}
        role="log"
        aria-live="polite"
      >
        {chatMessages.map((m) => {
          const socketId = socketRef.current?.id;
          const myUserId = localStorage.getItem("userId") || socketId;
          const isOwn = m.from === myUserId || m.userId === myUserId;

          const key = `${m.from ?? "anon"}:${m.ts ?? Math.random()}`;
          return (
            <motion.div
              key={key}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 8 }}
              style={{
                display: "flex",
                justifyContent: isOwn ? "flex-end" : "flex-start",
                margin: "6px 0",
              }}
            >
              <div
                style={{
                  background: isOwn ? "#DCF8C6" : "#FFF",
                  color: "#111",
                  padding: "8px 12px",
                  borderRadius: "16px",
                  maxWidth: "70%",
                  wordBreak: "break-word",
                  boxShadow: "0 1px 2px rgba(0,0,0,0.15)",
                }}
              >
                <div>{m.text}</div>
                <div
                  style={{
                    marginTop: 4,
                    fontSize: 11,
                    opacity: 0.7,
                    display: "flex",
                    justifyContent: "space-between",
                    gap: 8,
                  }}
                >
                  <span style={{ fontWeight: 600 }}>
                    {m.meta?.name || (isOwn ? "You" : m.from)}
                  </span>
                  <span>{new Date(m.ts).toLocaleTimeString()}</span>
                </div>
              </div>
            </motion.div>
          );
        })}
        <div ref={chatEndRef} />
      </div>

      <div className={styles.chattingArea}>
        <ChatInput onSend={(t) => sendChatMessage(t)} />
      </div>
    </motion.aside>
  )}
</AnimatePresence>


      <div className={styles.buttonContainers} role="toolbar" aria-label="Call controls">
        <button onClick={toggleMute} className={`${styles.iconButton} ${muted ? styles.active : ""}`} aria-pressed={muted} aria-label={muted ? "Unmute mic" : "Mute mic"} title="Mute / Unmute (M)">
          {muted ? <FaMicrophoneSlash /> : <FaMicrophone />}
        </button>

        <button onClick={toggleVideo} className={`${styles.iconButton} ${videoOff ? styles.active : ""}`} aria-pressed={videoOff} aria-label={videoOff ? "Turn camera on" : "Turn camera off"} title="Toggle Camera (V)">
          {videoOff ? <FaVideoSlash /> : <FaVideo />}
        </button>

        <button onClick={startScreenShare} className={styles.iconButton} aria-label="Share screen" title="Share screen">
          <FaDesktop />
        </button>

        <button onClick={() => setChatOpen((v) => !v)} className={styles.iconButton} aria-pressed={chatOpen} aria-label="Open chat" title="Toggle chat (C)">
          <FaComments />
        </button>

        {isHost ? (
  <button
    onClick={endMeeting}
    className={`${styles.iconButton} ${styles.leaveButton}`}
    aria-label="End meeting"
    title="End meeting"
  >
    <FaPhoneSlash />
  </button>
) : (
  <button
    onClick={leaveCall}
    className={`${styles.iconButton} ${styles.leaveButton}`}
    aria-label="Leave call"
    title="Leave call"
  >
    <FaPhoneSlash />
  </button>
)}

      </div>

      <EmotionServicePanel />
    </div>
  );
}

