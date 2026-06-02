import React, {
  useEffect,
  useRef,
  useState,
  useContext,
  useCallback,
  useMemo,
} from "react";
import { useParams, useNavigate } from "react-router-dom";
import { AuthContext } from "../contexts/AuthContext";
import useMediaBridge from "../hooks/useMediaBridge";
import useMeetingLifecycle from "../hooks/useMeetingLifecycle";
import { TRANSCRIPTS_ENABLED } from "../environment";
import ParticipantCard from "./ParticipantCard";
import SpotlightCard from "./SpotlightCard";
import useChat from "../hooks/useChat";
import useEmotionSocket from "../hooks/useEmotionSocket";
import {
  SOCKET_SERVER_URL,
  TRANSCRIPT_ENDPOINT,
  API_BASE,
  ICE_CONFIG,
} from "./meetConfig";
import { motion, AnimatePresence } from "framer-motion";
import useWebRTC from "../hooks/useWebRTC";
import useSocket from "../hooks/useSocket";
import useRecording from "../hooks/useRecording";
import useMediaControls from "../hooks/useMediaControls";
import EmotionServicePanel from "./EmotionServicePanel";
import useAudioAnalyzer from "../hooks/useAudioAnalyzer";
import useEmotionCapture from "../hooks/useEmotionCapture";
import MeetTopBar from "./MeetTopBar";
import MeetControlBar from "./MeetControlBar";
import MeetChatPanel from "./MeetChatPanel";
import MeetLocalPreview from "./MeetLocalPreview";
import MobilePanelSheet from "./MobilePanelSheet";
import s from "../styles/videoComponent.module.css";

const HOST_ONLY_EMOTION = true;

export function unwrapStream(entry) {
  return entry instanceof MediaStream ? entry : null;
}

export function videoKey(stream) {
  if (!stream) return "no-stream";
  const tracks = stream.getVideoTracks?.() ?? [];
  const ids = tracks.map((t) => `${t.id}_${t.readyState}`).join("|");
  return `${stream.id}__${ids || "novt"}`;
}

export function isRenderableVideo(el) {
  return (
    el != null &&
    el.readyState >= 2 &&
    el.videoWidth > 0 &&
    el.videoHeight > 0
  );
}

function useIsMobile(bp = 900) {
  const [val, setVal] = useState(() => window.innerWidth <= bp);
  useEffect(() => {
    const h = () => setVal(window.innerWidth <= bp);
    window.addEventListener("resize", h);
    return () => window.removeEventListener("resize", h);
  }, [bp]);
  return val;
}

function EmptyState() {
  return (
    <div className={s.emptyState}>
      <div className={s.emptyOrb} />
      <div className={s.emptyIcon}>
        <svg width="26" height="26" viewBox="0 0 24 24" fill="none"
          stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round">
          <path d="M23 7l-7 5 7 5V7z" />
          <rect x="1" y="5" width="15" height="14" rx="2" />
        </svg>
      </div>
      <p className={s.emptyTitle}>Waiting for others to join</p>
      <p className={s.emptySub}>Share your meeting link to get started</p>
    </div>
  );
}

export default function VideoMeet() {
  const { roomId } = useParams();
  const navigate = useNavigate();

  const localVideoRef = useRef(null);
  const socketRef = useRef(null);
  const localStreamRef = useRef(null);
  const prevLocalStreamRef = useRef(null);
  const pcsRef = useRef({});
  const remoteStreamsRef = useRef({});
  const chatContainerRef = useRef(null);
  const chatEndRef = useRef(null);
  const cleanupRef = useRef(null);
  const activeSpeakerIdRef = useRef(null);
  const spotlightPeerRef = useRef(null);
  const ignoreOfferRef = useRef({});
  const isSettingRemoteAnswerPending = useRef({});
  const updateParticipantMediaStateRef = useRef(null);
  const prevParticipantMuteStateRef = useRef({}); // pid → muted boolean

  const [endingMeeting, setEndingMeeting] = useState(false);
  const [remoteStreams, setRemoteStreams] = useState({});
  const [connecting, setConnecting] = useState(true);
  const [muted, setMuted] = useState(false);
  const [videoOff, setVideoOff] = useState(false);
  const [chatOpen, setChatOpen] = useState(false);
  const [participantsMeta, setParticipantsMeta] = useState([]);
  const [socketReady, setSocketReady] = useState(0);
  const [spotlightPeerId, setSpotlightPeerId] = useState(null);
  const [myId, setMyId] = useState(null);
  const myIdRef = useRef(null);

  const [shareEmotion, setShareEmotion] = useState(false);
  const [emotionLive, setEmotionLive] = useState(false);
  const [emotionsMap, setEmotionsMap] = useState({});
  const [stableSpeakerId, setStableSpeakerId] = useState(null);
  const [meetDuration, setMeetDuration] = useState(0);
  const [mobileSheet, setMobileSheet] = useState(null);
  const [unreadCount, setUnreadCount] = useState(0);

  const isMobile = useIsMobile(900);
  const participantsMetaRef = useRef([]);
  useEffect(() => { participantsMetaRef.current = participantsMeta; }, [participantsMeta]);

  const { addToUserHistory } = useContext(AuthContext);

  const [isHost, setIsHost] = useState(false);
  const onHostConfirmed = useCallback(() => setIsHost(true), []);
  const myUserId = useMemo(
    () => localStorage.getItem("userId") || "",
    []
  );

  const setMyIdSynced = useCallback((id) => {
    myIdRef.current = id;
    setMyId(id);
  }, []);

  const mutedRef = useRef(muted);
  const videoOffRef = useRef(videoOff);
  useEffect(() => { mutedRef.current = muted; }, [muted]);
  useEffect(() => { videoOffRef.current = videoOff; }, [videoOff]);
  useEffect(() => { activeSpeakerIdRef.current = stableSpeakerId; }, [stableSpeakerId]);

  useEffect(() => {
    const id = setInterval(() => setMeetDuration((d) => d + 1), 1000);
    return () => clearInterval(id);
  }, []);

  const unwrappedRemoteStreams = useMemo(() => {
    const out = {};
    for (const [id, entry] of Object.entries(remoteStreams)) {
      const stream = unwrapStream(entry);
      if (stream) out[id] = stream;
    }
    return out;
  }, [remoteStreams]);

  useEffect(() => {
    remoteStreamsRef.current = unwrappedRemoteStreams;
  }, [unwrappedRemoteStreams]);

  const {
    chatMessages, setChatMessages, sendChatMessage,
    handleIncomingMessage, handleAck, retryMessage, seenMsgIdsRef,
  } = useChat({
    socketRef, roomId, userId: myUserId,
    displayName: localStorage.getItem("displayName") ?? undefined,
  });

  const prevMsgCountRef = useRef(0);
  useEffect(() => {
    const isChatVisible = isMobile ? mobileSheet === "chat" : chatOpen;
    if (isChatVisible) {
      setUnreadCount(0);
      prevMsgCountRef.current = chatMessages.length;
      return;
    }
    const newCount = chatMessages.length - prevMsgCountRef.current;
    if (newCount > 0) setUnreadCount((n) => n + newCount);
    prevMsgCountRef.current = chatMessages.length;
  }, [chatMessages.length, isMobile, mobileSheet, chatOpen]);

  const participantsMetaMap = useMemo(() => {
    const map = {};
    (participantsMeta || []).forEach((p) => {
      map[p.id] = { muted: p?.meta?.muted === true };
    });
    return map;
  }, [participantsMeta]);

  const {
    activeSpeakerId,
    createAnalyzerForStream,
    removeAnalyzer,
    notifyPcsChanged,
  } = useAudioAnalyzer({
    remoteStreams: unwrappedRemoteStreams,
    localStreamRef,
    mutedRef,
    pcsRef,
    participantsMetaMap,
  });

  const {
    recordersRef,
    startRecordingForStream,
    stopAllRecorders,
    uploadRecordingsAndStoreTranscript,
    getSpeechActiveRecordings,
  } = useRecording({
    isHost,
    roomId,
    participantsMetaRef,
    TRANSCRIPTS_ENABLED,
    TRANSCRIPT_ENDPOINT,
    API_BASE,
  });

  const {
    createPeerConnection,
    handleSignal,
    politeRef,
    pendingCandidatesRef,
    makingOfferRef,
  } = useWebRTC({
    socketRef,
    localStreamRef,
    pcsRef,
    setRemoteStreams,
    createAnalyzerForStream,
    removeAnalyzer,
    startRecordingForStream,
    ICE_CONFIG,
    isHost,
  });

  const {
    ensureSocket,
    getSocketForParticipant,
    releaseSocket: releaseEmotionSocket,
    serverCapsRef,
    notifyMediaState,
  } = useEmotionSocket({
    setEmotionsMap,
    updateParticipantMediaStateRef,
  });

  const {
    startPeriodicEmotionCapture,
    stopPeriodicEmotionCapture,
    updateParticipantMediaState,
  } = useEmotionCapture({
    ensureSocket,
    getSocketForParticipant,
    remoteStreamsRef,
    participantsMetaRef,
    myId,
    roomId,
    isHost,
    DEBUG_SHOW_EMOTION_FOR_EVERYONE: false,
    activeSpeakerIdRef,
    localStreamRef,
    serverCapsRef,
  });

  updateParticipantMediaStateRef.current = updateParticipantMediaState;


  useEffect(() => {
    if (!isHost) return;
    const prev = prevParticipantMuteStateRef.current;
    const next = {};

    for (const p of participantsMeta) {
      const userId = p.meta?.userId || p.id;
      if (!userId || userId === myUserId) continue;

      const nowMuted = p.meta?.muted === true;
      next[userId] = nowMuted;

      if (prev[userId] === nowMuted) continue;
      const micEnabled = !nowMuted;

      try {
        notifyMediaState(userId, { micEnabled, cameraEnabled: true });
      } catch { }
    }

    prevParticipantMuteStateRef.current = next;
  }, [participantsMeta, isHost, myUserId, notifyMediaState]);

  const { toggleMute, toggleVideo, startScreenShare } = useMediaControls({
    localStreamRef,
    localVideoRef,
    pcsRef,
    socketRef,
    myUserId,
    createAnalyzerForStream,
    removeAnalyzer,
    startRecordingForStream,
    stopPeriodicEmotionCapture,
    startPeriodicEmotionCapture,
    notifyMediaState,
  });

  const {
    leaveCall, endMeeting, cleanupAll, persistHistorySnapshot,
  } = useMeetingLifecycle({
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
    getSpeechActiveRecordings,
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
    onSocketReady: () => setSocketReady((n) => n + 1),
    onHostConfirmed,
  });


  useEffect(() => { cleanupRef.current = cleanupAll; }, [cleanupAll]);

  useMediaBridge({
    localStreamRef,
    createAnalyzerForStream,
    removeAnalyzer,
    startRecordingForStream,
    stopAllRecorders,
    recordersRef,
    startPeriodicEmotionCapture,
    stopPeriodicEmotionCapture,
    notifyMediaState,
    updateParticipantMediaState,
  });

  const closePeer = useCallback(
    (peerId) => {
      const pc = pcsRef.current[peerId];
      if (pc) {
        try { pc.close(); } catch { }
        delete pcsRef.current[peerId];
      }
      try {
        const rec = recordersRef.current?.[peerId];
        if (rec?.recorder && rec.recorder.state !== "inactive") rec.recorder.stop();
        if (rec?.audioCtx?.state !== "closed") rec?.audioCtx?.close();
      } catch { }
      removeAnalyzer(peerId);
      notifyPcsChanged();

      try { releaseEmotionSocket(peerId); } catch { }
      setRemoteStreams((prev) => {
        if (!prev[peerId]) return prev;
        const next = { ...prev };
        delete next[peerId];
        return next;
      });
      setStableSpeakerId((prev) => (prev === peerId ? null : prev));
      setSpotlightPeerId((prev) => {
        if (prev !== peerId) return prev;
        spotlightPeerRef.current = null;
        return null;
      });
    },
    [
      pcsRef,
      recordersRef,
      removeAnalyzer,
      notifyPcsChanged,
      releaseEmotionSocket]
  );

  const stableSpeakerTimerRef = useRef(null);
  const pendingSpeakerRef = useRef(null);

  useEffect(() => {
    if (activeSpeakerId === stableSpeakerId) return;

    if (activeSpeakerId === null) {
      if (stableSpeakerTimerRef.current) return;
      stableSpeakerTimerRef.current = setTimeout(() => {
        stableSpeakerTimerRef.current = null;
        if (pendingSpeakerRef.current === null) setStableSpeakerId(null);
      }, 2000);
      pendingSpeakerRef.current = null;
      return;
    }

    pendingSpeakerRef.current = activeSpeakerId;

    if (stableSpeakerTimerRef.current) {
      clearTimeout(stableSpeakerTimerRef.current);
      stableSpeakerTimerRef.current = null;
    }

    setStableSpeakerId(activeSpeakerId);
  }, [activeSpeakerId]);

  useEffect(() => () => {
    if (stableSpeakerTimerRef.current) clearTimeout(stableSpeakerTimerRef.current);
  }, []);

  useEffect(() => {
    if (!isHost) return;
    if (shareEmotion) startPeriodicEmotionCapture({});
    else stopPeriodicEmotionCapture();
    if (socketRef.current?.connected) {
      socketRef.current.emit("emotion-status", { active: shareEmotion });
    }
  }, [shareEmotion, isHost, startPeriodicEmotionCapture, stopPeriodicEmotionCapture]);

  useEffect(() => {
    if (isHost) return;
    const socket = socketRef.current;
    if (!socket) return;

    const onEmotionStatus = ({ active }) => {
      setEmotionLive(Boolean(active));
    };

    socket.on("emotion-status", onEmotionStatus);
    socket.emit("get-emotion-status");

    return () => {
      socket.off("emotion-status", onEmotionStatus);
    };
  }, [socketReady, isHost]);

  useEffect(() => {
    const onKeyDown = (e) => {
      const tag = (e.target?.tagName || "").toUpperCase();
      if (tag === "INPUT" || tag === "TEXTAREA" || e.target?.isContentEditable) return;
      if (e.key === "m" || e.key === "M")
        toggleMute(muted, setMuted, mutedRef, TRANSCRIPTS_ENABLED, recordersRef);
      else if (e.key === "v" || e.key === "V")
        toggleVideo(videoOffRef.current, setVideoOff);
      else if (e.key === "c" || e.key === "C") {
        if (isMobile) setMobileSheet((v) => (v === "chat" ? null : "chat"));
        else setChatOpen((v) => !v);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [toggleMute, toggleVideo, isMobile]);

  useEffect(() => () => {
    Object.keys(pcsRef.current).forEach((pid) => closePeer(pid));
  }, []);

  useSocket({
    socketRef,
    roomId,
    setMyId: setMyIdSynced,
    setParticipantsMeta,
    setChatMessages,
    seenMsgIdsRef,
    createPeerConnection,
    politeRef,
    pendingCandidatesRef,
    pcsRef,
    closePeer,
    removeAnalyzer,
    recordersRef,
    setEmotionsMap,
    handleSignal,
    navigate,
    cleanupAll: () => cleanupRef.current?.(),
    persistHistorySnapshot,
    handleIncomingMessage,
    handleAck,
    notifyPcsChanged,
    makingOfferRef,
    socketReady,
    isHost,
    setEmotionLive,
  });

  const remoteEntries = useMemo(
    () =>
      Object.entries(unwrappedRemoteStreams)
        .filter(([peerId]) => {
          if (!peerId) return false;
          if (myIdRef.current && peerId === myIdRef.current) return false;
          if (myId && peerId === myId) return false;
          if (socketRef.current?.id && peerId === socketRef.current.id) return false;
          return true;
        })
        .sort(([a], [b]) => a.localeCompare(b, undefined, { numeric: true })),
    [unwrappedRemoteStreams, myId]
  );

  const effectiveSpotlightId = useMemo(() => {
    if (!remoteEntries.length) return null;
    if (spotlightPeerId && remoteEntries.some(([id]) => id === spotlightPeerId))
      return spotlightPeerId;
    if (stableSpeakerId && remoteEntries.some(([id]) => id === stableSpeakerId))
      return stableSpeakerId;
    return remoteEntries[0]?.[0] ?? null;
  }, [remoteEntries, spotlightPeerId, stableSpeakerId]);

  const activeEntry = useMemo(
    () => !effectiveSpotlightId
      ? null
      : remoteEntries.find(([id]) => id === effectiveSpotlightId) ?? null,
    [remoteEntries, effectiveSpotlightId]
  );

  const otherEntries = useMemo(
    () => !activeEntry
      ? remoteEntries
      : remoteEntries.filter(([id]) => id !== activeEntry[0]),
    [remoteEntries, activeEntry]
  );

  const participantMap = useMemo(() => {
    const map = {};
    participantsMeta.forEach((p) => { map[p.id] = p.meta; });
    return map;
  }, [participantsMeta]);

  const socketEmotionMap = useMemo(() => {
    if (!isHost && HOST_ONLY_EMOTION) return {};
    const map = {};
    const claimedKeys = new Set();
    participantsMeta.forEach((p) => {
      const userId = p.meta?.userId;
      let history = [];
      if (userId && emotionsMap[userId]) {
        history = emotionsMap[userId];
        claimedKeys.add(userId);
      } else if (!claimedKeys.has(p.id) && emotionsMap[p.id]) {
        history = emotionsMap[p.id];
        claimedKeys.add(p.id);
      }
      if (Array.isArray(history) && history.length) map[p.id] = history;
    });
    return map;
  }, [participantsMeta, emotionsMap, isHost]);

  const handleLeaveEnd = useCallback(async () => {
    if (endingMeeting) return;
    setEndingMeeting(true);
    try {
      if (isHost) await endMeeting(emotionsMap);
      else await leaveCall();
    } finally {
      setEndingMeeting(false);
    }
  }, [endingMeeting, isHost, endMeeting, leaveCall]);

  const handleToggleChat = useCallback(() => {
    if (isMobile) {
      setMobileSheet((v) => (v === "chat" ? null : "chat"));
      setUnreadCount(0);
    } else {
      setChatOpen((v) => !v);
    }
  }, [isMobile]);

  const handleToggleEmotion = useCallback(() => {
    if (isMobile && isHost) {
      setShareEmotion((v) => !v);
      setMobileSheet((v) => (v === "emotion" ? null : "emotion"));
    } else {
      setShareEmotion((v) => !v);
    }
  }, [isMobile, isHost]);

  const multiPartyLayout = remoteEntries.length > 0 && activeEntry;
  const showEmotionPanel = isHost && shareEmotion && !isMobile;

  return (
    <div className={s.shell}>
      <div className={s.bgMesh} aria-hidden="true" />

      <MeetTopBar
        roomId={roomId} isHost={isHost} duration={meetDuration}
        participantCount={participantsMeta.length + 1} connecting={connecting}
        emotionLive={!isHost && emotionLive}
      />

      <div className={s.body}>
        <div className={s.videoArea}>
          {remoteEntries.length === 0 && !connecting && <EmptyState />}

          {multiPartyLayout && (
            <div
              className={`${s.stageLayout} ${isMobile ? s.stageLayoutMobile : ""}`}
              style={isMobile ? { flexDirection: "column", height: "100%" } : undefined}
            >
              <div
                className={s.spotlightWrap}
                style={isMobile ? { flex: "1 1 0", minHeight: 0 } : undefined}
              >
                {activeEntry && (
                  <SpotlightCard
                    key={videoKey(activeEntry[1])}
                    id={activeEntry[0]}
                    stream={activeEntry[1]}
                    meta={participantMap[activeEntry[0]]}
                    emotion={isHost ? socketEmotionMap[activeEntry[0]]?.at(-1) : undefined}
                    isActive={stableSpeakerId === activeEntry[0]}
                    isHost={isHost}
                  />
                )}
              </div>

              {otherEntries.length > 0 && (
                <div
                  className={`${s.filmstrip} ${isMobile ? s.filmstripHoriz : ""}`}
                  style={isMobile ? {
                    flexDirection: "row", height: "120px", minHeight: "120px",
                    flexShrink: 0, width: "100%", overflowX: "auto", overflowY: "hidden",
                    display: "flex", gap: "6px", padding: "4px",
                  } : undefined}
                >
                  {otherEntries.map(([peerId, stream]) => (
                    <ParticipantCard
                      key={videoKey(stream)}
                      peerId={peerId}
                      stream={stream}
                      meta={participantMap[peerId]}
                      emotion={isHost ? socketEmotionMap[peerId]?.at(-1) : undefined}
                      isActive={stableSpeakerId === peerId}
                      isHost={isHost}
                      compact
                      style={isMobile ? {
                        width: "160px", height: "112px", minWidth: "160px",
                        flexShrink: 0, borderRadius: "10px",
                      } : undefined}
                      onClick={() => {
                        spotlightPeerRef.current = peerId;
                        setSpotlightPeerId(peerId);
                      }}
                    />
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        <AnimatePresence>
          {chatOpen && !isMobile && (
            <MeetChatPanel
              chatMessages={chatMessages} participantsMeta={participantsMeta}
              myUserId={myUserId} retryMessage={retryMessage}
              sendChatMessage={sendChatMessage}
              chatContainerRef={chatContainerRef} chatEndRef={chatEndRef}
              onClose={() => setChatOpen(false)}
            />
          )}
        </AnimatePresence>

        <AnimatePresence>
          {showEmotionPanel && (
            <motion.div
              key="emotion-panel"
              initial={{ opacity: 0, x: -40 }} animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -40 }}
              transition={{ duration: 0.22, ease: [0.16, 1, 0.3, 1] }}
              style={{ position: "absolute", inset: 0, pointerEvents: "none" }}
            >
              <div style={{ pointerEvents: "all" }}>
                <EmotionServicePanel
                  emotionsMap={socketEmotionMap} participantsMeta={participantsMeta}
                  isHost={isHost} DEBUG_SHOW_EMOTION_FOR_EVERYONE={false}
                />
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <MeetLocalPreview
        localVideoRef={localVideoRef}
        displayName={localStorage.getItem("displayName") || "You"}
        isHost={isHost} isSpeaking={stableSpeakerId === "local"}
        muted={muted} videoOff={videoOff} shareEmotion={shareEmotion}
        emotionLive={!isHost && emotionLive}
        chatOpen={chatOpen && !isMobile}
        onToggleMute={() => toggleMute(muted, setMuted, mutedRef, TRANSCRIPTS_ENABLED, recordersRef)}
        onToggleVideo={() => toggleVideo(videoOff, setVideoOff)}
        onToggleEmotion={handleToggleEmotion}
      />

      <MeetControlBar
        muted={muted} videoOff={videoOff}
        chatOpen={isMobile ? mobileSheet === "chat" : chatOpen}
        shareEmotion={shareEmotion} isHost={isHost}
        emotionLive={!isHost && emotionLive}
        endingMeeting={endingMeeting} unreadCount={unreadCount}
        onToggleMute={() => toggleMute(muted, setMuted, mutedRef, TRANSCRIPTS_ENABLED, recordersRef)}
        onToggleVideo={() => toggleVideo(videoOff, setVideoOff)}
        onScreenShare={() => startScreenShare(prevLocalStreamRef)}
        onToggleChat={handleToggleChat}
        onToggleEmotion={handleToggleEmotion}
        onLeaveEnd={handleLeaveEnd}
      />

      {isMobile && (
        <MobilePanelSheet
          activeSheet={mobileSheet}
          onClose={() => setMobileSheet(null)}
          onTabChange={(tab) => {
            setMobileSheet(tab);
            if (tab === "chat") setUnreadCount(0);
          }}
          showEmotionTab={isHost && shareEmotion}
          chatMessages={chatMessages} participantsMeta={participantsMeta}
          myUserId={myUserId} retryMessage={retryMessage}
          sendChatMessage={sendChatMessage}
          emotionsMap={socketEmotionMap} isHost={isHost}
        />
      )}
    </div>
  );
}