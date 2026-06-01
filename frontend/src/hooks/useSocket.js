import { useEffect, useRef } from "react";

export default function useSocket({
  socketRef,
  roomId,
  setMyId,
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
  cleanupAll,
  persistHistorySnapshot,
  handleIncomingMessage,
  handleAck,
  notifyPcsChanged,
  makingOfferRef,
  socketReady,
  isHost,
  setEmotionLive,
  onTranscriptRequestReceived,
  onTranscriptRequestUpdate,
}) {
  const h = useRef({});

  h.current = {
    setMyId,
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
    cleanupAll,
    persistHistorySnapshot,
    handleIncomingMessage,
    handleAck,
    notifyPcsChanged,
    makingOfferRef,
    isHost,
    setEmotionLive,
    onTranscriptRequestReceived,
    onTranscriptRequestUpdate,
  };

  useEffect(() => {
    const socket = socketRef.current;
    if (!socket) return;

    let disconnectTimer = null;
    let mySocketId = socket.connected ? socket.id : null;
    let wasEverConnected = false;

    const onConnect = () => {
      mySocketId = socket.id;
      h.current.setMyId(socket.id);
      wasEverConnected = true;

      if (disconnectTimer) {
        clearTimeout(disconnectTimer);
        disconnectTimer = null;
      }

      try {
        window.myId = socket.id;
      } catch { }
    };

    socket.on("connect", onConnect);
    if (socket.connected) onConnect();

    const onChatHistory = (history = []) => {
      const seen = h.current.seenMsgIdsRef.current;
      const unique = [];

      for (const m of history) {
        const id = m.id || `${m.userId}:${m.ts}`;
        if (seen.has(id)) continue;
        seen.add(id);
        unique.push(m);
      }

      h.current.setChatMessages((prev) => {
        if (!unique.length) return prev;

        const existingIds = new Set(
          prev.map((m) => m.id || `${m.userId}:${m.ts}`)
        );

        const fresh = unique.filter((m) => {
          const id = m.id || `${m.userId}:${m.ts}`;
          return !existingIds.has(id);
        });

        return fresh.length ? [...prev, ...fresh] : prev;
      });
    };

    socket.on("chat-history", onChatHistory);

    const syncParticipants = (list) => {
      const map = {};
      (list || []).forEach((p) => {
        if (!p?.id) return;
        if (p.id === mySocketId) return;
        map[p.id] = {
          id: p.id,
          meta: p.meta || {},
          polite:
            typeof p.polite === "boolean" ? p.polite : true,
        };
      });

      h.current.setParticipantsMeta((prev) => {
        const merged = { ...map };
        prev.forEach((p) => {
          if (!merged[p.id]) merged[p.id] = p;
        });
        return Object.values(merged);
      });

      Object.values(map).forEach((p) => {
        if (!h.current.pcsRef.current[p.id]) {
          h.current.politeRef.current[p.id] = p.polite;
          h.current.pendingCandidatesRef.current[p.id] ||= [];
          h.current.createPeerConnection(p.id);
        }
      });

      h.current.notifyPcsChanged?.();
    };

    socket.on("participants-updated", syncParticipants);
    socket.on("existing-participants", (list) => {
      syncParticipants(list);
      if (!h.current.isHost) {
        socket.emit("get-emotion-status");
      }
    });

    const onParticipantStateUpdate = ({ peerId, muted }) => {
      if (!peerId) return;

      h.current.setParticipantsMeta((prev) =>
        prev.map((p) =>
          p.id === peerId
            ? { ...p, meta: { ...p.meta, muted: muted === true } }
            : p
        )
      );
    };

    socket.on("update-participant-state", onParticipantStateUpdate);

    const onUserJoined = (peer) => {
      const peerId = peer?.id;
      if (!peerId || peerId === mySocketId) return;

      h.current.setParticipantsMeta((prev) => {
        if (prev.some((p) => p.id === peerId)) return prev;
        return [
          ...prev,
          {
            id: peerId,
            meta: peer.meta || {},
            polite:
              typeof peer.polite === "boolean"
                ? peer.polite
                : true,
          },
        ];
      });

      if (!h.current.pcsRef.current[peerId]) {
        h.current.politeRef.current[peerId] =
          typeof peer.polite === "boolean"
            ? peer.polite
            : true;

        h.current.pendingCandidatesRef.current[peerId] ||= [];

        h.current.createPeerConnection(peerId);
      }

      h.current.notifyPcsChanged?.();
    };

    socket.on("user-joined", onUserJoined);

    const onUserLeft = (peerId) => {

      h.current.closePeer(peerId);
      h.current.removeAnalyzer(peerId);

      h.current.setEmotionsMap((prev) => {
        const copy = { ...prev };
        delete copy[peerId];
        return copy;
      });


    };

    socket.on("user-left", onUserLeft);

    const onSignal = async (fromId, messageStr) => {
      if (!fromId || !messageStr) return;
      if (fromId === mySocketId) return;

      if (h.current.politeRef.current[fromId] === undefined) {
        h.current.politeRef.current[fromId] = true;
      }

      try {
        await h.current.handleSignal(fromId, messageStr);
      } catch { }
    };

    socket.on("signal", onSignal);

    const onChatMessage = (m) => {
      h.current.handleIncomingMessage(m);
    };

    socket.on("chat-message", onChatMessage);

    const onChatAck = (msg) => {
      if (!msg?.id) return;
      h.current.handleAck(msg.id);
    };

    socket.on("chat-ack", onChatAck);

    const onEndMeeting = async () => {
      if (h.current.isHost) return;
      try {
        await h.current.persistHistorySnapshot();
      } catch { }
      h.current.cleanupAll();
      h.current.navigate("/home");
    };

    socket.on("end-meeting", onEndMeeting);

    const onEmotionStatus = ({ active } = {}) => {
      if (!h.current.isHost && typeof h.current.setEmotionLive === "function") {
        h.current.setEmotionLive(Boolean(active));
      }
    };
    socket.on("emotion-status", onEmotionStatus);

    const onTranscriptRequestReceived = (payload) => {
      if (typeof h.current.onTranscriptRequestReceived === "function") {
        h.current.onTranscriptRequestReceived(payload);
      }
    };
    socket.on("transcript-request-received", onTranscriptRequestReceived);

    const onTranscriptRequestUpdate = (payload) => {
      if (typeof h.current.onTranscriptRequestUpdate === "function") {
        h.current.onTranscriptRequestUpdate(payload);
      }
    };
    socket.on("transcript-request-update", onTranscriptRequestUpdate);

    const onDisconnect = () => {
      if (disconnectTimer) return;

      disconnectTimer = setTimeout(() => {
        if (socket.connected) return;
        if (!wasEverConnected) return;

        const pcCount = Object.keys(
          h.current.pcsRef.current || {}
        ).length;

        if (pcCount > 0) {
          disconnectTimer = null;
          return;
        }

        h.current.cleanupAll();
        h.current.navigate("/home");
      }, 15000);
    };

    socket.on("disconnect", onDisconnect);

    return () => {
      if (disconnectTimer) {
        clearTimeout(disconnectTimer);
        disconnectTimer = null;
      }

      socket.off("connect", onConnect);
      socket.off("chat-history", onChatHistory);
      socket.off("participants-updated", syncParticipants);
      socket.off("existing-participants", syncParticipants);
      socket.off("user-joined", onUserJoined);
      socket.off("user-left", onUserLeft);
      socket.off("signal", onSignal);
      socket.off("chat-message", onChatMessage);
      socket.off("chat-ack", onChatAck);
      socket.off("end-meeting", onEndMeeting);
      socket.off("disconnect", onDisconnect);
      socket.off("update-participant-state", onParticipantStateUpdate);
      socket.off("emotion-status", onEmotionStatus);
      socket.off("transcript-request-received", onTranscriptRequestReceived);
      socket.off("transcript-request-update", onTranscriptRequestUpdate);
    };
  }, [socketReady]);
}