import { Server } from "socket.io";
import { createAdapter } from "@socket.io/redis-adapter";
import { makeLogger, safeRedisGet, safeRedisSet, safeRedisDel } from "../utils/redis.utils.js";

import { getState, mkdirp, UPLOAD_BASE } from "../services/socket.service.js";

import {
  handleJoinCall,
  handleUpdateParticipantState,
  handleUpdateMeta,
  handleChatMessage,
  handleTranscriptionUpdate,
  handleKeywordsUpdate,
  handleLeave,
  validateCode,
  getParticipants,
  REDIS_READ_FAILED,
} from "../services/socket.service.js";
import { startTimer, endTimer } from "../observability/latency/latency.service.js";

import { LATENCY_LABELS } from "../observability/latency/latency.constants.js";
import fs from "fs";
import { Meeting } from "../models/meeting.model.js";

const cfg = JSON.parse(
  fs.readFileSync(new URL("../config/config.json", import.meta.url))
);

const SOCKET_MAX_HTTP_BUFFER = parseInt(process.env.SOCKET_MAX_HTTP_BUFFER || `${100 * 1024 * 1024}`, 10);

const log = makeLogger("socket");

let broadcastDebounceTimers = new Map();

const EMOTION_KEY = (code) => `emotion:active:${code}`;

async function getEmotionState(code) {
  const val = await safeRedisGet(EMOTION_KEY(code));
  return val === "1";
}

async function setEmotionState(code, active) {
  if (active) {
    await safeRedisSet(EMOTION_KEY(code), "1");
  } else {
    await safeRedisDel(EMOTION_KEY(code));
  }
}

async function deleteEmotionState(code) {
  await safeRedisDel(EMOTION_KEY(code));
}

function broadcastParticipants(code, io) {
  if (broadcastDebounceTimers.has(code)) clearTimeout(broadcastDebounceTimers.get(code));

  broadcastDebounceTimers.set(code, setTimeout(async () => {
    broadcastDebounceTimers.delete(code);
    try {
      const participants_map = await getParticipants(code);
      if (participants_map === REDIS_READ_FAILED) {
        log.warn("broadcastParticipants skipped: redis unavailable", { code });
        return;
      }
      if (!participants_map.size) return;

      const participants = Array.from(participants_map.values())
        .map((p) => ({ id: p.socketId, meta: p.meta || {} }));

      io.in(`meeting:${code}`).emit("participants-updated", participants);
    } catch (e) {
      log.error("broadcastParticipants error", { code, err: e.message });
    }
  }, 150));
}

async function getHostSocketId(io, meetingCode) {
  try {
    const roomName = `meeting:${meetingCode}`;
    const socketsInRoom = await io.in(roomName).fetchSockets();
    const hostSocket = socketsInRoom.find((s) => s.data?.isHost === true);
    if (hostSocket) return hostSocket.id;
    const stateArr = await getState(meetingCode);
    if (stateArr === REDIS_READ_FAILED) return null;
    return stateArr?.[0] ?? null;
  } catch {
    const stateArr = await getState(meetingCode);
    if (stateArr === REDIS_READ_FAILED) return null;
    return stateArr?.[0] ?? null;
  }
}

let _io = null;

export function getIo() {
  return _io;
}

export async function notifyHostOfTranscriptRequest(meetingCode, requestPayload) {
  if (!_io) return;
  try {
    const roomName = `meeting:${meetingCode}`;
    const socketsInRoom = await _io.in(roomName).fetchSockets();
    const hostSocket = socketsInRoom.find((s) => s.data?.isHost === true);
    if (hostSocket) {
      hostSocket.emit("transcript-request-received", requestPayload);
      return;
    }

    const meeting = await Meeting.findOne({ meetingCode }).lean();
    const hostUserId = (meeting?.ownerId || meeting?.hostId)?.toString();
    if (!hostUserId) return;

    const allSockets = await _io.fetchSockets();
    const hostSockets = allSockets.filter((s) => s.data?.userId === hostUserId);
    hostSockets.forEach((s) => s.emit("transcript-request-received", requestPayload));
  } catch (err) {
    log.error("notifyHostOfTranscriptRequest error", { err: err.message, meetingCode });
  }
}

export async function notifyRequesterOfResolution(requesterSocketId, payload) {
  if (!_io) return;
  try {
    _io.to(requesterSocketId).emit("transcript-request-update", payload);
  } catch (err) {
    log.error("notifyRequesterOfResolution error", { err: err.message });
  }
}

export async function notifyUserOfResolution(userId, payload) {
  if (!_io) return;
  try {
    const allSockets = await _io.fetchSockets();
    const userSockets = allSockets.filter((s) => s.data?.userId === userId?.toString());
    userSockets.forEach((s) => s.emit("transcript-request-update", payload));
  } catch (err) {
    log.error("notifyUserOfResolution error", { err: err.message });
  }
}

export function connectToSocket(
  server,
  corsOptions = {
    origin: "http://localhost:3000",
    methods: ["GET", "POST"],
    credentials: true,
  },
  pubClient,
  subClient
) {
  const io = new Server(server, {
    cors: corsOptions,
    maxHttpBufferSize: SOCKET_MAX_HTTP_BUFFER,
    transports: cfg.socket?.transports ?? ["websocket", "polling"],
    allowEIO3: cfg.socket?.allowEIO3 ?? true,
  });

  io.adapter(createAdapter(pubClient, subClient));
  _io = io;
  mkdirp(UPLOAD_BASE).catch(() => { });

  io.on("connection", (socket) => {
    log.info("connected", { socketId: socket.id });

    socket.on("home-presence", ({ userId } = {}) => {
      if (userId) socket.data.userId = String(userId);
    });

    socket.on("join-call", async (meetingCodeRaw, meta = {}) => {
      try {
        const result = await handleJoinCall(socket, io, meetingCodeRaw, meta);
        if (result) broadcastParticipants(result.code, io);
      } catch (err) {
        log.error("join-call error", { err: err.message });
        socket.emit("error", "Failed to join call");
      }
    });

    socket.on("declare-host", async (meetingCodeRaw, secret, ack) => {
      const code = String(meetingCodeRaw || "").trim().toUpperCase();
      if (!code || !validateCode(code)) {
        if (typeof ack === "function") ack({ ok: false, reason: "invalid_code" });
        return;
      }
      if (socket.data?.meetingCode !== code) {
        if (typeof ack === "function") ack({ ok: false, reason: "not_in_room" });
        return;
      }
      const verified = await Meeting.verifyHostSecret(code, secret);
      if (!verified) {
        log.warn("declare-host failed: bad secret", { socketId: socket.id, code });
        if (typeof ack === "function") ack({ ok: false, reason: "unauthorized" });
        return;
      }
      socket.data.isHost = true;
      log.info("host declared and verified", { socketId: socket.id, code });
      if (typeof ack === "function") ack({ ok: true });
    });

    socket.on("update-participant-state", async (data = {}) => {
      try {
        await handleUpdateParticipantState(socket, io, data);
      } catch (err) {
        log.error("update-participant-state error", { err: err.message });
      }
    });

    socket.on("update-meta", async (metaUpdate = {}) => {
      try {
        const code = await handleUpdateMeta(socket, io, metaUpdate);
        if (code) broadcastParticipants(code, io);
      } catch (err) {
        log.error("update-meta error", { err: err.message });
      }
    });

    socket.on("signal", async (targetId, message) => {
      const start = startTimer();
      try {
        const code = socket.data?.meetingCode;
        if (!code) return;

        const sockets = await io.in(`meeting:${code}`).fetchSockets();
        if (!sockets.some((s) => s.id === targetId)) {
          log.warn("signal rejected: target not in room", { from: socket.id, targetId, code });
          return;
        }

        io.to(targetId).emit("signal", socket.id, message);
      } catch (err) {
        log.error("signal error", { err: err.message });
      } finally {
        endTimer(LATENCY_LABELS.SOCKET_SIGNAL, start, { from: socket.id, to: targetId });
      }
    });

    socket.on("chat-message", async (meetingCodeRaw, msg = {}, ack) => {
      try {
        const result = await handleChatMessage(socket, io, meetingCodeRaw, msg);
        if (typeof ack === "function") ack(result);
      } catch (err) {
        log.error("chat-message error", { err: err.message });
        if (typeof ack === "function") ack({ ok: false });
        socket.emit("error", "Failed to send chat message");
      }
    });

    socket.on("transcription-update", async (chunk) => {
      try {
        await handleTranscriptionUpdate(socket, io, chunk);
      } catch (err) {
        log.error("transcription-update error", { err: err.message });
      }
    });

    socket.on("keywords-update", async (keywords) => {
      try {
        await handleKeywordsUpdate(socket, io, keywords);
      } catch (err) {
        log.error("keywords-update error", { err: err.message });
      }
    });

    socket.on("emotion-status", async ({ active } = {}) => {
      const code = socket.data?.meetingCode;
      if (!code) return;
      if (!socket.data?.isHost) {
        log.warn("emotion-status ignored: not host", { socketId: socket.id, code });
        return;
      }
      const isActive = Boolean(active);
      await setEmotionState(code, isActive);
      socket.to(`meeting:${code}`).emit("emotion-status", { active: isActive });
      log.info("emotion-status broadcast", { code, active: isActive });
    });

    socket.on("get-emotion-status", async () => {
      const code = socket.data?.meetingCode;
      if (!code) {
        socket.once("join-call", () => {
          setTimeout(async () => {
            const c = socket.data?.meetingCode;
            if (!c) return;
            const active = await getEmotionState(c);
            socket.emit("emotion-status", { active });
          }, 200);
        });
        return;
      }
      const active = await getEmotionState(code);
      socket.emit("emotion-status", { active });
    });

    socket.on("end-meeting", async (meetingCodeRaw) => {
      try {
        const code = String(meetingCodeRaw || "").trim().toUpperCase();
        if (!code) return;

        if (!socket.data?.isHost) {
          log.warn("end-meeting ignored: not host", { socketId: socket.id, code });
          return;
        }

        await deleteEmotionState(code);
        const { userId } = socket.data || {};
        await handleLeave(socket, code, io, userId);
        broadcastParticipants(code, io);
        log.info("host left meeting (end-meeting)", { socketId: socket.id, code });
      } catch (err) {
        log.error("end-meeting error", { err: err.message });
      }
    });

    socket.on("leave-call", async (meetingCodeRaw) => {
      try {
        const code = String(meetingCodeRaw || "").trim().toUpperCase();
        const { userId } = socket.data || {};
        await handleLeave(socket, code, io, userId);
        broadcastParticipants(code, io);
      } catch (err) {
        log.error("leave-call error", { err: err.message });
      }
    });

    socket.on("disconnect", async () => {
      try {
        if (socket.data?.replaced) return;
        const code = socket.data?.meetingCode;
        const { userId } = socket.data || {};
        if (!code || !userId) return;
        await handleLeave(socket, code, io, userId);
        broadcastParticipants(code, io);
      } catch (err) {
        log.error("disconnect error", { err: err.message });
      }
    });
  });

  return io;
}