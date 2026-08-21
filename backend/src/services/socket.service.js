import sanitizeHtml from "sanitize-html";
import fs from "fs";
import path from "path";
import os from "os";
import crypto from "crypto";
import { withRoomLock } from "../infra/redisLock.js";
import { makeLogger, safeRedisGetResult, safeRedisSet, safeRedisDel, safeRedisIncr, safeRedisExpire, safeRedisHGetAllResult, safeRedisHSet, safeRedisHDel } from "../utils/redis.utils.js";
import { toBuffer } from "../utils/helpers.utils.js";

import {
    findMeetingByCode,
    markParticipantLeft,
    restoreMeetingParticipant,
    saveMeeting,
    updateMeetingParticipantMeta,
    addMeetingChatMessage,
    updateMeetingAnalytics,
} from "../data-access/socket.repository.js";

import { startTimer, endTimer } from "../observability/latency/latency.service.js";
import { LATENCY_LABELS } from "../observability/latency/latency.constants.js";

const cfg = JSON.parse(
    fs.readFileSync(new URL("../config/config.json", import.meta.url))
);

const PARTIAL_UPLOAD_MAX_BYTES = parseInt(process.env.PARTIAL_UPLOAD_MAX_BYTES || `${200 * 1024 * 1024}`, 10);

const PARTIAL_UPLOAD_TTL_MS = parseInt(process.env.PARTIAL_UPLOAD_TTL_MS || `${10 * 60 * 1000}`, 10);

const MAX_PARTICIPANTS_PER_ROOM = parseInt(process.env.MAX_PARTICIPANTS_PER_ROOM || "50", 10);
const SOCKET_CHAT_RATE_MAX = parseInt(process.env.SOCKET_CHAT_RATE_MAX || "20", 10);
const SOCKET_CHAT_RATE_WIN_SEC = parseInt(process.env.SOCKET_CHAT_RATE_WIN_SEC || "10", 10);

const SOCKET_CHUNK_RATE_MAX = parseInt(process.env.SOCKET_CHUNK_RATE_MAX || "100", 10);
const SOCKET_CHUNK_RATE_WIN_SEC = parseInt(process.env.SOCKET_CHUNK_RATE_WIN_SEC || "10", 10);
const PARTIAL_UPLOAD_TTL_SEC = Math.ceil(PARTIAL_UPLOAD_TTL_MS / 1000);

export const UPLOAD_BASE = path.join(
    os.tmpdir(),
    cfg.upload?.baseDir ?? "meet_uploads"
);
const MAX_CHUNKS = cfg.upload?.maxChunks ?? 50000;
const MAX_NAME_LEN = cfg.sanitize?.maxNameLength ?? 200;
const MAX_CHAT_LEN = cfg.sanitize?.maxChatLength ?? 2000;
const MAX_TRANSCRIPT_LEN = cfg.sanitize?.maxTranscriptionChunkLength ?? 500;
const MAX_KEYWORD_LEN = cfg.sanitize?.maxKeywordLength ?? 100;
const DEFAULT_NAME = cfg.sanitize?.defaultName ?? "Guest";

const MEETING_CODE_RE = /^[A-Z0-9\-]{3,32}$/;
const ALLOWED_CHUNK_TYPES = ["audio", "video"];

export const KEYS = {
    state: (code) => `${cfg.redisKeys?.meetingStatePrefix ?? "meeting:state:"}${code}`,
    participants: (code) => `${cfg.redisKeys?.meetingParticipantsPrefix ?? "meeting:participants:"}${code}`,
    partial: (key) => `${cfg.redisKeys?.partialPrefix ?? "partial:"}${key}`,
    partialMeta: (key) => `${cfg.redisKeys?.partialMetaPrefix ?? "partial:meta:"}${key}`,
    chatRate: (uid) => `socket:chat:rate:${uid}`,
    chunkRate: (uid) => `socket:chunk:rate:${uid}`,
};

const log = makeLogger("socket");

export const REDIS_READ_FAILED = Symbol("REDIS_READ_FAILED");

export const mkdirp = (p) => fs.promises.mkdir(p, { recursive: true });
export const unlink = (p) => fs.promises.rm(p,
    {
        force: true,
        maxRetries: 2, recursive: true
    }).catch((e) => log.warn("unlink failed", { p, err: e.message }));

export const stat = (p) => fs.promises.stat(p).catch(() => null);
export const writeFile = (p, buf) => fs.promises.writeFile(p, buf);

export async function getState(code) {
    const result = await safeRedisGetResult(KEYS.state(code));
    if (!result.ok) return REDIS_READ_FAILED;
    if (result.value === null) return null;
    return JSON.parse(result.value);
}
export async function setState(code, arr) {
    await safeRedisSet(
        KEYS.state(code),
        JSON.stringify(arr)
    );
}
export async function deleteState(code) {
    await safeRedisDel(KEYS.state(code));
}

export async function getParticipants(code) {
    const result = await safeRedisHGetAllResult(KEYS.participants(code));
    if (!result.ok) return REDIS_READ_FAILED;
    const raw = result.value;
    if (!raw || Object.keys(raw).length === 0) return new Map();
    const map = new Map();
    for (const [userId, json] of Object.entries(raw)) {
        try { map.set(userId, JSON.parse(json)); } catch { }
    }
    return map;
}

export async function setParticipants(code, map) {
    for (const [userId, participant] of map.entries()) {
        await safeRedisHSet(KEYS.participants(code), userId, JSON.stringify(participant));
    }
}

export async function deleteParticipantField(code, userId) {
    await safeRedisHDel(KEYS.participants(code), userId);
}

export async function deleteParticipants(code) {
    await safeRedisDel(KEYS.participants(code));
}

export async function getPartialMeta(key) {
    const result = await safeRedisGetResult(KEYS.partialMeta(key));
    if (!result.ok) return REDIS_READ_FAILED;
    if (result.value === null) return null;
    return JSON.parse(result.value);
}
export async function setPartialMeta(key, meta) {
    await safeRedisSet(
        KEYS.partialMeta(key),
        JSON.stringify(meta),
        {
            EX: PARTIAL_UPLOAD_TTL_SEC
        });
}
export async function deletePartialMeta(key) {

    await safeRedisDel(KEYS.partialMeta(key));
}

export async function isSocketRateLimited(userId, rateKey, max, winSec) {
    const count = await safeRedisIncr(rateKey);

    if (count === 1) await safeRedisExpire(rateKey, winSec);

    return count > max;
}

export function validateCode(code) {
    return MEETING_CODE_RE.test(code);
}

export function makeUploadDir(key) {
    const safeKey = key.replace(/[^\w\-_.]/g, "_");

    return path.join(
        UPLOAD_BASE,
        `${safeKey}_${crypto.randomBytes(6).toString("hex")}`
    );
}

export function sanitizeName(raw) {
    return sanitizeHtml(
        String(raw || DEFAULT_NAME))
        .slice(0, MAX_NAME_LEN) || DEFAULT_NAME;
}

export function resolveUserId(meta, socket) {
    return meta?.userId ? String(meta.userId) : socket.handshake.auth?.userId || socket.id;
}

export { toBuffer };

export async function finalizeAnalytics(meeting) {
    log.info("finalizing analytics", { code: meeting.meetingCode });
}

export async function handleLeave(socket, code, io, userId) {
    const meeting = await findMeetingByCode(code);
    if (!meeting) return;

    await meeting.markParticipantLeft(socket.id, userId);

    let stateArr = await getState(code);
    if (stateArr === REDIS_READ_FAILED) {
        log.warn("leave skipped state update: redis unavailable", { code, userId });
    } else if (stateArr) {
        stateArr = stateArr.filter((id) => id !== socket.id);
        if (stateArr.length === 0) {
            await deleteState(code);
            await finalizeAnalytics(meeting);
        } else {
            await setState(code, stateArr);
        }
    }

    const participants_map = await getParticipants(code);
    if (participants_map === REDIS_READ_FAILED) {
        log.warn("leave skipped participants update: redis unavailable", { code, userId });
    } else if (participants_map.size) {
        participants_map.delete(userId);
        if (participants_map.size === 0) {
            await deleteParticipants(code);
        } else {
            await deleteParticipantField(code, userId);
        }
    }

    socket.leave(`meeting:${code}`);

    socket.to(`meeting:${code}`).emit("user-left", socket.id);
}

export async function handleJoinCall(socket, io, meetingCodeRaw, meta = {}) {

    const start = startTimer();
    const code = String(meetingCodeRaw || "").trim().toUpperCase();
    if (!code || !validateCode(code)) {
        socket.emit("error", "Invalid meeting code");
        return;
    }

    const meeting = await findMeetingByCode(code);

    if (!meeting) {
        socket.emit("error", "Room does not exist. Please create it first.");
        return;
    }

    if (meta && typeof meta === "object" && JSON.stringify(meta).length > 4096) {
        socket.emit("error", "Meta payload too large");
        return;
    }

    const name = sanitizeName(meta.name);
    const userId = resolveUserId(meta, socket);

    const cleanMeta = { ...meta, name };

    await withRoomLock(code, async () => {
        let participants_map = await getParticipants(code);
        if (participants_map === REDIS_READ_FAILED) {
            log.warn("join aborted: redis unavailable (participants)", { code, userId });
            socket.emit("error", "Meeting service temporarily unavailable. Please try again.");
            return;
        }

        let stateArr = await getState(code);
        if (stateArr === REDIS_READ_FAILED) {
            log.warn("join aborted: redis unavailable (state)", { code, userId });
            socket.emit("error", "Meeting service temporarily unavailable. Please try again.");
            return;
        }
        stateArr = stateArr || [];

        if (!participants_map.has(userId) && participants_map.size >= MAX_PARTICIPANTS_PER_ROOM) {
            socket.emit("error", "Room is full");
            return;
        }

        if (participants_map.has(userId)) {
            const existing = participants_map.get(userId);
            const oldSocketId = existing.socketId;
            stateArr = stateArr.filter((id) => id !== oldSocketId);

            const oldSocket = io.sockets.sockets.get(oldSocketId);

            if (oldSocket) {
                oldSocket.data.replaced = true;
                oldSocket.leave(`meeting:${code}`);
                oldSocket.removeAllListeners();
                oldSocket.emit = () => { };

                try {
                    oldSocket.disconnect(true);
                } catch { }
            }

            existing.socketId = socket.id;
            existing.meta = cleanMeta;
            participants_map.set(userId, existing);

            if (!stateArr.includes(socket.id)) stateArr.push(socket.id);

            await meeting.restoreParticipant(

                socket.id, {
                userId, name
            }, cleanMeta);

        } else {

            participants_map.set(userId,
                {
                    socketId: socket.id, userId, meta: cleanMeta
                }
            );
            if (!stateArr.includes(socket.id)) stateArr.push(socket.id);

            await meeting.addParticipant(
                {
                    socketId: socket.id,
                    name,
                    meta: cleanMeta
                });
        }

        await setState(code, stateArr);
        await safeRedisHSet(
            KEYS.participants(code),
            userId,
            JSON.stringify(participants_map.get(userId))
        );

        socket.data = { meetingCode: code, name, meta: cleanMeta, userId };
        meeting.active = true;
        await meeting.save();

        const roomName = `meeting:${code}`;
        socket.join(roomName);

        const politeRole = stateArr.indexOf(socket.id) !== 0;

        const existingPeers = Array.from(participants_map.values())
            .filter((p) => p.socketId !== socket.id)
            .map((p) => ({ id: p.socketId, meta: p.meta || {}, polite: stateArr.indexOf(p.socketId) !== 0 }));

        socket.emit("existing-participants", existingPeers);
        socket.emit("assigned-role", { polite: politeRole });

        const rawHistory = Array.isArray(meeting.chat) ? meeting.chat : [];
        const normalizedHistory = rawHistory.map((m, idx) => {
            const id = m.id || m._id || (m.ts
                ? `${m.userId || m.from || "anon"}_${new Date(m.ts).getTime()}_${idx}`
                : crypto.randomBytes(8).toString("hex"));
            const ts = m.ts && typeof m.ts === "number" ? m.ts : m.ts ? new Date(m.ts).getTime() : Date.now();
            const stableUserId = m.userId || m.from || m.fromSocketId || m.sender || null;
            const msgName = m.meta?.name || m.name || DEFAULT_NAME;
            return { id, text: String(m.text || ""), from: stableUserId, userId: stableUserId, name: msgName, meta: { ...(m.meta || {}), name: msgName }, ts };
        });

        socket.emit("chat-history", normalizedHistory);
        socket.to(roomName).emit("user-joined", { id: socket.id, meta: cleanMeta, polite: politeRole });

        log.info("joined", { name, socketId: socket.id, code, polite: politeRole });
    });

    endTimer(LATENCY_LABELS.SOCKET_JOIN, start, { meetingCode: code, userId });
    return { code, userId };
}

export async function handleUpdateParticipantState(socket, io, data = {}) {
    const code = socket.data?.meetingCode;
    if (!code) return;

    const metaUpdate = {};
    if (data.muted !== undefined) metaUpdate.muted = !!data.muted;
    if (data.screen !== undefined) metaUpdate.screen = !!data.screen;

    socket.data.meta = { ...(socket.data.meta || {}), ...metaUpdate };

    await updateMeetingParticipantMeta(code, socket.id, socket.data.meta);

    const userId = socket.data.userId;
    if (userId) {
        const result = await safeRedisHGetAllResult(KEYS.participants(code));
        if (result.ok && result.value?.[userId]) {
            let entry;
            try { entry = JSON.parse(result.value[userId]); } catch { entry = {}; }
            entry.meta = socket.data.meta;
            await safeRedisHSet(KEYS.participants(code), userId, JSON.stringify(entry));
        } else if (!result.ok) {
            log.warn("update-participant-state skipped: redis unavailable", { code });
        }
    }

    io.in(`meeting:${code}`).emit("update-participant-state", { peerId: socket.id, muted: socket.data.meta.muted === true });
}

export async function handleUpdateMeta(socket, io, metaUpdate = {}) {
    const code = socket.data?.meetingCode;
    if (!code) return;

    const meeting = await findMeetingByCode(code);
    if (!meeting) return;

    if (!metaUpdate || typeof metaUpdate !== "object") metaUpdate = {};

    if (JSON.stringify(metaUpdate).length > 4096) return;

    const normalized = { ...metaUpdate };

    if (normalized.name !== undefined) normalized.name = sanitizeName(normalized.name) || socket.data.name;

    if (normalized.muted !== undefined) normalized.muted = !!normalized.muted;

    if (normalized.video !== undefined) normalized.video = !!normalized.video;

    if (normalized.screen !== undefined) normalized.screen = !!normalized.screen;

    socket.data.meta = { ...(socket.data.meta || {}), ...normalized };

    await meeting.updateParticipantMeta(socket.id, socket.data.meta);

    const userId = socket.data.userId;
    if (userId) {
        const result = await safeRedisHGetAllResult(KEYS.participants(code));
        if (result.ok && result.value?.[userId]) {
            let entry;
            try { entry = JSON.parse(result.value[userId]); } catch { entry = {}; }
            entry.meta = socket.data.meta;
            await safeRedisHSet(KEYS.participants(code), userId, JSON.stringify(entry));
        } else if (!result.ok) {
            log.warn("update-meta skipped: redis unavailable", { code });
        }
    }

    io.in(`meeting:${code}`).emit("participant-meta-updated", { id: socket.id, meta: socket.data.meta });
    return code;
}

export async function handleChatMessage(socket, io, meetingCodeRaw, msg = {}) {
    const start = startTimer();
    const code = String(meetingCodeRaw || "").trim().toUpperCase();
    if (!code || !validateCode(code)) return { ok: false };
    if (socket.data?.meetingCode !== code) return { ok: false, reason: "not_in_room" };

    const userId = socket.data?.userId || socket.id;
    const chatRateKey = KEYS.chatRate(userId);
    if (await isSocketRateLimited(userId, chatRateKey, SOCKET_CHAT_RATE_MAX, SOCKET_CHAT_RATE_WIN_SEC)) {
        return { ok: false, reason: "rate_limited" };
    }

    const meeting = await findMeetingByCode(code);
    if (!meeting) return { ok: false };

    const name = socket.data?.meta?.name || socket.data?.name || DEFAULT_NAME;
    const msgId = msg.id || crypto.randomBytes(10).toString("hex");
    const ts = Date.now();

    const chatMsg = {
        id: msgId,
        userId,
        from: userId,
        fromSocketId: socket.id,
        name,
        meta: { name, userId },
        text: sanitizeHtml(String(msg.text || "").slice(0, MAX_CHAT_LEN)),
        ts,
    };

    try {
        await meeting.addChatMessage(chatMsg);
    } catch (dbErr) {
        log.warn("chat persist failed", { err: dbErr?.message });
    }

    const payload = { id: chatMsg.id, text: chatMsg.text, from: chatMsg.from, fromSocketId: chatMsg.fromSocketId, userId: chatMsg.userId, name: chatMsg.name, meta: chatMsg.meta, ts: chatMsg.ts };

    socket.to(`meeting:${code}`).emit("chat-message", payload);
    socket.emit("chat-ack", payload);

    log.info("chat", { name, userId, code, text: chatMsg.text.slice(0, 50) });
    endTimer(LATENCY_LABELS.SOCKET_MESSAGE, start, { meetingCode: code, userId });
    return { ok: true };
}

export async function handleTranscriptionUpdate(socket, io, chunk) {
    const code = socket.data?.meetingCode;
    if (!code) return;

    const cleanChunk = sanitizeHtml(String(chunk || "").slice(0, MAX_TRANSCRIPT_LEN));
    await updateMeetingAnalytics(code, { transcription: cleanChunk });
    socket.to(`meeting:${code}`).emit("transcription-update", { from: socket.id, text: cleanChunk });
}

export async function handleKeywordsUpdate(socket, io, keywords) {
    const code = socket.data?.meetingCode;
    if (!code || !Array.isArray(keywords)) return;

    const cleanKeywords = keywords.map((k) => sanitizeHtml(String(k).slice(0, MAX_KEYWORD_LEN)));
    await updateMeetingAnalytics(code, { keywords: cleanKeywords });
    socket.to(`meeting:${code}`).emit("keywords-update", { from: socket.id, keywords: cleanKeywords });
}