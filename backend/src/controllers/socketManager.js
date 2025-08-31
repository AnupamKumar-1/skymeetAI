
import { Server } from "socket.io";
import sanitizeHtml from "sanitize-html";
import { Meeting } from "../models/meeting.model.js";
import { sendToEmotionService } from "./emotion.controller.js";
import fs from "fs";
import path from "path";
import os from "os";
import crypto from "crypto";

const mkdirp = (p) => fs.promises.mkdir(p, { recursive: true });
const unlink = (p) => fs.promises.rm(p, { force: true, maxRetries: 2, recursive: true });
const stat = (p) => fs.promises.stat(p).catch(() => null);
const writeFile = (p, buf) => fs.promises.writeFile(p, buf);
const readdir = (p) => fs.promises.readdir(p).catch(() => []);
const readFile = (p) => fs.promises.readFile(p);


const meetingState = {};
const meetingParticipants = {};

const PARTIAL_UPLOADS = new Map();

const UPLOAD_BASE = path.join(os.tmpdir(), "meet_uploads");
const PARTIAL_UPLOAD_MAX_BYTES = parseInt(process.env.PARTIAL_UPLOAD_MAX_BYTES || `${200 * 1024 * 1024}`, 10); // 200 MB default
const PARTIAL_UPLOAD_TTL_MS = parseInt(process.env.PARTIAL_UPLOAD_TTL_MS || `${10 * 60 * 1000}`, 10); // 10 minutes

function makeUploadDir(key) {
  // sanitize key for filesystem
  const safeKey = key.replace(/[^\w\-_.]/g, "_");
  const id = crypto.randomBytes(6).toString("hex");
  return path.join(UPLOAD_BASE, `${safeKey}_${id}`);
}


export function connectToSocket(
  server,
  corsOptions = { origin: "http://localhost:3000", methods: ["GET", "POST"], credentials: true }
) {

  const DEFAULT_MAX_HTTP_BUFFER = parseInt(process.env.SOCKET_MAX_HTTP_BUFFER || `${100 * 1024 * 1024}`, 10); // 100 MB default
  const DEFAULT_PING_INTERVAL = parseInt(process.env.SOCKET_PING_INTERVAL || `${20000}`, 10); // 20s
  const DEFAULT_PING_TIMEOUT = parseInt(process.env.SOCKET_PING_TIMEOUT || `${60000}`, 10); // 60s

  const io = new Server(server, {
    cors: corsOptions,
    maxHttpBufferSize: DEFAULT_MAX_HTTP_BUFFER,
    pingInterval: DEFAULT_PING_INTERVAL,
    pingTimeout: DEFAULT_PING_TIMEOUT,
    transports: ["websocket", "polling"],
    allowEIO3: true,
  });

  console.info(
    "[socket] initialized with options:",
    JSON.stringify(
      {
        cors: corsOptions,
        maxHttpBufferSize: DEFAULT_MAX_HTTP_BUFFER,
        pingInterval: DEFAULT_PING_INTERVAL,
        pingTimeout: DEFAULT_PING_TIMEOUT,
      },
      null,
      2
    )
  );


  mkdirp(UPLOAD_BASE).catch((e) => {
    console.warn("[socket] failed to create upload base dir:", e && e.message ? e.message : e);
  });

  const cleanupInterval = setInterval(async () => {
    const now = Date.now();
    for (const [key, meta] of PARTIAL_UPLOADS.entries()) {
      if (now - (meta.createdAt || 0) > PARTIAL_UPLOAD_TTL_MS) {
        console.warn(`[socket][cleanup] removing stale partial upload ${key} at ${meta.dir}`);
        PARTIAL_UPLOADS.delete(key);
        try {
          await unlink(meta.dir);
        } catch (e) {
          console.warn("[socket][cleanup] failed to remove dir:", e && e.message ? e.message : e);
        }
      }
    }

    try {
      const entries = await readdir(UPLOAD_BASE);
      for (const name of entries) {
        const p = path.join(UPLOAD_BASE, name);
        const s = await stat(p);
        if (!s) continue;
        if (now - s.mtimeMs > PARTIAL_UPLOAD_TTL_MS) {
          console.warn(`[socket][cleanup] removing stale dir ${p}`);
          await unlink(p);
        }
      }
    } catch (e) {

    }
  }, Math.max(60 * 1000, Math.floor(PARTIAL_UPLOAD_TTL_MS / 2)));

  global.io = io;
  global.meetingState = meetingState;
  global.meetingParticipants = meetingParticipants;

  io.on("connection", (socket) => {
    console.log("[socket] Client connected:", socket.id);

    // ===== JOIN CALL =====
    socket.on("join-call", async (meetingCodeRaw, meta = {}) => {
  try {
    const code = String(meetingCodeRaw || "").trim().toUpperCase();
    if (!code) return socket.emit("error", "Invalid meeting code");

    const meeting = await Meeting.findOne({ meetingCode: code });
    if (!meeting) {
      socket.emit("error", "Room does not exist. Please create it first.");
      return;
    }

    const name = sanitizeHtml(String(meta.name || "Guest")).slice(0, 200);

    const userId = meta.userId ? String(meta.userId) : socket.id;
    const cleanMeta = { ...(meta || {}), name };

    if (!meetingParticipants[code]) meetingParticipants[code] = new Map();

    let politeRole = false;

    if (meetingParticipants[code].has(userId)) {
      const existing = meetingParticipants[code].get(userId);
      existing.socketId = socket.id;
      existing.meta = cleanMeta;
      politeRole = meetingState[code]?.indexOf(existing.socketId) !== 0;

      await meeting.restoreParticipant(socket.id, { userId, name }, cleanMeta);
    } else {
      meetingParticipants[code].set(userId, {
        socketId: socket.id,
        userId,
        meta: cleanMeta,
      });

      if (!meetingState[code]) meetingState[code] = [];
      meetingState[code].push(socket.id);
      politeRole = meetingState[code].indexOf(socket.id) !== 0;

      await meeting.addParticipant({
        socketId: socket.id,
        name,
        meta: cleanMeta,
      });
    }

    socket.data = { meetingCode: code, name, meta: cleanMeta, userId };

    meeting.active = true;
    await meeting.save();

    const roomName = `meeting:${code}`;
    socket.join(roomName);

    const existing = Array.from(meetingParticipants[code].values())
      .filter((p) => p.socketId !== socket.id)
      .map((p) => ({
        id: p.socketId,
        meta: p.meta || {},
        polite: meetingState[code].indexOf(p.socketId) !== 0,
      }));

    socket.emit("existing-participants", existing);
    socket.emit("assigned-role", { polite: politeRole });


    const rawHistory = Array.isArray(meeting.chat) ? meeting.chat : [];
    const normalizedHistory = rawHistory.map((m, idx) => {
      const id =
        m.id ||
        m._id ||
        (m.ts
          ? `${m.userId || m.from || "anon"}_${new Date(m.ts).getTime()}_${idx}`
          : crypto.randomBytes(8).toString("hex"));

      const ts =
        m.ts && typeof m.ts === "number"
          ? m.ts
          : m.ts
          ? new Date(m.ts).getTime()
          : Date.now();

      // ✅ Always prefer stored userId
      const stableUserId =
        m.userId || m.from || m.fromSocketId || m.sender || null;

      const name = (m.meta && m.meta.name) || m.name || "Guest";

      return {
        id,
        text: String(m.text || ""),
        from: stableUserId,  // 👈 consistent id for client-side isOwn checks
        userId: stableUserId,
        name,
        meta: { ...(m.meta || {}), name },
        ts,
      };
    });

    socket.emit("chat-history", normalizedHistory);

    socket.to(roomName).emit("user-joined", {
      id: socket.id,
      meta: cleanMeta,
      polite: politeRole,
    });

    await broadcastParticipantsFromCache(code, io);

    console.log(
      `[socket] ${name} (${socket.id}) joined ${code} — polite:${politeRole}`
    );
  } catch (err) {
    console.error("[socket][join-call] error:", err);
    socket.emit("error", "Failed to join call");
  }
});

    socket.on("update-meta", async (metaUpdate = {}) => {
      try {
        const code = socket.data?.meetingCode;
        if (!code) return;

        const meeting = await Meeting.findOne({ meetingCode: code });
        if (!meeting) return;

        if (!metaUpdate || typeof metaUpdate !== "object") metaUpdate = {};

        const normalized = { ...(metaUpdate || {}) };

        if (typeof normalized.name !== "undefined") {
          normalized.name =
            sanitizeHtml(String(normalized.name || "")).slice(0, 200) ||
            socket.data.name ||
            "Guest";
        }
        if (typeof normalized.muted !== "undefined") normalized.muted = !!normalized.muted;
        if (typeof normalized.video !== "undefined") normalized.video = !!normalized.video;
        if (typeof normalized.screen !== "undefined") normalized.screen = !!normalized.screen;

        socket.data.meta = { ...(socket.data.meta || {}), ...normalized };

        await meeting.updateParticipantMeta(socket.id, socket.data.meta);
        if (meetingParticipants[code] && meetingParticipants[code].has(socket.data.userId)) {
          meetingParticipants[code].get(socket.data.userId).meta = socket.data.meta;
        }

        io.in(`meeting:${code}`).emit("participant-meta-updated", {
          id: socket.id,
          meta: socket.data.meta,
        });

        await broadcastParticipantsFromCache(code, io);

        console.log(
          `[socket] meta updated for ${socket.id} in ${code}:`,
          JSON.stringify(normalized)
        );
      } catch (err) {
        console.error("[socket][update-meta] error:", err);
      }
    });

    socket.on("signal", (targetId, message) => {
      try {
        const meetingCode = socket.data?.meetingCode;
        if (!meetingCode || !meetingState[meetingCode]) return;
        io.to(targetId).emit("signal", socket.id, message);
      } catch (err) {
        console.error("[socket][signal] error:", err);
      }
    });

    // 📌 inside your socket connection handler
socket.on("chat-message", async (meetingCodeRaw, msg = {}) => {
  try {
    const code = String(meetingCodeRaw || "").trim().toUpperCase();
    if (!code) return;

    const meeting = await Meeting.findOne({ meetingCode: code });
    if (!meeting) return;

    // ✅ Stable identity from socket.data (set at join)
    const userId = socket.data?.userId || socket.id;
    const name =
      (socket.data?.meta?.name || socket.data?.name) || "Guest";

    // ✅ Generate canonical message id + timestamp
    const msgId = msg.id || crypto.randomBytes(10).toString("hex");

    const ts = Date.now();

    // ✅ Build normalized message object
    const chatMsg = {
      id: msgId,
      userId,
      from: userId,               // stable user id (for isOwn checks on client)
      fromSocketId: socket.id,    // session id (changes per reconnect)
      name,
      meta: { name, userId },
      text: sanitizeHtml(String(msg.text || "").slice(0, 2000)),
      ts,
    };

    // ✅ Save in DB (best effort)
    try {
      await meeting.addChatMessage(chatMsg);
    } catch (dbErr) {
      console.warn(
        "[socket][chat-message] warning: failed to persist chat:",
        dbErr?.message || dbErr
      );
    }

    // ✅ Prepare payload for clients
    const payload = {
      id: chatMsg.id,
      text: chatMsg.text,
      from: chatMsg.from,
      fromSocketId: chatMsg.fromSocketId,
      userId: chatMsg.userId,
      name: chatMsg.name,
      meta: chatMsg.meta,
      ts: chatMsg.ts,
    };

    const roomName = `meeting:${code}`;

    // ✅ Emit to other participants
    socket.to(roomName).emit("chat-message", payload);

    // ✅ Emit ack back to sender (same shape, so client can reconcile/dedupe)
    socket.emit("chat-ack", payload);

    console.log(`[chat] ${name} (${userId}) -> ${code}: "${chatMsg.text}"`);
  } catch (err) {
    console.error("[socket][chat-message] error:", err);
    socket.emit("error", "Failed to send chat message");
  }
});




    // ===== TRANSCRIPTION UPDATE =====
    socket.on("transcription-update", async (chunk) => {
      try {
        const code = socket.data?.meetingCode;
        if (!code) return;
        const meeting = await Meeting.findOne({ meetingCode: code });
        if (!meeting) return;

        const cleanChunk = sanitizeHtml(String(chunk || "").slice(0, 500));
        await meeting.updateAnalytics({ transcription: cleanChunk });

        socket.to(`meeting:${code}`).emit("transcription-update", {
          from: socket.id,
          text: cleanChunk,
        });
      } catch (err) {
        console.error("[socket][transcription-update] error:", err);
      }
    });


    socket.on("emotion.chunk", async (payload = {}, ack) => {
      try {
        const meetingId =
          payload.meetingId || payload.meeting_id || payload.meeting || socket.data?.meetingCode;
        const participantId = payload.participantId || payload.participant_id || payload.from || socket.data?.userId;
        const seq = Number.isFinite(Number(payload.seq)) ? Number(payload.seq) : null;
        const totalChunks = Number.isFinite(Number(payload.totalChunks)) ? Number(payload.totalChunks) : null;
        const chunkRaw = payload.chunk;
        const filename = payload.filename || `upload_${Date.now()}.bin`;
        const expectedMaxBytes = Number(payload.maxBytes) || PARTIAL_UPLOAD_MAX_BYTES;

        if (!meetingId || !participantId || seq === null || totalChunks === null || !chunkRaw) {
          if (typeof ack === "function") ack({ ok: false, reason: "missing_fields" });
          return;
        }

        const key = `${String(meetingId).trim().toUpperCase()}__${String(participantId)}`;

        let meta = PARTIAL_UPLOADS.get(key);
        if (!meta) {
          if (Number(totalChunks) <= 0 || Number(totalChunks) > 50000) {
            if (typeof ack === "function") ack({ ok: false, reason: "invalid_totalChunks" });
            return;
          }
          const dir = makeUploadDir(key);
          await mkdirp(dir);
          meta = {
            dir,
            totalChunks: Number(totalChunks),
            receivedBytes: 0,
            receivedCount: 0,
            filename,
            createdAt: Date.now(),
            maxBytes: expectedMaxBytes,
          };
          PARTIAL_UPLOADS.set(key, meta);
        }

        // Convert chunk to Buffer
        let bufChunk;
        try {
          if (Buffer.isBuffer(chunkRaw)) {
            bufChunk = chunkRaw;
          } else if (chunkRaw instanceof ArrayBuffer) {
            bufChunk = Buffer.from(chunkRaw);
          } else if (ArrayBuffer.isView(chunkRaw)) {
            bufChunk = Buffer.from(chunkRaw.buffer, chunkRaw.byteOffset, chunkRaw.byteLength);
          } else if (chunkRaw && chunkRaw.data && Array.isArray(chunkRaw.data)) {
            bufChunk = Buffer.from(chunkRaw.data);
          } else {
            throw new Error("unsupported chunk type");
          }
        } catch (convErr) {
          console.warn("[emotion.chunk] failed to convert chunk:", convErr);
          if (typeof ack === "function") ack({ ok: false, reason: "invalid_chunk" });
          return;
        }

        // Check limits
        if (meta.receivedBytes + bufChunk.length > meta.maxBytes) {
          console.warn(`[emotion.chunk] upload for ${key} would exceed max allowed bytes (${meta.maxBytes}). Aborting.`);
          PARTIAL_UPLOADS.delete(key);
          try { await unlink(meta.dir); } catch (e) {}
          if (typeof ack === "function") ack({ ok: false, reason: "too_large" });
          return;
        }

        if (seq < 0 || seq >= meta.totalChunks) {
          if (typeof ack === "function") ack({ ok: false, reason: "invalid_seq" });
          return;
        }


        const chunkPath = path.join(meta.dir, `chunk_${seq}`);

        await writeFile(chunkPath, bufChunk);


        meta.receivedBytes += bufChunk.length;
        meta.receivedCount += 1;
        meta.createdAt = Date.now();

        if (typeof ack === "function") ack({ ok: true, seq, receivedCount: meta.receivedCount });
      } catch (err) {
        console.error("[socket][emotion.chunk] error:", err);
        if (typeof ack === "function") ack({ ok: false, reason: "internal" });
      }
    });

    socket.on("emotion.chunk.abort", async (metaReq = {}, ack) => {
      try {
        const meetingId = metaReq.meetingId || metaReq.meeting_id || metaReq.meeting || socket.data?.meetingCode;
        const participantId = metaReq.participantId || metaReq.participant_id || metaReq.from || socket.data?.userId;
        if (!meetingId || !participantId) {
          if (typeof ack === "function") ack({ ok: false, reason: "missing" });
          return;
        }
        const key = `${String(meetingId).trim().toUpperCase()}__${String(participantId)}`;
        const meta = PARTIAL_UPLOADS.get(key);
        if (meta) {
          PARTIAL_UPLOADS.delete(key);
          try {
            await unlink(meta.dir);
          } catch (e) {
            console.warn("[socket][emotion.chunk.abort] failed to remove dir:", e && e.message ? e.message : e);
          }
        }
        if (typeof ack === "function") ack({ ok: true });
      } catch (err) {
        console.error("[socket][emotion.chunk.abort] error:", err);
        if (typeof ack === "function") ack({ ok: false, reason: "internal" });
      }
    });

    socket.on("emotion.chunk.complete", async (metaReq = {}, ack) => {
      try {
        const meetingId = metaReq.meetingId || metaReq.meeting_id || metaReq.meeting || socket.data?.meetingCode;
        const participantId = metaReq.participantId || metaReq.participant_id || metaReq.from || socket.data?.userId;
        const type = (metaReq.type || "video").toLowerCase();

        if (!meetingId || !participantId) {
          if (typeof ack === "function") ack({ ok: false, reason: "missing" });
          return;
        }

        const key = `${String(meetingId).trim().toUpperCase()}__${String(participantId)}`;
        const meta = PARTIAL_UPLOADS.get(key);
        if (!meta) {
          if (typeof ack === "function") ack({ ok: false, reason: "missing_upload" });
          return;
        }


        const missing = [];
        for (let i = 0; i < meta.totalChunks; i++) {
          const p = path.join(meta.dir, `chunk_${i}`);
          const s = await stat(p);
          if (!s) missing.push(i);
        }
        if (missing.length > 0) {
          if (typeof ack === "function") ack({ ok: false, reason: "missing_chunks", missing });
          return;
        }

        const finalFilename = meta.filename || `upload_${Date.now()}.bin`;
        const finalPath = path.join(meta.dir, `assembled_${finalFilename}`);
        const writeStream = fs.createWriteStream(finalPath, { flags: "w" });

        for (let i = 0; i < meta.totalChunks; i++) {
          const chunkPath = path.join(meta.dir, `chunk_${i}`);
          await new Promise((res, rej) => {
            const rs = fs.createReadStream(chunkPath);
            rs.on("error", rej);
            rs.on("end", res);
            rs.pipe(writeStream, { end: false });
          });
        }

        // close the write stream
        await new Promise((res, rej) => {
          writeStream.end(() => res());
          writeStream.on("error", rej);
        });

        // Final safety size check
        const finalStat = await stat(finalPath);
        if (!finalStat) {
          if (typeof ack === "function") ack({ ok: false, reason: "assemble_failed" });
          // cleanup
          PARTIAL_UPLOADS.delete(key);
          try { await unlink(meta.dir); } catch (e) {}
          return;
        }

        if (finalStat.size > meta.maxBytes) {
          console.warn(`[emotion.chunk.complete] final file too large (${finalStat.size} > ${meta.maxBytes})`);
          PARTIAL_UPLOADS.delete(key);
          try { await unlink(meta.dir); } catch (e) {}
          if (typeof ack === "function") ack({ ok: false, reason: "too_large_final" });
          return;
        }

        // Remove partial entry now to avoid races
        PARTIAL_UPLOADS.delete(key);

        // Forward to emotion service using the assembled file path (no buffering in memory)
        let emotionRes;
        try {
          emotionRes = await sendToEmotionService(String(meetingId).trim().toUpperCase(), String(participantId), finalPath, type, {
            mime: metaReq.mime,
            filename: finalFilename,
            timeoutMs: metaReq.timeoutMs || undefined,
          });
        } catch (svcErr) {
          console.error("[Emotion] sendToEmotionService error (chunked):", svcErr && svcErr.message ? svcErr.message : svcErr);
          // cleanup
          try { await unlink(meta.dir); } catch (e) {}
          if (typeof ack === "function") ack({ ok: false, reason: "service_error", message: svcErr && svcErr.message ? svcErr.message : String(svcErr) });
          return;
        }

        // Persist analytics (best-effort)
        try {
          const meeting = await Meeting.findOne({ meetingCode: String(meetingId).trim().toUpperCase() });
          if (meeting) {
            await meeting.updateAnalytics({ emotionScores: emotionRes });
          }
        } catch (dbErr) {
          console.warn("[Emotion] failed to persist analytics (chunked):", dbErr);
        }

        // Emit to host if present
        const roomState = meetingState[String(meetingId).trim().toUpperCase()];
        const hostSocketId = Array.isArray(roomState) && roomState.length > 0 ? roomState[0] : null;

        if (hostSocketId && io) {
          io.to(hostSocketId).emit("emotion.update", {
            meeting_id: String(meetingId).trim().toUpperCase(),
            participant_id: String(participantId),
            type,
            emotion: emotionRes,
            ts: Date.now(),
          });
        } else {
          console.log(`[Emotion] no host for meeting ${meetingId}. skipping emit (chunked).`);
        }

        // Clean up temp dir (remove assembled file and chunk files)
        try {
          await unlink(meta.dir);
        } catch (e) {
          console.warn("[emotion.chunk.complete] cleanup failed:", e && e.message ? e.message : e);
        }

        if (typeof ack === "function") ack({ ok: true, result: Array.isArray(emotionRes) ? emotionRes.length : typeof emotionRes === "object" ? "object" : "ok" });
      } catch (err) {
        console.error("[socket][emotion.chunk.complete] error:", err);
        if (typeof ack === "function") ack({ ok: false, reason: "internal" });
      }
    });

    socket.on("emotion.frame", async (payload, ack) => {
      try {
        if (!payload || typeof payload !== "object") {
          if (typeof ack === "function") ack({ ok: false, reason: "invalid_payload" });
          return;
        }

        // normalize inputs
        const rawMeetingId = payload.meetingId || payload.meeting_id || payload.meeting || "";
        const meetingCode = String(rawMeetingId || "").trim().toUpperCase();
        const participantId = payload.participantId || payload.participant_id || payload.from || socket.data?.userId;
        const type = (payload.type || "frame").toLowerCase();
        const mime = payload.mime || payload.contentType || payload.content_type || undefined;
        const filename = payload.filename || undefined;
        const buffer = payload.buffer || payload.data || null;

        if (!meetingCode || !participantId || !buffer) {
          // client may be waiting for callback; call it
          if (typeof ack === "function") ack({ ok: false, reason: "missing meetingId/participantId/buffer" });
          // also emit legacy event for clients listening for that
          socket.emit("emotion.ack", { ok: false, reason: "missing meetingId/participantId/buffer" });
          return;
        }

        // convert various binary types to Buffer robustly (fast, synchronous)
        let buf;
        try {
          if (Buffer.isBuffer(buffer)) {
            buf = buffer;
          } else if (buffer instanceof ArrayBuffer) {
            buf = Buffer.from(buffer);
          } else if (ArrayBuffer.isView(buffer)) {
            buf = Buffer.from(buffer.buffer, buffer.byteOffset, buffer.byteLength);
          } else if (buffer && buffer.buffer instanceof ArrayBuffer) {
            buf = Buffer.from(buffer.buffer);
          } else if (buffer && buffer.data && Array.isArray(buffer.data)) {
            buf = Buffer.from(buffer.data);
          } else {
            throw new Error("unsupported buffer type");
          }
        } catch (convErr) {
          console.warn("[Emotion] failed to convert incoming buffer:", convErr);
          if (typeof ack === "function") ack({ ok: false, reason: "invalid buffer format" });
          socket.emit("emotion.ack", { ok: false, reason: "invalid buffer format" });
          return;
        }

        // Safety check: if buffer is very large, warn (and still try)
        try {
          const maxBuf = io?.opts?.maxHttpBufferSize || DEFAULT_MAX_HTTP_BUFFER;
          if (buf.length > maxBuf) {
            console.warn(`[Emotion] incoming buffer (${buf.length} bytes) exceeds configured maxHttpBufferSize (${maxBuf}). Consider increasing maxHttpBufferSize if this is expected.`);
          }
        } catch (e) {
          // ignore
        }

        if (typeof ack === "function") {
          try {
            ack({ ok: true, accepted: true });
          } catch (e) {

          }
        }

        (async () => {
          try {

            let emotionRes;
            try {
              emotionRes = await sendToEmotionService(meetingCode, participantId, buf, type, { mime, filename });
            } catch (svcErr) {
              console.error("[Emotion] sendToEmotionService error:", svcErr && svcErr.message ? svcErr.message : svcErr);

              try { socket.emit("emotion.ack", { ok: false, reason: svcErr && svcErr.message ? svcErr.message : "service error" }); } catch {}
              return;
            }

            try {
              const meeting = await Meeting.findOne({ meetingCode: meetingCode });
              if (meeting) {
                await meeting.updateAnalytics({ emotionScores: emotionRes });
              }
            } catch (dbErr) {
              console.warn("[Emotion] failed to persist analytics:", dbErr);
            }

            // Emit only to host (first joined socket for the meeting)
            const roomState = meetingState[meetingCode];
            const hostSocketId = Array.isArray(roomState) && roomState.length > 0 ? roomState[0] : null;

            if (hostSocketId && io) {
              io.to(hostSocketId).emit("emotion.update", {
                meeting_id: meetingCode,
                participant_id: participantId,
                type,
                emotion: emotionRes,
                ts: Date.now(),
              });
            } else {
              console.log(`[Emotion] no host for meeting ${meetingCode}. skipping emit.`);
            }

            try {
              socket.emit("emotion.ack", { ok: true, result: Array.isArray(emotionRes) ? emotionRes.length : typeof emotionRes === "object" ? "object" : "ok" });
            } catch (e) {

            }
          } catch (err) {
            console.error("[socket][emotion.frame async] processing error:", err);
            try { socket.emit("emotion.ack", { ok: false, reason: err && err.message ? err.message : "internal" }); } catch {}
          } finally {

          }
        })();

      } catch (err) {
        console.error("[socket][emotion.frame] error:", err);
        try {
          if (typeof ack === "function") ack({ ok: false, reason: err && err.message ? err.message : "internal" });
        } catch (e) {}
        try { socket.emit("emotion.ack", { ok: false, reason: err && err.message ? err.message : "internal" }); } catch {}
      }
    });

    socket.on("emotion-update", async (data) => {
      try {
        const code = socket.data?.meetingCode;
        if (!code || typeof data !== "object") return;
        const meeting = await Meeting.findOne({ meetingCode: code });
        if (!meeting) return;

        await meeting.updateAnalytics({ emotionScores: data });

        const hostSocketId = Array.isArray(meetingState[code]) && meetingState[code].length > 0
          ? meetingState[code][0]
          : null;

        if (hostSocketId) {
          io.to(hostSocketId).emit("emotion-update", {
            from: socket.id,
            scores: data,
          });
        } else {
          socket.to(`meeting:${code}`).emit("emotion-update", {
            from: socket.id,
            scores: data,
          });
        }
      } catch (err) {
        console.error("[socket][emotion-update] error:", err);
      }
    });

    socket.on("keywords-update", async (keywords) => {
      try {
        const code = socket.data?.meetingCode;
        if (!code || !Array.isArray(keywords)) return;
        const meeting = await Meeting.findOne({ meetingCode: code });
        if (!meeting) return;

        const cleanKeywords = keywords.map((k) =>
          sanitizeHtml(String(k).slice(0, 100))
        );
        await meeting.updateAnalytics({ keywords: cleanKeywords });

        socket.to(`meeting:${code}`).emit("keywords-update", {
          from: socket.id,
          keywords: cleanKeywords,
        });
      } catch (err) {
        console.error("[socket][keywords-update] error:", err);
      }
    });

    socket.on("leave-call", async (meetingCodeRaw) => {
      try {
        const code = String(meetingCodeRaw || "").trim().toUpperCase();
        const { userId } = socket.data || {};
        await handleLeave(socket, code, io, userId);
      } catch (err) {
        console.error("[socket][leave-call] error:", err);
      }
    });

    socket.on("disconnect", async () => {
      try {
        const code = socket.data?.meetingCode;
        const { userId } = socket.data || {};
        if (!code || !userId) return;
        await handleLeave(socket, code, io, userId);
      } catch (err) {
        console.error("[socket][disconnect] error:", err);
      }
    });
  });

  if (typeof process !== "undefined" && process && process.on) {
    process.on("exit", () => clearInterval(cleanupInterval));
    process.on("SIGINT", () => clearInterval(cleanupInterval));
    process.on("SIGTERM", () => clearInterval(cleanupInterval));
  }

  return io;
}

async function handleLeave(socket, code, io, userId) {
  try {
    const meeting = await Meeting.findOne({ meetingCode: code });
    if (!meeting) return;

    await meeting.markParticipantLeft(socket.id);

    if (meetingState[code]) {
      meetingState[code] = meetingState[code].filter((id) => id !== socket.id);
      if (meetingState[code].length === 0) {
        delete meetingState[code];
        await finalizeAnalytics(meeting);
      }
    }

    if (meetingParticipants[code]) {
      meetingParticipants[code].delete(userId);
      if (meetingParticipants[code].size === 0) delete meetingParticipants[code];
    }

    socket.leave(`meeting:${code}`);
    socket.to(`meeting:${code}`).emit("user-left", socket.id);
    await broadcastParticipantsFromCache(code, io);

    console.log(`[socket] ${socket.id} left meeting ${code}`);
  } catch (err) {
    console.error("[socket][handleLeave] error:", err);
  }
}

async function broadcastParticipantsFromCache(code, io) {
  try {
    if (!meetingParticipants[code]) return;
    const participants = Array.from(meetingParticipants[code].values()).map((p) => ({
      id: p.socketId,
      meta: p.meta || {},
    }));
    io.in(`meeting:${code}`).emit("participants-updated", participants);
  } catch (err) {
    console.error("[socket][broadcastParticipantsFromCache] error:", err);
  }
}

async function finalizeAnalytics(meeting) {
  console.log(`[socket] Finalizing analytics for meeting ${meeting.meetingCode}...`);
}
