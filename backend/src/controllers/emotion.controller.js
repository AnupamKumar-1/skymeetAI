// backend/src/controllers/emotion.controller.js
import axios from "axios";
import FormData from "form-data";
import fs from "fs";
import path from "path";
import os from "os";

const EMOTION_SERVICE_URL =
  process.env.EMOTION_SERVICE_URL || "http://localhost:5002/analyze";

/**
 * Build a FormData instance for the emotion service request.
 * Accepts Buffer, file path (string), or a readable stream.
 *
 * @param {string} meetingId
 * @param {string} participantId
 * @param {string|Buffer|stream.Readable} fileOrBuffer
 * @param {string} type - 'audio'|'video'|'frame'
 * @param {Object} opts - optional { mime, filename }
 * @returns {FormData}
 */
function buildForm(meetingId, participantId, fileOrBuffer, type = "audio", opts = {}) {
  const form = new FormData();
  form.append("meeting_id", meetingId);
  form.append("participant_id", participantId);
  form.append("type", type);

  const mime = opts.mime || "";
  let filename = opts.filename || "";

  if (Buffer.isBuffer(fileOrBuffer)) {
    // Ensure a sensible default filename/extension if none provided
    if (!filename) {
      const ext = type === "audio" ? "webm" : type === "video" ? "webm" : "jpg";
      filename = `${participantId}.${ext}`;
    }
    // Attach Buffer directly with optional contentType
    const opt = {};
    if (mime) opt.contentType = mime;
    opt.filename = filename;
    form.append("file", fileOrBuffer, opt);
  } else if (typeof fileOrBuffer === "string") {
    // treat as file path
    const resolved = fileOrBuffer;
    const base = path.basename(resolved);
    // prefer explicit filename, else use file basename
    filename = filename || base;
    // create read stream and append (form-data will set content-type by stream, if not provided we can set it via opts.mime)
    if (mime) {
      form.append("file", fs.createReadStream(resolved), { filename, contentType: mime });
    } else {
      form.append("file", fs.createReadStream(resolved), { filename });
    }
  } else if (fileOrBuffer && typeof fileOrBuffer.pipe === "function") {
    // readable stream (e.g. fs.createReadStream supplied by caller)
    if (!filename) {
      const ext = type === "audio" ? "webm" : type === "video" ? "webm" : "jpg";
      filename = `${participantId}.${ext}`;
    }
    if (mime) {
      form.append("file", fileOrBuffer, { filename, contentType: mime });
    } else {
      form.append("file", fileOrBuffer, { filename });
    }
  } else {
    throw new Error("fileOrBuffer must be a Buffer, file path string, or readable stream");
  }

  return form;
}

/**
 * Helper to compute FormData length (wraps form.getLength)
 * @param {FormData} form
 * @returns {Promise<number>}
 */
function getFormLength(form) {
  return new Promise((resolve, reject) => {
    form.getLength((err, length) => {
      if (err) return reject(err);
      resolve(length);
    });
  });
}

/**
 * Post a FormData to the EMOTION_SERVICE_URL using axios.
 * This helper computes Content-Length when possible and is tolerant of large uploads.
 *
 * @param {FormData} form
 * @param {number} timeoutMs
 */
async function postForm(form, timeoutMs = 120000) {
  // Ensure headers from form-data
  const headers = form.getHeaders();

  // Try to compute the content-length and add to headers (helps with proxies)
  try {
    const length = await getFormLength(form);
    if (typeof length === "number") {
      headers["Content-Length"] = length;
    }
  } catch (lenErr) {
    // Not fatal — some form-data combinations cannot compute length (streams). We continue without it.
    console.warn("[EmotionService] could not compute form length:", lenErr && (lenErr.message || lenErr));
  }

  const axiosCfg = {
    headers,
    timeout: timeoutMs,
    maxContentLength: Infinity,
    maxBodyLength: Infinity,
  };

  return axios.post(EMOTION_SERVICE_URL, form, axiosCfg);
}

/**
 * Send audio/video/frame data to external emotion service
 * (Supports Buffer, file path string or readable stream - streams are written to a temp file to allow retries)
 *
 * @param {string} meetingId
 * @param {string} participantId
 * @param {string|Buffer|stream.Readable} fileOrBuffer - Path to file OR raw Buffer OR readable stream
 * @param {string} type - 'audio'|'video'|'frame'
 * @param {Object} opts - optional { mime, filename, timeoutMs }
 * @returns {Promise<Object>} emotion service response body
 */
export async function sendToEmotionService(
  meetingId,
  participantId,
  fileOrBuffer,
  type = "audio",
  opts = {}
) {
  if (!meetingId || !participantId || !fileOrBuffer) {
    throw new Error("meetingId, participantId and fileOrBuffer are required");
  }

  const timeoutMs = typeof opts.timeoutMs === "number" ? opts.timeoutMs : 120000; // default 120s

  // If caller passed a readable stream that is not a file-path-backed stream,
  // write it to a temp file so we can safely retry the request.
  let cleanupTemp = null;
  try {
    // detect readable stream (duck-typing)
    const isStream =
      fileOrBuffer &&
      typeof fileOrBuffer === "object" &&
      typeof fileOrBuffer.pipe === "function";

    if (isStream) {
      // If the stream has a .path property (fs.createReadStream), use the path directly
      if (fileOrBuffer.path && typeof fileOrBuffer.path === "string") {
        fileOrBuffer = String(fileOrBuffer.path);
      } else {
        // Otherwise, drain stream to a temp file
        const tmpName = path.join(
          os.tmpdir(),
          `emotion_upload_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`
        );

        await new Promise((resolve, reject) => {
          const out = fs.createWriteStream(tmpName);
          fileOrBuffer.pipe(out);
          out.on("finish", resolve);
          out.on("error", reject);
          fileOrBuffer.on("error", reject);
        });

        fileOrBuffer = tmpName;
        cleanupTemp = tmpName;
      }
    }

    // Build form for first attempt
    let form = buildForm(meetingId, participantId, fileOrBuffer, type, opts);

    console.log(
      `[EmotionService] Posting ${type} for meeting=${meetingId} participant=${participantId} -> ${EMOTION_SERVICE_URL}`
    );

    try {
      const res = await postForm(form, timeoutMs);
      console.log(`[EmotionService] status ${res.status}`);
      return res.data;
    } catch (err) {
      // Log details
      if (err.response) {
        console.error(
          `[EmotionService] HTTP ${err.response.status}: ${JSON.stringify(err.response.data)}`
        );
      } else {
        console.error(`[EmotionService] Request error: ${err.message || err}`);
      }

      // If it's a network error (no response) and transient, attempt one retry.
      const transient =
        !err.response &&
        (err.code === "ECONNREFUSED" ||
          err.code === "ETIMEDOUT" ||
          err.code === "EPIPE" ||
          err.code === "ECONNRESET");

      if (transient) {
        console.log("[EmotionService] Network error — retrying once...");
        try {
          // rebuild form (for file path this re-creates streams; for Buffer it's OK)
          form = buildForm(meetingId, participantId, fileOrBuffer, type, opts);
          const retryRes = await postForm(form, timeoutMs);
          console.log(`[EmotionService] retry status ${retryRes.status}`);
          return retryRes.data;
        } catch (retryErr) {
          console.error("[EmotionService] Retry failed:", retryErr && (retryErr.message || retryErr));
          throw retryErr;
        }
      }

      throw err;
    }
  } finally {
    // Clean up any temporary file we created from a stream
    if (cleanupTemp) {
      try {
        fs.unlinkSync(cleanupTemp);
      } catch (e) {
        console.warn("[EmotionService] failed to remove temp upload file:", e && (e.message || e));
      }
    }
  }
}

/**
 * Express handler for multipart uploads (multer will populate req.file)
 * Expects fields:
 *  - meeting_id
 *  - participant_id
 *  - type (optional) 'frame'|'audio'|'video'
 *  - file (multipart file) provided by multer
 *
 * Emits 'emotion.update' ONLY to host socket for that meeting (host = first joined socket in meetingState[meetingId])
 */
export async function uploadEmotionFileHandler(req, res) {
  try {
    const meetingId = req.body?.meeting_id || req.body?.meetingId;
    const participantId = req.body?.participant_id || req.body?.participantId;
    const type = req.body?.type || "frame";
    const file = req.file;

    if (!meetingId || !participantId || !file) {
      return res.status(400).json({
        ok: false,
        error: "meeting_id, participant_id and file are required",
      });
    }

    // Multer provides an on-disk file at file.path
    const filePath = file.path;
    const mime = file.mimetype || undefined;
    const filename = file.originalname || path.basename(file.path);

    // Forward to emotion service using file path and explicit mime/filename
    const emotionResult = await sendToEmotionService(meetingId, participantId, filePath, type, {
      mime,
      filename,
      timeoutMs: 120000,
    });

    // Emit only to host if possible
    try {
      const io = global.io;
      const meetingState = global.meetingState || {};
      const roomState = meetingState[String(meetingId).trim().toUpperCase()];

      const hostSocketId = Array.isArray(roomState) && roomState.length > 0 ? roomState[0] : null;

      if (io && hostSocketId) {
  // try to fetch participant name from the server's in-memory cache so UI can show it immediately
  let nameFromCache;
  try {
    const mpStore = global.meetingParticipants || {};
    const key = String(meetingId).trim().toUpperCase();
    const mp = mpStore[key];

    // common structures used in your socketManager:
    // - mp is a Map -> values() yields objects { socketId, meta }
    // - or mp may be an array of { socketId, meta } or similar
    if (mp && typeof mp.values === "function") {
      for (const v of mp.values()) {
        if (v && (v.socketId === participantId || v.id === participantId)) {
          nameFromCache = v.meta && (v.meta.name || v.meta.displayName);
          break;
        }
      }
    } else if (Array.isArray(mp)) {
      const found = mp.find((p) => p.socketId === participantId || p.id === participantId);
      if (found) nameFromCache = found.meta && (found.meta.name || found.meta.displayName);
    } else if (mp && typeof mp === "object") {
      // if mp keyed by userId and stores values with socketId
      for (const k of Object.keys(mp)) {
        const v = mp[k];
        if (v && (v.socketId === participantId || v.id === participantId)) {
          nameFromCache = v.meta && (v.meta.name || v.meta.displayName);
          break;
        }
      }
    }
  } catch (e) {
    console.warn("[Emotion] failed to lookup participant name from cache:", e && (e.message || e));
  }

  const payload = {
    meeting_id: meetingId,
    participant_id: participantId,
    type,
    emotion: emotionResult,
    ts: Date.now(),
  };
  if (nameFromCache) payload.name = nameFromCache;

  // log when we attach a name to help debug
  if (nameFromCache) {
    console.info(`[Emotion] emitting update for ${participantId} name=${nameFromCache} meeting=${meetingId}`);
  }

  io.to(hostSocketId).emit("emotion.update", payload);
} else {
  console.log(`[Emotion] No host socket found for meeting=${meetingId}. skipping emit.`);
}

    } catch (emitErr) {
      console.error("[Emotion] emit error:", emitErr);
    }

    return res.json({ ok: true, result: emotionResult });
  } catch (err) {
    console.error("[Emotion] upload handler error:", err && (err.stack || err));
    return res.status(500).json({ ok: false, error: err.message || "internal error" });
  } finally {
    // Clean up multer temp file if present (best-effort)
    try {
      if (req.file && req.file.path) {
        fs.unlink(req.file.path, (unlinkErr) => {
          if (unlinkErr) {
            console.warn("[Emotion] failed to unlink temp file:", unlinkErr && unlinkErr.message);
          }
        });
      }
    } catch (e) {
      // ignore
    }
  }
}
