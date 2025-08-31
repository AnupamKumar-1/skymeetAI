// backend/src/routes/emotion.routes.js
import express from "express";
import multer from "multer";
import path from "path";
import fs from "fs";
import {
  uploadEmotionFileHandler,
  sendToEmotionService, // optional export if you want direct use elsewhere
} from "../controllers/emotion.controller.js";

const router = express.Router();

/**
 * Optional auth middleware placeholder.
 * Replace with your real auth (JWT/session) checker.
 */
function ensureAuth(req, res, next) {
  // Example: check req.user set by passport or a session middleware
  // if (req.user) return next();
  // OR check Authorization header: Bearer <token>
  // if (!req.headers.authorization) return res.status(401).json({ error: 'Unauthorized' });
  //
  // Replace the following with your checks:
  return next();
}

/* ---------- Multer setup ---------- */

// Destination directory for temporary uploads
const TMP_UPLOAD_DIR = process.env.EMOTION_UPLOAD_TMP_DIR || "/tmp/emotion_uploads";

// Ensure directory exists
if (!fs.existsSync(TMP_UPLOAD_DIR)) {
  try {
    fs.mkdirSync(TMP_UPLOAD_DIR, { recursive: true });
  } catch (err) {
    console.warn(`[emotion.routes] could not create tmp dir ${TMP_UPLOAD_DIR}:`, err.message);
  }
}

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, TMP_UPLOAD_DIR);
  },
  filename: (req, file, cb) => {
    // keep extension, but generate unique filename
    const ext = path.extname(file.originalname) || "";
    const safeBase = Date.now() + "-" + Math.round(Math.random() * 1e9);
    cb(null, `${safeBase}${ext}`);
  },
});

// Accept images, audio and video. Adjust as needed.
function fileFilter(req, file, cb) {
  const mime = file.mimetype || "";
  if (
    mime.startsWith("image/") ||
    mime.startsWith("audio/") ||
    mime.startsWith("video/")
  ) {
    cb(null, true);
  } else {
    cb(new Error("Only image/audio/video files are allowed"));
  }
}

const upload = multer({
  storage,
  fileFilter,
  limits: {
    // 6 MB default - tune per your model/input requirements
    fileSize: parseInt(process.env.EMOTION_MAX_FILE_BYTES || String(6 * 1024 * 1024), 10),
  },
});

/* ---------- Routes ---------- */

/**
 * POST /upload
 * multipart/form-data:
 *  - meeting_id
 *  - participant_id
 *  - type (optional: 'frame'|'audio'|'video')
 *  - file (the binary)
 *
 * Protected by ensureAuth (optional) — replace ensureAuth with your real auth middleware.
 */
router.post(
  "/upload",
  ensureAuth,
  upload.single("file"),
  async (req, res, next) => {
    try {
      // Delegate to controller
      return await uploadEmotionFileHandler(req, res);
    } catch (err) {
      // Multer errors are passed here; normalize response
      console.error("[emotion.routes] /upload handler error:", err);
      return res.status(500).json({ ok: false, error: err.message || "internal error" });
    }
  }
);

/**
 * Lightweight health/status endpoint for the emotion route
 */
router.get("/status", (req, res) => {
  res.json({
    ok: true,
    msg: "emotion routes healthy",
    tmpUploadDir: TMP_UPLOAD_DIR,
  });
});

/**
 * Optional: direct test route to proxy to EMOTION_SERVICE (debug)
 * Use with care — this bypasses storage and only intended for local debug.
 */
router.post("/proxy-test", ensureAuth, upload.single("file"), async (req, res) => {
  try {
    const meetingId = req.body?.meeting_id || req.body?.meetingId;
    const participantId = req.body?.participant_id || req.body?.participantId;
    const type = req.body?.type || "frame";
    if (!meetingId || !participantId || !req.file) {
      return res.status(400).json({ ok: false, error: "meeting_id, participant_id and file required" });
    }

    // sendToEmotionService is defined in controller and supports file path or Buffer
    const result = await sendToEmotionService(meetingId, participantId, req.file.path, type);

    // cleanup file (controller also attempts to remove; double-safety)
    try {
      fs.unlinkSync(req.file.path);
    } catch (e) {
      // ignore
    }

    return res.json({ ok: true, result });
  } catch (err) {
    console.error("[emotion.routes] /proxy-test error:", err);
    return res.status(500).json({ ok: false, error: err.message || "internal error" });
  }
});

export default router;
