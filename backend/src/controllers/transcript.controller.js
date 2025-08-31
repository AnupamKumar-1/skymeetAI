// backend/src/controllers/transcript.controller.js
import Transcript from "../models/transcript.model.js";

/**
 * POST /api/v1/transcript
 * Body: { meetingCode, transcriptText, fileName?, createdAt?, metadata? }
 * Upserts the transcript by meetingCode and returns the saved doc.
 */
export async function createTranscript(req, res) {
  try {
    const { meetingCode, transcriptText, fileName, createdAt, metadata } = req.body;

    if (!meetingCode) {
      return res.status(400).json({ success: false, message: "meetingCode is required" });
    }

    const code = String(meetingCode).toUpperCase();

    const update = {
      meetingCode: code,
      transcriptText: transcriptText || "",
      fileName: fileName || null,
      metadata: metadata || {},
      updatedAt: new Date(),
    };
    if (createdAt) update.createdAt = new Date(createdAt);

    const doc = await Transcript.findOneAndUpdate(
      { meetingCode: code },
      { $set: update },
      { new: true, upsert: true, setDefaultsOnInsert: true }
    ).lean();

    return res.json({ success: true, transcript: doc });
  } catch (err) {
    console.error("createTranscript error:", err);
    return res.status(500).json({ success: false, message: "Server error", error: err.message });
  }
}

// Backwards-compatible alias
export const saveTranscript = createTranscript;

/**
 * GET /api/v1/transcript/:id
 * Accepts either a MongoDB _id or a meetingCode (case-insensitive)
 */
export async function getTranscript(req, res) {
  try {
    const idOrCode = String(req.params.id || "").trim();
    if (!idOrCode) return res.status(400).json({ success: false, message: "Missing id/code" });

    let doc = null;

    // Try as ObjectId (findById) — wrap in try/catch because invalid ObjectId will throw a CastError
    try {
      doc = await Transcript.findById(idOrCode).lean();
    } catch (e) {
      // not a valid ObjectId; ignore and try by meetingCode
    }

    if (!doc) {
      const code = idOrCode.toUpperCase();
      doc = await Transcript.findOne({ meetingCode: code }).lean();
    }

    if (!doc) return res.status(404).json({ success: false, message: "Not found" });

    return res.json({ success: true, transcript: doc });
  } catch (err) {
    console.error("getTranscript error:", err);
    return res.status(500).json({ success: false, message: "Server error", error: err.message });
  }
}

// Backwards-compatible alias
export const getTranscriptByCode = getTranscript;

/**
 * GET /api/v1/transcript
 * Optional: ?meeting_code=CODE or ?limit=50
 */
export async function listTranscripts(req, res) {
  try {
    const { meeting_code, limit = 50 } = req.query;
    const q = {};
    if (meeting_code) q.meetingCode = String(meeting_code).toUpperCase();

    const docs = await Transcript.find(q)
      .sort({ createdAt: -1 })
      .limit(Math.min(parseInt(limit, 10) || 50, 200))
      .lean();

    return res.json({ success: true, transcripts: docs });
  } catch (err) {
    console.error("listTranscripts error:", err);
    return res.status(500).json({ success: false, message: "Server error", error: err.message });
  }
}
