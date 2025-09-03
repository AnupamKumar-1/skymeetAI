
// backend/src/controllers/transcript.controller.js
import crypto from "crypto";
import Transcript from "../models/transcript.model.js";
import { Meeting } from "../models/meeting.model.js";

/**
 * Helper: read hostSecret from header / body / query
 */
function getHostSecretFromReq(req) {
  return (
    req.headers["x-host-secret"] ||
    req.body?.hostSecret ||
    req.query?.hostSecret ||
    null
  );
}

/**
 * Helper: compute sha256 hex of provided secret
 */
function sha256Hex(input) {
  return crypto.createHash("sha256").update(String(input)).digest("hex");
}

/**
 * Normalize user id from req.user, supporting both .id and ._id
 */
function getUserIdFromReq(req) {
  if (!req?.user) return null;
  if (req.user.id) return String(req.user.id);
  if (req.user._id) return String(req.user._id);
  return null;
}

/**
 * Verify that provided hostSecret is valid for meetingCode.
 * Returns the meeting doc (lean) when valid, otherwise null.
 * Uses Meeting.verifyHostSecret if available; otherwise falls back to manual hash compare.
 */
async function verifyHostSecretForMeeting(meetingCode, providedSecret) {
  if (!meetingCode || !providedSecret) return null;

  if (typeof Meeting.verifyHostSecret === "function") {
    try {
      const meeting = await Meeting.verifyHostSecret(meetingCode, providedSecret);
      return meeting || null;
    } catch (e) {
      console.warn("verifyHostSecret error, falling back to manual check:", e?.message || e);
      // fall-through to manual check
    }
  }

  const providedHash = sha256Hex(providedSecret);
  const meeting = await Meeting.findOne({
    meetingCode: String(meetingCode).toUpperCase(),
    hostSecretHash: providedHash,
  }).lean();

  return meeting || null;
}

/**
 * Find meetings that match the provided hostSecret (returns array of meetings)
 */
async function findMeetingsByHostSecret(providedSecret) {
  if (!providedSecret) return [];
  const providedHash = sha256Hex(providedSecret);
  const meetings = await Meeting.find({ hostSecretHash: providedHash }).lean();
  return meetings || [];
}

/**
 * Find meetings by ownerId (returns array of meetings)
 */
async function findMeetingsByOwnerId(ownerId) {
  if (!ownerId) return [];
  const meetings = await Meeting.find({ ownerId }).lean();
  return meetings || [];
}

/**
 * Authorize access to a meeting given either:
 *  - a valid hostSecret for the meeting, or
 *  - an authenticated user who matches meeting.ownerId
 *
 * Returns the meeting doc (lean) if authorized, otherwise null.
 *
 * meetingCode: code string (required)
 * providedSecret: hostSecret string (optional)
 * reqUser: req.user object (optional)
 */
async function authorizeMeetingAccess(meetingCode, providedSecret, reqUser) {
  if (!meetingCode) return null;
  const code = String(meetingCode).toUpperCase();

  // 1) If hostSecret provided, prefer that (backwards-compatible)
  if (providedSecret) {
    const meeting = await verifyHostSecretForMeeting(code, providedSecret);
    if (meeting) return meeting;
    // if invalid secret, do not immediately return; we will still allow owner check below
  }

  // 2) If authenticated user, check ownership
  if (reqUser) {
    const userId = reqUser.id ? String(reqUser.id) : reqUser._id ? String(reqUser._id) : null;
    if (userId) {
      const meeting = await Meeting.findOne({ meetingCode: code, ownerId: userId }).lean();
      if (meeting) return meeting;
    }
  }

  // no access
  return null;
}

/**
 * POST /api/v1/transcript
 * Body: { meetingCode, transcriptText, fileName?, createdAt?, metadata?, hostSecret? }
 * Upserts the transcript by meetingCode and returns the saved doc.
 *
 * Authorization: Accepts either hostSecret (legacy) OR authenticated owner (req.user).
 */
export async function createTranscript(req, res) {
  try {
    const { meetingCode, transcriptText, fileName, createdAt, metadata } = req.body;
    const hostSecret = getHostSecretFromReq(req);
    const reqUser = req.user;

    if (!meetingCode) {
      return res.status(400).json({ success: false, message: "meetingCode is required" });
    }

    const code = String(meetingCode).toUpperCase();

    // Authorize either by hostSecret or by owner (authenticated user)
    const meeting = await authorizeMeetingAccess(code, hostSecret, reqUser);
    if (!meeting) {
      return res.status(403).json({ success: false, message: "not authorized to create/update transcript for this meeting" });
    }

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
 *
 * Authorization: Either hostSecret that matches the transcript's meeting OR authenticated owner (req.user).
 */
export async function getTranscript(req, res) {
  try {
    const idOrCode = String(req.params.id || "").trim();
    if (!idOrCode) return res.status(400).json({ success: false, message: "Missing id/code" });

    const hostSecret = getHostSecretFromReq(req);
    const reqUser = req.user;

    let doc = null;
    let meetingCode = null;

    // Try as ObjectId (findById). If found, extract meetingCode for verification.
    try {
      doc = await Transcript.findById(idOrCode).lean();
    } catch (e) {
      // not a valid ObjectId; ignore and try by meetingCode
    }

    if (doc) {
      meetingCode = doc.meetingCode;
      if (!meetingCode) {
        return res.status(404).json({ success: false, message: "Transcript missing meetingCode" });
      }
      const meeting = await authorizeMeetingAccess(meetingCode, hostSecret, reqUser);
      if (!meeting) {
        return res.status(403).json({ success: false, message: "not authorized to access this transcript" });
      }
      return res.json({ success: true, transcript: doc });
    }

    // If not found by id, treat idOrCode as a meetingCode
    const code = idOrCode.toUpperCase();
    const meeting = await authorizeMeetingAccess(code, hostSecret, reqUser);
    if (!meeting) {
      return res.status(403).json({ success: false, message: "not authorized to access transcript for this meeting code" });
    }

    doc = await Transcript.findOne({ meetingCode: code }).lean();
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
 * Query:
 *   - ?meeting_code=CODE  (optional)
 *   - ?limit=50
 *   - ?mine=true          (optional, when authenticated)
 * Authorization:
 *   - If meeting_code provided: either hostSecret or authenticated owner for that meeting.
 *   - If meeting_code NOT provided:
 *       - If ?mine=true and authenticated: return transcripts for meetings owned by user.
 *       - Else if hostSecret provided: return transcripts for all meetings matching that hostSecret.
 *       - Otherwise: 403
 */
export async function listTranscripts(req, res) {
  try {
    const { meeting_code, limit = 50, mine } = req.query;
    const hostSecret = getHostSecretFromReq(req);
    const reqUser = req.user;
    const userId = getUserIdFromReq(req);

    // limit sanity
    const finalLimit = Math.min(parseInt(limit, 10) || 50, 200);

    if (meeting_code) {
      const code = String(meeting_code).toUpperCase();
      const meeting = await authorizeMeetingAccess(code, hostSecret, reqUser);
      if (!meeting) {
        return res.status(403).json({ success: false, message: "not authorized for this meeting code" });
      }

      const docs = await Transcript.find({ meetingCode: code })
        .sort({ createdAt: -1 })
        .limit(finalLimit)
        .lean();

      return res.json({ success: true, transcripts: docs });
    }

    // If explicitly asked for transcripts for the authenticated user's meetings
    if (String(mine) === "true") {
      if (!userId) {
        return res.status(401).json({ success: false, message: "authentication required for mine=true" });
      }
      const meetings = await findMeetingsByOwnerId(userId);
      const meetingCodes = meetings.map((m) => m.meetingCode).filter(Boolean);
      if (meetingCodes.length === 0) {
        return res.json({ success: true, transcripts: [] });
      }
      const docs = await Transcript.find({ meetingCode: { $in: meetingCodes } })
        .sort({ createdAt: -1 })
        .limit(finalLimit)
        .lean();
      return res.json({ success: true, transcripts: docs });
    }

    // No meeting_code and not requesting mine=true:
    // If authenticated user is present, return transcripts for user's meetings (conservative default)
    if (userId) {
      const meetings = await findMeetingsByOwnerId(userId);
      const meetingCodes = meetings.map((m) => m.meetingCode).filter(Boolean);
      if (meetingCodes.length === 0) {
        return res.json({ success: true, transcripts: [] });
      }
      const docs = await Transcript.find({ meetingCode: { $in: meetingCodes } })
        .sort({ createdAt: -1 })
        .limit(finalLimit)
        .lean();
      return res.json({ success: true, transcripts: docs });
    }

    // Fallback: require hostSecret and return transcripts for meetings matching that secret
    if (!hostSecret) {
      return res.status(403).json({ success: false, message: "hostSecret required or authenticate to list transcripts" });
    }

    const meetings = await findMeetingsByHostSecret(hostSecret);
    if (!meetings || meetings.length === 0) {
      return res.status(403).json({ success: false, message: "no meetings found for provided host secret" });
    }

    const meetingCodes = meetings.map((m) => m.meetingCode).filter(Boolean);
    const docs = await Transcript.find({ meetingCode: { $in: meetingCodes } })
      .sort({ createdAt: -1 })
      .limit(finalLimit)
      .lean();

    return res.json({ success: true, transcripts: docs });
  } catch (err) {
    console.error("listTranscripts error:", err);
    return res.status(500).json({ success: false, message: "Server error", error: err.message });
  }
}
