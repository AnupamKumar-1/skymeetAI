


// backend/src/routes/rooms.js
import express from "express";
import crypto from "crypto";
import { Meeting } from "../models/meeting.model.js";

const router = express.Router();

/**
 * Helper: generate a random host secret (raw) and its SHA256 hash.
 */
function generateHostSecretPair() {
  // raw secret (64 hex chars)
  const hostSecret = crypto.randomBytes(32).toString("hex");
  const hostSecretHash = crypto.createHash("sha256").update(hostSecret).digest("hex");
  return { hostSecret, hostSecretHash };
}

/**
 * Generate a candidate room code.
 * Existing code used crypto.randomBytes(4).toString("hex").toUpperCase()
 * — keep that but ensure uniqueness with a small retry loop.
 */
function generateRoomCode() {
  return crypto.randomBytes(4).toString("hex").toUpperCase();
}

/**
 * POST /api/rooms
 * Create a new meeting room and return the raw hostSecret (only here).
 *
 * If the request is authenticated (i.e. req.user is set by your auth middleware),
 * we record ownerId on the Meeting document so that this account can later
 * query rooms/transcripts across devices.
 */
router.post("/", async (req, res) => {
  try {
    const { hostName } = req.body;

    if (!hostName || typeof hostName !== "string" || !hostName.trim()) {
      return res.status(400).json({ error: "Host name is required" });
    }

    // generate a unique room code (small retry loop)
    let roomCode;
    let tries = 0;
    const maxTries = 5;
    do {
      roomCode = generateRoomCode();
      const existing = await Meeting.findOne({ meetingCode: roomCode }).lean();
      if (!existing) break;
      tries += 1;
    } while (tries < maxTries);

    if (tries >= maxTries) {
      // extremely unlikely, but fail gracefully
      return res.status(500).json({ error: "Failed to generate unique room code, try again" });
    }

    // generate host secret and store only the hash in DB
    const { hostSecret, hostSecretHash } = generateHostSecretPair();

    const meetingPayload = {
      meetingCode: roomCode,
      hostName: hostName.trim(),
      participants: [],
      chat: [],
      active: true,
      createdAt: new Date(),
      transcription: null,
      emotionAnalysis: null,
      // STORE ONLY HASH server-side
      hostSecretHash,
    };

    // If an authenticated user created the room, persist ownerId
    // Note: ensure your auth middleware sets req.user.id (or adjust accordingly)
    if (req.user && req.user.id) {
      meetingPayload.ownerId = req.user.id;
    }

    const meeting = await Meeting.create(meetingPayload);

    console.log(`[Room Created] ${roomCode} by ${hostName} ${meetingPayload.ownerId ? `(ownerId=${meetingPayload.ownerId})` : ""}`);
    // Return raw hostSecret *only here* — client should persist it (e.g. localStorage)
    res.status(201).json({
      message: "Room created successfully",
      roomCode: meeting.meetingCode,
      hostSecret, // IMPORTANT: return raw secret only ONCE
      owner: !!meetingPayload.ownerId,
    });
  } catch (err) {
    console.error("Error creating room:", err);
    res.status(500).json({ error: "Internal server error" });
  }
});

/**
 * GET /api/rooms/:roomCode
 * Check if a room exists and return non-sensitive details.
 * DO NOT return hostSecret or hostSecretHash here.
 */
router.get("/:roomCode", async (req, res) => {
  try {
    const { roomCode } = req.params;

    if (!roomCode || typeof roomCode !== "string") {
      return res.status(400).json({ error: "Room code is required" });
    }

    const meeting = await Meeting.findOne({
      meetingCode: roomCode.toUpperCase(),
      active: true,
    }).lean();

    if (!meeting) {
      return res.status(404).json({ error: "Room not found" });
    }

    res.json({
      roomCode: meeting.meetingCode,
      hostName: meeting.hostName,
      createdAt: meeting.createdAt,
      participantsCount: Array.isArray(meeting.participants) ? meeting.participants.length : 0,
      // indicate whether this room has an owner recorded (useful client-side)
      hasOwner: !!meeting.ownerId,
    });
  } catch (err) {
    console.error("Error fetching room:", err);
    res.status(500).json({ error: "Internal server error" });
  }
});

/**
 * GET /api/rooms/mine
 * Return all rooms created by the authenticated user.
 * This endpoint requires authentication: it checks for req.user.
 *
 * Use this on frontend login to list rooms (and then fetch transcripts for each room),
 * enabling cross-device visibility for hosts.
 */
router.get("/mine", async (req, res) => {
  try {
    if (!req.user || !req.user.id) {
      return res.status(401).json({ error: "Authentication required" });
    }

    const ownerId = req.user.id;
    // limit / sort to reasonable defaults
    const rooms = await Meeting.find({ ownerId })
      .sort({ createdAt: -1 })
      .select("meetingCode hostName createdAt active")
      .lean();

    res.json({ rooms });
  } catch (err) {
    console.error("Error fetching owner rooms:", err);
    res.status(500).json({ error: "Internal server error" });
  }
});

export default router;
