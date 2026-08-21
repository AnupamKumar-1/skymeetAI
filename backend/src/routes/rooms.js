import express from "express";
import crypto from "crypto";
import passport from "passport";
import rateLimit from "express-rate-limit";
import "../../config/passport.js";
import { startTimer, endTimer } from "../observability/latency/latency.service.js";
import { LATENCY_LABELS } from "../observability/latency/latency.constants.js";
import {
  findMeetingByCode,
  findActiveMeetingByCode,
  createMeetingRoom,
  findRoomsByOwner,
} from "../data-access/rooms.repository.js";

const router = express.Router();

const roomsLimiter = rateLimit({
  windowMs: 60 * 1000,
  max: 30,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: "Too many requests" },
});

router.use(roomsLimiter);

const requireAuth = passport.authenticate("jwt", { session: false });

const softAuth = (req, _res, next) => {
  passport.authenticate("jwt", { session: false }, (_err, user) => {
    if (user) req.user = user;
    next();
  })(req, _res, next);
};

function generateHostSecretPair() {
  const hostSecret = crypto.randomBytes(32).toString("hex");
  const hostSecretHash = crypto.createHash("sha256").update(hostSecret).digest("hex");
  return { hostSecret, hostSecretHash };
}

function generateRoomCode() {
  return crypto.randomBytes(4).toString("hex").toUpperCase();
}

router.post("/", softAuth, async (req, res) => {
  const start = startTimer();
  try {
    const { hostName } = req.body;

    if (!hostName || typeof hostName !== "string" || !hostName.trim()) {
      endTimer(LATENCY_LABELS.SOCKET_JOIN, start, { route: "POST /rooms", error: true });
      return res.status(400).json({ error: "Host name is required" });
    }

    const maxTries = 5;
    let roomCode = null;

    for (let i = 0; i < maxTries; i++) {
      const code = generateRoomCode();
      const existing = await findMeetingByCode(code);
      if (!existing) {
        roomCode = code;
        break;
      }
    }

    if (!roomCode) {
      endTimer(LATENCY_LABELS.SOCKET_JOIN, start, { route: "POST /rooms", error: true });
      return res.status(500).json({ error: "Failed to generate unique room code, try again" });
    }

    const { hostSecret, hostSecretHash } = generateHostSecretPair();

    const ownerId = req.user && (req.user._id || req.user.id)
      ? String(req.user._id || req.user.id)
      : null;

    const meetingPayload = {
      meetingCode: roomCode,
      hostName: hostName.trim(),
      hostInfo: { name: hostName.trim(), userId: ownerId },
      participants: [{
        socketId: `init-${ownerId || "anon"}-${Date.now()}`,
        name: hostName.trim(),
        userId: ownerId,
        meta: { userId: ownerId },
        joinedAt: new Date(),
      }],
      chat: [],
      active: true,
      createdAt: new Date(),
      transcription: null,
      emotionAnalysis: null,
      hostSecretHash,
    };

    if (ownerId) {
      meetingPayload.ownerId = ownerId;
    }

    const meeting = await createMeetingRoom(meetingPayload);

    endTimer(LATENCY_LABELS.SOCKET_JOIN, start, {
      route: "POST /rooms",
      roomCode,
    });

    console.log(`[Room Created] ${roomCode} by ${hostName} ${meetingPayload.ownerId ? `(ownerId=${meetingPayload.ownerId})` : ""}`);
    res.status(201).json({
      message: "Room created successfully",
      roomCode: meeting.meetingCode,
      hostSecret,
      owner: !!meetingPayload.ownerId,
    });
  } catch (err) {
    endTimer(LATENCY_LABELS.SOCKET_JOIN, start, { route: "POST /rooms", error: true });
    console.error("Error creating room:", err);
    res.status(500).json({ error: "Internal server error" });
  }
});

router.get("/mine", requireAuth, async (req, res) => {
  try {
    if (!req.user || !(req.user._id || req.user.id)) {
      return res.status(401).json({ error: "Authentication required" });
    }

    const rooms = await findRoomsByOwner(req.user._id || req.user.id);
    res.json({ rooms });
  } catch (err) {
    console.error("Error fetching owner rooms:", err);
    res.status(500).json({ error: "Internal server error" });
  }
});

router.get("/:roomCode", async (req, res) => {
  try {
    const { roomCode } = req.params;

    if (!roomCode || typeof roomCode !== "string") {
      return res.status(400).json({ error: "Room code is required" });
    }

    const meeting = await findActiveMeetingByCode(roomCode);

    if (!meeting) {
      return res.status(404).json({ error: "Room not found" });
    }

    res.json({
      roomCode: meeting.meetingCode,
      hostName: meeting.hostName,
      createdAt: meeting.createdAt,
      participantsCount: Array.isArray(meeting.participants) ? meeting.participants.length : 0,
      hasOwner: !!meeting.ownerId,
    });
  } catch (err) {
    console.error("Error fetching room:", err);
    res.status(500).json({ error: "Internal server error" });
  }
});

export default router;