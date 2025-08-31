// backend/src/routes/rooms.js
import express from "express";
import crypto from "crypto";
import { Meeting } from "../models/meeting.model.js";

const router = express.Router();

/**
 * POST /api/rooms
 * Create a new meeting room
 */
router.post("/", async (req, res) => {
  try {
    const { hostName } = req.body;

    if (!hostName || typeof hostName !== "string" || !hostName.trim()) {
      return res.status(400).json({ error: "Host name is required" });
    }

    const roomCode = crypto.randomBytes(4).toString("hex").toUpperCase();

    const meeting = await Meeting.create({
      meetingCode: roomCode,
      hostName: hostName.trim(),
      participants: [],
      chat: [],
      active: true,
      createdAt: new Date(),
      // reserved fields for future
      transcription: null,
      emotionAnalysis: null,
    });

    console.log(`[Room Created] ${roomCode} by ${hostName}`);
    res.status(201).json({
      message: "Room created successfully",
      roomCode: meeting.meetingCode,
    });
  } catch (err) {
    console.error("Error creating room:", err);
    res.status(500).json({ error: "Internal server error" });
  }
});

/**
 * GET /api/rooms/:roomCode
 * Check if a room exists and return details
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
    });

    if (!meeting) {
      return res.status(404).json({ error: "Room not found" });
    }

    res.json({
      roomCode: meeting.meetingCode,
      hostName: meeting.hostName,
      createdAt: meeting.createdAt,
      participantsCount: meeting.participants.length,
    });
  } catch (err) {
    console.error("Error fetching room:", err);
    res.status(500).json({ error: "Internal server error" });
  }
});

export default router;


