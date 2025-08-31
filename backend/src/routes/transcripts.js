// backend/src/routes/transcripts.js
import express from "express";
import * as ctrl from "../controllers/transcript.controller.js";

const router = express.Router();

function pickHandler(...names) {
  for (const n of names) {
    if (typeof ctrl[n] === "function") return ctrl[n];
  }
  return null;
}

const available = Object.keys(ctrl || {});

// try several common names so route works even if controller uses a different name
const createHandler =
  pickHandler(
    "createTranscript",
    "saveTranscript",
    "upsertTranscript",
    "create",
    "save"
  ) ||
  ((req, res) =>
    res
      .status(501)
      .json({
        success: false,
        message:
          "Transcript create handler not implemented on server. Available exports: " +
          available.join(", "),
      }));

const listHandler =
  pickHandler("listTranscripts", "getTranscripts", "list", "listAll") ||
  ((req, res) =>
    res
      .status(501)
      .json({
        success: false,
        message:
          "Transcript list handler not implemented on server. Available exports: " +
          available.join(", "),
      }));

const getHandler =
  pickHandler(
    "getTranscript",
    "getTranscriptByCode",
    "getById",
    "fetchTranscript",
    "findTranscript"
  ) ||
  ((req, res) =>
    res
      .status(501)
      .json({
        success: false,
        message:
          "Transcript get handler not implemented on server. Available exports: " +
          available.join(", "),
      }));

// helpful console log for debugging import/export mismatches
if (available.length === 0) {
  console.warn(
    "[transcripts route] controller exported nothing (empty). Check ../controllers/transcript.controller.js"
  );
} else {
  console.log("[transcripts route] controller exports:", available);
}

router.post("/", createHandler);       // create & save transcript (.txt + DB)
router.get("/", listHandler);          // list transcripts (optionally filter by ?meeting_code=)
router.get("/:id", getHandler);        // download transcript by DB id (or fetch by code)

export default router;
