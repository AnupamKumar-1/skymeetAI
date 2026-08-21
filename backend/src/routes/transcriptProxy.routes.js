import express from "express";
import multer from "multer";
import { Blob } from "buffer";
import { Meeting } from "../models/meeting.model.js";

const router = express.Router();

const PROXY_MAX_FILE_BYTES = parseInt(process.env.PROXY_MAX_FILE_BYTES || `${50 * 1024 * 1024}`, 10);
const PROXY_MAX_FILES = parseInt(process.env.PROXY_MAX_FILES || "10", 10);

const upload = multer({
    limits: {
        fileSize: PROXY_MAX_FILE_BYTES,
        files: PROXY_MAX_FILES,
    },
});

export async function verifyProxyAuth(meetingCode, hostSecret) {
    if (!meetingCode || !hostSecret) return false;
    const meeting = await Meeting.verifyHostSecret(meetingCode, hostSecret);
    return !!meeting;
}

function handleMulterError(err, _req, res, next) {
    if (err instanceof multer.MulterError || err?.message) {
        return res.status(422).json({ success: false, error: err.message });
    }
    next(err);
}

router.post(["/", "/process_meeting"], upload.any(), handleMulterError, async (req, res) => {
    const tsServiceUrl = process.env.Ts_SERVICE_URL;
    if (!tsServiceUrl) {
        console.error("Proxy error: Ts_SERVICE_URL environment variable is not set");
        return res.status(503).json({ success: false, error: "Transcription service not configured (Ts_SERVICE_URL missing)" });
    }

    const meetingCode = String(req.body?.meeting_code || req.body?.meetingCode || "").trim().toUpperCase();
    const hostSecret = req.headers["x-host-secret"] || "";

    const authorized = await verifyProxyAuth(meetingCode, hostSecret);
    if (!authorized) {
        return res.status(403).json({ success: false, error: "Not authorized for this meeting" });
    }

    try {
        const form = new FormData();

        Object.entries(req.body || {}).forEach(([key, value]) => {
            form.append(
                key,
                typeof value === "string" ? value : JSON.stringify(value)
            );
        });

        (req.files || []).forEach((file) => {
            const blob = new Blob([file.buffer], { type: file.mimetype });
            form.append("audio_files", blob, file.originalname);
        });

        const response = await fetch(tsServiceUrl, {
            method: "POST",
            headers: {
                "x-host-secret": hostSecret,
                "x-user-token": req.headers["x-user-token"] || "",
            },
            body: form,
        });

        const text = await response.text();

        try {
            res.status(response.status).json(JSON.parse(text));
        } catch {
            res.status(response.status).send(text);
        }
    } catch (err) {
        console.error("Proxy error:", err);
        res.status(500).json({ success: false, error: "Proxy failed" });
    }
});

export default router;
