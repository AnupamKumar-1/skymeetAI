import express from "express";
import multer from "multer";
import { Blob } from "buffer";

const router = express.Router();
const upload = multer();

router.post(["/", "/process_meeting"], upload.any(), async (req, res) => {
    const tsServiceUrl = process.env.Ts_SERVICE_URL;
    if (!tsServiceUrl) {
        console.error("Proxy error: Ts_SERVICE_URL environment variable is not set");
        return res.status(503).json({ success: false, error: "Transcription service not configured (Ts_SERVICE_URL missing)" });
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
                "x-host-secret": req.headers["x-host-secret"] || "",
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