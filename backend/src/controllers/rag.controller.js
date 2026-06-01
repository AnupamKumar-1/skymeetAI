import {
    indexTranscriptService,
    getIndexStatusService,
    ragQueryService,
    getRagSessionService,
    clearRagSessionService,
} from "../services/rag.service.js";

export async function indexTranscript(req, res) {
    try {
        const { status, body } = await indexTranscriptService(req);
        return res.status(status).json(body);
    } catch (err) {
        return res.status(500).json({ success: false, message: "Server error" });
    }
}

export async function getIndexStatus(req, res) {
    try {
        const { status, body } = await getIndexStatusService(req);
        return res.status(status).json(body);
    } catch (err) {
        return res.status(500).json({ success: false, message: "Server error" });
    }
}

export async function ragQuery(req, res) {
    try {
        await ragQueryService(req, res);
    } catch (err) {
        if (!res.headersSent) {
            return res.status(500).json({ success: false, message: "Server error" });
        }
        res.end();
    }
}

export async function getRagSession(req, res) {
    try {
        const { status, body } = await getRagSessionService(req);
        return res.status(status).json(body);
    } catch (err) {
        return res.status(500).json({ success: false, message: "Server error" });
    }
}

export async function clearRagSession(req, res) {
    try {
        const { status, body } = await clearRagSessionService(req);
        return res.status(status).json(body);
    } catch (err) {
        return res.status(500).json({ success: false, message: "Server error" });
    }
}