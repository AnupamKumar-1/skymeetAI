import { makeLogger } from "../utils/redis.utils.js";
import {
    requestTranscriptService,
    listPendingRequestsService,
    resolveRequestService,
    myRequestsService,
} from "../services/transcriptRequest.service.js";

const log = makeLogger("transcript-request");

export async function requestTranscript(req, res) {
    try {
        const { status, body } = await requestTranscriptService(req);
        return res.status(status).json(body);
    } catch (err) {
        log.error("requestTranscript error", { err: err.message });
        return res.status(500).json({ success: false, message: "Server error." });
    }
}

export async function listPendingRequests(req, res) {
    try {
        const { status, body } = await listPendingRequestsService(req);
        return res.status(status).json(body);
    } catch (err) {
        log.error("listPendingRequests error", { err: err.message });
        return res.status(500).json({ success: false, message: "Server error." });
    }
}

export async function resolveRequest(req, res) {
    try {
        const { status, body } = await resolveRequestService(req);
        return res.status(status).json(body);
    } catch (err) {
        log.error("resolveRequest error", { err: err.message });
        return res.status(500).json({ success: false, message: "Server error." });
    }
}

export async function myRequests(req, res) {
    try {
        const { status, body } = await myRequestsService(req);
        return res.status(status).json(body);
    } catch (err) {
        log.error("myRequests error", { err: err.message });
        return res.status(500).json({ success: false, message: "Server error." });
    }
}