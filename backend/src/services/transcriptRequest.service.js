import { Meeting } from "../models/meeting.model.js";
import Transcript from "../models/transcript.model.js";
import {
    createRequest,
    findRequestById,
    findRequestsByHost,
    findRequestsByRequester,
    findRequestByMeetingAndRequester,
    updateRequestStatus,
    countPendingForHost,
} from "../data-access/transcriptRequest.repository.js";
import { makeLogger } from "../utils/redis.utils.js";
import { notifyHostOfTranscriptRequest, notifyUserOfResolution } from "../controllers/socket.controller.js";

const log = makeLogger("transcript-request");

export async function requestTranscriptService(req) {
    const userId = req.user?._id;
    const requesterName = req.user?.name || req.user?.username || "Participant";

    if (!userId) {
        return { status: 401, body: { success: false, message: "Authentication required." } };
    }

    const meetingCode = String(req.body?.meetingCode || "").trim().toUpperCase();
    if (!meetingCode || !/^[A-Z0-9\-]{3,32}$/.test(meetingCode)) {
        return { status: 400, body: { success: false, message: "Invalid meeting code." } };
    }

    const meeting = await Meeting.findOne({ meetingCode }).lean();
    if (!meeting) {
        return { status: 404, body: { success: false, message: "Meeting not found." } };
    }

    const hostId = meeting.ownerId || meeting.host || null;

    if (hostId && userId.toString() === hostId.toString()) {
        return { status: 400, body: { success: false, message: "Hosts can view their own transcripts directly." } };
    }

    const wasParticipant = Array.isArray(meeting.participants)
        ? meeting.participants.some((p) => {
            const pid = p?.userId || p?.user || p;
            return pid && pid.toString() === userId.toString();
        })
        : false;

    if (!wasParticipant) {
        return { status: 403, body: { success: false, message: "You can only request transcripts for meetings you participated in." } };
    }

    const transcript = await Transcript.findOne({ meetingCode }).lean();
    if (!transcript) {
        return { status: 404, body: { success: false, message: "No transcript exists for this meeting yet." } };
    }

    const existing = await findRequestByMeetingAndRequester(meetingCode, userId);
    if (existing) {
        if (existing.status === "approved") {
            return { status: 409, body: { success: false, message: "Your request has already been approved.", status: existing.status } };
        }
        if (existing.status === "pending") {
            return { status: 409, body: { success: false, message: "You already have a pending request for this meeting.", status: existing.status } };
        }
        if (existing.status === "denied") {
            const updated = await updateRequestStatus(existing._id, "pending", hostId || undefined);
            log.info("transcript request re-submitted after denial", { requestId: updated._id, meetingCode, userId });

            notifyHostOfTranscriptRequest(meetingCode, {
                requestId: updated._id.toString(),
                meetingCode,
                requesterId: userId.toString(),
                requesterName,
            }).catch(() => { });

            return { status: 200, body: { success: true, request: updated } };
        }
    }

    const created = await createRequest({ meetingCode, requesterId: userId, requesterName, hostId });
    log.info("transcript request created", { requestId: created._id, meetingCode, userId });

    notifyHostOfTranscriptRequest(meetingCode, {
        requestId: created._id.toString(),
        meetingCode,
        requesterId: userId.toString(),
        requesterName,
    }).catch(() => { });

    return { status: 201, body: { success: true, request: created } };
}

export async function listPendingRequestsService(req) {
    const userId = req.user?._id;
    if (!userId) {
        return { status: 401, body: { success: false, message: "Authentication required." } };
    }

    const status = req.query?.status || "pending";
    const requests = await findRequestsByHost(userId, status);
    const pendingCount = await countPendingForHost(userId);

    return { status: 200, body: { success: true, requests, pendingCount } };
}

export async function resolveRequestService(req) {
    const userId = req.user?._id;
    if (!userId) {
        return { status: 401, body: { success: false, message: "Authentication required." } };
    }

    const requestId = req.params?.id;
    const resolution = String(req.body?.status || "").trim();

    if (!resolution || !["approved", "denied"].includes(resolution)) {
        return { status: 400, body: { success: false, message: "Status must be 'approved' or 'denied'." } };
    }

    const request = await findRequestById(requestId);
    if (!request) {
        return { status: 404, body: { success: false, message: "Request not found." } };
    }

    if (!request.hostId || request.hostId?.toString() !== userId.toString()) {
        const meeting = await Meeting.findOne({ meetingCode: request.meetingCode }).lean();
        const meetingHostId = meeting?.ownerId || meeting?.host;
        if (!meeting || meetingHostId?.toString() !== userId.toString()) {
            return { status: 403, body: { success: false, message: "Only the host of this meeting can resolve requests." } };
        }
        if (!request.hostId) {
            await updateRequestStatus(request._id, "pending", meetingHostId);
        }
    }

    if (request.status !== "pending") {
        return { status: 409, body: { success: false, message: `Request is already ${request.status}.` } };
    }

    const updated = await updateRequestStatus(requestId, resolution);
    log.info("transcript request resolved", { requestId, resolution, hostId: userId });

    notifyUserOfResolution(request.requesterId?.toString(), {
        requestId,
        meetingCode: request.meetingCode,
        status: resolution,
    }).catch(() => { });

    return { status: 200, body: { success: true, request: updated } };
}

export async function myRequestsService(req) {
    const userId = req.user?._id;
    if (!userId) {
        return { status: 401, body: { success: false, message: "Authentication required." } };
    }

    const requests = await findRequestsByRequester(userId);
    return { status: 200, body: { success: true, requests } };
}