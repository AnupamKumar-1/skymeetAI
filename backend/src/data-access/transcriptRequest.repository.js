import { TranscriptRequest } from "../models/transcriptRequest.model.js";

export async function createRequest({ meetingCode, requesterId, requesterName, hostId }) {
    return TranscriptRequest.findOneAndUpdate(
        { meetingCode, requesterId },
        { $setOnInsert: { meetingCode, requesterId, requesterName, hostId, status: "pending", resolvedAt: null } },
        { upsert: true, new: true, setDefaultsOnInsert: true }
    );
}

export async function findRequestById(id) {
    return TranscriptRequest.findById(id).lean();
}

export async function findRequestsByHost(hostId, status) {
    const query = { hostId };
    if (status) query.status = status;
    return TranscriptRequest.find(query).sort({ createdAt: -1 }).lean();
}

export async function findRequestsByRequester(requesterId) {
    return TranscriptRequest.find({ requesterId }).sort({ createdAt: -1 }).lean();
}

export async function findRequestByMeetingAndRequester(meetingCode, requesterId) {
    return TranscriptRequest.findOne({ meetingCode, requesterId }).lean();
}

export async function updateRequestStatus(id, status) {
    return TranscriptRequest.findByIdAndUpdate(
        id,
        { status, resolvedAt: status !== "pending" ? new Date() : null },
        { new: true }
    ).lean();
}

export async function countPendingForHost(hostId) {
    return TranscriptRequest.countDocuments({ hostId, status: "pending" });
}