import Transcript from "../models/transcript.model.js";
import { Meeting } from "../models/meeting.model.js";

export async function createTranscriptDoc({
    meetingCode,
    ownerId,
    hostSecretHash,
    transcriptText,
    fileName,
    metadata
}) {
    const meeting = await Meeting.findOne({ meetingCode }).lean();
    const resolvedOwnerId = meeting?.ownerId?.toString() || ownerId || null;
    const resolvedSecretHash = meeting?.hostSecretHash || hostSecretHash || null;

    try {
        return await Transcript.findOneAndUpdate(
            { meetingCode },
            {
                $set: {
                    transcriptText,
                    fileName: fileName || null,
                    metadata,
                },
                $setOnInsert: {
                    meetingCode,
                    ownerId: resolvedOwnerId,
                    hostSecretHash: resolvedSecretHash,
                },
            },
            { upsert: true, new: true }
        );
    } catch (err) {
        if (err.code === 11000) {
            return Transcript.findOne({ meetingCode });
        }
        throw err;
    }
}

export async function findTranscriptById(id) {
    return Transcript.findById(id).lean();
}

export async function findTranscriptByCode(meetingCode) {
    return Transcript.findOne({ meetingCode }).lean();
}

export async function listTranscriptDocs({ query, meetingCode, limit }) {
    const baseQuery = { ...query };

    if (meetingCode) baseQuery.meetingCode = meetingCode;

    return Transcript.find(
        baseQuery,
        {
            transcriptText: 1,
            meetingCode: 1,
            fileName: 1,
            metadata: 1,
            createdAt: 1,
            ownerId: 1,
            aiSummary: 1,
        }
    )
        .sort({ createdAt: -1 })
        .limit(limit)
        .lean();
}