import mongoose from "mongoose";

const transcriptRequestSchema = new mongoose.Schema(
    {
        meetingCode: {
            type: String,
            required: true,
            uppercase: true,
            trim: true,
            match: /^[A-Z0-9\-]{3,32}$/,
            index: true,
        },
        requesterId: {
            type: mongoose.Schema.Types.ObjectId,
            ref: "User",
            required: true,
            index: true,
        },
        requesterName: {
            type: String,
            required: true,
            trim: true,
            maxlength: 128,
        },
        hostId: {
            type: mongoose.Schema.Types.ObjectId,
            ref: "User",
            index: true,
        },
        status: {
            type: String,
            enum: ["pending", "approved", "denied"],
            default: "pending",
            index: true,
        },
        resolvedAt: {
            type: Date,
            default: null,
        },
    },
    {
        timestamps: true,
    }
);

transcriptRequestSchema.index({ meetingCode: 1, requesterId: 1 }, { unique: true });
transcriptRequestSchema.index({ hostId: 1, status: 1 });

export const TranscriptRequest = mongoose.model("TranscriptRequest", transcriptRequestSchema);