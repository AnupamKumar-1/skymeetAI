import mongoose from "mongoose";

const MessageSchema = new mongoose.Schema(
    {
        role: { type: String, enum: ["user", "assistant", "system"], required: true },
        content: { type: String, required: true },
        sourceChunkIds: { type: [mongoose.Schema.Types.ObjectId], default: [] },
        latencyMs: { type: Number, default: null },
    },
    { _id: false, timestamps: false }
);

const RagSessionSchema = new mongoose.Schema(
    {
        transcriptId: {
            type: mongoose.Schema.Types.ObjectId,
            ref: "Transcript",
            required: true,
            index: true,
        },
        meetingCode: { type: String, required: true, uppercase: true, index: true },
        userId: { type: String, required: true, index: true },
        messages: { type: [MessageSchema], default: [] },
        contextWindowSize: { type: Number, default: 10 },
        totalTokensUsed: { type: Number, default: 0 },
        lastActivityAt: { type: Date, default: Date.now },
    },
    { timestamps: true }
);

RagSessionSchema.index({ transcriptId: 1, userId: 1 }, { unique: true });

RagSessionSchema.index(
    { lastActivityAt: 1 },
    { expireAfterSeconds: 24 * 60 * 60 }
);

export default mongoose.models.RagSession || mongoose.model("RagSession", RagSessionSchema);