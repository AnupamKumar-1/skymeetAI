import mongoose from "mongoose";

const RagChunkSchema = new mongoose.Schema(
  {
    transcriptId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "Transcript",
      required: true,
      index: true,
    },
    meetingCode: {
      type: String,
      required: true,
      uppercase: true,
      trim: true,
      index: true,
    },
    ownerId: { type: String, default: null, index: true },
    chunkIndex: { type: Number, required: true },
    chunkText: { type: String, required: true },
    tokenCount: { type: Number, default: 0 },
    startSec: { type: Number, default: null },
    endSec: { type: Number, default: null },
    speakers: { type: [String], default: [] },
    embedding: { type: [Number], default: [] },
    embeddingModel: { type: String, default: "nomic-embed-text-v1.5" },
    chunkType: {
      type: String,
      enum: ["segment", "window", "summary"],
      default: "window",
    },
    contentHash: { type: String, index: true },
  },
  { timestamps: true }
);

RagChunkSchema.index({ transcriptId: 1, chunkIndex: 1 }, { unique: true });
RagChunkSchema.index({ meetingCode: 1, chunkIndex: 1 });

export default mongoose.models.RagChunk || mongoose.model("RagChunk", RagChunkSchema);