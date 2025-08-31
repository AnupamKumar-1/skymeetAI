// backend/models/transcript.model.js
import mongoose from "mongoose";

const TranscriptSchema = new mongoose.Schema(
  {
    meetingCode: { type: String, required: true, index: true, unique: true },
    transcriptText: { type: String, default: "" },
    fileName: { type: String, default: null },
    metadata: { type: mongoose.Schema.Types.Mixed, default: {} },
    createdAt: { type: Date, default: Date.now },
    updatedAt: { type: Date, default: Date.now },
  },
  {
    timestamps: true,
  }
);

TranscriptSchema.pre("save", function (next) {
  this.updatedAt = new Date();
  next();
});

export default mongoose.models.Transcript || mongoose.model("Transcript", TranscriptSchema);
