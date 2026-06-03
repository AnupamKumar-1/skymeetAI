import mongoose from "mongoose";
import crypto from "crypto";

const { Schema, Types, model } = mongoose;

const ParticipantSchema = new Schema(
  {
    socketId: { type: String, index: true, sparse: true },
    userId: { type: Schema.Types.Mixed, ref: "UserDb", default: null },
    name: { type: String, trim: true, default: "Guest" },
    meta: { type: Schema.Types.Mixed, default: {} },
    joinedAt: { type: Date, default: Date.now },
    leftAt: { type: Date, default: null },
  },
  { _id: false }
);

const ChatSchema = new Schema(
  {
    id: { type: String, required: true },
    userId: { type: Schema.Types.Mixed, required: true },
    fromSocketId: { type: String, required: true },
    name: { type: String, trim: true, default: "Guest" },
    text: { type: String, required: true, maxlength: 2000 },
    meta: { type: Schema.Types.Mixed, default: {} },
    ts: { type: Date, default: Date.now },
  },
  { _id: false }
);

const AnalyticsSchema = new Schema(
  {
    transcription: { type: String, default: "" },
    emotionScores: { type: Schema.Types.Mixed, default: {} },
    keywords: { type: [String], default: [] },
  },
  { _id: false }
);

const meetingSchema = new Schema(
  {
    meetingCode: {
      type: String,
      required: true,
      unique: true,
      trim: true,
      uppercase: true,
      index: true,
    },

    ownerId: { type: Types.ObjectId, ref: "UserDb", default: null, index: true, sparse: true },

    host: { type: Types.ObjectId, ref: "UserDb", default: null },
    hostInfo: {
      userId: { type: Types.ObjectId, ref: "UserDb", default: null },
      name: { type: String, trim: true, default: null },
    },

    hostSecretHash: { type: String, default: null },
    hostSecretExpiresAt: { type: Date, default: null },

    participants: { type: [ParticipantSchema], default: [] },
    chat: { type: [ChatSchema], default: [] },
    analytics: { type: AnalyticsSchema, default: {} },
    active: { type: Boolean, default: true },
    lastActivityAt: { type: Date, default: Date.now },
  },
  { timestamps: true }
);

meetingSchema.methods.getHostName = function () {
  if (this.host && typeof this.host === "object") {
    if (this.host.name) return this.host.name;
    if (this.host.username) return this.host.username;
  }
  if (this.hostInfo?.name) return this.hostInfo.name;
  return null;
};

meetingSchema.methods.addParticipant = async function (participant) {
  if (participant.meta && participant.meta.userId) {
    participant.meta.userId = String(participant.meta.userId);
    if (!participant.userId) {
      try {
        participant.userId = Types.ObjectId(participant.meta.userId);
      } catch (e) {
        participant.userId = participant.meta.userId;
      }
    }
    participant.userId = String(participant.userId);
  }

  let idx = -1;
  // Always try userId first — catches the init-* placeholder entry created at room-creation
  // time, so we update it in-place rather than creating a duplicate leftAt:null entry.
  if (participant.userId) {
    idx = this.participants.findIndex((p) => {
      if (p.userId) return String(p.userId) === String(participant.userId);
      if (p.meta && p.meta.userId) return String(p.meta.userId) === String(participant.userId);
      return false;
    });
  }
  // Fall back to socketId match only when there is no userId or no userId match
  if (idx === -1 && participant.socketId) {
    idx = this.participants.findIndex((p) => p.socketId === participant.socketId);
  }

  if (idx !== -1) {
    const existing = this.participants[idx];
    existing.name = participant.name || existing.name || "Guest";
    existing.meta = { ...(existing.meta || {}), ...(participant.meta || {}) };
    if (participant.userId) existing.userId = participant.userId;
    existing.joinedAt = new Date();
    existing.leftAt = null;
    this.participants[idx] = existing;
  } else {
    const newP = {
      socketId: participant.socketId || null,
      userId: participant.userId || (participant.meta ? participant.meta.userId : null),
      name: participant.name || "Guest",
      meta: participant.meta || {},
      joinedAt: new Date(),
      leftAt: null,
    };
    this.participants.push(newP);
  }

  this.active = true;
  this.lastActivityAt = new Date();
  return this.save();
};

meetingSchema.methods.updateParticipantMeta = async function (socketId, metaUpdate) {
  if (metaUpdate && metaUpdate.userId) {
    metaUpdate.userId = String(metaUpdate.userId);
  }

  const participant = this.participants.find((p) => p.socketId === socketId);
  if (participant) {
    participant.meta = { ...(participant.meta || {}), ...metaUpdate };
    if (metaUpdate.userId && !participant.userId) {
      try {
        participant.userId = Types.ObjectId(metaUpdate.userId);
      } catch (e) {
        participant.userId = metaUpdate.userId;
      }
    }
    this.lastActivityAt = new Date();
    return this.save();
  }
  return this;
};

meetingSchema.methods.restoreParticipant = async function (socketId, identifier, meta) {
  let participant;
  if (identifier.userId) {
    participant = this.participants.find(
      (p) => (p.meta?.userId === identifier.userId || (p.userId && String(p.userId) === String(identifier.userId))) && p.leftAt
    );
  } else if (identifier.name) {
    const cutoff = Date.now() - 5 * 60 * 1000;
    participant = this.participants.find(
      (p) =>
        p.name === identifier.name &&
        p.leftAt &&
        new Date(p.leftAt).getTime() >= cutoff
    );
  }

  if (participant) {
    participant.socketId = socketId;
    participant.leftAt = null;
    participant.meta = { ...(participant.meta || {}), ...meta };
    this.lastActivityAt = new Date();
    await this.save();
    return true;
  }
  return false;
};

meetingSchema.methods.removeParticipant = async function (socketId) {
  await this.constructor.updateOne(
    { _id: this._id },
    {
      $pull: { participants: { socketId } },
      $set: { lastActivityAt: new Date() },
    }
  );
  const updated = await this.constructor.findById(this._id);
  if (updated && updated.participants.length === 0) {
    await this.constructor.updateOne(
      { _id: this._id },
      { $set: { active: false } }
    );
  }
  return updated;
};

meetingSchema.methods.markParticipantLeft = async function (socketId, userId) {
  const now = new Date();

  // Step 1: stamp leftAt by socketId (the live socket connection)
  const bySocket = await this.constructor.updateOne(
    { _id: this._id, "participants.socketId": socketId },
    { $set: { "participants.$.leftAt": now, lastActivityAt: now } }
  );

  // Step 2: if socketId matched nothing (participant was stored with an init-* or user-* 
  // placeholder socketId and the real socket ID was never persisted), fall back to userId.
  // This is the common case — stamp leftAt on ALL entries for this userId that have no leftAt.
  if (userId && (bySocket.modifiedCount === 0 || bySocket.matchedCount === 0)) {
    const uid = String(userId);
    await this.constructor.updateMany(
      { _id: this._id },
      { $set: { "participants.$[p].leftAt": now, lastActivityAt: now } },
      { arrayFilters: [{ "p.leftAt": null, $or: [{ "p.userId": uid }, { "p.meta.userId": uid }] }] }
    );
  } else if (userId) {
    // socketId matched, but also clean up any duplicate placeholder entries for the same
    // userId (init-* / user-* entries that were never merged) so they don't block active:false.
    const uid = String(userId);
    await this.constructor.updateMany(
      { _id: this._id },
      { $set: { "participants.$[stale].leftAt": now } },
      { arrayFilters: [{ "stale.leftAt": null, "stale.socketId": { $regex: /^(init|user)-/ }, $or: [{ "stale.userId": uid }, { "stale.meta.userId": uid }] }] }
    );
  }

  // Step 3: re-fetch and check in plain JS
  const updated = await this.constructor.findById(this._id);
  if (!updated) return null;

  const anyoneStillPresent = updated.participants.some((p) => !p.leftAt);
  if (!anyoneStillPresent) {
    await this.constructor.updateOne(
      { _id: this._id },
      { $set: { active: false, lastActivityAt: now } }
    );
    updated.active = false;
  }

  return updated;
};

meetingSchema.methods.addChatMessage = async function (msg) {
  this.chat.push({
    id: msg.id,
    userId: msg.userId,
    fromSocketId: msg.fromSocketId,
    name: msg.name || "Guest",
    text: msg.text,
    meta: msg.meta || {},
    ts: msg.ts ? new Date(msg.ts) : new Date(),
  });

  if (this.chat.length > 500) {
    this.chat = this.chat.slice(-500);
  }

  this.lastActivityAt = new Date();
  return this.save();
};

meetingSchema.methods.updateAnalytics = async function (data) {
  this.analytics = { ...(this.analytics || {}), ...data };
  this.lastActivityAt = new Date();
  return this.save();
};

meetingSchema.methods.setHostSecretHash = async function (rawSecret) {
  if (!rawSecret || typeof rawSecret !== "string") {
    throw new Error("rawSecret string required");
  }
  this.hostSecretHash = crypto.createHash("sha256").update(rawSecret).digest("hex");
  return this.save();
};

meetingSchema.statics.verifyHostSecret = async function (meetingCode, providedSecret) {
  if (!meetingCode || !providedSecret) return null;
  const code = String(meetingCode).toUpperCase().trim();
  const meeting = await this.findOne({ meetingCode: code }).lean();
  if (!meeting || !meeting.hostSecretHash) return null;

  if (meeting.hostSecretExpiresAt && new Date(meeting.hostSecretExpiresAt) < new Date()) {
    return null;
  }

  const providedHash = crypto.createHash("sha256").update(providedSecret).digest("hex");
  if (providedHash === meeting.hostSecretHash) {
    return meeting;
  }
  return null;
};

meetingSchema.statics.upsertByMeetingCode = async function (meetingCode, payload = {}) {
  if (!meetingCode) throw new Error("meetingCode is required for upsert");
  const code = String(meetingCode).toUpperCase().trim();
  const setFields = { ...payload, meetingCode: code, lastActivityAt: new Date() };
  const opts = { upsert: true, new: true, setDefaultsOnInsert: true };

  const doc = await this.findOneAndUpdate({ meetingCode: code }, { $set: setFields }, opts).exec();
  return doc;
};

meetingSchema.statics.cleanupOldMeetings = async function (maxAgeHours = 24) {
  const cutoff = new Date(Date.now() - maxAgeHours * 60 * 60 * 1000);
  const result = await this.deleteMany({
    active: false,
    updatedAt: { $lt: cutoff },
  });
  if (result.deletedCount > 0) {
    console.log(`Cleaned up ${result.deletedCount} inactive meeting(s)`);
  }
};

meetingSchema.index({ hostSecretHash: 1 });
meetingSchema.index({ ownerId: 1, createdAt: -1 });

const Meeting = model("Meeting", meetingSchema);

setInterval(() => {
  Meeting.cleanupOldMeetings().catch((err) =>
    console.error("Cleanup error:", err)
  );
}, 60 * 60 * 1000);

export { Meeting };