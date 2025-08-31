// // // backend/src/models/meeting.model.js
// // import mongoose from "mongoose";
// // const { Schema, model, Types } = mongoose;

// // // --- Participant Schema ---
// // const ParticipantSchema = new Schema(
// //   {
// //     socketId: { type: String, required: true, index: true },
// //     name: { type: String, trim: true, default: "Guest" },
// //     meta: { type: Object, default: {} },
// //     joinedAt: { type: Date, default: Date.now },
// //     leftAt: { type: Date, default: null },
// //   },
// //   { _id: false }
// // );

// // // --- Chat Schema ---
// // const ChatSchema = new Schema(
// //   {
// //     id: { type: String, required: true },             // stable message id
// //     userId: { type: String, required: true },         // stable user id
// //     fromSocketId: { type: String, required: true },   // current session socket
// //     name: { type: String, trim: true, default: "Guest" },
// //     text: { type: String, required: true, maxlength: 2000 },
// //     meta: { type: Object, default: {} },
// //     ts: { type: Date, default: Date.now },            // timestamp
// //   },
// //   { _id: false }
// // );

// // // --- Analytics Schema ---
// // const AnalyticsSchema = new Schema(
// //   {
// //     transcription: { type: String, default: "" },
// //     emotionScores: { type: Object, default: {} },
// //     keywords: { type: [String], default: [] },
// //   },
// //   { _id: false }
// // );

// // // --- Main Meeting Schema ---
// // const meetingSchema = new Schema(
// //   {
// //     meetingCode: {
// //       type: String,
// //       required: true,
// //       unique: true,
// //       trim: true,
// //       uppercase: true,
// //       index: true,
// //     },
// //     host: { type: Types.ObjectId, ref: "User" },
// //     participants: { type: [ParticipantSchema], default: [] },
// //     chat: { type: [ChatSchema], default: [] },
// //     analytics: { type: AnalyticsSchema, default: {} },
// //     active: { type: Boolean, default: true },
// //     lastActivityAt: { type: Date, default: Date.now },
// //   },
// //   { timestamps: true }
// // );

// // // --- Methods ---
// // meetingSchema.methods.addParticipant = async function (participant) {
// //   if (participant.meta && participant.meta.userId) {
// //     participant.meta.userId = String(participant.meta.userId);
// //   }

// //   const idx = this.participants.findIndex(
// //     (p) => p.socketId === participant.socketId
// //   );

// //   if (idx !== -1) {
// //     this.participants[idx] = {
// //       ...this.participants[idx].toObject(),
// //       ...participant,
// //       joinedAt: new Date(),
// //       leftAt: null,
// //     };
// //   } else {
// //     this.participants.push({
// //       socketId: participant.socketId,
// //       name: participant.name || "Guest",
// //       meta: participant.meta || {},
// //       joinedAt: new Date(),
// //     });
// //   }

// //   this.active = true;
// //   this.lastActivityAt = new Date();
// //   return this.save();
// // };

// // meetingSchema.methods.updateParticipantMeta = async function (
// //   socketId,
// //   metaUpdate
// // ) {
// //   if (metaUpdate && metaUpdate.userId) {
// //     metaUpdate.userId = String(metaUpdate.userId);
// //   }

// //   const participant = this.participants.find((p) => p.socketId === socketId);
// //   if (participant) {
// //     participant.meta = { ...(participant.meta || {}), ...metaUpdate };
// //     this.lastActivityAt = new Date();
// //     return this.save();
// //   }
// //   return this;
// // };

// // // Restore participant (safe rejoin without losing state)
// // meetingSchema.methods.restoreParticipant = async function (
// //   socketId,
// //   identifier,
// //   meta
// // ) {
// //   let participant;

// //   if (identifier.userId) {
// //     participant = this.participants.find(
// //       (p) => p.meta?.userId === identifier.userId && p.leftAt
// //     );
// //   } else if (identifier.name) {
// //     const cutoff = Date.now() - 5 * 60 * 1000;
// //     participant = this.participants.find(
// //       (p) =>
// //         p.name === identifier.name &&
// //         p.leftAt &&
// //         new Date(p.leftAt).getTime() >= cutoff
// //     );
// //   }

// //   if (participant) {
// //     participant.socketId = socketId;
// //     participant.leftAt = null;
// //     participant.meta = { ...(participant.meta || {}), ...meta };
// //     this.lastActivityAt = new Date();
// //     await this.save();
// //     return true;
// //   }

// //   return false;
// // };

// // // Remove participant
// // meetingSchema.methods.removeParticipant = async function (socketId) {
// //   await this.constructor.updateOne(
// //     { _id: this._id },
// //     {
// //       $pull: { participants: { socketId } },
// //       $set: { lastActivityAt: new Date() },
// //     }
// //   );
// //   const updated = await this.constructor.findById(this._id);
// //   if (updated.participants.length === 0) {
// //     await this.constructor.updateOne(
// //       { _id: this._id },
// //       { $set: { active: false } }
// //     );
// //   }
// //   return updated;
// // };

// // // Mark participant left
// // meetingSchema.methods.markParticipantLeft = async function (socketId) {
// //   await this.constructor.updateOne(
// //     { _id: this._id, "participants.socketId": socketId },
// //     {
// //       $set: {
// //         "participants.$.leftAt": new Date(),
// //         lastActivityAt: new Date(),
// //       },
// //     }
// //   );
// //   const updated = await this.constructor.findById(this._id);
// //   const stillActive = updated.participants.some((p) => !p.leftAt);
// //   if (!stillActive) {
// //     await this.constructor.updateOne(
// //       { _id: this._id },
// //       { $set: { active: false } }
// //     );
// //   }
// //   return updated;
// // };

// // // --- Chat ---
// // meetingSchema.methods.addChatMessage = async function (msg) {
// //   this.chat.push({
// //     id: msg.id,
// //     userId: msg.userId,
// //     fromSocketId: msg.fromSocketId,
// //     name: msg.name || "Guest",
// //     text: msg.text,
// //     meta: msg.meta || {},
// //     ts: msg.ts ? new Date(msg.ts) : new Date(),
// //   });

// //   if (this.chat.length > 500) {
// //     this.chat = this.chat.slice(-500);
// //   }

// //   this.lastActivityAt = new Date();
// //   return this.save();
// // };

// // // --- Analytics ---
// // meetingSchema.methods.updateAnalytics = async function (data) {
// //   this.analytics = { ...(this.analytics || {}), ...data };
// //   this.lastActivityAt = new Date();
// //   return this.save();
// // };

// // // --- Static Cleanup ---
// // meetingSchema.statics.cleanupOldMeetings = async function (maxAgeHours = 24) {
// //   const cutoff = new Date(Date.now() - maxAgeHours * 60 * 60 * 1000);
// //   const result = await this.deleteMany({
// //     active: false,
// //     updatedAt: { $lt: cutoff },
// //   });
// //   if (result.deletedCount > 0) {
// //     console.log(`🧹 Cleaned up ${result.deletedCount} inactive meeting(s)`);
// //   }
// // };

// // const Meeting = model("Meeting", meetingSchema);

// // setInterval(() => {
// //   Meeting.cleanupOldMeetings().catch((err) =>
// //     console.error("Cleanup error:", err)
// //   );
// // }, 60 * 60 * 1000);

// // export { Meeting };


// // backend/src/models/meeting.model.js
// import mongoose from "mongoose";
// const { Schema, model, Types } = mongoose;

// /**
//  * Meeting model (updated)
//  *
//  * - socketId on Participant is optional to allow HTTP-created participants
//  *   (we create synthetic socketIds in controllers when needed).
//  * - Added participant.userId (ObjectId) for easier querying by user.
//  * - Added hostInfo as an embedded fallback shape so older documents or
//  *   non-User-hosts remain supported.
//  * - Added helper getHostName() and a convenience static upsertByMeetingCode().
//  */

// // --- Participant Schema ---
// const ParticipantSchema = new Schema(
//   {
//     socketId: { type: String, index: true, sparse: true }, // optional now
//     userId: { type: Types.ObjectId, ref: "User", default: null }, // optional stable reference
//     name: { type: String, trim: true, default: "Guest" },
//     meta: { type: Schema.Types.Mixed, default: {} }, // keep flexible meta
//     joinedAt: { type: Date, default: Date.now },
//     leftAt: { type: Date, default: null },
//   },
//   { _id: false }
// );

// // --- Chat Schema ---
// const ChatSchema = new Schema(
//   {
//     id: { type: String, required: true }, // stable message id
//     userId: { type: Schema.Types.Mixed, required: true }, // allow either ObjectId or string
//     fromSocketId: { type: String, required: true }, // current session socket
//     name: { type: String, trim: true, default: "Guest" },
//     text: { type: String, required: true, maxlength: 2000 },
//     meta: { type: Schema.Types.Mixed, default: {} },
//     ts: { type: Date, default: Date.now }, // timestamp
//   },
//   { _id: false }
// );

// // --- Analytics Schema ---
// const AnalyticsSchema = new Schema(
//   {
//     transcription: { type: String, default: "" },
//     emotionScores: { type: Schema.Types.Mixed, default: {} },
//     keywords: { type: [String], default: [] },
//   },
//   { _id: false }
// );

// // --- Main Meeting Schema ---
// const meetingSchema = new Schema(
//   {
//     meetingCode: {
//       type: String,
//       required: true,
//       unique: true,
//       trim: true,
//       uppercase: true,
//       index: true,
//     },
//     // Preferred: store host as ObjectId ref to User
//     host: { type: Types.ObjectId, ref: "User", default: null },
//     // Fallback / legacy host shape (optional)
//     hostInfo: {
//       userId: { type: Types.ObjectId, ref: "User", default: null },
//       name: { type: String, trim: true, default: null },
//     },
//     participants: { type: [ParticipantSchema], default: [] },
//     chat: { type: [ChatSchema], default: [] },
//     analytics: { type: AnalyticsSchema, default: {} },
//     active: { type: Boolean, default: true },
//     lastActivityAt: { type: Date, default: Date.now },
//   },
//   { timestamps: true }
// );

// /**
//  * Instance helper: getHostName()
//  * Returns a best-effort host display name regardless of host shape.
//  */
// meetingSchema.methods.getHostName = function () {
//   // If host is populated (Object), prefer name/username
//   if (this.host && typeof this.host === "object") {
//     if (this.host.name) return this.host.name;
//     if (this.host.username) return this.host.username;
//     // if it's an ObjectId populated to an object without name, fall through
//   }

//   // If host is an ObjectId (not populated), we can't resolve name here
//   // Fall back to hostInfo if present
//   if (this.hostInfo && this.hostInfo.name) return this.hostInfo.name;

//   // Try older shapes where host was stored as an object in host (unpopulated)
//   if (this.host && this.host.name) return this.host.name;
//   if (this.host && this.host.userId && this.host.name) return this.host.name;

//   return null;
// };

// /**
//  * addParticipant(participant)
//  * - participant: { socketId?, userId?, name?, meta? }
//  * Works even when socketId is absent (HTTP-created participants).
//  */
// meetingSchema.methods.addParticipant = async function (participant) {
//   // Normalize userId inside meta and also set userId field if present
//   if (participant.meta && participant.meta.userId) {
//     participant.meta.userId = String(participant.meta.userId);
//     try {
//       participant.userId = Types.ObjectId(participant.meta.userId);
//     } catch (e) {
//       // not a valid ObjectId; keep as-is
//     }
//   }

//   let idx = -1;
//   if (participant.socketId) {
//     idx = this.participants.findIndex((p) => p.socketId === participant.socketId);
//   } else if (participant.userId) {
//     // try to find by userId for idempotency
//     idx = this.participants.findIndex((p) => {
//       if (p.userId) return String(p.userId) === String(participant.userId);
//       if (p.meta && p.meta.userId) return String(p.meta.userId) === String(participant.userId);
//       return false;
//     });
//   }

//   if (idx !== -1) {
//     // update existing participant entry
//     const existing = this.participants[idx];
//     existing.name = participant.name || existing.name || "Guest";
//     existing.meta = { ...(existing.meta || {}), ...(participant.meta || {}) };
//     if (participant.userId) existing.userId = participant.userId;
//     existing.joinedAt = new Date();
//     existing.leftAt = null;
//     this.participants[idx] = existing;
//   } else {
//     // create a new participant entry (socketId optional)
//     const newP = {
//       socketId: participant.socketId || null,
//       userId: participant.userId || (participant.meta ? participant.meta.userId : null),
//       name: participant.name || "Guest",
//       meta: participant.meta || {},
//       joinedAt: new Date(),
//       leftAt: null,
//     };
//     this.participants.push(newP);
//   }

//   this.active = true;
//   this.lastActivityAt = new Date();
//   return this.save();
// };

// meetingSchema.methods.updateParticipantMeta = async function (socketId, metaUpdate) {
//   if (metaUpdate && metaUpdate.userId) {
//     metaUpdate.userId = String(metaUpdate.userId);
//   }

//   const participant = this.participants.find((p) => p.socketId === socketId);
//   if (participant) {
//     participant.meta = { ...(participant.meta || {}), ...metaUpdate };
//     if (metaUpdate.userId && !participant.userId) {
//       try {
//         participant.userId = Types.ObjectId(metaUpdate.userId);
//       } catch (e) {
//         participant.userId = metaUpdate.userId;
//       }
//     }
//     this.lastActivityAt = new Date();
//     return this.save();
//   }
//   return this;
// };

// meetingSchema.methods.restoreParticipant = async function (socketId, identifier, meta) {
//   let participant;

//   if (identifier.userId) {
//     participant = this.participants.find(
//       (p) => (p.meta?.userId === identifier.userId || (p.userId && String(p.userId) === String(identifier.userId))) && p.leftAt
//     );
//   } else if (identifier.name) {
//     const cutoff = Date.now() - 5 * 60 * 1000;
//     participant = this.participants.find(
//       (p) =>
//         p.name === identifier.name &&
//         p.leftAt &&
//         new Date(p.leftAt).getTime() >= cutoff
//     );
//   }

//   if (participant) {
//     participant.socketId = socketId;
//     participant.leftAt = null;
//     participant.meta = { ...(participant.meta || {}), ...meta };
//     this.lastActivityAt = new Date();
//     await this.save();
//     return true;
//   }

//   return false;
// };

// meetingSchema.methods.removeParticipant = async function (socketId) {
//   await this.constructor.updateOne(
//     { _id: this._id },
//     {
//       $pull: { participants: { socketId } },
//       $set: { lastActivityAt: new Date() },
//     }
//   );
//   const updated = await this.constructor.findById(this._id);
//   if (updated && updated.participants.length === 0) {
//     await this.constructor.updateOne(
//       { _id: this._id },
//       { $set: { active: false } }
//     );
//   }
//   return updated;
// };

// meetingSchema.methods.markParticipantLeft = async function (socketId) {
//   await this.constructor.updateOne(
//     { _id: this._id, "participants.socketId": socketId },
//     {
//       $set: {
//         "participants.$.leftAt": new Date(),
//         lastActivityAt: new Date(),
//       },
//     }
//   );
//   const updated = await this.constructor.findById(this._id);
//   const stillActive = updated.participants.some((p) => !p.leftAt);
//   if (!stillActive) {
//     await this.constructor.updateOne(
//       { _id: this._id },
//       { $set: { active: false } }
//     );
//   }
//   return updated;
// };

// // --- Chat ---
// meetingSchema.methods.addChatMessage = async function (msg) {
//   this.chat.push({
//     id: msg.id,
//     userId: msg.userId,
//     fromSocketId: msg.fromSocketId,
//     name: msg.name || "Guest",
//     text: msg.text,
//     meta: msg.meta || {},
//     ts: msg.ts ? new Date(msg.ts) : new Date(),
//   });

//   if (this.chat.length > 500) {
//     this.chat = this.chat.slice(-500);
//   }

//   this.lastActivityAt = new Date();
//   return this.save();
// };

// // --- Analytics ---
// meetingSchema.methods.updateAnalytics = async function (data) {
//   this.analytics = { ...(this.analytics || {}), ...data };
//   this.lastActivityAt = new Date();
//   return this.save();
// };

// /**
//  * Static: upsertByMeetingCode(meetingCode, payload)
//  * Creates or updates a meeting document with the provided payload.
//  * Useful for the frontend upsert route.
//  */
// meetingSchema.statics.upsertByMeetingCode = async function (meetingCode, payload = {}) {
//   if (!meetingCode) throw new Error("meetingCode is required for upsert");
//   const code = String(meetingCode).toUpperCase().trim();
//   const update = { ...payload, meetingCode: code, lastActivityAt: new Date() };
//   const opts = { upsert: true, new: true, setDefaultsOnInsert: true };
//   const doc = await this.findOneAndUpdate({ meetingCode: code }, update, opts).exec();
//   return doc;
// };

// // --- Static Cleanup ---
// meetingSchema.statics.cleanupOldMeetings = async function (maxAgeHours = 24) {
//   const cutoff = new Date(Date.now() - maxAgeHours * 60 * 60 * 1000);
//   const result = await this.deleteMany({
//     active: false,
//     updatedAt: { $lt: cutoff },
//   });
//   if (result.deletedCount > 0) {
//     console.log(`🧹 Cleaned up ${result.deletedCount} inactive meeting(s)`);
//   }
// };

// const Meeting = model("Meeting", meetingSchema);

// // Run periodic cleanup (keeps original behavior)
// setInterval(() => {
//   Meeting.cleanupOldMeetings().catch((err) =>
//     console.error("Cleanup error:", err)
//   );
// }, 60 * 60 * 1000);

// export { Meeting };

// backend/src/models/meeting.model.js
import mongoose from "mongoose";
const { Schema, model, Types } = mongoose;

/**
 * Meeting model (updated)
 *
 * - Accept participant.userId as Mixed so UUIDs or ObjectIds are accepted without casting errors.
 * - Use the actual model name "UserDb" (matches your user.model.js).
 */

// --- Participant Schema ---
const ParticipantSchema = new Schema(
  {
    socketId: { type: String, index: true, sparse: true }, // optional now
    // Accept either ObjectId or a string (UUID). Keep ref to UserDb for populated ObjectIds.
    userId: { type: Schema.Types.Mixed, ref: "UserDb", default: null },
    name: { type: String, trim: true, default: "Guest" },
    meta: { type: Schema.Types.Mixed, default: {} }, // keep flexible meta
    joinedAt: { type: Date, default: Date.now },
    leftAt: { type: Date, default: null },
  },
  { _id: false }
);

// --- Chat Schema ---
const ChatSchema = new Schema(
  {
    id: { type: String, required: true }, // stable message id
    userId: { type: Schema.Types.Mixed, required: true }, // allow either ObjectId or string
    fromSocketId: { type: String, required: true }, // current session socket
    name: { type: String, trim: true, default: "Guest" },
    text: { type: String, required: true, maxlength: 2000 },
    meta: { type: Schema.Types.Mixed, default: {} },
    ts: { type: Date, default: Date.now }, // timestamp
  },
  { _id: false }
);

// --- Analytics Schema ---
const AnalyticsSchema = new Schema(
  {
    transcription: { type: String, default: "" },
    emotionScores: { type: Schema.Types.Mixed, default: {} },
    keywords: { type: [String], default: [] },
  },
  { _id: false }
);

// --- Main Meeting Schema ---
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
    // Preferred: store host as ObjectId ref to UserDb (matches user.model.js)
    host: { type: Types.ObjectId, ref: "UserDb", default: null },
    // Fallback / legacy host shape (optional)
    hostInfo: {
      userId: { type: Types.ObjectId, ref: "UserDb", default: null },
      name: { type: String, trim: true, default: null },
    },
    participants: { type: [ParticipantSchema], default: [] },
    chat: { type: [ChatSchema], default: [] },
    analytics: { type: AnalyticsSchema, default: {} },
    active: { type: Boolean, default: true },
    lastActivityAt: { type: Date, default: Date.now },
  },
  { timestamps: true }
);

/* --- helper methods (unchanged in behaviour, minor safety for mixed userId) --- */

meetingSchema.methods.getHostName = function () {
  if (this.host && typeof this.host === "object") {
    if (this.host.name) return this.host.name;
    if (this.host.username) return this.host.username;
  }
  if (this.hostInfo && this.hostInfo.name) return this.hostInfo.name;
  if (this.host && this.host.name) return this.host.name;
  if (this.host && this.host.userId && this.host.name) return this.host.name;
  return null;
};

meetingSchema.methods.addParticipant = async function (participant) {
  if (participant.meta && participant.meta.userId) {
    participant.meta.userId = String(participant.meta.userId);
    // try to normalize to ObjectId if it's a valid hex 24 char string
    if (!participant.userId) {
      try {
        participant.userId = Types.ObjectId(participant.meta.userId);
      } catch (e) {
        // leave as string (UUID) — schema allows Mixed
        participant.userId = participant.meta.userId;
      }
    }
  }

  let idx = -1;
  if (participant.socketId) {
    idx = this.participants.findIndex((p) => p.socketId === participant.socketId);
  } else if (participant.userId) {
    idx = this.participants.findIndex((p) => {
      if (p.userId) return String(p.userId) === String(participant.userId);
      if (p.meta && p.meta.userId) return String(p.meta.userId) === String(participant.userId);
      return false;
    });
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

meetingSchema.methods.markParticipantLeft = async function (socketId) {
  await this.constructor.updateOne(
    { _id: this._id, "participants.socketId": socketId },
    {
      $set: {
        "participants.$.leftAt": new Date(),
        lastActivityAt: new Date(),
      },
    }
  );
  const updated = await this.constructor.findById(this._id);
  const stillActive = updated.participants.some((p) => !p.leftAt);
  if (!stillActive) {
    await this.constructor.updateOne(
      { _id: this._id },
      { $set: { active: false } }
    );
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

meetingSchema.statics.upsertByMeetingCode = async function (meetingCode, payload = {}) {
  if (!meetingCode) throw new Error("meetingCode is required for upsert");
  const code = String(meetingCode).toUpperCase().trim();
  const update = { ...payload, meetingCode: code, lastActivityAt: new Date() };
  const opts = { upsert: true, new: true, setDefaultsOnInsert: true };
  const doc = await this.findOneAndUpdate({ meetingCode: code }, update, opts).exec();
  return doc;
};

meetingSchema.statics.cleanupOldMeetings = async function (maxAgeHours = 24) {
  const cutoff = new Date(Date.now() - maxAgeHours * 60 * 60 * 1000);
  const result = await this.deleteMany({
    active: false,
    updatedAt: { $lt: cutoff },
  });
  if (result.deletedCount > 0) {
    console.log(`🧹 Cleaned up ${result.deletedCount} inactive meeting(s)`);
  }
};

const Meeting = model("Meeting", meetingSchema);

setInterval(() => {
  Meeting.cleanupOldMeetings().catch((err) =>
    console.error("Cleanup error:", err)
  );
}, 60 * 60 * 1000);

export { Meeting };
