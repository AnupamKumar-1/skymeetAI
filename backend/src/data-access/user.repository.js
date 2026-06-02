import { User } from "../models/user.model.js";
import { Meeting } from "../models/meeting.model.js";
import fs from "fs";

const cfg = JSON.parse(
    fs.readFileSync(new URL("../config/config.json", import.meta.url))
);

const HOST_POPULATE_FIELDS = cfg.user?.hostPopulateFields ?? "name username";
const ME_POPULATE_FIELDS = cfg.user?.mePopulateFields ?? "_id username name";
const MEETINGS_QUERY_LIMIT = cfg.user?.meetingsQueryLimit ?? 200;

export async function findUserByUsername(username) {
    return User.findOne({ username: username.toLowerCase().trim() })
        .select("_id username name password")
        .lean();
}

export async function findUserById(userId) {
    return User.findById(userId)
        .select("_id username name bio timezone avatar email createdAt")
        .lean();
}

export async function findUserByUsernameLean(username) {
    return User.findOne({ username: username.toLowerCase().trim() })
        .select("_id")
        .lean();
}

export async function createUser({ name, username, hashedPassword, email }) {
    const newUser = new User({
        name: name.trim(),
        username: username.toLowerCase().trim(),
        password: hashedPassword,
        ...(email ? { email: email.toLowerCase().trim() } : {}),
    });
    return newUser.save();
}

export async function findUserProfileById(userId) {
    return User.findById(userId)
        .select("_id username name bio timezone avatar email createdAt")
        .lean();
}

export async function updateUserProfile(userId, { name, bio, timezone, email }) {
    const fields = {};
    if (name !== undefined) fields.name = name?.trim();
    if (bio !== undefined) fields.bio = bio?.trim();
    if (timezone !== undefined) fields.timezone = timezone?.trim();
    if (email !== undefined) fields.email = email ? email.toLowerCase().trim() : null;
    return User.findByIdAndUpdate(
        userId,
        { $set: fields },
        { new: true, runValidators: true }
    )
        .select("_id username name bio timezone avatar email createdAt")
        .lean();
}

export async function updateUserAvatar(userId, { url, publicId }) {
    return User.findByIdAndUpdate(
        userId,
        { $set: { "avatar.url": url, "avatar.publicId": publicId } },
        { new: true }
    )
        .select("_id username name bio timezone avatar email createdAt")
        .lean();
}

export async function removeUserAvatar(userId) {
    return User.findByIdAndUpdate(
        userId,
        { $set: { "avatar.url": null, "avatar.publicId": null } },
        { new: true }
    )
        .select("_id username name bio timezone avatar email createdAt")
        .lean();
}

export async function findUserWithPasswordById(userId) {
    return User.findById(userId)
        .select("_id password")
        .lean();
}

export async function updateUserPassword(userId, hashedPassword) {
    return User.findByIdAndUpdate(
        userId,
        { $set: { password: hashedPassword } },
        { new: true }
    )
        .select("_id")
        .lean();
}

export async function findMeetingsByUser(objectUserId, userId) {
    const query = {
        $or: [
            { host: objectUserId },
            { ownerId: objectUserId },
            {
                participants: {
                    $elemMatch: {
                        $or: [
                            { "meta.userId": userId },
                            { userId: objectUserId },
                            { userId },
                        ],
                    },
                },
            },
        ],
    };

    return Meeting.find(query)
        .sort({ createdAt: -1 })
        .populate("host", HOST_POPULATE_FIELDS)
        .lean()
        .exec();
}

export async function findMeetingByCode(meetingCode) {
    return Meeting.findOne({ meetingCode })
        .select("_id meetingCode")
        .lean()
        .exec();
}

export async function createMeeting({ meetingCode, link, objectUserId, userId, name }) {
    const newMeeting = new Meeting({
        meetingCode,
        link,
        host: objectUserId,
        ownerId: objectUserId,
        participants: [{
            socketId: `init-${userId}-${Date.now()}`,
            name,
            userId,
            meta: { userId },
            joinedAt: new Date(),
        }],
    });
    return newMeeting.save();
}

export async function findMeetingForParticipant(meetingCode) {
    return Meeting.findOne({ meetingCode });
}

export async function saveMeeting(meeting) {
    return meeting.save();
}

export async function findMeetingsForUser(objectUserId, userId, mineOnly, limit) {
    let filter = {};

    if (objectUserId && mineOnly) {
        filter = {
            $or: [
                { ownerId: objectUserId },
                { host: objectUserId },
                { "participants.meta.userId": userId },
            ],
        };
    } else if (objectUserId) {
        filter = {
            $or: [
                { host: objectUserId },
                { ownerId: objectUserId },
                { "participants.meta.userId": userId },
                { active: true },
            ],
        };
    } else {
        filter = { active: true };
    }

    const projection = {
        meetingCode: 1,
        link: 1,
        active: 1,
        hostInfo: 1,
        host: 1,
        ownerId: 1,
        createdAt: 1,
        lastActivityAt: 1,
        participants: 1,
    };

    return Meeting.find(filter, projection)
        .sort({ lastActivityAt: -1, createdAt: -1 })
        .limit(limit ?? MEETINGS_QUERY_LIMIT)
        .populate({ path: "host", model: "UserDb", select: HOST_POPULATE_FIELDS })
        .lean()
        .exec();
}

export async function upsertMeetingByCode(meetingCode, payload) {
    return Meeting.upsertByMeetingCode(meetingCode, payload);
}

export async function ensureMeetingIndexes() {
    const col = Meeting.collection;
    await Promise.all([
        col.createIndex({ meetingCode: 1 }, { unique: true, background: true }),
        col.createIndex({ ownerId: 1 }, { background: true }),
        col.createIndex({ host: 1 }, { background: true }),
        col.createIndex({ "participants.meta.userId": 1 }, { background: true }),
    ]);
}