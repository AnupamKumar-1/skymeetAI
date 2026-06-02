import httpStatus from "http-status";
import crypto from "crypto";
import mongoose from "mongoose";
import bcrypt from "bcrypt";
import jwt from "jsonwebtoken";
import fs from "fs";
import { makeLogger, safeRedisGet, safeRedisSet, safeRedisDel, safeRedisIncr, safeRedisExpire, batchDel, isRateLimited } from "../utils/redis.utils.js";
import {
    findUserByUsername,
    findUserById,
    findUserByUsernameLean,
    createUser,
    findMeetingsByUser,
    findMeetingByCode,
    createMeeting,
    findMeetingForParticipant,
    saveMeeting,
    findMeetingsForUser,
    upsertMeetingByCode,
    ensureMeetingIndexes as repoEnsureMeetingIndexes,
    findUserProfileById,
    updateUserProfile,
    updateUserAvatar,
    removeUserAvatar,
    findUserWithPasswordById,
    updateUserPassword,
} from "../data-access/user.repository.js";
import { uploadAvatar, destroyAvatar, AVATAR_MAX_BYTES, ALLOWED_FORMATS } from "../config/cloudinary.js";

const cfg = JSON.parse(
    fs.readFileSync(new URL("../config/config.json", import.meta.url))
);

const HISTORY_CACHE_TTL_SEC = parseInt(process.env.HISTORY_CACHE_TTL_SEC || "120", 10);
const MEETINGS_CACHE_TTL_SEC = parseInt(process.env.MEETINGS_CACHE_TTL_SEC || "60", 10);
const USER_CACHE_TTL_SEC = parseInt(process.env.USER_CACHE_TTL_SEC || "300", 10);
const LOGIN_RATE_MAX = parseInt(process.env.LOGIN_RATE_MAX || "10", 10);
const LOGIN_RATE_WIN_SEC = parseInt(process.env.LOGIN_RATE_WIN_SEC || "60", 10);
const REGISTER_RATE_MAX = parseInt(process.env.REGISTER_RATE_MAX || "5", 10);
const REGISTER_RATE_WIN_SEC = parseInt(process.env.REGISTER_RATE_WIN_SEC || "60", 10);
const MAX_NAME_LEN = parseInt(process.env.MAX_NAME_LEN || "100", 10);
const MAX_USERNAME_LEN = parseInt(process.env.MAX_USERNAME_LEN || "50", 10);
const MAX_MEETINGCODE_LEN = parseInt(process.env.MAX_MEETINGCODE_LEN || "32", 10);
const ACCOUNT_LOCK_THRESHOLD = parseInt(process.env.ACCOUNT_LOCK_THRESHOLD || "10", 10);
const ACCOUNT_LOCK_SEC = parseInt(process.env.ACCOUNT_LOCK_SEC || "900", 10);

const REFRESH_TOKEN_TTL_SEC = parseInt(process.env.REFRESH_TOKEN_TTL_SEC || `${7 * 24 * 3600}`, 10);

const BCRYPT_SALT_ROUNDS = cfg.user?.bcryptSaltRounds ?? 10;
const JWT_EXPIRES_IN = process.env.JWT_EXPIRES_IN ?? cfg.user?.jwtExpiresIn ?? "1h";
const MEETINGS_QUERY_LIMIT = cfg.user?.meetingsQueryLimit ?? 200;

const PASSWORD_MIN_LEN = 8;
const USERNAME_RE = /^[a-z0-9_.\-]+$/;

if (!process.env.JWT_SECRET) {
    console.error("[UserService] FATAL: JWT_SECRET is not set");
    process.exit(1);
}

if (process.env.JWT_SECRET && process.env.JWT_SECRET.length < 32) {
    console.warn("[UserService] WARNING: JWT_SECRET is shorter than 32 characters");
}

export const RKEYS = {
    history: (uid) => `meetings:history:{${uid}}`,
    meetingsList: (uid, mine) => `meetings:list:{${uid}}:${mine}`,
    user: (uid) => `user:{${uid}}`,
    loginByUser: (u) => `login:rate:{${u}}`,
    loginByIp: (ip) => `login:rate:{ip}:${ip}`,
    registerByIp: (ip) => `register:rate:{ip}:${ip}`,
    accountLock: (u) => `login:lock:{${u}}`,
    loginFails: (u) => `login:fails:{${u}}`,
    refreshToken: (token) => `refresh:${token}`,
};

const log = makeLogger("user");

export function getClientIp(req) {
    const forwarded = req.headers["x-forwarded-for"];
    if (forwarded) return String(forwarded).split(",")[0].trim();
    return req.socket?.remoteAddress || "unknown";
}

async function isAccountLocked(username) {
    const locked = await safeRedisGet(RKEYS.accountLock(username));
    return locked !== null;
}

async function recordLoginFailure(username) {
    const key = RKEYS.loginFails(username);
    const count = await safeRedisIncr(key);
    if (count === 1) await safeRedisExpire(key, ACCOUNT_LOCK_SEC);
    if (count >= ACCOUNT_LOCK_THRESHOLD) {
        await safeRedisSet(RKEYS.accountLock(username), "1", { EX: ACCOUNT_LOCK_SEC });
        log.warn("account locked after repeated failures", { username, count });
    }
}

async function clearLoginFailures(username) {
    await safeRedisDel(RKEYS.loginFails(username));
    await safeRedisDel(RKEYS.accountLock(username));
}

function validatePassword(password) {
    if (!password || password.length < PASSWORD_MIN_LEN) {
        return `Password must be at least ${PASSWORD_MIN_LEN} characters.`;
    }
    return null;
}

function validateUsername(username) {
    if (!username || username.length < 3) return "Username must be at least 3 characters.";
    if (!USERNAME_RE.test(username)) return "Username may only contain letters, numbers, underscores, dots, and hyphens.";
    return null;
}

export const getUserId = (user) => {
    if (!user) return null;
    return user._id || user.id || user.sub || (typeof user === "string" ? user : null);
};

function parseExpiresInToSeconds(expiresIn) {
    if (typeof expiresIn === "number") return expiresIn;
    if (typeof expiresIn !== "string") return 3600;
    const match = expiresIn.match(/^(\d+)(s|m|h|d)?$/);
    if (!match) return 3600;
    const value = parseInt(match[1], 10);
    const unit = match[2] || "s";
    const multipliers = { s: 1, m: 60, h: 3600, d: 86400 };
    return value * (multipliers[unit] ?? 1);
}

function issueTokens(user) {
    const payload = {
        _id: user._id.toString(),
        sub: user._id.toString(),
        username: user.username,
        name: user.name,
    };
    const accessToken = jwt.sign(payload, process.env.JWT_SECRET, { expiresIn: JWT_EXPIRES_IN });
    const refreshToken = crypto.randomBytes(40).toString("hex");
    return { accessToken, refreshToken, payload };
}

export async function loginService(req) {
    const rawUsername = req.body?.username;
    const password = req.body?.password;

    if (!rawUsername?.trim() || !password?.trim()) {
        return { status: httpStatus.BAD_REQUEST, body: { success: false, message: "Username and password are required." } };
    }

    const username = rawUsername.trim().toLowerCase();
    if (username.length > MAX_USERNAME_LEN) {
        return { status: httpStatus.BAD_REQUEST, body: { success: false, message: `Username must be ${MAX_USERNAME_LEN} characters or fewer.` } };
    }

    if (await isAccountLocked(username)) {
        return {
            status: httpStatus.TOO_MANY_REQUESTS,
            body: { success: false, message: "Account temporarily locked due to repeated failed login attempts." },
        };
    }

    const ip = getClientIp(req);
    const blockedByUser = await isRateLimited(RKEYS.loginByUser(username), LOGIN_RATE_MAX, LOGIN_RATE_WIN_SEC);
    const blockedByIp = await isRateLimited(RKEYS.loginByIp(ip), LOGIN_RATE_MAX, LOGIN_RATE_WIN_SEC);

    if (blockedByUser || blockedByIp) {
        return {
            status: httpStatus.TOO_MANY_REQUESTS,
            body: { success: false, message: "Too many login attempts, please wait before trying again." },
        };
    }

    try {
        const user = await findUserByUsername(username);
        if (!user) {
            await recordLoginFailure(username);
            return { status: httpStatus.UNAUTHORIZED, body: { success: false, message: "Invalid username or password." } };
        }

        const isPasswordCorrect = await bcrypt.compare(password, user.password);
        if (!isPasswordCorrect) {
            await recordLoginFailure(username);
            return { status: httpStatus.UNAUTHORIZED, body: { success: false, message: "Invalid username or password." } };
        }

        await clearLoginFailures(username);
        await safeRedisDel(RKEYS.loginByUser(username));

        const { accessToken, refreshToken, payload } = issueTokens(user);

        await safeRedisSet(
            RKEYS.refreshToken(refreshToken),
            JSON.stringify(payload),
            { EX: REFRESH_TOKEN_TTL_SEC }
        );

        log.info("login success", { username });

        return {
            status: httpStatus.OK,
            body: {
                success: true,
                accessToken,
                expiresIn: JWT_EXPIRES_IN,
                message: "Login successful.",
                user: { _id: user._id, username: user.username, name: user.name, avatar: user.avatar ?? null },
            },
            cookies: {
                refreshToken: { value: refreshToken, ttlSec: REFRESH_TOKEN_TTL_SEC },
            },
        };
    } catch (error) {
        log.error("login error", { err: error.message });
        return { status: httpStatus.INTERNAL_SERVER_ERROR, body: { success: false, message: "Something went wrong." } };
    }
}

export async function refreshTokenService(req) {
    const token = req.cookies?.refreshToken ?? null;

    if (!token || typeof token !== "string") {
        return { status: httpStatus.BAD_REQUEST, body: { success: false, message: "Refresh token required." } };
    }

    try {
        const raw = await safeRedisGet(RKEYS.refreshToken(token));
        if (!raw) {
            return { status: httpStatus.UNAUTHORIZED, body: { success: false, message: "Invalid or expired refresh token." } };
        }

        const stored = JSON.parse(raw);
        const userId = stored._id || stored.sub;
        if (!userId) {
            return { status: httpStatus.UNAUTHORIZED, body: { success: false, message: "Invalid refresh token payload." } };
        }

        const user = await findUserById(userId);
        if (!user) {
            await safeRedisDel(RKEYS.refreshToken(token));
            return { status: httpStatus.UNAUTHORIZED, body: { success: false, message: "User not found." } };
        }

        await safeRedisDel(RKEYS.refreshToken(token));

        const { accessToken, refreshToken: newRefreshToken, payload } = issueTokens(user);

        await safeRedisSet(
            RKEYS.refreshToken(newRefreshToken),
            JSON.stringify(payload),
            { EX: REFRESH_TOKEN_TTL_SEC }
        );

        log.info("token refreshed", { userId });

        return {
            status: httpStatus.OK,
            body: {
                success: true,
                accessToken,
                expiresIn: JWT_EXPIRES_IN,
            },
            cookies: {
                refreshToken: { value: newRefreshToken, ttlSec: REFRESH_TOKEN_TTL_SEC },
            },
        };
    } catch (err) {
        log.error("refreshToken error", { err: err.message });
        return { status: httpStatus.INTERNAL_SERVER_ERROR, body: { success: false, message: "Something went wrong." } };
    }
}

export async function registerService(req) {
    const rawName = req.body?.name;
    const rawUsername = req.body?.username;
    const password = req.body?.password;
    const rawEmail = req.body?.email;

    if (!rawName?.trim() || !rawUsername?.trim() || !password?.trim()) {
        return { status: httpStatus.BAD_REQUEST, body: { success: false, message: "Name, username, and password are required." } };
    }

    if (!rawEmail?.trim()) {
        return { status: httpStatus.BAD_REQUEST, body: { success: false, message: "Email is required." } };
    }

    const emailRe = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRe.test(rawEmail.trim())) {
        return { status: httpStatus.BAD_REQUEST, body: { success: false, message: "A valid email address is required." } };
    }

    const name = rawName.trim();
    const username = rawUsername.trim().toLowerCase();
    const email = rawEmail.trim().toLowerCase();

    if (name.length > MAX_NAME_LEN) {
        return { status: httpStatus.BAD_REQUEST, body: { success: false, message: `Name must be ${MAX_NAME_LEN} characters or fewer.` } };
    }

    const usernameErr = validateUsername(username);
    if (usernameErr) return { status: httpStatus.BAD_REQUEST, body: { success: false, message: usernameErr } };

    const passwordErr = validatePassword(password);
    if (passwordErr) return { status: httpStatus.BAD_REQUEST, body: { success: false, message: passwordErr } };

    const ip = getClientIp(req);
    const blocked = await isRateLimited(RKEYS.registerByIp(ip), REGISTER_RATE_MAX, REGISTER_RATE_WIN_SEC);

    if (blocked) {
        return { status: httpStatus.TOO_MANY_REQUESTS, body: { success: false, message: "Too many registration attempts, please wait before trying again." } };
    }

    try {
        const existingUser = await findUserByUsernameLean(username);
        if (existingUser) {
            return { status: httpStatus.CONFLICT, body: { success: false, message: "User already exists." } };
        }

        const hashedPassword = await bcrypt.hash(password, BCRYPT_SALT_ROUNDS);
        await createUser({ name, username, hashedPassword, email });
        log.info("registered", { username });
        return { status: httpStatus.CREATED, body: { success: true, message: "User registered successfully." } };
    } catch (error) {
        log.error("register error", { err: error.message });
        return { status: httpStatus.INTERNAL_SERVER_ERROR, body: { success: false, message: "Something went wrong." } };
    }
}

export async function getUserHistoryService(req) {
    try {
        const userId = getUserId(req.user);
        if (!userId) {
            return { status: httpStatus.UNAUTHORIZED, body: { success: false, message: "Unauthorized. Missing user id." } };
        }

        const objectUserId = new mongoose.Types.ObjectId(userId);
        const cacheKey = RKEYS.history(userId);

        const cached = await safeRedisGet(cacheKey);
        if (cached !== null) {
            log.info("getUserHistory cache hit", { userId });
            return { status: httpStatus.OK, body: { success: true, meetings: JSON.parse(cached) } };
        }

        const meetings = await findMeetingsByUser(objectUserId, userId);

        const clientOrigin =
            process.env.CLIENT_ORIGIN ||
            `http://localhost:${process.env.CLIENT_PORT || 3000}`;

        const withLinks = meetings.map((m) => {
            if (m.meetingCode) {
                m.meetingCode = String(m.meetingCode).toUpperCase();
            }

            if (!m.link && m.meetingCode) {
                m.link = `${clientOrigin}/room/${m.meetingCode}`;
            }

            let hostName = "Unknown";
            if (m.hostInfo?.name) {
                hostName = m.hostInfo.name;
            } else if (m.host && typeof m.host === "object") {
                hostName = m.host.name || m.host.username || "Unknown";
            }
            m.hostName = hostName;

            m.participants = (m.participants || []).map((p) => ({
                socketId: p?.socketId || null,
                userId: p?.meta?.userId || p?.userId || null,
                name: p?.name || p?.meta?.name || p?.meta?.display || "Guest",
                joinedAt: p?.joinedAt || p?.createdAt || null,
                leftAt: p?.leftAt || null,
            }));

            return m;
        });

        await safeRedisSet(cacheKey, JSON.stringify(withLinks), { EX: HISTORY_CACHE_TTL_SEC });

        return { status: httpStatus.OK, body: { success: true, meetings: withLinks } };
    } catch (error) {
        log.error("getUserHistory error", { err: error.message });
        return { status: httpStatus.INTERNAL_SERVER_ERROR, body: { success: false, message: "Something went wrong." } };
    }
}

export async function addToHistoryService(req) {
    const rawCode = (req.body.meeting_code || req.body.meetingCode || "").toString().trim();

    if (!rawCode) return { status: httpStatus.BAD_REQUEST, body: { success: false, message: "Meeting code is required." } };

    if (rawCode.length > MAX_MEETINGCODE_LEN) {
        return { status: httpStatus.BAD_REQUEST, body: { success: false, message: `Meeting code must be ${MAX_MEETINGCODE_LEN} characters or fewer.` } };
    }

    const meeting_code = rawCode.toUpperCase();

    try {
        const userId = getUserId(req.user);
        if (!userId) return { status: httpStatus.UNAUTHORIZED, body: { success: false, message: "Unauthorized. Missing user id." } };

        const objectUserId = new mongoose.Types.ObjectId(userId);

        const existing = await findMeetingByCode(meeting_code);
        if (existing) {
            return { status: httpStatus.OK, body: { success: true, message: "Meeting already exists.", meeting: existing } };
        }

        const link = req.body.link || req.body.url || null;
        const newMeeting = await createMeeting({
            meetingCode: meeting_code,
            link,
            objectUserId,
            userId,
            name: req.user?.name || req.user?.username || "Host",
        });

        await batchDel(RKEYS.history(userId), RKEYS.meetingsList(userId, "true"), RKEYS.meetingsList(userId, "false"));

        return { status: httpStatus.CREATED, body: { success: true, message: "Meeting created and saved to history.", meeting: newMeeting } };
    } catch (error) {
        log.error("addToHistory error", { err: error.message });
        return { status: httpStatus.INTERNAL_SERVER_ERROR, body: { success: false, message: "Something went wrong." } };
    }
}

export async function addParticipantService(req) {
    try {
        const userId = getUserId(req.user);
        if (!userId) return { status: httpStatus.UNAUTHORIZED, body: { success: false, message: "Unauthorized. Missing user id." } };
        const objectUserId = new mongoose.Types.ObjectId(userId);

        const codeParam = (req.params?.code || req.body?.meeting_code || req.body?.meetingCode || "").toString().trim();
        if (!codeParam) return { status: httpStatus.BAD_REQUEST, body: { success: false, message: "Meeting code is required (param or body)." } };

        if (codeParam.length > MAX_MEETINGCODE_LEN) {
            return { status: httpStatus.BAD_REQUEST, body: { success: false, message: `Meeting code must be ${MAX_MEETINGCODE_LEN} characters or fewer.` } };
        }

        const meetingCode = codeParam.toUpperCase();

        const meeting = await findMeetingForParticipant(meetingCode);
        if (!meeting) return { status: httpStatus.NOT_FOUND, body: { success: false, message: "Meeting not found." } };

        const participantName = (req.body.name || req.user?.name || req.user?.username || "Guest").toString();

        const existingParticipant = meeting.participants.find((p) => {
            if (!p) return false;
            return String(p.meta?.userId) === userId || String(p.userId) === userId;
        });

        if (existingParticipant) {
            existingParticipant.joinedAt = new Date();
            existingParticipant.leftAt = null;
            existingParticipant.name = participantName;
        } else {
            meeting.participants.push({
                socketId: `user-${userId}-${Date.now()}`,
                name: participantName,
                userId,
                meta: { userId },
                joinedAt: new Date(),
            });
        }

        if (!meeting.host) meeting.host = objectUserId;
        if (!meeting.ownerId) meeting.ownerId = objectUserId;
        await saveMeeting(meeting);

        await batchDel(RKEYS.history(userId), RKEYS.meetingsList(userId, "true"), RKEYS.meetingsList(userId, "false"));
        return { status: httpStatus.OK, body: { success: true, meeting } };
    } catch (error) {
        log.error("addParticipant error", { err: error.message });
        return { status: httpStatus.INTERNAL_SERVER_ERROR, body: { success: false, message: "Something went wrong." } };
    }
}

export async function getMeetingsService(req) {
    try {
        const userId = getUserId(req.user);
        const objectUserId = userId ? new mongoose.Types.ObjectId(userId) : null;
        const mineOnly = String(req.query?.mine || "false").toLowerCase() === "true";

        if (userId) {
            const cacheKey = RKEYS.meetingsList(userId, String(mineOnly));
            const cached = await safeRedisGet(cacheKey);
            if (cached !== null) {
                log.info("getMeetings cache hit", { userId, mineOnly });
                return { status: httpStatus.OK, body: { meetings: JSON.parse(cached) } };
            }
        }

        const meetings = await findMeetingsForUser(objectUserId, userId, mineOnly, MEETINGS_QUERY_LIMIT);

        if (userId) {
            const cacheKey = RKEYS.meetingsList(userId, String(mineOnly));
            await safeRedisSet(cacheKey, JSON.stringify(meetings), { EX: MEETINGS_CACHE_TTL_SEC });
        }

        return { status: httpStatus.OK, body: { meetings } };
    } catch (err) {
        log.error("getMeetings error", { err: err.message });
        return { status: httpStatus.INTERNAL_SERVER_ERROR, body: { success: false, message: "Failed to fetch meetings" } };
    }
}

export async function upsertMeetingService(req) {
    try {
        const body = req.body || {};
        const meetingCodeRaw = body.meetingCode || body.meeting_code || body.code || body.meeting;

        if (!meetingCodeRaw) {
            return { status: 400, body: { success: false, message: "meetingCode is required" } };
        }

        const meetingCodeStr = String(meetingCodeRaw).trim();
        if (meetingCodeStr.length > MAX_MEETINGCODE_LEN) {
            return { status: 400, body: { success: false, message: `Meeting code must be ${MAX_MEETINGCODE_LEN} characters or fewer.` } };
        }

        const meetingCode = meetingCodeStr.toUpperCase();
        const userId = getUserId(req.user);
        const objectUserId = userId ? new mongoose.Types.ObjectId(userId) : null;

        const payload = {};
        if (objectUserId) {
            payload.host = objectUserId;
            payload.ownerId = objectUserId;
        }

        payload.hostInfo = { name: body.hostName || body.host_name || null, userId: userId || null };

        const existing = await findMeetingByCode(meetingCode);
        let rawSecret = null;
        if (!existing) {
            rawSecret = crypto.randomBytes(32).toString("hex");
            payload.hostSecretHash = crypto.createHash("sha256").update(rawSecret).digest("hex");
        }

        const saved = await upsertMeetingByCode(meetingCode, payload);

        if (userId) {
            await batchDel(RKEYS.meetingsList(userId, "true"), RKEYS.meetingsList(userId, "false"));
        }

        return {
            status: 200,
            body: {
                success: true,
                meeting: saved,
                ...(rawSecret ? { hostSecret: rawSecret } : {}),
            },
        };
    } catch (err) {
        log.error("upsertMeeting error", { err: err.message });
        return { status: 500, body: { success: false, message: "Failed to upsert meeting" } };
    }
}

export async function getMeService(req) {
    try {
        const userId = getUserId(req.user);
        if (!userId) return { status: 401, body: { success: false, message: "Unauthorized" } };

        const cacheKey = RKEYS.user(userId);
        const cached = await safeRedisGet(cacheKey);
        if (cached !== null) {
            log.info("getMe cache hit", { userId });
            return { status: 200, body: { success: true, user: JSON.parse(cached) } };
        }

        const user = await findUserProfileById(userId);
        if (!user) return { status: 404, body: { success: false, message: "User not found" } };

        await safeRedisSet(cacheKey, JSON.stringify(user), { EX: USER_CACHE_TTL_SEC });
        return { status: 200, body: { success: true, user } };
    } catch (err) {
        log.error("getMe error", { err: err.message });
        return { status: 500, body: { success: false, message: "Server error" } };
    }
}

export async function ensureMeetingIndexes() {
    const log2 = makeLogger("user");
    try {
        await repoEnsureMeetingIndexes();
        log2.info("Meeting indexes verified");
    } catch (err) {
        log2.error("Failed to create Meeting indexes", { err: err.message });
    }
}

export async function logoutService(req) {
    try {
        const authHeader = req.headers?.authorization;
        const accessToken = authHeader?.startsWith("Bearer ") ? authHeader.slice(7) : null;
        if (accessToken) {
            const ttl = parseExpiresInToSeconds(JWT_EXPIRES_IN);
            await safeRedisSet(`blacklist:${accessToken}`, "1", { EX: ttl });
        }

        const refreshToken = req.cookies?.refreshToken ?? null;
        if (refreshToken) {
            await safeRedisDel(RKEYS.refreshToken(refreshToken));
        }

        return { status: httpStatus.OK, body: { success: true, message: "Logged out" } };
    } catch (err) {
        log.error("logout error", { err: err.message });
        return { status: httpStatus.INTERNAL_SERVER_ERROR, body: { success: false, message: "Failed to logout" } };
    }
}

export async function getProfileService(req) {
    try {
        const userId = req.user?.id || req.user?._id;
        if (!userId) return { status: httpStatus.UNAUTHORIZED, body: { success: false, message: "Unauthorized" } };

        const user = await findUserProfileById(userId);
        if (!user) return { status: httpStatus.NOT_FOUND, body: { success: false, message: "User not found" } };

        return { status: 200, body: { success: true, profile: user } };
    } catch (err) {
        log.error("getProfile error", { err: err.message });
        return { status: 500, body: { success: false, message: "Server error" } };
    }
}

export async function updateProfileService(req) {
    try {
        const userId = req.user?.id || req.user?._id;
        if (!userId) return { status: httpStatus.UNAUTHORIZED, body: { success: false, message: "Unauthorized" } };

        const { name, bio, timezone, email } = req.body || {};

        if (name !== undefined) {
            if (typeof name !== "string" || name.trim().length < 1 || name.trim().length > 64) {
                return { status: 422, body: { success: false, message: "Name must be 1–64 characters." } };
            }
        }
        if (bio !== undefined && typeof bio === "string" && bio.length > 280) {
            return { status: 422, body: { success: false, message: "Bio must be under 280 characters." } };
        }
        if (email !== undefined && email !== null && email !== "") {
            const emailRe = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
            if (typeof email !== "string" || !emailRe.test(email.trim())) {
                return { status: 422, body: { success: false, message: "Must be a valid email address." } };
            }
            if (email.trim().length > 254) {
                return { status: 422, body: { success: false, message: "Email must be under 254 characters." } };
            }
        }

        const updated = await updateUserProfile(userId, {
            name: name !== undefined ? name : undefined,
            bio: bio !== undefined ? bio : undefined,
            timezone: timezone !== undefined ? timezone : undefined,
            email: email !== undefined ? email : undefined,
        });

        await safeRedisDel(RKEYS.user(userId.toString()));

        return { status: 200, body: { success: true, profile: updated } };
    } catch (err) {
        log.error("updateProfile error", { err: err.message });
        return { status: 500, body: { success: false, message: "Server error" } };
    }
}

export async function updateAvatarService(req) {
    try {
        const userId = req.user?.id || req.user?._id;
        log.info("updateAvatar called", { userId, hasFile: !!req.file, fileSize: req.file?.size, mimetype: req.file?.mimetype });
        if (!userId) return { status: httpStatus.UNAUTHORIZED, body: { success: false, message: "Unauthorized" } };

        if (!req.file) {
            return { status: 422, body: { success: false, message: "An image file is required." } };
        }

        if (req.file.size > AVATAR_MAX_BYTES) {
            return { status: 413, body: { success: false, message: `Avatar must be under ${AVATAR_MAX_BYTES / (1024 * 1024)} MB.` } };
        }

        const currentProfile = await findUserProfileById(userId);
        const oldPublicId = currentProfile?.avatar?.publicId ?? null;

        log.info("uploading to Cloudinary", { userId, oldPublicId });
        const { url, publicId } = await uploadAvatar(req.file.buffer, userId);
        log.info("Cloudinary upload success", { url, publicId });

        if (oldPublicId && oldPublicId !== publicId) {
            await destroyAvatar(oldPublicId).catch((e) =>
                log.warn("Failed to delete old avatar from Cloudinary", { err: e.message, oldPublicId })
            );
        }

        const updated = await updateUserAvatar(userId, { url, publicId });

        await safeRedisDel(RKEYS.user(userId.toString()));

        return { status: 200, body: { success: true, profile: updated } };
    } catch (err) {
        log.error("updateAvatar error", { err: err.message, stack: err.stack });
        return { status: 500, body: { success: false, message: err.message || "Server error" } };
    }
}

export async function deleteAvatarService(req) {
    try {
        const userId = req.user?.id || req.user?._id;
        if (!userId) return { status: httpStatus.UNAUTHORIZED, body: { success: false, message: "Unauthorized" } };

        const currentProfile = await findUserProfileById(userId);
        const publicId = currentProfile?.avatar?.publicId ?? null;

        if (publicId) {
            await destroyAvatar(publicId).catch((e) =>
                log.warn("Failed to delete avatar from Cloudinary", { err: e.message, publicId })
            );
        }

        const updated = await removeUserAvatar(userId);

        await safeRedisDel(RKEYS.user(userId.toString()));

        return { status: 200, body: { success: true, profile: updated } };
    } catch (err) {
        log.error("deleteAvatar error", { err: err.message });
        return { status: 500, body: { success: false, message: "Server error" } };
    }
}

export async function changePasswordService(req) {
    try {
        const userId = req.user?.id || req.user?._id;
        if (!userId) return { status: httpStatus.UNAUTHORIZED, body: { success: false, message: "Unauthorized" } };

        const { currentPassword, newPassword } = req.body || {};

        if (!currentPassword) {
            return { status: 422, body: { success: false, message: "Current password is required." } };
        }
        if (!newPassword || newPassword.length < 8) {
            return { status: 422, body: { success: false, message: "New password must be at least 8 characters." } };
        }
        if (!/[A-Z]/.test(newPassword)) {
            return { status: 422, body: { success: false, message: "New password must contain at least one uppercase letter." } };
        }
        if (!/[0-9]/.test(newPassword)) {
            return { status: 422, body: { success: false, message: "New password must contain at least one number." } };
        }
        if (newPassword.length > 128) {
            return { status: 422, body: { success: false, message: "New password must be under 128 characters." } };
        }

        const user = await findUserWithPasswordById(userId);
        if (!user) return { status: httpStatus.NOT_FOUND, body: { success: false, message: "User not found." } };

        const isCorrect = await bcrypt.compare(currentPassword, user.password);
        if (!isCorrect) {
            return { status: httpStatus.UNAUTHORIZED, body: { success: false, message: "Current password is incorrect." } };
        }

        const hashed = await bcrypt.hash(newPassword, BCRYPT_SALT_ROUNDS);
        await updateUserPassword(userId, hashed);

        log.info("password changed", { userId });
        return { status: 200, body: { success: true, message: "Password changed successfully." } };
    } catch (err) {
        log.error("changePassword error", { err: err.message });
        return { status: 500, body: { success: false, message: "Server error" } };
    }
}