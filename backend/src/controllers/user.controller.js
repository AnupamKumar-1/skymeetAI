// backend/src/controllers/user.controller.js
import httpStatus from "http-status";
import { User } from "../models/user.model.js";
import bcrypt from "bcrypt";
import jwt from "jsonwebtoken";
import { Meeting } from "../models/meeting.model.js";

/**
 * Helper to send a JSON error response
 */
const sendError = (res, status, message) =>
  res.status(status).json({ success: false, message });

/**
 * Normalize req.user into a usable string id
 */
const getUserId = (user) => {
  if (!user) return null;
  return user._id || user.id || user.sub || (typeof user === "string" ? user : null);
};

/**
 * POST /auth/logout
 *
 * Clears the refresh cookie (httpOnly). If you store refresh tokens
 * server-side (in the DB), invalidate/remove that token here.
 *
 * Notes:
 * - Ensure you have cookie-parser enabled in your app (app.use(cookieParser()))
 * - Ensure CORS allows credentials and frontend calls fetch(..., { credentials: "include" })
 */
const logout = async (req, res) => {
  try {
    // Optional: read refresh token from cookies (if you set it that way)
    const refreshToken = req.cookies ? req.cookies.refreshToken : null;

    // If you manage refresh tokens server-side (DB) and want to invalidate them,
    // you can do it here. Example (pseudo):
    // if (refreshToken) { await RefreshTokenModel.invalidate(refreshToken); }
    //
    // Or, if you stored a refreshToken on the user record:
    // if (req.user && req.user._id) { await User.updateOne({ _id: req.user._id }, { $unset: { refreshToken: "" } }); }

    // Clear the cookie that holds the refresh token.
    res.clearCookie("refreshToken", {
      httpOnly: true,
      secure: process.env.NODE_ENV === "production",
      sameSite: "Strict",
      path: "/", // match the path that was used when setting the cookie
    });

    return res.status(httpStatus.OK).json({ success: true, message: "Logged out" });
  } catch (err) {
    console.error("logout error:", err.stack || err);

    // Try to clear cookie even on error
    try {
      res.clearCookie("refreshToken", {
        httpOnly: true,
        secure: process.env.NODE_ENV === "production",
        sameSite: "Strict",
        path: "/",
      });
    } catch (e) {
      // ignore
    }

    return sendError(res, httpStatus.INTERNAL_SERVER_ERROR, "Failed to logout");
  }
};

/**
 * POST /login
 */
const login = async (req, res) => {
  const { username, password } = req.body;

  if (!username?.trim() || !password?.trim()) {
    return sendError(res, httpStatus.BAD_REQUEST, "Username and password are required.");
  }

  try {
    const user = await User.findOne({ username });
    if (!user) {
      return sendError(res, httpStatus.NOT_FOUND, "User not found.");
    }

    const isPasswordCorrect = await bcrypt.compare(password, user.password);
    if (!isPasswordCorrect) {
      return sendError(res, httpStatus.UNAUTHORIZED, "Invalid username or password.");
    }

    const payload = { sub: user._id, username: user.username, name: user.name };
    const expiresIn = "1h";
    const token = jwt.sign(payload, process.env.JWT_SECRET, { expiresIn });

    res.status(httpStatus.OK).json({
      success: true,
      accessToken: token,
      expiresIn,
      message: "Login successful.",
      user: { _id: user._id, username: user.username, name: user.name },
    });
  } catch (error) {
    console.error("login error:", error.stack || error);
    sendError(res, httpStatus.INTERNAL_SERVER_ERROR, `Something went wrong: ${error.message}`);
  }
};

/**
 * POST /register
 */
const register = async (req, res) => {
  const { name, username, password } = req.body;
  if (!name?.trim() || !username?.trim() || !password?.trim()) {
    return sendError(res, httpStatus.BAD_REQUEST, "Name, username, and password are required.");
  }

  try {
    const existingUser = await User.findOne({ username });
    if (existingUser) {
      return sendError(res, httpStatus.CONFLICT, "User already exists.");
    }

    const hashedPassword = await bcrypt.hash(password, 10);
    const newUser = new User({ name, username, password: hashedPassword });

    await newUser.save();
    res.status(httpStatus.CREATED).json({ success: true, message: "User registered successfully." });
  } catch (error) {
    console.error("register error:", error.stack || error);
    sendError(res, httpStatus.INTERNAL_SERVER_ERROR, `Something went wrong: ${error.message}`);
  }
};

/**
 * GET /get_all_activity
 */
const getUserHistory = async (req, res) => {
  try {
    const userId = getUserId(req.user);
    if (!userId) {
      return sendError(res, httpStatus.UNAUTHORIZED, "Unauthorized. Missing user id.");
    }

    const query = {
      $or: [
        { host: userId },
        { "host.userId": userId },
        { "participants.meta.userId": String(userId) },
        { "participants.userId": String(userId) },
      ],
    };

    const meetings = await Meeting.find(query)
      .sort({ createdAt: -1 })
      .populate("host", "name username")
      .lean()
      .exec();

    const clientOrigin = process.env.CLIENT_ORIGIN || `http://localhost:${process.env.CLIENT_PORT || 3000}`;

    const withLinks = meetings.map((m) => {
      if (m.meetingCode) m.meetingCode = String(m.meetingCode).toUpperCase();
      if (!m.link && m.meetingCode) {
        m.link = `${clientOrigin}/room/${m.meetingCode}`;
      }

      let hostName = "Unknown";
      if (m.host) {
        if (typeof m.host === "object" && (m.host.name || m.host.username)) {
          hostName = m.host.name || m.host.username;
        } else if (m.host.name) {
          hostName = m.host.name;
        } else if (m.host.userId && m.host.name) {
          hostName = m.host.name;
        }
      }
      m.hostName = hostName;

      m.participants = (m.participants || []).map((p) => {
        const userId = p?.meta?.userId || p?.userId || null;
        const name = p?.name || p?.meta?.name || p?.meta?.display || "Guest";
        return {
          socketId: p?.socketId || null,
          userId,
          name,
          joinedAt: p?.joinedAt || p?.createdAt || null,
          leftAt: p?.leftAt || null,
        };
      });

      return m;
    });

    return res.status(httpStatus.OK).json({ success: true, meetings: withLinks });
  } catch (error) {
    console.error("getUserHistory error:", error.stack || error);
    sendError(res, httpStatus.INTERNAL_SERVER_ERROR, `Something went wrong: ${error.message}`);
  }
};

/**
 * POST /add_to_activity
 */
const addToHistory = async (req, res) => {
  const rawCode = (req.body.meeting_code || req.body.meetingCode || "").toString().trim();
  if (!rawCode) return sendError(res, httpStatus.BAD_REQUEST, "Meeting code is required.");
  const meeting_code = rawCode.toUpperCase();

  try {
    const userId = getUserId(req.user);
    if (!userId) return sendError(res, httpStatus.UNAUTHORIZED, "Unauthorized. Missing user id.");

    const existing = await Meeting.findOne({ meetingCode: meeting_code }).lean().exec();
    if (existing) {
      return res.status(httpStatus.OK).json({ success: true, message: "Meeting already exists.", meeting: existing });
    }

    const link = req.body.link || req.body.url || null;
    const synthSocketId = `init-${String(userId)}-${Date.now()}`;
    const participantEntry = {
      socketId: synthSocketId,
      name: req.user?.name || req.user?.username || "Host",
      meta: { userId: String(userId) },
      joinedAt: new Date(),
    };

    const newMeeting = new Meeting({ meetingCode: meeting_code, link, host: userId, participants: [participantEntry] });
    await newMeeting.save();

    res.status(httpStatus.CREATED).json({ success: true, message: "Meeting created and saved to history.", meeting: newMeeting });
  } catch (error) {
    console.error("addToHistory error:", error.stack || error);
    sendError(res, httpStatus.INTERNAL_SERVER_ERROR, `Something went wrong: ${error.message}`);
  }
};

/**
 * POST /meetings/:code/participants
 */
const addParticipant = async (req, res) => {
  try {
    const userId = getUserId(req.user);
    if (!userId) return sendError(res, httpStatus.UNAUTHORIZED, "Unauthorized. Missing user id.");

    const codeParam = (req.params?.code || req.body?.meeting_code || req.body?.meetingCode || "").toString().trim();
    if (!codeParam) return sendError(res, httpStatus.BAD_REQUEST, "Meeting code is required (param or body).");
    const meetingCode = codeParam.toUpperCase();

    const meeting = await Meeting.findOne({ meetingCode });
    if (!meeting) return sendError(res, httpStatus.NOT_FOUND, "Meeting not found.");

    const participantName = (req.body.name || req.user?.name || req.user?.username || "Guest").toString();

    const existingParticipant = meeting.participants.find((p) => {
      if (!p) return false;
      const metaUserId = p.meta?.userId ? String(p.meta.userId) : null;
      const directUserId = p.userId ? String(p.userId) : null;
      return metaUserId === String(userId) || directUserId === String(userId);
    });

    if (existingParticipant) {
      existingParticipant.joinedAt = new Date();
      existingParticipant.leftAt = null;
      existingParticipant.name = participantName;
      await meeting.save();
    } else {
      const synthSocketId = `user-${String(userId)}-${Date.now()}`;
      meeting.participants.push({
        socketId: synthSocketId,
        name: participantName,
        meta: { userId: String(userId) },
        joinedAt: new Date(),
      });
      await meeting.save();
    }

    if (!meeting.host) {
      meeting.host = userId;
      await meeting.save();
    }

    res.status(httpStatus.OK).json({ success: true, meeting });
  } catch (error) {
    console.error("addParticipant error:", error.stack || error);
    sendError(res, httpStatus.INTERNAL_SERVER_ERROR, `Something went wrong: ${error.message}`);
  }
};

/**
 * GET /meetings
 */
const getMeetings = async (req, res) => {
  try {
    const userId = getUserId(req.user);
    const mineOnly = String(req.query?.mine || "false").toLowerCase() === "true";

    let filter = {};
    if (userId && mineOnly) {
      filter = {
        $or: [
          { host: userId },
          { "host.userId": userId },
          { "participants.userId": userId },
          { "participants.meta.userId": String(userId) },
        ],
      };
    } else if (userId) {
      filter = {
        $or: [{ host: userId }, { "participants.userId": userId }, { active: true }],
      };
    } else {
      filter = { active: true };
    }

    const meetings = await Meeting.find(filter)
      .sort({ lastActivityAt: -1, createdAt: -1 })
      .limit(200)
      .populate({ path: "host", model: "UserDb", select: "name username" })
      .lean()
      .exec();

    return res.status(httpStatus.OK).json({ meetings });
  } catch (err) {
    console.error("getMeetings error:", err.stack || err);
    return res.status(httpStatus.INTERNAL_SERVER_ERROR).json({ success: false, message: "Failed to fetch meetings", detail: err.message });
  }
};

/**
 * POST /meetings
 */
const upsertMeeting = async (req, res) => {
  try {
    const body = req.body || {};
    const meetingCodeRaw = body.meetingCode || body.meeting_code || body.code || body.meeting;
    if (!meetingCodeRaw || !String(meetingCodeRaw).trim()) {
      return res.status(httpStatus.BAD_REQUEST).json({ success: false, message: "meetingCode is required" });
    }
    const meetingCode = String(meetingCodeRaw).toUpperCase().trim();

    const payload = {};

    if (req.user && req.user._id) payload.host = req.user._id;

    if (body.hostName || body.host_name || body.host) {
      payload.hostInfo = {
        name: body.hostName || body.host_name || (typeof body.host === "string" ? body.host : null) || null,
        userId: req.user && req.user._id ? req.user._id : null,
      };
    }

    if (Array.isArray(body.participants) && body.participants.length > 0) {
      payload.participants = body.participants.map((p) => {
        if (!p) return null;
        if (typeof p === "string") {
          return { socketId: null, userId: null, name: p, meta: {}, joinedAt: new Date() };
        }
        return {
          socketId: p.socketId || p.id || null,
          userId: p.userId || p.user_id || p.user || null,
          name: p.name || p.display || "Guest",
          meta: p.meta || p.info || {},
          joinedAt: p.joinedAt ? new Date(p.joinedAt) : new Date(),
          leftAt: p.leftAt ? new Date(p.leftAt) : null,
        };
      }).filter(Boolean);
    }

    if (body.analytics) payload.analytics = body.analytics;
    if (body.createdAt || body.created_at) payload.createdAt = new Date(body.createdAt || body.created_at);
    if (body.link) payload.link = body.link;

    const saved = await Meeting.upsertByMeetingCode(meetingCode, payload);
    return res.status(httpStatus.OK).json(saved);
  } catch (err) {
    console.error("upsertMeeting error:", err.stack || err);
    return res.status(httpStatus.INTERNAL_SERVER_ERROR).json({ success: false, message: "Failed to upsert meeting", detail: err.message });
  }
};

export {
  login,
  register,
  getUserHistory,
  addToHistory,
  addParticipant,
  getMeetings,
  upsertMeeting,
  logout, // <-- exported
};
