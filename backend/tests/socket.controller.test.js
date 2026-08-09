import { describe, it, expect, vi, beforeEach } from "vitest";

const ioHolder = vi.hoisted(() => {
    class FakeRoom {
        constructor() {
            this.fetchSockets = vi.fn().mockResolvedValue([]);
            this.emit = vi.fn();
        }
    }

    class FakeIo {
        constructor(server, opts) {
            this.server = server;
            this.opts = opts;
            this._handlers = {};
            this._middleware = null;
            this._rooms = new Map();
            this.use = vi.fn((fn) => {
                this._middleware = fn;
            });
            this.adapter = vi.fn();
            this.fetchSockets = vi.fn().mockResolvedValue([]);
            this.on = vi.fn((event, cb) => {
                this._handlers[event] = cb;
            });
            this.to = vi.fn(() => ({ emit: vi.fn() }));
        }
        in(room) {
            if (!this._rooms.has(room)) this._rooms.set(room, new FakeRoom());
            return this._rooms.get(room);
        }
    }

    return { instance: null, FakeIo };
});

vi.mock("socket.io", () => ({
    Server: vi.fn().mockImplementation((server, opts) => {
        const io = new ioHolder.FakeIo(server, opts);
        ioHolder.instance = io;
        return io;
    }),
}));

vi.mock("@socket.io/redis-adapter", () => ({
    createAdapter: vi.fn(() => "fake-adapter"),
}));

vi.mock("jsonwebtoken", () => ({
    default: {
        verify: vi.fn(),
    },
}));

vi.mock("fs", () => ({
    default: {
        readFileSync: vi.fn(() =>
            JSON.stringify({
                socket: { transports: ["websocket"], allowEIO3: true },
            })
        ),
        mkdirSync: vi.fn(),
        createWriteStream: vi.fn(() => ({
            write: vi.fn(),
            end: vi.fn(),
            on: vi.fn(),
        })),
    },
}));

vi.mock("../src/utils/redis.utils.js", () => ({
    makeLogger: vi.fn(() => ({
        info: vi.fn(),
        warn: vi.fn(),
        error: vi.fn(),
    })),
    safeRedisGet: vi.fn(),
    safeRedisSet: vi.fn(),
    safeRedisDel: vi.fn(),
}));

vi.mock("../src/services/socket.service.js", () => ({
    getState: vi.fn(),
    mkdirp: vi.fn().mockResolvedValue(undefined),
    UPLOAD_BASE: "/tmp/uploads",
    handleJoinCall: vi.fn(),
    handleUpdateParticipantState: vi.fn(),
    handleUpdateMeta: vi.fn(),
    handleChatMessage: vi.fn(),
    handleTranscriptionUpdate: vi.fn(),
    handleKeywordsUpdate: vi.fn(),
    handleLeave: vi.fn(),
    validateCode: vi.fn(),
    getParticipants: vi.fn(),
    REDIS_READ_FAILED: "REDIS_READ_FAILED",
}));

vi.mock("../observability/latency/latency.service.js", () => ({
    startTimer: vi.fn(() => 0),
    endTimer: vi.fn(),
}));

vi.mock("../src/models/meeting.model.js", () => ({
    Meeting: {
        findOne: vi.fn(),
        verifyHostSecret: vi.fn(),
    },
}));

import jwt from "jsonwebtoken";
import { safeRedisGet, safeRedisSet, safeRedisDel } from "../src/utils/redis.utils.js";
import {
    handleJoinCall,
    handleUpdateParticipantState,
    handleUpdateMeta,
    handleChatMessage,
    handleTranscriptionUpdate,
    handleKeywordsUpdate,
    handleLeave,
    validateCode,
    getParticipants,
} from "../src/services/socket.service.js";
import { Meeting } from "../src/models/meeting.model.js";
import {
    connectToSocket,
    getIo,
    notifyHostOfTranscriptRequest,
    notifyRequesterOfResolution,
    notifyUserOfResolution,
} from "../src/controllers/socket.controller.js";

function makeFakeSocket(id = "socket-1") {
    return {
        id,
        data: {},
        handlers: {},
        emit: vi.fn(),
        once: vi.fn(),
        to: vi.fn(() => ({ emit: vi.fn() })),
        on(event, cb) {
            this.handlers[event] = cb;
        },
        async trigger(event, ...args) {
            if (this.handlers[event]) return this.handlers[event](...args);
        },
    };
}

function setupConnection() {
    const server = {};
    const pubClient = {};
    const subClient = {};
    const io = connectToSocket(server, undefined, pubClient, subClient);
    const socket = makeFakeSocket();
    io._handlers["connection"](socket);
    return { io, socket };
}

beforeEach(() => {
    vi.clearAllMocks();
});

describe("module-level exports before connectToSocket", () => {
    it("getIo returns null", () => {
        expect(getIo()).toBeNull();
    });
});

describe("connectToSocket setup", () => {
    it("creates the Server with adapter and default cors options", () => {
        const server = {};
        const io = connectToSocket(server, undefined, {}, {});
        expect(io.adapter).toHaveBeenCalledWith("fake-adapter");
        expect(getIo()).toBe(io);
    });

    it("registers auth middleware that sets userId on valid token", async () => {
        process.env.JWT_SECRET = "secret";
        jwt.verify.mockReturnValue({ _id: "user-123" });
        const { io, socket: connSocket } = setupConnection();
        const socket = makeFakeSocket("auth-socket");
        socket.handshake = { auth: { token: "valid-token" } };
        const next = vi.fn();
        io._middleware(socket, next);
        expect(socket.data.userId).toBe("user-123");
        expect(next).toHaveBeenCalled();
    });

    it("does not throw and still calls next on invalid token", () => {
        process.env.JWT_SECRET = "secret";
        jwt.verify.mockImplementation(() => {
            throw new Error("bad token");
        });
        const { io } = setupConnection();
        const socket = makeFakeSocket("auth-socket-2");
        socket.handshake = { auth: { token: "bad-token" } };
        const next = vi.fn();
        expect(() => io._middleware(socket, next)).not.toThrow();
        expect(socket.data.userId).toBeUndefined();
        expect(next).toHaveBeenCalled();
    });
});

describe("join-call event", () => {
    it("calls handleJoinCall with socket, io, code, and meta", async () => {
        handleJoinCall.mockResolvedValue({ code: "ABCD" });
        const { io, socket } = setupConnection();
        await socket.trigger("join-call", "abcd", { name: "Test" });
        expect(handleJoinCall).toHaveBeenCalledWith(socket, io, "abcd", { name: "Test" });
    });

    it("emits an error event when handleJoinCall throws", async () => {
        handleJoinCall.mockRejectedValue(new Error("join failed"));
        const { socket } = setupConnection();
        await socket.trigger("join-call", "abcd", {});
        expect(socket.emit).toHaveBeenCalledWith("error", "Failed to join call");
    });
});

describe("declare-host event", () => {
    it("acks invalid_code for a bad meeting code", async () => {
        validateCode.mockReturnValue(false);
        const { socket } = setupConnection();
        const ack = vi.fn();
        await socket.trigger("declare-host", "bad code", "secret", ack);
        expect(ack).toHaveBeenCalledWith({ ok: false, reason: "invalid_code" });
    });

    it("acks not_in_room when socket is not in the target meeting", async () => {
        validateCode.mockReturnValue(true);
        const { socket } = setupConnection();
        socket.data.meetingCode = "OTHER";
        const ack = vi.fn();
        await socket.trigger("declare-host", "abcd", "secret", ack);
        expect(ack).toHaveBeenCalledWith({ ok: false, reason: "not_in_room" });
    });

    it("acks unauthorized when the host secret is invalid", async () => {
        validateCode.mockReturnValue(true);
        Meeting.verifyHostSecret.mockResolvedValue(false);
        const { socket } = setupConnection();
        socket.data.meetingCode = "ABCD";
        const ack = vi.fn();
        await socket.trigger("declare-host", "abcd", "wrong-secret", ack);
        expect(ack).toHaveBeenCalledWith({ ok: false, reason: "unauthorized" });
        expect(socket.data.isHost).toBeUndefined();
    });

    it("marks socket as host and acks ok on success", async () => {
        validateCode.mockReturnValue(true);
        Meeting.verifyHostSecret.mockResolvedValue(true);
        const { socket } = setupConnection();
        socket.data.meetingCode = "ABCD";
        const ack = vi.fn();
        await socket.trigger("declare-host", "abcd", "correct-secret", ack);
        expect(socket.data.isHost).toBe(true);
        expect(ack).toHaveBeenCalledWith({ ok: true });
    });
});

describe("signal event", () => {
    it("does nothing when target socket is not in the room", async () => {
        const { io, socket } = setupConnection();
        socket.data.meetingCode = "ABCD";
        const room = io.in("meeting:ABCD");
        room.fetchSockets.mockResolvedValue([{ id: "other-socket" }]);
        await socket.trigger("signal", "target-id", { sdp: "x" });
        expect(io.to).not.toHaveBeenCalled();
    });

    it("forwards the signal when target socket is in the room", async () => {
        const { io, socket } = setupConnection();
        socket.data.meetingCode = "ABCD";
        const room = io.in("meeting:ABCD");
        room.fetchSockets.mockResolvedValue([{ id: "target-id" }]);
        const emit = vi.fn();
        io.to.mockReturnValue({ emit });
        await socket.trigger("signal", "target-id", { sdp: "x" });
        expect(io.to).toHaveBeenCalledWith("target-id");
        expect(emit).toHaveBeenCalledWith("signal", socket.id, { sdp: "x" });
    });
});

describe("chat-message event", () => {
    it("acks with the result from handleChatMessage", async () => {
        handleChatMessage.mockResolvedValue({ ok: true, id: "msg-1" });
        const { socket } = setupConnection();
        const ack = vi.fn();
        await socket.trigger("chat-message", "ABCD", { text: "hi" }, ack);
        expect(handleChatMessage).toHaveBeenCalled();
        expect(ack).toHaveBeenCalledWith({ ok: true, id: "msg-1" });
    });

    it("acks with ok:false and emits error when handleChatMessage throws", async () => {
        handleChatMessage.mockRejectedValue(new Error("fail"));
        const { socket } = setupConnection();
        const ack = vi.fn();
        await socket.trigger("chat-message", "ABCD", { text: "hi" }, ack);
        expect(ack).toHaveBeenCalledWith({ ok: false });
        expect(socket.emit).toHaveBeenCalledWith("error", "Failed to send chat message");
    });
});

describe("emotion-status event", () => {
    it("ignores the event when socket is not host", async () => {
        const { socket } = setupConnection();
        socket.data.meetingCode = "ABCD";
        socket.data.isHost = false;
        await socket.trigger("emotion-status", { active: true });
        expect(safeRedisSet).not.toHaveBeenCalled();
    });

    it("sets emotion state and broadcasts when socket is host", async () => {
        const { socket } = setupConnection();
        socket.data.meetingCode = "ABCD";
        socket.data.isHost = true;
        const toEmit = vi.fn();
        socket.to.mockReturnValue({ emit: toEmit });
        await socket.trigger("emotion-status", { active: true });
        expect(safeRedisSet).toHaveBeenCalledWith("emotion:active:ABCD", "1");
        expect(socket.to).toHaveBeenCalledWith("meeting:ABCD");
        expect(toEmit).toHaveBeenCalledWith("emotion-status", { active: true });
    });

    it("clears emotion state when active is false", async () => {
        const { socket } = setupConnection();
        socket.data.meetingCode = "ABCD";
        socket.data.isHost = true;
        await socket.trigger("emotion-status", { active: false });
        expect(safeRedisDel).toHaveBeenCalledWith("emotion:active:ABCD");
    });
});

describe("end-meeting event", () => {
    it("does nothing when socket is not host", async () => {
        const { socket } = setupConnection();
        socket.data.isHost = false;
        await socket.trigger("end-meeting", "ABCD");
        expect(handleLeave).not.toHaveBeenCalled();
    });

    it("clears emotion state and leaves the meeting when socket is host", async () => {
        const { socket } = setupConnection();
        socket.data.isHost = true;
        socket.data.userId = "user-1";
        await socket.trigger("end-meeting", "abcd");
        expect(safeRedisDel).toHaveBeenCalledWith("emotion:active:ABCD");
        expect(handleLeave).toHaveBeenCalledWith(socket, "ABCD", expect.anything(), "user-1");
    });
});

describe("leave-call event", () => {
    it("calls handleLeave with the normalized code and userId", async () => {
        const { socket } = setupConnection();
        socket.data.userId = "user-2";
        await socket.trigger("leave-call", "abcd");
        expect(handleLeave).toHaveBeenCalledWith(socket, "ABCD", expect.anything(), "user-2");
    });
});

describe("disconnect event", () => {
    it("does nothing when socket was replaced", async () => {
        const { socket } = setupConnection();
        socket.data.replaced = true;
        socket.data.meetingCode = "ABCD";
        socket.data.userId = "user-3";
        await socket.trigger("disconnect");
        expect(handleLeave).not.toHaveBeenCalled();
    });

    it("does nothing when there is no meetingCode or userId", async () => {
        const { socket } = setupConnection();
        await socket.trigger("disconnect");
        expect(handleLeave).not.toHaveBeenCalled();
    });

    it("calls handleLeave when meetingCode and userId are present", async () => {
        const { socket } = setupConnection();
        socket.data.meetingCode = "ABCD";
        socket.data.userId = "user-4";
        await socket.trigger("disconnect");
        expect(handleLeave).toHaveBeenCalledWith(socket, "ABCD", expect.anything(), "user-4");
    });
});

describe("update-participant-state and update-meta events", () => {
    it("calls handleUpdateParticipantState with the payload", async () => {
        const { socket } = setupConnection();
        await socket.trigger("update-participant-state", { muted: true });
        expect(handleUpdateParticipantState).toHaveBeenCalledWith(socket, expect.anything(), { muted: true });
    });

    it("calls handleUpdateMeta with the payload", async () => {
        handleUpdateMeta.mockResolvedValue("ABCD");
        const { socket } = setupConnection();
        await socket.trigger("update-meta", { name: "New Name" });
        expect(handleUpdateMeta).toHaveBeenCalledWith(socket, expect.anything(), { name: "New Name" });
    });
});

describe("transcription-update and keywords-update events", () => {
    it("calls handleTranscriptionUpdate with the chunk", async () => {
        const { socket } = setupConnection();
        await socket.trigger("transcription-update", { text: "hello" });
        expect(handleTranscriptionUpdate).toHaveBeenCalledWith(socket, expect.anything(), { text: "hello" });
    });

    it("calls handleKeywordsUpdate with the keywords", async () => {
        const { socket } = setupConnection();
        await socket.trigger("keywords-update", ["a", "b"]);
        expect(handleKeywordsUpdate).toHaveBeenCalledWith(socket, expect.anything(), ["a", "b"]);
    });
});

describe("notifyHostOfTranscriptRequest", () => {
    it("does nothing when io has not been initialized", async () => {
        ioHolder.instance = null;
        await expect(notifyHostOfTranscriptRequest("ABCD", { foo: "bar" })).resolves.toBeUndefined();
    });

    it("emits directly to the host socket when present in the room", async () => {
        const { io } = setupConnection();
        const room = io.in("meeting:ABCD");
        const hostSocket = { id: "host-1", data: { isHost: true }, emit: vi.fn() };
        room.fetchSockets.mockResolvedValue([hostSocket]);
        await notifyHostOfTranscriptRequest("ABCD", { foo: "bar" });
        expect(hostSocket.emit).toHaveBeenCalledWith("transcript-request-received", { foo: "bar" });
    });

    it("falls back to Meeting lookup and emits to matching user sockets", async () => {
        const { io } = setupConnection();
        const room = io.in("meeting:ABCD");
        room.fetchSockets.mockResolvedValue([]);
        Meeting.findOne.mockReturnValue({
            lean: vi.fn().mockResolvedValue({ ownerId: "owner-1" }),
        });
        const matchingSocket = { data: { userId: "owner-1" }, emit: vi.fn() };
        const nonMatchingSocket = { data: { userId: "other" }, emit: vi.fn() };
        io.fetchSockets.mockResolvedValue([matchingSocket, nonMatchingSocket]);
        await notifyHostOfTranscriptRequest("ABCD", { foo: "bar" });
        expect(matchingSocket.emit).toHaveBeenCalledWith("transcript-request-received", { foo: "bar" });
        expect(nonMatchingSocket.emit).not.toHaveBeenCalled();
    });
});

describe("notifyRequesterOfResolution", () => {
    it("does nothing when io has not been initialized", async () => {
        ioHolder.instance = null;
        await expect(notifyRequesterOfResolution("socket-1", { status: "resolved" })).resolves.toBeUndefined();
    });

    it("emits to the requester socket id", async () => {
        const { io } = setupConnection();
        const emit = vi.fn();
        io.to.mockReturnValue({ emit });
        await notifyRequesterOfResolution("socket-1", { status: "resolved" });
        expect(io.to).toHaveBeenCalledWith("socket-1");
        expect(emit).toHaveBeenCalledWith("transcript-request-update", { status: "resolved" });
    });
});

describe("notifyUserOfResolution", () => {
    it("does nothing when io has not been initialized", async () => {
        ioHolder.instance = null;
        await expect(notifyUserOfResolution("user-1", { status: "resolved" })).resolves.toBeUndefined();
    });

    it("emits to all sockets matching the userId", async () => {
        const { io } = setupConnection();
        const matchingSocket = { data: { userId: "user-1" }, emit: vi.fn() };
        const nonMatchingSocket = { data: { userId: "user-2" }, emit: vi.fn() };
        io.fetchSockets.mockResolvedValue([matchingSocket, nonMatchingSocket]);
        await notifyUserOfResolution("user-1", { status: "resolved" });
        expect(matchingSocket.emit).toHaveBeenCalledWith("transcript-request-update", { status: "resolved" });
        expect(nonMatchingSocket.emit).not.toHaveBeenCalled();
    });
});