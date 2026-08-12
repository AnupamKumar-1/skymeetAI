import { describe, it, expect, vi, beforeAll, beforeEach, afterEach } from "vitest";
import crypto from "crypto";

vi.mock("bullmq", () => {
    const addMock = vi.fn().mockResolvedValue({ id: "job-1" });
    const onMock = vi.fn();
    return {
        Queue: vi.fn().mockImplementation(() => ({ add: addMock })),
        Worker: vi.fn().mockImplementation(() => ({ on: onMock })),
    };
});

vi.mock("../../src/models/ragChunk.model.js", () => ({
    default: {
        deleteMany: vi.fn(),
        insertMany: vi.fn(),
        aggregate: vi.fn(),
        countDocuments: vi.fn(),
    },
}));

vi.mock("../../src/models/ragSession.model.js", () => ({
    default: {
        findOne: vi.fn(),
        findOneAndUpdate: vi.fn(),
        findByIdAndUpdate: vi.fn(),
        deleteOne: vi.fn(),
    },
}));

vi.mock("../../src/utils/redis.utils.js", () => ({
    makeLogger: vi.fn(() => ({ info: vi.fn(), warn: vi.fn(), error: vi.fn() })),
    safeRedisGet: vi.fn(),
    safeRedisSet: vi.fn(),
    safeRedisDel: vi.fn(),
    safeRedisIncr: vi.fn(),
    safeRedisExpire: vi.fn(),
}));

vi.mock("../../src/services/transcript.service.js", () => ({
    isAuthorized: vi.fn(),
    resolveAuth: vi.fn(),
}));

vi.mock("../../src/data-access/transcript.repository.js", () => ({
    findTranscriptById: vi.fn(),
    findTranscriptByCode: vi.fn(),
}));

vi.mock("../../src/infra/redis.js", () => ({
    connectRedis: vi.fn(),
}));

vi.mock("../../src/models/transcriptRequest.model.js", () => ({
    TranscriptRequest: { exists: vi.fn() },
}));

import RagChunk from "../../src/models/ragChunk.model.js";
import RagSession from "../../src/models/ragSession.model.js";
import {
    safeRedisGet,
    safeRedisSet,
    safeRedisDel,
    safeRedisIncr,
    safeRedisExpire,
} from "../../src/utils/redis.utils.js";
import { isAuthorized, resolveAuth } from "../../src/services/transcript.service.js";
import {
    findTranscriptById,
    findTranscriptByCode,
} from "../../src/data-access/transcript.repository.js";
import { TranscriptRequest } from "../../src/models/transcriptRequest.model.js";

function computeChecksum(rawText, segments, aiSummary) {
    return crypto
        .createHash("sha256")
        .update(rawText + JSON.stringify(segments) + JSON.stringify(aiSummary))
        .digest("hex");
}

async function loadService(env = {}) {
    vi.resetModules();
    process.env.GROQ_API_KEY = env.GROQ_API_KEY ?? "test-groq-key";
    process.env.NOMIC_API_KEY = env.NOMIC_API_KEY ?? "test-nomic-key";
    process.env.REDIS_URL = "redis://127.0.0.1:6379";
    return await import("../../src/services/rag.service.js");
}

let ragService;

beforeAll(async () => {
    ragService = await loadService();
});

beforeEach(() => {
    vi.clearAllMocks();
    safeRedisIncr.mockResolvedValue(1);
    safeRedisExpire.mockResolvedValue(true);
    safeRedisGet.mockResolvedValue(null);
    safeRedisSet.mockResolvedValue(true);
    safeRedisDel.mockResolvedValue(true);
});

afterEach(() => {
    vi.unstubAllGlobals();
});

describe("indexTranscriptService", () => {
    it("returns 400 when id is missing", async () => {
        const req = { params: {}, };
        const result = await ragService.indexTranscriptService(req);
        expect(result.status).toBe(400);
    });

    it("returns 403 when not authorized at all", async () => {
        resolveAuth.mockReturnValue({ secret: null, userId: null, secretHash: null });
        const req = { params: { id: "MEET1" } };
        const result = await ragService.indexTranscriptService(req);
        expect(result.status).toBe(403);
    });

    it("returns 429 when index rate limit exceeded", async () => {
        resolveAuth.mockReturnValue({ secret: null, userId: "user1", secretHash: null });
        safeRedisIncr.mockResolvedValue(999);
        const req = { params: { id: "MEET1" } };
        const result = await ragService.indexTranscriptService(req);
        expect(result.status).toBe(429);
    });

    it("returns 404 when transcript not found", async () => {
        resolveAuth.mockReturnValue({ secret: null, userId: "user1", secretHash: null });
        findTranscriptById.mockResolvedValue(null);
        findTranscriptByCode.mockResolvedValue(null);
        const req = { params: { id: "MEET1" } };
        const result = await ragService.indexTranscriptService(req);
        expect(result.status).toBe(404);
    });

    it("returns 403 when user not authorized on a non legacy doc and not approved", async () => {
        resolveAuth.mockReturnValue({ secret: null, userId: "user1", secretHash: null });
        findTranscriptByCode.mockResolvedValue({
            _id: "aaaaaaaaaaaaaaaaaaaaaaaa",
            ownerId: "owner1",
            hostSecretHash: "h",
            meetingCode: "MEET1",
            metadata: { segments: [] },
            transcriptText: "hello",
            aiSummary: null,
        });
        isAuthorized.mockReturnValue(false);
        TranscriptRequest.exists.mockResolvedValue(null);
        const req = { params: { id: "MEET1" } };
        const result = await ragService.indexTranscriptService(req);
        expect(result.status).toBe(403);
    });

    it("returns 409 when indexing already in progress", async () => {
        resolveAuth.mockReturnValue({ secret: null, userId: "user1", secretHash: null });
        findTranscriptByCode.mockResolvedValue({
            _id: "aaaaaaaaaaaaaaaaaaaaaaaa",
            ownerId: "user1",
            hostSecretHash: "h",
            meetingCode: "MEET1",
            metadata: { segments: [] },
            transcriptText: "hello",
            aiSummary: null,
        });
        isAuthorized.mockReturnValue(true);
        safeRedisGet.mockImplementation(async (key) => {
            if (key.includes("rag:index:status:")) return "indexing";
            return null;
        });
        const req = { params: { id: "MEET1" } };
        const result = await ragService.indexTranscriptService(req);
        expect(result.status).toBe(409);
    });

    it("returns 200 ready when checksum matches an already ready index", async () => {
        resolveAuth.mockReturnValue({ secret: null, userId: "user1", secretHash: null });
        const rawText = "hello world";
        const segments = [];
        const aiSummary = null;
        const checksum = computeChecksum(rawText, segments, aiSummary);
        findTranscriptByCode.mockResolvedValue({
            _id: "aaaaaaaaaaaaaaaaaaaaaaaa",
            ownerId: "user1",
            hostSecretHash: "h",
            meetingCode: "MEET1",
            metadata: { segments },
            transcriptText: rawText,
            aiSummary,
        });
        isAuthorized.mockReturnValue(true);
        safeRedisGet.mockImplementation(async (key) => {
            if (key.includes("rag:index:status:")) return "ready";
            if (key.includes("rag:checksum:index:")) return checksum;
            return null;
        });
        const req = { params: { id: "MEET1" } };
        const result = await ragService.indexTranscriptService(req);
        expect(result.status).toBe(200);
        expect(result.body.indexStatus).toBe("ready");
    });

    it("enqueues an indexing job and returns 202 for a new index request", async () => {
        resolveAuth.mockReturnValue({ secret: null, userId: "user1", secretHash: null });
        findTranscriptByCode.mockResolvedValue({
            _id: "aaaaaaaaaaaaaaaaaaaaaaaa",
            ownerId: "user1",
            hostSecretHash: "h",
            meetingCode: "MEET1",
            metadata: { segments: [] },
            transcriptText: "hello world",
            aiSummary: null,
        });
        isAuthorized.mockReturnValue(true);
        safeRedisGet.mockResolvedValue(null);
        const req = { params: { id: "MEET1" } };
        const result = await ragService.indexTranscriptService(req);
        expect(result.status).toBe(202);
        expect(result.body.indexStatus).toBe("indexing");
        expect(ragService.indexQueue.add).toHaveBeenCalledTimes(1);
        expect(safeRedisSet).toHaveBeenCalledWith(
            expect.stringContaining("rag:index:status:"),
            "indexing",
            expect.any(Object)
        );
    });
});

describe("getIndexStatusService", () => {
    it("returns 400 when id is missing", async () => {
        const req = { params: {} };
        const result = await ragService.getIndexStatusService(req);
        expect(result.status).toBe(400);
    });

    it("returns 404 when transcript is not found", async () => {
        findTranscriptById.mockResolvedValue(null);
        findTranscriptByCode.mockResolvedValue(null);
        const req = { params: { id: "MEET1" } };
        const result = await ragService.getIndexStatusService(req);
        expect(result.status).toBe(404);
    });

    it("returns 403 when not authorized and not approved", async () => {
        findTranscriptByCode.mockResolvedValue({
            _id: "aaaaaaaaaaaaaaaaaaaaaaaa",
            ownerId: "owner1",
            hostSecretHash: "h",
            meetingCode: "MEET1",
        });
        resolveAuth.mockReturnValue({ secret: null, userId: "user1", secretHash: null });
        isAuthorized.mockReturnValue(false);
        TranscriptRequest.exists.mockResolvedValue(null);
        const req = { params: { id: "MEET1" } };
        const result = await ragService.getIndexStatusService(req);
        expect(result.status).toBe(403);
    });

    it("returns not_indexed with zero chunk count when no status exists", async () => {
        findTranscriptByCode.mockResolvedValue({
            _id: "aaaaaaaaaaaaaaaaaaaaaaaa",
            ownerId: "user1",
            hostSecretHash: "h",
            meetingCode: "MEET1",
        });
        resolveAuth.mockReturnValue({ secret: null, userId: "user1", secretHash: null });
        isAuthorized.mockReturnValue(true);
        safeRedisGet.mockResolvedValue(null);
        const req = { params: { id: "MEET1" } };
        const result = await ragService.getIndexStatusService(req);
        expect(result.status).toBe(200);
        expect(result.body.indexStatus).toBe("not_indexed");
        expect(result.body.chunkCount).toBe(0);
        expect(RagChunk.countDocuments).not.toHaveBeenCalled();
    });

    it("returns ready status with chunk count from the database", async () => {
        findTranscriptByCode.mockResolvedValue({
            _id: "aaaaaaaaaaaaaaaaaaaaaaaa",
            ownerId: "user1",
            hostSecretHash: "h",
            meetingCode: "MEET1",
        });
        resolveAuth.mockReturnValue({ secret: null, userId: "user1", secretHash: null });
        isAuthorized.mockReturnValue(true);
        safeRedisGet.mockImplementation(async (key) => (key.includes("rag:index:status:") ? "ready" : null));
        RagChunk.countDocuments.mockResolvedValue(42);
        const req = { params: { id: "MEET1" } };
        const result = await ragService.getIndexStatusService(req);
        expect(result.status).toBe(200);
        expect(result.body.indexStatus).toBe("ready");
        expect(result.body.chunkCount).toBe(42);
    });
});

describe("ragQueryService", () => {
    it("returns 400 when id is missing", async () => {
        const req = { params: {}, body: { question: "hi" } };
        const result = await ragService.ragQueryService(req);
        expect(result.status).toBe(400);
    });

    it("returns 400 when question is missing", async () => {
        const req = { params: { id: "MEET1" }, body: {} };
        const result = await ragService.ragQueryService(req);
        expect(result.status).toBe(400);
    });

    it("returns 403 when not authorized at all", async () => {
        resolveAuth.mockReturnValue({ secret: null, userId: null, secretHash: null });
        const req = { params: { id: "MEET1" }, body: { question: "hi" } };
        const result = await ragService.ragQueryService(req);
        expect(result.status).toBe(403);
    });

    it("returns 429 when query rate limit exceeded", async () => {
        resolveAuth.mockReturnValue({ secret: null, userId: "user1", secretHash: null });
        safeRedisIncr.mockResolvedValue(999);
        const req = { params: { id: "MEET1" }, body: { question: "hi" } };
        const result = await ragService.ragQueryService(req);
        expect(result.status).toBe(429);
    });

    it("returns 500 when GROQ_API_KEY is not configured", async () => {
        const svc = await loadService({ GROQ_API_KEY: "" });
        resolveAuth.mockReturnValue({ secret: null, userId: "user1", secretHash: null });
        const req = { params: { id: "MEET1" }, body: { question: "hi" } };
        const result = await svc.ragQueryService(req);
        expect(result.status).toBe(500);
        ragService = await loadService();
    });

    it("returns 500 when NOMIC_API_KEY is not configured", async () => {
        const svc = await loadService({ NOMIC_API_KEY: "" });
        resolveAuth.mockReturnValue({ secret: null, userId: "user1", secretHash: null });
        const req = { params: { id: "MEET1" }, body: { question: "hi" } };
        const result = await svc.ragQueryService(req);
        expect(result.status).toBe(500);
        ragService = await loadService();
    });

    it("returns 404 when transcript is not found", async () => {
        resolveAuth.mockReturnValue({ secret: null, userId: "user1", secretHash: null });
        findTranscriptById.mockResolvedValue(null);
        findTranscriptByCode.mockResolvedValue(null);
        const req = { params: { id: "MEET1" }, body: { question: "hi" } };
        const result = await ragService.ragQueryService(req);
        expect(result.status).toBe(404);
    });

    it("returns 403 when not authorized on a non legacy doc", async () => {
        resolveAuth.mockReturnValue({ secret: null, userId: "user1", secretHash: null });
        findTranscriptByCode.mockResolvedValue({
            _id: "aaaaaaaaaaaaaaaaaaaaaaaa",
            ownerId: "owner1",
            hostSecretHash: "h",
            meetingCode: "MEET1",
        });
        isAuthorized.mockReturnValue(false);
        TranscriptRequest.exists.mockResolvedValue(null);
        const req = { params: { id: "MEET1" }, body: { question: "hi" } };
        const result = await ragService.ragQueryService(req);
        expect(result.status).toBe(403);
    });

    it("returns 424 when transcript is not indexed", async () => {
        resolveAuth.mockReturnValue({ secret: null, userId: "user1", secretHash: null });
        findTranscriptByCode.mockResolvedValue({
            _id: "aaaaaaaaaaaaaaaaaaaaaaaa",
            ownerId: "user1",
            hostSecretHash: "h",
            meetingCode: "MEET1",
        });
        isAuthorized.mockReturnValue(true);
        safeRedisGet.mockResolvedValue(null);
        const req = { params: { id: "MEET1" }, body: { question: "hi" } };
        const result = await ragService.ragQueryService(req);
        expect(result.status).toBe(424);
        expect(result.body.indexStatus).toBe("not_indexed");
    });

    it("returns 424 when transcript is currently indexing", async () => {
        resolveAuth.mockReturnValue({ secret: null, userId: "user1", secretHash: null });
        findTranscriptByCode.mockResolvedValue({
            _id: "aaaaaaaaaaaaaaaaaaaaaaaa",
            ownerId: "user1",
            hostSecretHash: "h",
            meetingCode: "MEET1",
        });
        isAuthorized.mockReturnValue(true);
        safeRedisGet.mockResolvedValue("indexing");
        const req = { params: { id: "MEET1" }, body: { question: "hi" } };
        const result = await ragService.ragQueryService(req);
        expect(result.status).toBe(424);
        expect(result.body.indexStatus).toBe("indexing");
    });

    it("returns a fallback answer when no chunks are retrieved", async () => {
        resolveAuth.mockReturnValue({ secret: null, userId: "user1", secretHash: null });
        findTranscriptByCode.mockResolvedValue({
            _id: "aaaaaaaaaaaaaaaaaaaaaaaa",
            ownerId: "user1",
            hostSecretHash: "h",
            meetingCode: "MEET1",
        });
        isAuthorized.mockReturnValue(true);
        safeRedisGet.mockImplementation(async (key) => (key.includes("rag:index:status:") ? "ready" : null));
        RagChunk.countDocuments.mockResolvedValue(0);
        RagChunk.aggregate.mockResolvedValue([]);
        const fetchMock = vi.fn().mockResolvedValue({
            ok: true,
            json: async () => ({ embeddings: [[0.1, 0.2, 0.3]] }),
        });
        vi.stubGlobal("fetch", fetchMock);
        const req = { params: { id: "MEET1" }, body: { question: "what happened?" } };
        const result = await ragService.ragQueryService(req);
        expect(result.status).toBe(200);
        expect(result.body.sources).toEqual([]);
        expect(result.body.sessionId).toBeNull();
    });

    it("returns an answer with sources on a successful non streaming query", async () => {
        resolveAuth.mockReturnValue({ secret: null, userId: "user1", secretHash: null });
        findTranscriptByCode.mockResolvedValue({
            _id: "aaaaaaaaaaaaaaaaaaaaaaaa",
            ownerId: "user1",
            hostSecretHash: "h",
            meetingCode: "MEET1",
        });
        isAuthorized.mockReturnValue(true);
        safeRedisGet.mockImplementation(async (key) => (key.includes("rag:index:status:") ? "ready" : null));
        RagChunk.countDocuments.mockResolvedValue(2);
        RagChunk.aggregate.mockResolvedValue([
            {
                _id: "chunk1",
                chunkIndex: 0,
                chunkText: "Speaker A discussed the roadmap.",
                embedding: [0.1, 0.2, 0.3],
                speakers: ["Speaker A"],
                startSec: 0,
                endSec: 10,
                chunkType: "segment",
                score: 0.9,
            },
            {
                _id: "chunk2",
                chunkIndex: 1,
                chunkText: "Speaker B raised concerns about the budget.",
                embedding: [0.2, 0.1, 0.4],
                speakers: ["Speaker B"],
                startSec: 10,
                endSec: 20,
                chunkType: "segment",
                score: 0.8,
            },
        ]);
        RagSession.findOneAndUpdate.mockResolvedValue({ _id: "session1", messages: [] });
        RagSession.findByIdAndUpdate.mockResolvedValue({});

        const fetchMock = vi
            .fn()
            .mockResolvedValueOnce({
                ok: true,
                json: async () => ({ embeddings: [[0.15, 0.15, 0.35]] }),
            })
            .mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    choices: [{ message: { content: "Here is the answer." } }],
                }),
            });
        vi.stubGlobal("fetch", fetchMock);

        const req = { params: { id: "MEET1" }, body: { question: "what happened?" } };
        const result = await ragService.ragQueryService(req);

        expect(result.status).toBe(200);
        expect(result.body.success).toBe(true);
        expect(result.body.answer).toBe("Here is the answer.");
        expect(result.body.sources).toHaveLength(2);
        expect(result.body.sessionId).toBe("session1");
        expect(RagSession.findByIdAndUpdate).toHaveBeenCalledTimes(1);
        expect(fetchMock).toHaveBeenCalledTimes(2);
    });

    it("returns 502 when the LLM provider responds with an error", async () => {
        resolveAuth.mockReturnValue({ secret: null, userId: "user1", secretHash: null });
        findTranscriptByCode.mockResolvedValue({
            _id: "aaaaaaaaaaaaaaaaaaaaaaaa",
            ownerId: "user1",
            hostSecretHash: "h",
            meetingCode: "MEET1",
        });
        isAuthorized.mockReturnValue(true);
        safeRedisGet.mockImplementation(async (key) => (key.includes("rag:index:status:") ? "ready" : null));
        RagChunk.countDocuments.mockResolvedValue(1);
        RagChunk.aggregate.mockResolvedValue([
            {
                _id: "chunk1",
                chunkIndex: 0,
                chunkText: "Some transcript content.",
                embedding: [0.1, 0.2, 0.3],
                speakers: [],
                startSec: null,
                endSec: null,
                chunkType: "segment",
                score: 0.5,
            },
        ]);
        RagSession.findOneAndUpdate.mockResolvedValue({ _id: "session1", messages: [] });

        const fetchMock = vi
            .fn()
            .mockResolvedValueOnce({
                ok: true,
                json: async () => ({ embeddings: [[0.1, 0.2, 0.3]] }),
            })
            .mockResolvedValueOnce({
                ok: false,
                status: 503,
                json: async () => ({ error: { message: "service unavailable" } }),
            });
        vi.stubGlobal("fetch", fetchMock);

        const req = { params: { id: "MEET1" }, body: { question: "what happened?" } };
        const result = await ragService.ragQueryService(req);
        expect(result.status).toBe(502);
    });
});

describe("getRagSessionService", () => {
    it("returns 403 when no userId is present", async () => {
        resolveAuth.mockReturnValue({ secret: null, userId: null, secretHash: null });
        const req = { params: { id: "MEET1" } };
        const result = await ragService.getRagSessionService(req);
        expect(result.status).toBe(403);
    });

    it("returns 404 when transcript is not found", async () => {
        resolveAuth.mockReturnValue({ secret: null, userId: "user1", secretHash: null });
        findTranscriptById.mockResolvedValue(null);
        findTranscriptByCode.mockResolvedValue(null);
        const req = { params: { id: "MEET1" } };
        const result = await ragService.getRagSessionService(req);
        expect(result.status).toBe(404);
    });

    it("returns null session when none exists", async () => {
        resolveAuth.mockReturnValue({ secret: null, userId: "user1", secretHash: null });
        findTranscriptByCode.mockResolvedValue({ _id: "aaaaaaaaaaaaaaaaaaaaaaaa", meetingCode: "MEET1" });
        RagSession.findOne.mockReturnValue({ lean: vi.fn().mockResolvedValue(null) });
        const req = { params: { id: "MEET1" } };
        const result = await ragService.getRagSessionService(req);
        expect(result.status).toBe(200);
        expect(result.body.session).toBeNull();
    });

    it("returns session data when it exists", async () => {
        resolveAuth.mockReturnValue({ secret: null, userId: "user1", secretHash: null });
        findTranscriptByCode.mockResolvedValue({ _id: "aaaaaaaaaaaaaaaaaaaaaaaa", meetingCode: "MEET1" });
        RagSession.findOne.mockReturnValue({
            lean: vi.fn().mockResolvedValue({
                messages: [{ role: "user", content: "hi" }],
                totalTokensUsed: 12,
                lastActivityAt: new Date("2026-01-01"),
            }),
        });
        const req = { params: { id: "MEET1" } };
        const result = await ragService.getRagSessionService(req);
        expect(result.status).toBe(200);
        expect(result.body.session.totalTokensUsed).toBe(12);
        expect(result.body.session.messages).toHaveLength(1);
    });
});

describe("clearRagSessionService", () => {
    it("returns 403 when no userId is present", async () => {
        resolveAuth.mockReturnValue({ secret: null, userId: null, secretHash: null });
        const req = { params: { id: "MEET1" } };
        const result = await ragService.clearRagSessionService(req);
        expect(result.status).toBe(403);
    });

    it("returns 404 when transcript is not found", async () => {
        resolveAuth.mockReturnValue({ secret: null, userId: "user1", secretHash: null });
        findTranscriptById.mockResolvedValue(null);
        findTranscriptByCode.mockResolvedValue(null);
        const req = { params: { id: "MEET1" } };
        const result = await ragService.clearRagSessionService(req);
        expect(result.status).toBe(404);
    });

    it("deletes the session and returns 200", async () => {
        resolveAuth.mockReturnValue({ secret: null, userId: "user1", secretHash: null });
        findTranscriptByCode.mockResolvedValue({ _id: "aaaaaaaaaaaaaaaaaaaaaaaa", meetingCode: "MEET1" });
        RagSession.deleteOne.mockResolvedValue({ deletedCount: 1 });
        const req = { params: { id: "MEET1" } };
        const result = await ragService.clearRagSessionService(req);
        expect(result.status).toBe(200);
        expect(RagSession.deleteOne).toHaveBeenCalledWith({
            transcriptId: "aaaaaaaaaaaaaaaaaaaaaaaa",
            userId: "user1",
        });
    });
});

describe("deleteRagIndexService", () => {
    it("deletes chunks and clears the index status", async () => {
        RagChunk.deleteMany.mockResolvedValue({ deletedCount: 3 });
        await ragService.deleteRagIndexService("transcript123");
        expect(RagChunk.deleteMany).toHaveBeenCalledWith({ transcriptId: "transcript123" });
        expect(safeRedisDel).toHaveBeenCalledWith(expect.stringContaining("transcript123"));
    });
});