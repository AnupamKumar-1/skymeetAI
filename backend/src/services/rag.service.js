import crypto from "crypto";
import { Queue, Worker } from "bullmq";
import mongoose from "mongoose";
import RagChunk from "../models/ragChunk.model.js";
import RagSession from "../models/ragSession.model.js";
import {
    makeLogger,
    safeRedisGet,
    safeRedisSet,
    safeRedisDel,
    safeRedisIncr,
    safeRedisExpire,
} from "../utils/redis.utils.js";
import { isAuthorized, resolveAuth } from "./transcript.service.js";
import {
    findTranscriptById,
    findTranscriptByCode,
} from "../data-access/transcript.repository.js";

import { connectRedis } from "../infra/redis.js";

const log = makeLogger("rag");

const GROQ_API_KEY = process.env.GROQ_API_KEY;
const GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions";

const NOMIC_API_KEY = process.env.NOMIC_API_KEY;
const NOMIC_EMBED_URL = "https://api-atlas.nomic.ai/v1/embedding/text";

const CHUNK_TOKENS = parseInt(process.env.RAG_CHUNK_TOKENS || "600", 10);
const CHUNK_OVERLAP = parseInt(process.env.RAG_CHUNK_OVERLAP || "100", 10);

const RETRIEVAL_TOP_K = parseInt(process.env.RAG_RETRIEVAL_TOP_K || "30", 10);
const MMR_LAMBDA = parseFloat(process.env.RAG_MMR_LAMBDA || "0.6");
const MMR_FINAL_K = parseInt(process.env.RAG_MMR_FINAL_K || "5", 10);

const LLM_MODEL = process.env.RAG_LLM_MODEL || "llama-3.3-70b-versatile";
const EMBED_MODEL = process.env.RAG_EMBED_MODEL || "nomic-embed-text-v1.5";
const LLM_MAX_TOKENS = parseInt(process.env.RAG_LLM_MAX_TOKENS || "1024", 10);
const LLM_TEMPERATURE = parseFloat(process.env.RAG_LLM_TEMPERATURE || "0.2");

const SESSION_CONTEXT_WINDOW = parseInt(process.env.RAG_SESSION_CONTEXT_WINDOW || "30", 10);

const RAG_QUERY_RATE_MAX = parseInt(process.env.RAG_QUERY_RATE_MAX || "20", 10);
const RAG_QUERY_RATE_WIN_SEC = parseInt(process.env.RAG_QUERY_RATE_WIN_SEC || "3600", 10);
const RAG_INDEX_RATE_MAX = parseInt(process.env.RAG_INDEX_RATE_MAX || "5", 10);
const RAG_INDEX_RATE_WIN_SEC = parseInt(process.env.RAG_INDEX_RATE_WIN_SEC || "3600", 10);

const RKEYS = {
    indexStatus: (tid) => `rag:index:status:${tid}`,
    queryRate: (uid) => `rag:query:rate:${uid}`,
    indexRate: (uid) => `rag:index:rate:${uid}`,
    embeddingCache: (hash) => `rag:embed:${hash}`,
    payloadLock: (tid) => `rag:lock:index:${tid}`,
    payloadChecksum: (tid) => `rag:checksum:index:${tid}`
};
const EMBED_CACHE_TTL = 7 * 24 * 3600;

const _redisUrl = new URL(process.env.REDIS_URL || "redis://127.0.0.1:6379");
const queueOptions = {
    connection: {
        host: _redisUrl.hostname,
        port: parseInt(_redisUrl.port || "6379", 10),
        password: _redisUrl.password || undefined,
        tls: (process.env.REDIS_URL || "").startsWith("rediss://") ? {} : undefined,
    }
};

export const indexQueue = new Queue("transcriptIndexing", queueOptions);

function estimateTokens(text) {
    return Math.ceil(text.length / 4);
}

function textHash(str) {
    return crypto.createHash("sha256").update(str).digest("hex");
}

function cosineSim(a, b) {
    if (!a?.length || a.length !== b?.length) return 0;
    let dot = 0, normA = 0, normB = 0;
    for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    const denom = Math.sqrt(normA) * Math.sqrt(normB);
    return denom === 0 ? 0 : dot / denom;
}

function mmrRerank(queryEmbed, candidates, lambda = MMR_LAMBDA, k = MMR_FINAL_K) {
    if (candidates.length <= k) return candidates;

    const selected = [];
    const remaining = [...candidates];

    while (selected.length < k && remaining.length > 0) {
        let bestIdx = 0;
        let bestScore = -Infinity;

        for (let i = 0; i < remaining.length; i++) {
            const relevance = cosineSim(queryEmbed, remaining[i].embedding);
            const maxSim =
                selected.length === 0
                    ? 0
                    : Math.max(...selected.map((s) => cosineSim(s.embedding, remaining[i].embedding)));
            const mmrScore = lambda * relevance - (1 - lambda) * maxSim;
            if (mmrScore > bestScore) {
                bestScore = mmrScore;
                bestIdx = i;
            }
        }

        selected.push(remaining[bestIdx]);
        remaining.splice(bestIdx, 1);
    }

    return selected;
}

async function checkRateLimit(key, max, windowSec) {
    const count = await safeRedisIncr(key);
    if (count === 1) await safeRedisExpire(key, windowSec);
    return count > max;
}

async function embedText(text, taskType = "search_document") {
    const cacheKey = RKEYS.embeddingCache(textHash(text));
    const cached = await safeRedisGet(cacheKey);
    if (cached) return JSON.parse(cached);

    if (!NOMIC_API_KEY) throw new Error("NOMIC_API_KEY not configured");
    const res = await fetch(NOMIC_EMBED_URL, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${NOMIC_API_KEY}`,
        },
        body: JSON.stringify({ model: EMBED_MODEL, texts: [text], task_type: taskType }),
    });

    if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(`Nomic embed error ${res.status}: ${err?.message || res.statusText}`);
    }

    const data = await res.json();
    const vector = data.embeddings?.[0];
    if (!vector?.length) throw new Error("Empty embedding response from Nomic");

    await safeRedisSet(cacheKey, JSON.stringify(vector), { EX: EMBED_CACHE_TTL });
    return vector;
}

async function embedBatch(texts) {
    if (!texts.length) return [];

    const results = new Array(texts.length).fill(null);
    const uncachedIndices = [];

    for (let i = 0; i < texts.length; i++) {
        const cacheKey = RKEYS.embeddingCache(textHash(texts[i]));
        const cached = await safeRedisGet(cacheKey);
        if (cached) results[i] = JSON.parse(cached);
        else uncachedIndices.push(i);
    }

    if (!uncachedIndices.length) return results;

    const uncachedTexts = uncachedIndices.map((i) => texts[i]);

    if (!NOMIC_API_KEY) throw new Error("NOMIC_API_KEY not configured");
    const res = await fetch(NOMIC_EMBED_URL, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${NOMIC_API_KEY}`,
        },
        body: JSON.stringify({ model: EMBED_MODEL, texts: uncachedTexts, task_type: "search_document" }),
    });

    if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(`Nomic batch embed error ${res.status}: ${err?.message || res.statusText}`);
    }

    const data = await res.json();
    const embeddings = data.embeddings || [];

    for (let j = 0; j < uncachedIndices.length; j++) {
        const origIdx = uncachedIndices[j];
        const vector = embeddings[j];
        if (vector?.length) {
            results[origIdx] = vector;
            const cacheKey = RKEYS.embeddingCache(textHash(texts[origIdx]));
            safeRedisSet(cacheKey, JSON.stringify(vector), { EX: EMBED_CACHE_TTL });
        }
    }

    return results;
}

function buildChunks(segments, transcriptText) {
    if (Array.isArray(segments) && segments.length > 0) {
        return buildSegmentChunks(segments);
    }
    return buildWindowChunks(transcriptText);
}

function buildSegmentChunks(segments) {
    const chunks = [];
    let buffer = [];
    let bufferTokens = 0;
    let chunkIdx = 0;

    function flush(overlapBuffer) {
        if (!buffer.length) return;

        const text = buffer.map((s) => `[${s.speaker || "Unknown"}] ${s.text}`).join("\n");
        const speakers = [...new Set(buffer.map((s) => s.speaker).filter(Boolean))];
        const startSec = buffer[0]?.start ?? null;
        const endSec = buffer[buffer.length - 1]?.end ?? null;

        chunks.push({
            chunkIndex: chunkIdx++,
            chunkText: text,
            tokenCount: estimateTokens(text),
            speakers,
            startSec,
            endSec,
            chunkType: "segment",
        });

        buffer = overlapBuffer || [];
        bufferTokens = buffer.reduce((acc, s) => acc + estimateTokens(s.text), 0);
    }

    for (const seg of segments) {
        const segTokens = estimateTokens(seg.text || "");

        if (bufferTokens + segTokens > CHUNK_TOKENS && buffer.length > 0) {
            const overlapSegs = [];
            let overlapTokens = 0;
            for (let i = buffer.length - 1; i >= 0; i--) {
                const t = estimateTokens(buffer[i].text || "");
                if (overlapTokens + t > CHUNK_OVERLAP) break;
                overlapSegs.unshift(buffer[i]);
                overlapTokens += t;
            }
            flush(overlapSegs);
        }

        buffer.push(seg);
        bufferTokens += segTokens;
    }

    if (buffer.length) flush([]);
    return chunks;
}

function buildWindowChunks(text) {
    if (!text) return [];

    const words = text.split(/\s+/).filter(Boolean);
    const chunks = [];
    let chunkIdx = 0;
    let i = 0;

    while (i < words.length) {
        const slice = words.slice(i, i + CHUNK_TOKENS);
        chunks.push({
            chunkIndex: chunkIdx++,
            chunkText: slice.join(" "),
            tokenCount: slice.length,
            speakers: [],
            startSec: null,
            endSec: null,
            chunkType: "window",
        });
        i += CHUNK_TOKENS - CHUNK_OVERLAP;
        if (i >= words.length) break;
    }

    return chunks;
}

function buildSummaryChunks(aiSummary, startIdx = 0) {
    if (!aiSummary) return [];
    const ins = aiSummary.insights || aiSummary;
    const parts = [];

    if (ins.summary) parts.push(`Meeting Summary:\n${ins.summary}`);
    if (ins.headline) parts.push(`Headline:\n${ins.headline}`);

    if (Array.isArray(ins.top_topics) && ins.top_topics.length) {
        parts.push(`Top Topics:\n${ins.top_topics.map((t) => `- ${typeof t === "string" ? t : t.topic || JSON.stringify(t)}`).join("\n")}`);
    }

    if (ins.emotion_distribution && typeof ins.emotion_distribution === "object") {
        const emotions = Object.entries(ins.emotion_distribution)
            .map(([k, v]) => `${k}: ${v}`)
            .join(", ");
        parts.push(`Emotion Distribution:\n${emotions}`);
    }

    if (typeof ins.speaking_pace_wpm === "number") {
        parts.push(`Speaking Pace: ${ins.speaking_pace_wpm} words per minute`);
    }

    if (typeof ins.total_words === "number") {
        parts.push(`Total Words Spoken: ${ins.total_words}`);
    }

    if (Array.isArray(ins.discrepancies) && ins.discrepancies.length) {
        parts.push(`Key Discrepancies:\n${ins.discrepancies.map((d) => `- ${typeof d === "string" ? d : JSON.stringify(d)}`).join("\n")}`);
    }

    if (Array.isArray(ins.emotional_moments) && ins.emotional_moments.length) {
        parts.push(`Emotional Moments:\n${ins.emotional_moments.map((m) => `- ${typeof m === "string" ? m : JSON.stringify(m)}`).join("\n")}`);
    }

    const extraKeys = Object.keys(ins).filter(
        (k) => !["summary", "headline", "top_topics", "emotion_distribution",
            "speaking_pace_wpm", "total_words", "discrepancies", "emotional_moments"].includes(k)
    );
    for (const key of extraKeys) {
        const val = ins[key];
        if (val == null) continue;
        const text = typeof val === "string" ? val : JSON.stringify(val);
        if (text.trim()) parts.push(`${key}:\n${text}`);
    }

    return parts
        .filter((p) => p.trim())
        .map((text, i) => ({
            chunkIndex: startIdx + i,
            chunkText: text,
            tokenCount: estimateTokens(text),
            speakers: [],
            startSec: null,
            endSec: null,
            chunkType: "summary",
        }));
}

export async function indexTranscriptService(req) {
    const idOrCode = String(req.params.id || "").trim();
    if (!idOrCode) return { status: 400, body: { success: false, message: "id or meetingCode required" } };

    const { secret, userId, secretHash } = resolveAuth(req);
    if (!secret && !userId) return { status: 403, body: { success: false, message: "Not authorized" } };

    if (userId && await checkRateLimit(RKEYS.indexRate(userId), RAG_INDEX_RATE_MAX, RAG_INDEX_RATE_WIN_SEC)) {
        return { status: 429, body: { success: false, message: "RAG index rate limit reached" } };
    }

    const isMongoId = /^[a-f\d]{24}$/i.test(idOrCode);
    let doc = null;
    if (isMongoId) doc = await findTranscriptById(idOrCode);
    if (!doc) doc = await findTranscriptByCode(idOrCode.toUpperCase());
    if (!doc) return { status: 404, body: { success: false, message: "Transcript not found" } };

    const isLegacyDoc = !doc.ownerId && !doc.hostSecretHash;
    if (!isLegacyDoc && !isAuthorized(doc, userId, secretHash)) {
        return { status: 403, body: { success: false, message: "Not authorized" } };
    }

    const transcriptId = String(doc._id);
    const lockKey = RKEYS.payloadLock(transcriptId);
    const statusKey = RKEYS.indexStatus(transcriptId);
    const checksumKey = RKEYS.payloadChecksum(transcriptId);

    const currentStatus = await safeRedisGet(statusKey);
    if (currentStatus === "indexing") {
        return { status: 409, body: { success: false, message: "Indexing already in progress" } };
    }
    if (currentStatus === "failed") {
        await safeRedisDel(statusKey);
    }

    const segments = doc.metadata?.segments || [];
    const rawText = doc.transcriptText || "";
    const aiSummary = doc.aiSummary || null;
    const computedChecksum = textHash(rawText + JSON.stringify(segments) + JSON.stringify(aiSummary));
    const activeChecksum = await safeRedisGet(checksumKey);

    if (currentStatus === "ready" && activeChecksum === computedChecksum) {
        return {
            status: 200,
            body: { success: true, indexStatus: "ready", transcriptId }
        };
    }

    await safeRedisSet(statusKey, "indexing", { EX: 300 });

    await indexQueue.add(`index-${transcriptId}`, {
        transcriptId,
        meetingCode: doc.meetingCode,
        ownerId: doc.ownerId || null,
        segments,
        rawText,
        aiSummary,
        computedChecksum
    });

    return {
        status: 202,
        body: { success: true, indexStatus: "indexing" }
    };
}

const indexWorker = new Worker("transcriptIndexing", async (job) => {
    const { transcriptId, meetingCode, ownerId, segments, rawText, aiSummary, computedChecksum } = job.data;
    const statusKey = RKEYS.indexStatus(transcriptId);
    const checksumKey = RKEYS.payloadChecksum(transcriptId);

    try {
        const transcriptChunks = buildChunks(segments, rawText);
        const summaryChunks = buildSummaryChunks(aiSummary, transcriptChunks.length);
        const chunks = [...transcriptChunks, ...summaryChunks];

        if (!chunks.length) {
            await safeRedisSet(statusKey, "no_content", { EX: 3600 });
            return;
        }

        const EMBED_BATCH_SIZE = 96;
        const allEmbeddings = [];

        for (let i = 0; i < chunks.length; i += EMBED_BATCH_SIZE) {
            const batch = chunks.slice(i, i + EMBED_BATCH_SIZE);
            const embeddings = await embedBatch(batch.map((c) => c.chunkText));
            allEmbeddings.push(...embeddings);
        }

        await RagChunk.deleteMany({ transcriptId });

        const docs = chunks.map((c, i) => ({
            transcriptId,
            meetingCode,
            ownerId,
            ...c,
            embedding: allEmbeddings[i] || [],
            embeddingModel: EMBED_MODEL,
        }));

        await RagChunk.insertMany(docs, { ordered: false });
        await safeRedisSet(checksumKey, computedChecksum, { EX: 30 * 24 * 3600 });
        await safeRedisSet(statusKey, "ready", { EX: 7 * 24 * 3600 });

    } catch (err) {
        await safeRedisSet(statusKey, "failed", { EX: 3600 });
        throw err;
    }
}, queueOptions);

indexWorker.on("failed", (job, err) => {
    log.error("Background chunk parsing thread failed:", { jobId: job?.id, error: err.message });
});

export async function getIndexStatusService(req) {
    const idOrCode = String(req.params.id || "").trim();
    if (!idOrCode) return { status: 400, body: { success: false, message: "id required" } };

    const isMongoId = /^[a-f\d]{24}$/i.test(idOrCode);
    let doc = null;
    if (isMongoId) doc = await findTranscriptById(idOrCode);
    if (!doc) doc = await findTranscriptByCode(idOrCode.toUpperCase());
    if (!doc) return { status: 404, body: { success: false, message: "Transcript not found" } };

    const { secret, userId, secretHash } = resolveAuth(req);
    const isLegacyDoc = !doc.ownerId && !doc.hostSecretHash;
    if (!isLegacyDoc && !isAuthorized(doc, userId, secretHash)) {
        return { status: 403, body: { success: false, message: "Not authorized" } };
    }

    const transcriptId = String(doc._id);
    const status = await safeRedisGet(RKEYS.indexStatus(transcriptId));
    const chunkCount = status === "ready"
        ? await RagChunk.countDocuments({ transcriptId })
        : 0;

    return {
        status: 200,
        body: {
            success: true,
            indexStatus: status || "not_indexed",
            chunkCount,
            transcriptId,
            meetingCode: doc.meetingCode,
        },
    };
}

async function retrieveChunks(transcriptId, queryEmbedding) {
    const vectorLimit = Math.max(MMR_FINAL_K * 2, 10);
    const vectorCandidates = Math.max(RETRIEVAL_TOP_K, vectorLimit * 10, 100);

    const tid = mongoose.Types.ObjectId.isValid(transcriptId)
        ? new mongoose.Types.ObjectId(transcriptId)
        : transcriptId;

    const chunks = await RagChunk.aggregate([
        {
            $vectorSearch: {
                index: "vector_index",
                path: "embedding",
                queryVector: queryEmbedding,
                numCandidates: vectorCandidates,
                limit: vectorLimit,
                filter: { transcriptId: tid }
            }
        },
        { $addFields: { score: { $meta: "vectorSearchScore" } } }
    ]);

    if (!chunks.length) return [];
    return mmrRerank(queryEmbedding, chunks, MMR_LAMBDA, MMR_FINAL_K);
}

function buildSystemPrompt(meetingCode) {
    return `You are an expert meeting assistant with access to the transcript and AI-generated summary of meeting "${meetingCode}".
Your job is to answer questions about this meeting accurately and concisely.

Rules:
- Only answer based on the provided context (transcript chunks and/or summary) enclosed inside the markup tags. If the answer isn't explicitly supported in the context, say so clearly.
- Context chunks labelled as summary come from the AI-generated meeting summary — use them for high-level questions about topics, emotions, pace, and key insights.
- Context chunks from the transcript contain the actual spoken content — prefer these for speaker attribution, quotes, and specific details.
- When quoting speakers, attribute them by name.
- If asked about timestamps or timing, use the time ranges provided in transcript chunks.
- Keep answers focused and professional.
- For follow-up questions, consider the conversation history.`;
}

function buildContextBlock(chunks) {
    if (!chunks.length) return "No relevant context found.";

    return chunks
        .map((c, i) => {
            if (c.chunkType === "summary") {
                return `--- Summary Context ${i + 1} ---\n${c.chunkText}`;
            }
            const timeRange = c.startSec != null
                ? ` [${formatTime(c.startSec)} – ${formatTime(c.endSec)}]`
                : "";
            const speakers = c.speakers?.length ? ` (${c.speakers.join(", ")})` : "";
            return `--- Transcript Context ${i + 1}${timeRange}${speakers} ---\n${c.chunkText}`;
        })
        .join("\n\n");
}

function formatTime(sec) {
    if (sec == null) return "?";
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}:${String(s).padStart(2, "0")}`;
}

export async function ragQueryService(req, res = null) {
    const reply = (status, body) => {
        if (res && !res.headersSent) res.status(status).json(body);
        return { status, body };
    };

    const idOrCode = String(req.params.id || "").trim();
    let question = String(req.body?.question || "").trim();

    log.info("ragQuery hit", { idOrCode, question: question.slice(0, 50), userId: req.user?._id || req.user?.id || null });

    if (!idOrCode) return reply(400, { success: false, message: "id or meetingCode required" });
    if (!question) return reply(400, { success: false, message: "question is required" });

    question = question.slice(0, 2000).replace(/<\/?[^>]+(>|$)/g, "");

    const { secret, userId, secretHash } = resolveAuth(req);
    log.info("ragQuery: auth resolved", { userId, hasSecret: !!secret });
    if (!secret && !userId) return reply(403, { success: false, message: "Not authorized" });

    if (userId && await checkRateLimit(RKEYS.queryRate(userId), RAG_QUERY_RATE_MAX, RAG_QUERY_RATE_WIN_SEC)) {
        log.warn("ragQuery: rate limited", { userId });
        return reply(429, { success: false, message: "RAG query rate limit reached (20/hour)" });
    }

    if (!GROQ_API_KEY) return reply(500, { success: false, message: "AI service not configured (GROQ_API_KEY missing)" });
    if (!NOMIC_API_KEY) return reply(500, { success: false, message: "Embedding service not configured (NOMIC_API_KEY missing)" });
    log.info("ragQuery: keys ok");

    const isMongoId = /^[a-f\d]{24}$/i.test(idOrCode);
    let doc = null;
    if (isMongoId) doc = await findTranscriptById(idOrCode);
    if (!doc) doc = await findTranscriptByCode(idOrCode.toUpperCase());
    if (!doc) return reply(404, { success: false, message: "Transcript not found" });
    log.info("ragQuery: doc found", { docId: String(doc._id), ownerId: doc.ownerId });

    if (!isAuthorized(doc, userId, secretHash)) {
        const isLegacyDoc = !doc.ownerId && !doc.hostSecretHash;
        log.warn("ragQuery: isAuthorized false", { userId, ownerId: doc.ownerId, isLegacyDoc });
        if (!isLegacyDoc) {
            return reply(403, { success: false, message: "Not authorized" });
        }
    }

    const transcriptId = String(doc._id);
    const indexStatus = await safeRedisGet(RKEYS.indexStatus(transcriptId));
    log.info("ragQuery: indexStatus", { indexStatus, transcriptId });
    if (!indexStatus || indexStatus === "indexing") {
        return reply(424, {
            success: false,
            message: indexStatus === "indexing"
                ? "Transcript is currently being indexed. Try again in a moment."
                : "Transcript not indexed yet. Call POST /rag/:id/index first.",
            indexStatus: indexStatus || "not_indexed",
        });
    }

    const t0 = Date.now();

    try {
        log.info("ragQuery: embedding start", { transcriptId });
        const queryEmbedding = await embedText(question, "search_query");
        log.info("ragQuery: embedding done, retrieving chunks", { transcriptId });
        const totalChunks = await RagChunk.countDocuments({ transcriptId });
        log.info("ragQuery: total chunks in DB", { totalChunks, transcriptId });
        const chunks = await retrieveChunks(transcriptId, queryEmbedding);
        log.info("ragQuery: chunks retrieved", { count: chunks.length, transcriptId });

        if (!chunks.length) {
            return {
                status: 200,
                body: {
                    success: true,
                    answer: "I couldn't find relevant information in this transcript to answer your question.",
                    sources: [],
                    sessionId: null,
                },
            };
        }

        let session = null;
        if (userId) {
            session = await RagSession.findOneAndUpdate(
                { transcriptId, userId },
                {
                    $setOnInsert: {
                        transcriptId,
                        meetingCode: doc.meetingCode,
                        userId,
                        contextWindowSize: SESSION_CONTEXT_WINDOW,
                    },
                    $set: { lastActivityAt: new Date() },
                },
                { upsert: true, new: true }
            );
        }

        const contextBlock = buildContextBlock(chunks);
        const historyMessages = session
            ? session.messages
                .slice(-SESSION_CONTEXT_WINDOW)
                .map((m) => ({ role: m.role, content: m.content }))
            : [];

        const messages = [
            { role: "system", content: buildSystemPrompt(doc.meetingCode) },
            ...historyMessages,
            {
                role: "user",
                content: `<context>\n${contextBlock}\n</context>\n\n<query>\n${question}\n</query>`,
            },
        ];

        const shouldStream = res && typeof res.write === "function";

        const llmRes = await fetch(GROQ_CHAT_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                Authorization: `Bearer ${GROQ_API_KEY}`,
            },
            body: JSON.stringify({
                model: LLM_MODEL,
                messages,
                temperature: LLM_TEMPERATURE,
                max_tokens: LLM_MAX_TOKENS,
                stream: shouldStream
            }),
        });

        if (!llmRes.ok) {
            const errBody = await llmRes.json().catch(() => ({}));
            log.error("Groq LLM error", { status: llmRes.status, msg: errBody?.error?.message });
            return { status: 502, body: { success: false, message: `AI provider error: ${llmRes.status}` } };
        }

        const sources = chunks.map((c) => ({
            chunkIndex: c.chunkIndex,
            speakers: c.speakers,
            startSec: c.startSec,
            endSec: c.endSec,
            preview: c.chunkText.slice(0, 150) + (c.chunkText.length > 150 ? "…" : ""),
            score: Math.round((c.score || 0) * 100) / 100,
        }));

        if (shouldStream) {
            res.setHeader("Content-Type", "text/event-stream");
            res.setHeader("Cache-Control", "no-cache");
            res.setHeader("Connection", "keep-alive");
            res.flushHeaders();

            res.write(`data: ${JSON.stringify({ sources, sessionId: session?._id?.toString() || null })}\n\n`);
            if (typeof res.flush === "function") res.flush();

            let fullAnswer = "";
            let buffer = "";

            for await (const rawChunk of llmRes.body) {
                buffer += Buffer.isBuffer(rawChunk)
                    ? rawChunk.toString("utf8")
                    : new TextDecoder().decode(rawChunk);

                const lines = buffer.split("\n");
                buffer = lines.pop() ?? "";

                for (const line of lines) {
                    const trimmed = line.trim();
                    if (!trimmed.startsWith("data: ")) continue;
                    const message = trimmed.slice(6).trim();
                    if (message === "[DONE]") continue;

                    try {
                        const parsed = JSON.parse(message);
                        const token = parsed.choices?.[0]?.delta?.content || "";
                        if (token) {
                            fullAnswer += token;
                            res.write(`data: ${JSON.stringify({ token })}\n\n`);
                        }
                    } catch (e) {
                    }
                }
            }

            if (buffer.trim().startsWith("data: ")) {
                const message = buffer.trim().slice(6).trim();
                if (message && message !== "[DONE]") {
                    try {
                        const parsed = JSON.parse(message);
                        const token = parsed.choices?.[0]?.delta?.content || "";
                        if (token) {
                            fullAnswer += token;
                            res.write(`data: ${JSON.stringify({ token })}\n\n`);
                        }
                    } catch (e) { }
                }
            }

            const latencyMs = Date.now() - t0;
            if (session) {
                const sourceChunkIds = chunks.map((c) => c._id).filter(Boolean);
                const estimatedTokens = estimateTokens(question) + estimateTokens(fullAnswer);

                await RagSession.findByIdAndUpdate(session._id, {
                    $push: {
                        messages: {
                            $each: [
                                { role: "user", content: question, sourceChunkIds: [], latencyMs: null },
                                { role: "assistant", content: fullAnswer, sourceChunkIds, latencyMs },
                            ],
                            $slice: -30
                        },
                    },
                    $inc: { totalTokensUsed: estimatedTokens },
                    $set: { lastActivityAt: new Date() },
                });
            }

            res.write("data: [DONE]\n\n");
            res.end();
            return;
        }

        const llmData = await llmRes.json();
        const answer = llmData.choices?.[0]?.message?.content?.trim() || "No answer generated.";
        const latencyMs = Date.now() - t0;

        if (session) {
            const sourceChunkIds = chunks.map((c) => c._id).filter(Boolean);
            const estimatedTokens = estimateTokens(question) + estimateTokens(answer);

            await RagSession.findByIdAndUpdate(session._id, {
                $push: {
                    messages: {
                        $each: [
                            { role: "user", content: question, sourceChunkIds: [], latencyMs: null },
                            { role: "assistant", content: answer, sourceChunkIds, latencyMs },
                        ],
                        $slice: -30
                    },
                },
                $inc: { totalTokensUsed: estimatedTokens },
                $set: { lastActivityAt: new Date() },
            });
        }

        return {
            status: 200,
            body: {
                success: true,
                answer,
                sources,
                sessionId: session?._id?.toString() || null,
                latencyMs,
            },
        };
    } catch (err) {
        log.error("ragQuery error", { err: err.message });
        return { status: 500, body: { success: false, message: "RAG query failed" } };
    }
}

export async function getRagSessionService(req) {
    const idOrCode = String(req.params.id || "").trim();
    const { userId } = resolveAuth(req);
    if (!userId) return { status: 403, body: { success: false, message: "Auth required for session history" } };

    const isMongoId = /^[a-f\d]{24}$/i.test(idOrCode);
    let doc = null;
    if (isMongoId) doc = await findTranscriptById(idOrCode);
    if (!doc) doc = await findTranscriptByCode(idOrCode.toUpperCase());
    if (!doc) return { status: 404, body: { success: false, message: "Transcript not found" } };

    const session = await RagSession.findOne(
        { transcriptId: String(doc._id), userId },
        { messages: 1, totalTokensUsed: 1, lastActivityAt: 1 }
    ).lean();

    return {
        status: 200,
        body: {
            success: true,
            session: session
                ? {
                    messages: session.messages,
                    totalTokensUsed: session.totalTokensUsed,
                    lastActivityAt: session.lastActivityAt,
                }
                : null,
        },
    };
}

export async function clearRagSessionService(req) {
    const idOrCode = String(req.params.id || "").trim();
    const { userId } = resolveAuth(req);
    if (!userId) return { status: 403, body: { success: false, message: "Auth required" } };

    const isMongoId = /^[a-f\d]{24}$/i.test(idOrCode);
    let doc = null;
    if (isMongoId) doc = await findTranscriptById(idOrCode);
    if (!doc) doc = await findTranscriptByCode(idOrCode.toUpperCase());
    if (!doc) return { status: 404, body: { success: false, message: "Transcript not found" } };

    await RagSession.deleteOne({ transcriptId: String(doc._id), userId });

    return { status: 200, body: { success: true, message: "Session cleared" } };
}

export async function deleteRagIndexService(transcriptId) {
    await RagChunk.deleteMany({ transcriptId });
    await safeRedisDel(RKEYS.indexStatus(transcriptId));
    log.info("rag index deleted", { transcriptId });
}