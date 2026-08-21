import { describe, it, expect, vi, beforeEach } from "vitest";
import {
    indexTranscript,
    getIndexStatus,
    ragQuery,
    getRagSession,
    clearRagSession,
} from "../../../src/controllers/rag.controller.js";

import {
    indexTranscriptService,
    getIndexStatusService,
    ragQueryService,
    getRagSessionService,
    clearRagSessionService,
} from "../../../src/services/rag.service.js";

vi.mock("../../../src/services/rag.service.js", () => ({
    indexTranscriptService: vi.fn(),
    getIndexStatusService: vi.fn(),
    ragQueryService: vi.fn(),
    getRagSessionService: vi.fn(),
    clearRagSessionService: vi.fn(),
}));

function mockRes() {
    const res = {};
    res.status = vi.fn().mockReturnValue(res);
    res.json = vi.fn().mockReturnValue(res);
    res.end = vi.fn().mockReturnValue(res);
    res.headersSent = false;
    return res;
}

function mockReq() {
    return {};
}

beforeEach(() => {
    vi.clearAllMocks();
});

describe("indexTranscript", () => {
    it("returns success response from service", async () => {
        const req = mockReq();
        const res = mockRes();
        indexTranscriptService.mockResolvedValue({
            status: 200,
            body: { success: true, message: "Indexed" },
        });

        await indexTranscript(req, res);

        expect(indexTranscriptService).toHaveBeenCalledWith(req);
        expect(res.status).toHaveBeenCalledWith(200);
        expect(res.json).toHaveBeenCalledWith({ success: true, message: "Indexed" });
    });

    it("returns 500 when service throws", async () => {
        const req = mockReq();
        const res = mockRes();
        indexTranscriptService.mockRejectedValue(new Error("fail"));

        await indexTranscript(req, res);

        expect(res.status).toHaveBeenCalledWith(500);
        expect(res.json).toHaveBeenCalledWith({ success: false, message: "Server error" });
    });
});

describe("getIndexStatus", () => {
    it("returns success response from service", async () => {
        const req = mockReq();
        const res = mockRes();
        getIndexStatusService.mockResolvedValue({
            status: 200,
            body: { success: true, status: "ready" },
        });

        await getIndexStatus(req, res);

        expect(getIndexStatusService).toHaveBeenCalledWith(req);
        expect(res.status).toHaveBeenCalledWith(200);
        expect(res.json).toHaveBeenCalledWith({ success: true, status: "ready" });
    });

    it("returns 500 when service throws", async () => {
        const req = mockReq();
        const res = mockRes();
        getIndexStatusService.mockRejectedValue(new Error("fail"));

        await getIndexStatus(req, res);

        expect(res.status).toHaveBeenCalledWith(500);
        expect(res.json).toHaveBeenCalledWith({ success: false, message: "Server error" });
    });
});

describe("ragQuery", () => {
    it("calls ragQueryService with req and res", async () => {
        const req = mockReq();
        const res = mockRes();
        ragQueryService.mockResolvedValue(undefined);

        await ragQuery(req, res);

        expect(ragQueryService).toHaveBeenCalledWith(req, res);
    });

    it("returns 500 when service throws and headersSent is false", async () => {
        const req = mockReq();
        const res = mockRes();
        res.headersSent = false;
        ragQueryService.mockRejectedValue(new Error("fail"));

        await ragQuery(req, res);

        expect(res.status).toHaveBeenCalledWith(500);
        expect(res.json).toHaveBeenCalledWith({ success: false, message: "Server error" });
    });

    it("calls res.end when service throws and headersSent is true", async () => {
        const req = mockReq();
        const res = mockRes();
        res.headersSent = true;
        ragQueryService.mockRejectedValue(new Error("fail"));

        await ragQuery(req, res);

        expect(res.status).not.toHaveBeenCalled();
        expect(res.json).not.toHaveBeenCalled();
        expect(res.end).toHaveBeenCalled();
    });
});

describe("getRagSession", () => {
    it("returns success response from service", async () => {
        const req = mockReq();
        const res = mockRes();
        getRagSessionService.mockResolvedValue({
            status: 200,
            body: { success: true, session: {} },
        });

        await getRagSession(req, res);

        expect(getRagSessionService).toHaveBeenCalledWith(req);
        expect(res.status).toHaveBeenCalledWith(200);
        expect(res.json).toHaveBeenCalledWith({ success: true, session: {} });
    });

    it("returns 500 when service throws", async () => {
        const req = mockReq();
        const res = mockRes();
        getRagSessionService.mockRejectedValue(new Error("fail"));

        await getRagSession(req, res);

        expect(res.status).toHaveBeenCalledWith(500);
        expect(res.json).toHaveBeenCalledWith({ success: false, message: "Server error" });
    });
});

describe("clearRagSession", () => {
    it("returns success response from service", async () => {
        const req = mockReq();
        const res = mockRes();
        clearRagSessionService.mockResolvedValue({
            status: 200,
            body: { success: true, message: "Cleared" },
        });

        await clearRagSession(req, res);

        expect(clearRagSessionService).toHaveBeenCalledWith(req);
        expect(res.status).toHaveBeenCalledWith(200);
        expect(res.json).toHaveBeenCalledWith({ success: true, message: "Cleared" });
    });

    it("returns 500 when service throws", async () => {
        const req = mockReq();
        const res = mockRes();
        clearRagSessionService.mockRejectedValue(new Error("fail"));

        await clearRagSession(req, res);

        expect(res.status).toHaveBeenCalledWith(500);
        expect(res.json).toHaveBeenCalledWith({ success: false, message: "Server error" });
    });
});