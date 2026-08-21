import { describe, it, expect, vi, beforeEach } from "vitest";

vi.mock("../../../src/models/meeting.model.js", () => ({
    Meeting: {
        verifyHostSecret: vi.fn(),
    },
}));

import { Meeting } from "../../../src/models/meeting.model.js";
import { verifyProxyAuth } from "../../../src/routes/transcriptProxy.routes.js";

beforeEach(() => {
    vi.clearAllMocks();
});

describe("verifyProxyAuth", () => {
    it("returns false when meetingCode is missing", async () => {
        const result = await verifyProxyAuth("", "some-secret");
        expect(result).toBe(false);
        expect(Meeting.verifyHostSecret).not.toHaveBeenCalled();
    });

    it("returns false when hostSecret is missing", async () => {
        const result = await verifyProxyAuth("ABCD1234", "");
        expect(result).toBe(false);
        expect(Meeting.verifyHostSecret).not.toHaveBeenCalled();
    });

    it("returns false when Meeting.verifyHostSecret resolves null", async () => {
        Meeting.verifyHostSecret.mockResolvedValue(null);
        const result = await verifyProxyAuth("ABCD1234", "wrong-secret");
        expect(Meeting.verifyHostSecret).toHaveBeenCalledWith("ABCD1234", "wrong-secret");
        expect(result).toBe(false);
    });

    it("returns true when Meeting.verifyHostSecret resolves a meeting", async () => {
        Meeting.verifyHostSecret.mockResolvedValue({ meetingCode: "ABCD1234" });
        const result = await verifyProxyAuth("ABCD1234", "correct-secret");
        expect(result).toBe(true);
    });
});
