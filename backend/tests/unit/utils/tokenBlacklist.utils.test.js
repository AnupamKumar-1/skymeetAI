import { describe, it, expect, vi, beforeEach } from "vitest";

vi.mock("../../../src/utils/redis.utils.js", () => ({
    safeRedisGet: vi.fn(),
    safeRedisSet: vi.fn(),
}));

import { safeRedisGet, safeRedisSet } from "../../../src/utils/redis.utils.js";
import {
    blacklistKey,
    blacklistAccessToken,
    isTokenBlacklisted,
} from "../../../src/utils/tokenBlacklist.utils.js";

beforeEach(() => {
    vi.clearAllMocks();
});

describe("blacklistKey", () => {
    it("prefixes the token with blacklist:", () => {
        expect(blacklistKey("abc.def.ghi")).toBe("blacklist:abc.def.ghi");
    });
});

describe("blacklistAccessToken", () => {
    it("sets the token in redis with the given ttl", async () => {
        safeRedisSet.mockResolvedValue("OK");
        const result = await blacklistAccessToken("token-1", 3600);
        expect(safeRedisSet).toHaveBeenCalledWith("blacklist:token-1", "1", { EX: 3600 });
        expect(result).toBe("OK");
    });

    it("does nothing when token is missing", async () => {
        const result = await blacklistAccessToken(null, 3600);
        expect(safeRedisSet).not.toHaveBeenCalled();
        expect(result).toBeNull();
    });

    it("does nothing when ttl is not positive", async () => {
        const result = await blacklistAccessToken("token-1", 0);
        expect(safeRedisSet).not.toHaveBeenCalled();
        expect(result).toBeNull();
    });
});

describe("isTokenBlacklisted", () => {
    it("returns true when redis has a value for the token", async () => {
        safeRedisGet.mockResolvedValue("1");
        const result = await isTokenBlacklisted("token-1");
        expect(safeRedisGet).toHaveBeenCalledWith("blacklist:token-1");
        expect(result).toBe(true);
    });

    it("returns false when redis has no value for the token", async () => {
        safeRedisGet.mockResolvedValue(null);
        const result = await isTokenBlacklisted("token-1");
        expect(result).toBe(false);
    });

    it("returns false when token is missing", async () => {
        const result = await isTokenBlacklisted(null);
        expect(safeRedisGet).not.toHaveBeenCalled();
        expect(result).toBe(false);
    });
});
