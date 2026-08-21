import { describe, it, expect, vi, beforeEach } from "vitest";
import httpStatus from "http-status";

vi.mock("../../../src/utils/redis.utils.js", () => ({
    makeLogger: vi.fn(() => ({ info: vi.fn(), warn: vi.fn(), error: vi.fn() })),
    safeRedisGet: vi.fn(),
    safeRedisSet: vi.fn(),
    safeRedisDel: vi.fn(),
    safeRedisIncr: vi.fn(),
    safeRedisExpire: vi.fn(),
    batchDel: vi.fn(),
    isRateLimited: vi.fn().mockResolvedValue(false),
}));

vi.mock("../../../src/utils/tokenBlacklist.utils.js", () => ({
    blacklistAccessToken: vi.fn(),
}));

vi.mock("../../../src/data-access/user.repository.js", () => ({
    findUserByUsername: vi.fn(),
    findUserById: vi.fn(),
    findUserByUsernameLean: vi.fn(),
    createUser: vi.fn(),
    findMeetingsByUser: vi.fn(),
    findMeetingByCode: vi.fn(),
    createMeeting: vi.fn(),
    findMeetingForParticipant: vi.fn(),
    saveMeeting: vi.fn(),
    findMeetingsForUser: vi.fn(),
    upsertMeetingByCode: vi.fn(),
    ensureMeetingIndexes: vi.fn(),
    findUserProfileById: vi.fn(),
    updateUserProfile: vi.fn(),
    updateUserAvatar: vi.fn(),
    removeUserAvatar: vi.fn(),
    findUserWithPasswordById: vi.fn(),
    updateUserPassword: vi.fn(),
}));

vi.mock("../../../src/config/cloudinary.js", () => ({
    uploadAvatar: vi.fn(),
    destroyAvatar: vi.fn(),
    AVATAR_MAX_BYTES: 5 * 1024 * 1024,
    ALLOWED_FORMATS: ["png", "jpg", "jpeg"],
}));

import { safeRedisGet, safeRedisSet, safeRedisDel } from "../../../src/utils/redis.utils.js";
import { blacklistAccessToken } from "../../../src/utils/tokenBlacklist.utils.js";
import { findUserById } from "../../../src/data-access/user.repository.js";
import { refreshTokenService, logoutService } from "../../../src/services/user.service.js";

beforeEach(() => {
    vi.clearAllMocks();
});

describe("refreshTokenService", () => {
    it("returns 400 when the refresh cookie is missing", async () => {
        const req = { cookies: {} };
        const result = await refreshTokenService(req);
        expect(result.status).toBe(httpStatus.BAD_REQUEST);
        expect(result.body.success).toBe(false);
    });

    it("returns 401 when the refresh token is not found in redis", async () => {
        safeRedisGet.mockResolvedValue(null);
        const req = { cookies: { refreshToken: "missing-token" } };
        const result = await refreshTokenService(req);
        expect(safeRedisGet).toHaveBeenCalledWith("refresh:missing-token");
        expect(result.status).toBe(httpStatus.UNAUTHORIZED);
    });

    it("returns 401 and cleans up when the referenced user no longer exists", async () => {
        safeRedisGet.mockResolvedValue(JSON.stringify({ _id: "user-1" }));
        findUserById.mockResolvedValue(null);
        const req = { cookies: { refreshToken: "orphan-token" } };
        const result = await refreshTokenService(req);
        expect(result.status).toBe(httpStatus.UNAUTHORIZED);
        expect(safeRedisDel).toHaveBeenCalledWith("refresh:orphan-token");
    });

    it("rotates the refresh token and issues a new access token on success", async () => {
        safeRedisGet.mockResolvedValue(JSON.stringify({ _id: "user-1" }));
        findUserById.mockResolvedValue({
            _id: "user-1",
            username: "alice",
            name: "Alice",
        });
        const req = { cookies: { refreshToken: "old-token" } };

        const result = await refreshTokenService(req);

        expect(result.status).toBe(httpStatus.OK);
        expect(result.body.success).toBe(true);
        expect(result.body.accessToken).toBeTruthy();
        expect(safeRedisDel).toHaveBeenCalledWith("refresh:old-token");
        expect(result.cookies.refreshToken.value).not.toBe("old-token");
        expect(safeRedisSet).toHaveBeenCalledWith(
            expect.stringMatching(/^refresh:/),
            expect.any(String),
            expect.objectContaining({ EX: expect.any(Number) })
        );
    });
});

describe("logoutService", () => {
    it("blacklists the bearer access token when present", async () => {
        const req = {
            headers: { authorization: "Bearer access-token-value" },
            cookies: {},
        };

        const result = await logoutService(req);

        expect(blacklistAccessToken).toHaveBeenCalledWith("access-token-value", expect.any(Number));
        expect(result.status).toBe(httpStatus.OK);
        expect(result.body.success).toBe(true);
    });

    it("deletes the refresh token cookie value from redis when present", async () => {
        const req = {
            headers: {},
            cookies: { refreshToken: "session-refresh-token" },
        };

        await logoutService(req);

        expect(safeRedisDel).toHaveBeenCalledWith("refresh:session-refresh-token");
    });

    it("does not blacklist anything when there is no bearer token", async () => {
        const req = { headers: {}, cookies: {} };
        await logoutService(req);
        expect(blacklistAccessToken).not.toHaveBeenCalled();
    });
});
