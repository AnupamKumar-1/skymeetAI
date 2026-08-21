import { describe, it, expect, vi, beforeEach } from "vitest";

vi.mock("../../../src/models/user.model.js", () => ({
    User: {
        findById: vi.fn(),
    },
}));

vi.mock("../../../src/utils/tokenBlacklist.utils.js", () => ({
    isTokenBlacklisted: vi.fn(),
}));

import passport from "../../../config/passport.js";
import { User } from "../../../src/models/user.model.js";
import { isTokenBlacklisted } from "../../../src/utils/tokenBlacklist.utils.js";

function runVerify(req, payload) {
    return new Promise((resolve, reject) => {
        const strategy = passport._strategy("jwt");
        strategy._verify(req, payload, (err, user, info) => {
            if (err) return reject(err);
            resolve({ user, info });
        });
    });
}

function fakeReq(token) {
    return { headers: { authorization: token ? `Bearer ${token}` : undefined } };
}

beforeEach(() => {
    vi.clearAllMocks();
});

describe("passport jwt strategy", () => {
    it("rejects payloads without a sub claim", async () => {
        const { user } = await runVerify(fakeReq("token-a"), {});
        expect(user).toBe(false);
        expect(isTokenBlacklisted).not.toHaveBeenCalled();
    });

    it("rejects blacklisted access tokens without querying the user", async () => {
        isTokenBlacklisted.mockResolvedValue(true);
        const { user } = await runVerify(fakeReq("blacklisted-token"), { sub: "user-1" });
        expect(isTokenBlacklisted).toHaveBeenCalledWith("blacklisted-token");
        expect(user).toBe(false);
        expect(User.findById).not.toHaveBeenCalled();
    });

    it("rejects when the user no longer exists", async () => {
        isTokenBlacklisted.mockResolvedValue(false);
        User.findById.mockReturnValue({
            select: () => ({ lean: () => Promise.resolve(null) }),
        });
        const { user } = await runVerify(fakeReq("valid-token"), { sub: "user-1" });
        expect(user).toBe(false);
    });

    it("resolves the user when the token is valid and not blacklisted", async () => {
        isTokenBlacklisted.mockResolvedValue(false);
        User.findById.mockReturnValue({
            select: () => ({
                lean: () =>
                    Promise.resolve({ _id: "user-1", username: "alice", name: "Alice" }),
            }),
        });
        const { user } = await runVerify(fakeReq("valid-token"), { sub: "user-1" });
        expect(user).toEqual({
            _id: "user-1",
            id: "user-1",
            username: "alice",
            name: "Alice",
        });
    });
});
