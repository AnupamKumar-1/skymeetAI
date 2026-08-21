import { describe, it, expect } from "vitest";
import crypto from "crypto";
import { hashHostSecret, hostSecretsMatch } from "../../../src/utils/hostSecret.utils.js";

describe("hashHostSecret", () => {
    it("produces a sha256 hex digest of the input", () => {
        const raw = "super-secret";
        const expected = crypto.createHash("sha256").update(raw).digest("hex");
        expect(hashHostSecret(raw)).toBe(expected);
    });

    it("coerces non-string input to a string before hashing", () => {
        const expected = crypto.createHash("sha256").update("123").digest("hex");
        expect(hashHostSecret(123)).toBe(expected);
    });
});

describe("hostSecretsMatch", () => {
    it("returns true for identical hashes", () => {
        const hash = hashHostSecret("abc");
        expect(hostSecretsMatch(hash, hash)).toBe(true);
    });

    it("returns false for different hashes of the same length", () => {
        const hashA = hashHostSecret("abc");
        const hashB = hashHostSecret("xyz");
        expect(hostSecretsMatch(hashA, hashB)).toBe(false);
    });

    it("returns false when lengths differ instead of throwing", () => {
        expect(hostSecretsMatch("short", "a-much-longer-value")).toBe(false);
    });

    it("returns false when either hash is missing", () => {
        expect(hostSecretsMatch(null, "abc")).toBe(false);
        expect(hostSecretsMatch("abc", undefined)).toBe(false);
        expect(hostSecretsMatch("", "")).toBe(false);
    });
});
