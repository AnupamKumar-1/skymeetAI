import { safeRedisGet, safeRedisSet } from "./redis.utils.js";

export function blacklistKey(token) {
    return `blacklist:${token}`;
}

export async function blacklistAccessToken(token, ttlSec) {
    if (!token || !ttlSec || ttlSec <= 0) return null;
    return safeRedisSet(blacklistKey(token), "1", { EX: ttlSec });
}

export async function isTokenBlacklisted(token) {
    if (!token) return false;
    const value = await safeRedisGet(blacklistKey(token));
    return value !== null;
}
