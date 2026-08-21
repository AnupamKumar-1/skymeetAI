import crypto from "crypto";

export function hashHostSecret(rawSecret) {
    return crypto.createHash("sha256").update(String(rawSecret)).digest("hex");
}

export function hostSecretsMatch(hashA, hashB) {
    if (!hashA || !hashB) return false;
    const bufA = Buffer.from(String(hashA), "utf8");
    const bufB = Buffer.from(String(hashB), "utf8");
    if (bufA.length !== bufB.length) return false;
    return crypto.timingSafeEqual(bufA, bufB);
}
