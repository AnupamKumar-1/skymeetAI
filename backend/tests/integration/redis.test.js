import { createClient } from "redis";
import { strict as assert } from "assert";
import crypto from "crypto";
import { execSync, spawn } from "child_process";

const REDIS_URL = process.env.REDIS_URL || "redis://localhost:6379";
const SKIP_RECOVERY = process.env.SKIP_RECOVERY_TESTS === "true";

function makeClient(name) {
    const c = createClient({
        url: REDIS_URL,
        socket: {
            tls: REDIS_URL.startsWith("rediss://"),
            reconnectStrategy: (r) => Math.min(r * 100 + Math.random() * 100, 3000),
        },
    });
    c.on("error", (err) => {
        if (!err.message.includes("ECONNREFUSED")) {
            console.error(`[redis:${name}]`, err.message);
        }
    });
    return c;
}

const LOCK_TTL_MS = 10000;
const LOCK_RETRY_INTERVAL_MS = 50;
const LOCK_MAX_WAIT_MS = 8000;

function lockKey(code) {
    return `lock:room:${code}`;
}

async function acquireLock(client, code, token) {
    const result = await client.set(lockKey(code), token, {
        NX: true,
        PX: LOCK_TTL_MS,
    });
    return result === "OK";
}

async function releaseLock(client, code, token) {
    const script = `
    if redis.call("GET", KEYS[1]) == ARGV[1]
    then return redis.call("DEL", KEYS[1])
    else return 0
    end
  `;
    await client.eval(script, { keys: [lockKey(code)], arguments: [token] });
}

async function withRoomLock(client, code, fn) {
    const token = crypto.randomUUID();
    const deadline = Date.now() + LOCK_MAX_WAIT_MS;
    while (true) {
        const acquired = await acquireLock(client, code, token);
        if (acquired) break;
        if (Date.now() >= deadline) throw new Error(`timeout acquiring lock for room: ${code}`);
        const jitter = Math.random() * 50;
        await new Promise((r) => setTimeout(r, LOCK_RETRY_INTERVAL_MS + jitter));
    }
    try {
        return await fn();
    } finally {
        await releaseLock(client, code, token);
    }
}

async function isRateLimited(client, key, max, windowSec) {
    try {
        const script = `
      local count = redis.call("INCR", KEYS[1])
      if count == 1 then redis.call("EXPIRE", KEYS[1], ARGV[1]) end
      return count
    `;
        const count = await client.eval(script, {
            keys: [key],
            arguments: [String(windowSec)],
        });
        return count > max;
    } catch {
        return false;
    }
}

async function sleep(ms) {
    return new Promise((r) => setTimeout(r, ms));
}

async function waitForReady(client, timeoutMs = 15000) {
    const deadline = Date.now() + timeoutMs;
    if (client.isReady) return;
    return new Promise((resolve, reject) => {
        const check = setInterval(() => {
            if (client.isReady) {
                clearInterval(check);
                clearTimeout(timer);
                resolve();
            } else if (Date.now() >= deadline) {
                clearInterval(check);
                reject(new Error("Redis ready timeout"));
            }
        }, 100);
        const timer = setTimeout(() => {
            clearInterval(check);
            reject(new Error("Redis ready timeout"));
        }, timeoutMs);
        client.once("ready", () => {
            clearInterval(check);
            clearTimeout(timer);
            resolve();
        });
    });
}

async function scanKeys(client, pattern) {
    const keys = [];
    let cursor = "0";
    do {
        const reply = await client.scan(cursor, { MATCH: pattern, COUNT: "100" });
        cursor = String(reply.cursor);
        keys.push(...reply.keys);
    } while (cursor !== "0");
    return keys;
}

function isRedisRunning() {
    try {
        execSync("redis-cli ping", { stdio: "pipe" });
        return true;
    } catch {
        return false;
    }
}

function stopRedis() {
    try {
        execSync("redis-cli shutdown nosave", { stdio: "pipe" });
    } catch { }
}

function startRedis() {

    const proc = spawn("redis-server", [], {
        stdio: "ignore",
        detached: true,
    });
    proc.unref();
}

async function waitForRedisCli(timeoutMs = 12000) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        try {
            execSync("redis-cli ping", { stdio: "pipe" });
            return; // Redis is up
        } catch {
            await sleep(150);
        }
    }
    throw new Error("Redis did not come back up in time");
}

const results = { passed: 0, failed: 0, skipped: 0 };

async function test(name, fn) {
    try {
        await fn();
        console.log(`  ✅ ${name}`);
        results.passed++;
    } catch (err) {
        console.log(`  ❌ ${name}`);
        console.log(`     ${err.message}`);
        results.failed++;
    }
}

async function runBasicTests(client) {
    console.log("\n── Basic Cache ──");

    await test("set and get a value", async () => {
        await client.set("test:basic", "hello", { EX: 10 });
        const val = await client.get("test:basic");
        assert.equal(val, "hello");
    });

    await test("key expires after TTL", async () => {
        await client.set("test:ttl", "bye", { PX: 200 });
        await sleep(300);
        const val = await client.get("test:ttl");
        assert.equal(val, null);
    });

    await test("del removes key", async () => {
        await client.set("test:del", "x");
        await client.del("test:del");
        const val = await client.get("test:del");
        assert.equal(val, null);
    });

    await test("incr increments atomically", async () => {
        await client.del("test:counter");
        const a = await client.incr("test:counter");
        const b = await client.incr("test:counter");
        assert.equal(a, 1);
        assert.equal(b, 2);
    });

    await test("set JSON and parse back", async () => {
        const obj = { userId: "abc", name: "Pratibha", score: 42 };
        await client.set("test:json", JSON.stringify(obj), { EX: 10 });
        const raw = await client.get("test:json");
        const parsed = JSON.parse(raw);
        assert.deepEqual(parsed, obj);
    });
}

async function runLockTests(client) {
    console.log("\n── Distributed Lock ──");

    await test("acquires and releases lock", async () => {
        const code = `room_${crypto.randomUUID()}`;
        let inside = false;
        await withRoomLock(client, code, async () => { inside = true; });
        assert.ok(inside);
        const remaining = await client.get(lockKey(code));
        assert.equal(remaining, null);
    });

    await test("second caller waits while lock is held", async () => {
        const code = `room_${crypto.randomUUID()}`;
        const order = [];
        const first = withRoomLock(client, code, async () => {
            order.push("first-start");
            await sleep(300);
            order.push("first-end");
        });
        await sleep(50);
        const second = withRoomLock(client, code, async () => { order.push("second"); });
        await Promise.all([first, second]);
        assert.deepEqual(order, ["first-start", "first-end", "second"]);
    });

    await test("only lock owner can release", async () => {
        const code = `room_${crypto.randomUUID()}`;
        const token = crypto.randomUUID();
        await acquireLock(client, code, token);
        await releaseLock(client, code, "wrong-token");
        const val = await client.get(lockKey(code));
        assert.equal(val, token);
        await releaseLock(client, code, token);
    });

    await test("lock times out if not acquired", async () => {
        const code = `room_${crypto.randomUUID()}`;
        const token = crypto.randomUUID();
        await acquireLock(client, code, token);
        try {
            await withRoomLock(client, code, async () => { });
            assert.fail("should have thrown");
        } catch (err) {
            assert.ok(err.message.includes("timeout"));
        } finally {
            await releaseLock(client, code, token);
        }
    });

    await test("concurrent locks on different rooms dont block", async () => {
        const codeA = `room_${crypto.randomUUID()}`;
        const codeB = `room_${crypto.randomUUID()}`;
        const res = [];
        await Promise.all([
            withRoomLock(client, codeA, async () => { await sleep(100); res.push("A"); }),
            withRoomLock(client, codeB, async () => { await sleep(100); res.push("B"); }),
        ]);
        assert.ok(res.includes("A") && res.includes("B"));
    });
}

async function runRateLimitTests(client) {
    console.log("\n── Rate Limiting ──");

    await test("allows requests under the limit", async () => {
        const key = `rate:test:${crypto.randomUUID()}`;
        for (let i = 0; i < 5; i++) {
            const limited = await isRateLimited(client, key, 5, 10);
            assert.equal(limited, false);
        }
    });

    await test("blocks requests over the limit", async () => {
        const key = `rate:test:${crypto.randomUUID()}`;
        for (let i = 0; i < 5; i++) await isRateLimited(client, key, 5, 10);
        const limited = await isRateLimited(client, key, 5, 10);
        assert.equal(limited, true);
    });

    await test("different keys have independent counters", async () => {
        const key1 = `rate:test:${crypto.randomUUID()}`;
        const key2 = `rate:test:${crypto.randomUUID()}`;
        for (let i = 0; i < 5; i++) await isRateLimited(client, key1, 5, 10);
        const limited = await isRateLimited(client, key2, 5, 10);
        assert.equal(limited, false);
    });

    await test("counter resets after window", async () => {
        const key = `rate:test:${crypto.randomUUID()}`;
        for (let i = 0; i < 5; i++) await isRateLimited(client, key, 5, 1);
        await sleep(1100);
        const limited = await isRateLimited(client, key, 5, 1);
        assert.equal(limited, false);
    });
}

async function runPubSubTests(pub, sub) {
    console.log("\n── Pub/Sub ──");

    await test("message published on one client received by subscriber", async () => {
        const channel = `test:channel:${crypto.randomUUID()}`;
        const received = [];
        await sub.subscribe(channel, (msg) => received.push(msg));
        await sleep(50);
        await pub.publish(channel, "hello");
        await sleep(100);
        await sub.unsubscribe(channel);
        assert.ok(received.includes("hello"));
    });

    await test("multiple messages received in order", async () => {
        const channel = `test:channel:${crypto.randomUUID()}`;
        const received = [];
        await sub.subscribe(channel, (msg) => received.push(msg));
        await sleep(50);
        await pub.publish(channel, "one");
        await pub.publish(channel, "two");
        await pub.publish(channel, "three");
        await sleep(150);
        await sub.unsubscribe(channel);
        assert.deepEqual(received, ["one", "two", "three"]);
    });

    await test("unsubscribed channel stops receiving", async () => {
        const channel = `test:channel:${crypto.randomUUID()}`;
        const received = [];
        await sub.subscribe(channel, (msg) => received.push(msg));
        await sleep(50);
        await pub.publish(channel, "before");
        await sleep(100);
        await sub.unsubscribe(channel);
        await pub.publish(channel, "after");
        await sleep(100);
        assert.ok(received.includes("before"));
        assert.ok(!received.includes("after"));
    });

    await test("two subscribers both receive the same message", async () => {
        const sub2 = makeClient("sub2-test");
        await sub2.connect();
        const channel = `test:channel:${crypto.randomUUID()}`;
        const recv1 = [];
        const recv2 = [];
        await sub.subscribe(channel, (msg) => recv1.push(msg));
        await sub2.subscribe(channel, (msg) => recv2.push(msg));
        await sleep(50);
        await pub.publish(channel, "broadcast");
        await sleep(150);
        await sub.unsubscribe(channel);
        await sub2.unsubscribe(channel);
        await sub2.disconnect();
        assert.ok(recv1.includes("broadcast"));
        assert.ok(recv2.includes("broadcast"));
    });
}

async function runBatchDelTest(client) {
    console.log("\n── Batch Operations ──");

    await test("batch delete removes multiple keys", async () => {
        const keys = ["batch:a", "batch:b", "batch:c"];
        for (const k of keys) await client.set(k, "val");
        const multi = client.multi();
        keys.forEach((k) => multi.del(k));
        await multi.exec();
        for (const k of keys) {
            const val = await client.get(k);
            assert.equal(val, null);
        }
    });

    await test("multi exec is atomic", async () => {
        await client.set("atomic:a", "1");
        await client.set("atomic:b", "2");
        const multi = client.multi();
        multi.incr("atomic:a");
        multi.incr("atomic:b");
        const res = await multi.exec();
        assert.equal(res[0], 2);
        assert.equal(res[1], 3);
    });
}

async function runRecoveryTests() {
    console.log("\n── Recovery After Redis Death ──");

    if (!isRedisRunning()) {
        console.log("  ⚠ Redis not running locally via redis-cli — skipping recovery tests");
        results.skipped += 5;
        return;
    }

    await test("client reconnects automatically after redis restart", async () => {
        const client = makeClient("recovery-test");
        await client.connect();

        stopRedis();
        await sleep(300);
        startRedis();
        await waitForRedisCli();

        await waitForReady(client);

        await client.set("recovery:after", "alive");
        const val = await client.get("recovery:after");
        assert.equal(val, "alive");
        await client.disconnect();
    });

    await test("pub client reconnects and can publish after restart", async () => {
        const pub = makeClient("pub-recovery");
        const sub = makeClient("sub-recovery");
        await pub.connect();
        await sub.connect();

        const channel = `recovery:channel:${crypto.randomUUID()}`;
        const received = [];
        await sub.subscribe(channel, (msg) => received.push(msg));

        stopRedis();
        await sleep(300);
        startRedis();
        await waitForRedisCli();

        await sub.unsubscribe(channel);
        await Promise.all([waitForReady(pub), waitForReady(sub)]);

        await sub.subscribe(channel, (msg) => received.push(msg));
        await pub.publish(channel, "after-restart");
        await sleep(200);

        await sub.unsubscribe(channel);
        await pub.disconnect();
        await sub.disconnect();

        assert.ok(received.includes("after-restart"));
    });

    await test("lock can be acquired after redis restart", async () => {
        const client = makeClient("lock-recovery");
        await client.connect();

        stopRedis();
        await sleep(300);
        startRedis();
        await waitForRedisCli();

        await waitForReady(client);

        const code = `room_${crypto.randomUUID()}`;
        let ran = false;
        await withRoomLock(client, code, async () => { ran = true; });
        assert.ok(ran);
        await client.disconnect();
    });

    await test("rate limiter works after redis restart", async () => {
        const client = makeClient("rate-recovery");
        await client.connect();

        stopRedis();
        await sleep(300);
        startRedis();
        await waitForRedisCli();

        await waitForReady(client);

        const key = `rate:recovery:${crypto.randomUUID()}`;
        const limited = await isRateLimited(client, key, 5, 10);
        assert.equal(limited, false);
        await client.disconnect();
    });

    await test("publish during outage does not crash, resumes after restart", async () => {
        const pub = makeClient("partition-pub");
        const sub = makeClient("partition-sub");
        await pub.connect();
        await sub.connect();

        const channel = `partition:channel:${crypto.randomUUID()}`;
        const received = [];
        await sub.subscribe(channel, (msg) => received.push(msg));

        stopRedis();
        await sleep(300);

        try { await pub.publish(channel, "msg-during-outage"); } catch { }

        startRedis();
        await waitForRedisCli();
        await sub.unsubscribe(channel);
        await Promise.all([waitForReady(pub), waitForReady(sub)]);

        await sub.subscribe(channel, (msg) => received.push(msg));
        await pub.publish(channel, "msg-after-restart");
        await sleep(200);

        await sub.unsubscribe(channel);
        await pub.disconnect();
        await sub.disconnect();

        assert.ok(received.includes("msg-after-restart"), "resumes after restart");
    });
}

async function cleanup(client) {
    const patterns = ["test:*", "lock:room:room_*", "rate:test:*", "rate:recovery:*", "batch:*", "atomic:*", "recovery:*", "partition:*"];
    const all = [];
    for (const pattern of patterns) {
        const keys = await scanKeys(client, pattern);
        all.push(...keys);
    }
    if (all.length) await client.del(all);
}

async function main() {
    console.log("Redis Test Suite");
    console.log("=".repeat(40));

    const client = makeClient("data");
    const pub = makeClient("pub");
    const sub = makeClient("sub");

    try {
        await Promise.all([client.connect(), pub.connect(), sub.connect()]);
        console.log("Connected to Redis:", REDIS_URL);
    } catch (err) {
        console.error("Could not connect to Redis:", err.message);
        process.exit(1);
    }

    await runBasicTests(client);
    await runLockTests(client);
    await runRateLimitTests(client);
    await runPubSubTests(pub, sub);
    await runBatchDelTest(client);
    if (!SKIP_RECOVERY) {
        await runRecoveryTests();
    } else {
        console.log("\nSkipping recovery tests (CI mode)");
    }

    await cleanup(client);

    try {
        await Promise.all([client.disconnect(), pub.disconnect(), sub.disconnect()]);
    } catch { }

    console.log("\n" + "=".repeat(40));
    console.log(`Results: ${results.passed} passed, ${results.failed} failed, ${results.skipped} skipped`);

    if (results.failed > 0) process.exit(1);
}

main();