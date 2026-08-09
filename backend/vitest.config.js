import { defineConfig } from "vitest/config";

export default defineConfig({
    test: {
        include: ["tests/**/*.test.js"],
        exclude: ["tests/redis.test.js"],
    },
});