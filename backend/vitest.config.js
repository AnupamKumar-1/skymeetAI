import { defineConfig } from "vitest/config";

export default defineConfig({
    test: {
        include: ["tests/unit/**/*.test.js"],
        exclude: ["tests/integration/**"],
        setupFiles: ["./tests/setup.unit.js"],
    },
});
