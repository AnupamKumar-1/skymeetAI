import dotenv from "dotenv";
dotenv.config();

import express from "express";
import { createServer } from "node:http";
import mongoose from "mongoose";
import cors from "cors";
import passport from "passport";
import "../config/passport.js";
import "./models/user.model.js";
import "./models/meeting.model.js";
import "./models/transcriptRequest.model.js";
import userRoutes from "./routes/users.routes.js";
import roomsRoutes from "./routes/rooms.js";
import meetingsRoutes from "./routes/meetings.routes.js";
import transcriptRoutes from "./routes/transcripts.js";
import { connectToSocket } from "./controllers/socket.controller.js";
import { logout } from "./controllers/user.controller.js";
import { connectRedis, redisPub, redisSub } from "./infra/redis.js";
import transcriptProxyRoutes from "./routes/transcriptProxy.routes.js";
import transcriptRequestRoutes from "./routes/transcriptRequests.routes.js";
import { createProxyMiddleware } from "http-proxy-middleware";

const app = express();
app.set("trust proxy", 1);
const server = createServer(app);

const allowedOrigins = [
  "http://localhost:3000",
  ...(process.env.CLIENT_ORIGIN ? [process.env.CLIENT_ORIGIN] : []),
];

app.use(cors({

  origin: function (origin, callback) {

    if (!origin || allowedOrigins.includes(origin)) {

      callback(null, true);
    } else {
      callback(new Error("Not allowed by CORS"));
    }

  },

  credentials: true,
}));

app.options("*", cors());

app.use(passport.initialize());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.use("/api/v1/users", userRoutes);
app.use("/api/v1/rooms", roomsRoutes);
app.use("/api/v1/transcripts/proxy", transcriptProxyRoutes);
app.use("/api/v1/transcripts", transcriptRoutes);
app.use("/api/v1/meetings", meetingsRoutes);
app.use("/api/v1/transcript-requests", transcriptRequestRoutes);

app.post("/api/v1/auth/logout", logout);

app.use("/api", (req, res) => {
  res.status(404).json({
    success: false,
    message: `Route not found: ${req.method} ${req.originalUrl}`,
  });
});

app.use((err, req, res, next) => {
  console.error(`server: unhandled error on ${req.method} ${req.originalUrl} —`, err.message);
  const status = err?.status ?? 500;
  res.status(status).json({
    success: false,
    message: err?.message ?? "Internal server error",
  });
});

app.set("port", process.env.PORT || 8000);

const start = async () => {
  try {
    const db = await mongoose.connect(process.env.MONGO_URI);
    console.info(`server: MongoDB connected to ${db.connection.host}`);
  } catch (err) {
    console.error("server: MongoDB connection failed —", err.message);
    process.exit(1);
  }

  try {
    await connectRedis();
    console.info("server: Redis connected");
  } catch (err) {
    console.error("server: Redis connection failed —", err.message);
    process.exit(1);
  }

  server.listen(app.get("port"), "0.0.0.0", () => {
    console.info(`server: listening on port ${app.get("port")}`);

    try {
      connectToSocket(
        server,
        {
          origin: [
            "http://localhost:3000",
            ...(process.env.CLIENT_ORIGIN ? [process.env.CLIENT_ORIGIN] : []),
          ],
          credentials: true,
        },
        redisPub,
        redisSub
      );
      console.info("server: socket manager initialized");
    } catch (err) {
      console.error("server: socket manager failed to initialize —", err.message);
    }
  });

  server.on("error", (err) => {
    console.error("server: HTTP server error —", err.message);
  });
};

console.info(`server: PORT env -> ${process.env.PORT}`);

process.on("unhandledRejection", (reason) => {
  console.error("server: unhandled promise rejection —", reason?.message ?? reason);
});

process.on("uncaughtException", (err) => {
  console.error("server: uncaught exception —", err.message);
  process.exit(1);
});

start();