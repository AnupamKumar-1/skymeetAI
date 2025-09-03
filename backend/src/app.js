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
import userRoutes from "./routes/users.routes.js";
import roomsRoutes from "./routes/rooms.js";
import meetingsRoutes from "./routes/meetings.routes.js";
import transcriptRoutes from "./routes/transcripts.js";
import emotionRoutes from "./routes/emotion.routes.js";
import { connectToSocket } from "./controllers/socketManager.js";
import { logout } from './controllers/user.controller.js';
const app = express();
const server = createServer(app);

const CLIENT_ORIGIN = process.env.CLIENT_ORIGIN || "http://localhost:3000";
const corsOptions = {
  origin: CLIENT_ORIGIN,
  methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
  credentials: true,
};

app.use(cors(corsOptions));
app.options("*", cors(corsOptions));
app.use(passport.initialize());

app.use(express.json({ limit: process.env.REQUEST_JSON_LIMIT || "4mb" }));
app.use(express.urlencoded({ limit: process.env.REQUEST_URLENCODED_LIMIT || "4mb", extended: true }));

app.use("/api/v1/users", userRoutes);
app.use("/api/v1/rooms", roomsRoutes);
app.use("/api/v1/transcript", transcriptRoutes);
app.use("/api/v1/emotion", emotionRoutes);
app.use("/api/v1/meetings", meetingsRoutes);
app.post('/api/v1/auth/logout', logout);

app.use("/api", (req, res) => {
  res.status(404).json({ success: false, message: `API route not found: ${req.method} ${req.originalUrl}` });
});

app.use((err, req, res, next) => {
  console.error("Unhandled error:", err && (err.stack || err));
  const status = err && err.status ? err.status : 500;
  res.status(status).json({ success: false, message: err && err.message ? err.message : "Internal Server Error" });
});

app.set("port", process.env.PORT || 8000);

const start = async () => {
  try {
    const connectionDb = await mongoose.connect(process.env.MONGO_URI);
    console.log(`MONGO Connected DB Host: ${connectionDb.connection.host}`);
  } catch (error) {
    console.error("MongoDB connection error:", error && (error.stack || error));
    process.exit(1);
  }

  server.listen(app.get("port"), () => {
    console.log(`LISTENING ON PORT ${app.get("port")}`);
    try {
      connectToSocket(server, corsOptions);
      console.log("Socket manager initialized.");
    } catch (socketErr) {
      console.error("Failed to initialize sockets:", socketErr && (socketErr.stack || socketErr));
    }

    try {
      const routes = [];
      app._router.stack.forEach((middleware) => {
        if (middleware.route) {
          const r = middleware.route;
          routes.push(`${Object.keys(r.methods).join(",").toUpperCase()} ${r.path}`);
        } else if (middleware.name === "router" && middleware.handle && middleware.handle.stack) {
          middleware.handle.stack.forEach((handler) => {
            const route = handler.route;
            if (route) routes.push(`${Object.keys(route.methods).join(",").toUpperCase()} ${route.path}`);
          });
        }
      });
      console.log("Registered routes:\n", routes.join("\n"));
    } catch (listErr) {
      console.debug("listRoutes error", listErr);
    }
  });

  server.on("error", (err) => {
    console.error("Server error:", err && (err.stack || err));
  });
};

process.on("unhandledRejection", (reason) => {
  console.error("Unhandled Rejection:", reason && (reason.stack || reason));
});
process.on("uncaughtException", (err) => {
  console.error("Uncaught Exception:", err && (err.stack || err));
});

start();