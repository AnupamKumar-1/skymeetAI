// backend/src/routes/meetings.routes.js
import { Router } from "express";
import passport from "passport";
import "../../config/passport.js";

import {
  addParticipant,
  getMeetings,
  upsertMeeting,
} from "../controllers/user.controller.js";

const router = Router();

const jwtAuth = passport.authenticate("jwt", { session: false });

// optionalAuth that sets req.user when JWT present
const optionalAuth = (req, res, next) =>
  passport.authenticate("jwt", { session: false }, (err, user) => {
    if (err) return next(err);
    if (user) req.user = user;
    return next();
  })(req, res, next);

// safe wrapper to catch sync/async errors and return a JSON 500 with stack printed
const safe = (fn, name = "handler") => async (req, res, next) => {
  console.debug(`[meetings] ${req.method} ${req.originalUrl} - query=${JSON.stringify(req.query)} bodyPresent=${!!req.body}`);
  try {
    await Promise.resolve(fn(req, res, next));
  } catch (err) {
    console.error(`[meetings] error in ${name}:`, err && (err.stack || err));
    // prefer to return structured JSON so client-side code sees the failure
    res.status(err && err.status ? err.status : 500).json({
      success: false,
      message: err && err.message ? err.message : "Internal Server Error",
      // don't return full stack to clients in production; useful for dev
      ...(process.env.NODE_ENV !== "production" ? { stack: err && err.stack } : {}),
    });
  }
};

/* Meetings API */
router.get("/", optionalAuth, safe(getMeetings, "getMeetings"));
router.post("/", jwtAuth, safe(upsertMeeting, "upsertMeeting"));

/* Participants for a meeting */
router.post("/:code/participants", jwtAuth, safe(addParticipant, "addParticipant"));
router.post("/add_participant", jwtAuth, safe(addParticipant, "addParticipant"));

export default router;