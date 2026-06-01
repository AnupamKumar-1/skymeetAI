import express from "express";
import passport from "passport";
import rateLimit from "express-rate-limit";
import * as ctrl from "../controllers/rag.controller.js";

const router = express.Router();

const ragLimiter = rateLimit({
    windowMs: 60 * 1000,
    max: 200,
    standardHeaders: true,
    legacyHeaders: false,
    message: { success: false, message: "Too many requests" },
});

const optionalAuth = (req, _res, next) => {
    passport.authenticate("jwt", { session: false }, (_err, user) => {
        if (user) req.user = user;
        next();
    })(req, _res, next);
};

router.use(ragLimiter);

router.post("/:id/index", optionalAuth, ctrl.indexTranscript);
router.get("/:id/index", optionalAuth, ctrl.getIndexStatus);

router.post("/:id/query", optionalAuth, ctrl.ragQuery);

router.get("/:id/session", optionalAuth, ctrl.getRagSession);
router.delete("/:id/session", optionalAuth, ctrl.clearRagSession);

export default router;