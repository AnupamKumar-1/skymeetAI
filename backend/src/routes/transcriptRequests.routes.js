import express from "express";
import passport from "passport";
import {
    requestTranscript,
    listPendingRequests,
    resolveRequest,
    myRequests,
} from "../controllers/transcriptRequest.controller.js";

const router = express.Router();
const auth = passport.authenticate("jwt", { session: false });

router.post("/", auth, requestTranscript);
router.get("/mine", auth, myRequests);
router.get("/host", auth, listPendingRequests);
router.patch("/:id/resolve", auth, resolveRequest);

export default router;