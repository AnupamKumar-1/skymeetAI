// //users.routes.js
// import { Router } from "express";
// import passport from "passport";
// import "../../config/passport.js";

// import {
//   addToHistory,
//   getUserHistory,
//   login,
//   register,
//   addParticipant,
// } from "../controllers/user.controller.js";

// const router = Router();

// const jwtAuth = passport.authenticate("jwt", { session: false });


// /* Public / auth routes */
// router.post("/login", login);
// router.post("/register", register);

// /* Activity / history (protected) */
// router.post("/add_to_activity", jwtAuth, addToHistory);
// router.get("/get_all_activity", jwtAuth, getUserHistory);

// /* Participants (protected) */
// router.post("/meetings/:code/participants", jwtAuth, addParticipant);
// router.post("/add_participant", jwtAuth, addParticipant);


// export default router;


// backend/src/routes/users.routes.js
import { Router } from "express";
import passport from "passport";
import "../../config/passport.js";

import {
  addToHistory,
  getUserHistory,
  login,
  register,
  addParticipant,
  logout, // <-- added
} from "../controllers/user.controller.js";

const router = Router();

const jwtAuth = passport.authenticate("jwt", { session: false });

/* Public / auth routes */
router.post("/login", login);
router.post("/register", register);

// Logout (public endpoint which clears refresh cookie)
router.post("/logout", logout);

/* Activity / history (protected) */
router.post("/add_to_activity", jwtAuth, addToHistory);
router.get("/get_all_activity", jwtAuth, getUserHistory);

/* Participants (protected) */
router.post("/meetings/:code/participants", jwtAuth, addParticipant);
router.post("/add_participant", jwtAuth, addParticipant);

export default router;
