import { Router } from "express";
import passport from "passport";
import { body, validationResult } from "express-validator";
import rateLimit from "express-rate-limit";
import multer from "multer";
import "../../config/passport.js";
import { AVATAR_MAX_BYTES, ALLOWED_FORMATS } from "../config/cloudinary.js";
import {
  addToHistory,
  getUserHistory,
  login,
  register,
  refreshToken,
  addParticipant,
  logout,
  upsertMeeting,
  getMe,
  getProfile,
  updateProfile,
  updateAvatar,
  deleteAvatar,
  changePassword,
} from "../controllers/user.controller.js";

const router = Router();

const jwtAuth = passport.authenticate("jwt", { session: false });

function validate(req, res, next) {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(422).json({ message: errors.array()[0].msg });
  }
  next();
}

const authLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 20,
  standardHeaders: true,
  legacyHeaders: false,
  message: { message: "Too many attempts, please try again later." },
  skipSuccessfulRequests: true,
});

const refreshLimiter = rateLimit({
  windowMs: 5 * 60 * 1000,
  max: 30,
  standardHeaders: true,
  legacyHeaders: false,
  message: { message: "Too many refresh attempts." },
});

const profileLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 40,
  standardHeaders: true,
  legacyHeaders: false,
  message: { message: "Too many profile requests." },
});

const avatarUpload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: AVATAR_MAX_BYTES, files: 1 },
  fileFilter(_req, file, cb) {
    const ext = file.originalname.split(".").pop().toLowerCase();
    const mime = file.mimetype.startsWith("image/");
    if (!mime || !ALLOWED_FORMATS.includes(ext)) {
      return cb(new Error(`Only image files are allowed (${ALLOWED_FORMATS.join(", ")}).`));
    }
    cb(null, true);
  },
});

function handleMulterError(err, _req, res, next) {
  if (err instanceof multer.MulterError || err?.message) {
    return res.status(422).json({ success: false, message: err.message });
  }
  next(err);
}

const loginRules = [
  body("username")
    .trim()
    .toLowerCase()
    .isLength({ min: 3, max: 32 })
    .withMessage("Username must be 3–32 characters.")
    .matches(/^[a-z0-9_.-]+$/)
    .withMessage("Username may only contain letters, numbers, _, ., and -."),
  body("password")
    .isLength({ min: 8, max: 128 })
    .withMessage("Password must be 8–128 characters."),
];

const registerRules = [
  body("name")
    .trim()
    .isLength({ min: 1, max: 64 })
    .withMessage("Name is required and must be under 64 characters.")
    .escape(),
  body("username")
    .trim()
    .toLowerCase()
    .isLength({ min: 3, max: 32 })
    .withMessage("Username must be 3–32 characters.")
    .matches(/^[a-z0-9_.-]+$/)
    .withMessage("Username may only contain letters, numbers, _, ., and -."),
  body("password")
    .isLength({ min: 8, max: 128 })
    .withMessage("Password must be 8–128 characters.")
    .matches(/[A-Z]/)
    .withMessage("Password must contain at least one uppercase letter.")
    .matches(/[0-9]/)
    .withMessage("Password must contain at least one number."),
  body("email")
    .trim()
    .isEmail()
    .withMessage("A valid email address is required.")
    .isLength({ max: 254 })
    .withMessage("Email must be under 254 characters.")
    .normalizeEmail(),
];

const updateProfileRules = [
  body("name")
    .optional()
    .trim()
    .isLength({ min: 1, max: 64 })
    .withMessage("Name must be 1–64 characters.")
    .escape(),
  body("bio")
    .optional()
    .trim()
    .isLength({ max: 280 })
    .withMessage("Bio must be under 280 characters.")
    .escape(),
  body("timezone")
    .optional()
    .trim()
    .isLength({ max: 64 })
    .withMessage("Timezone must be under 64 characters."),
  body("email")
    .optional({ nullable: true, checkFalsy: true })
    .trim()
    .isEmail()
    .withMessage("Must be a valid email address.")
    .isLength({ max: 254 })
    .withMessage("Email must be under 254 characters.")
    .normalizeEmail(),
];

const changePasswordRules = [
  body("currentPassword")
    .isLength({ min: 1 })
    .withMessage("Current password is required."),
  body("newPassword")
    .isLength({ min: 8, max: 128 })
    .withMessage("New password must be 8–128 characters.")
    .matches(/[A-Z]/)
    .withMessage("New password must contain at least one uppercase letter.")
    .matches(/[0-9]/)
    .withMessage("New password must contain at least one number."),
];

router.post("/login", authLimiter, loginRules, validate, login);
router.post("/register", authLimiter, registerRules, validate, register);
router.post("/refresh", refreshLimiter, refreshToken);
router.post("/logout", jwtAuth, logout);

router.post("/add_to_activity", jwtAuth, addToHistory);
router.get("/get_all_activity", jwtAuth, getUserHistory);
router.post("/meetings/:code/participants", jwtAuth, addParticipant);
router.post("/add_participant", jwtAuth, addParticipant);
router.post("/meetings", jwtAuth, upsertMeeting);
router.get("/me", jwtAuth, getMe);

router.get("/profile", jwtAuth, profileLimiter, getProfile);
router.patch("/profile", jwtAuth, profileLimiter, updateProfileRules, validate, updateProfile);
router.patch("/profile/password", jwtAuth, profileLimiter, changePasswordRules, validate, changePassword);
router.put(
  "/profile/avatar",
  jwtAuth,
  profileLimiter,
  avatarUpload.single("avatar"),
  handleMulterError,
  updateAvatar
);
router.delete("/profile/avatar", jwtAuth, profileLimiter, deleteAvatar);

export default router;