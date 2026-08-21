import dotenv from "dotenv";
dotenv.config();

import passport from "passport";
import { Strategy as JwtStrategy, ExtractJwt } from "passport-jwt";
import { User } from "../src/models/user.model.js";
import { isTokenBlacklisted } from "../src/utils/tokenBlacklist.utils.js";

if (!process.env.JWT_SECRET) {
  throw new Error("FATAL: JWT_SECRET environment variable is not set.");
}

const jwtFromRequest = ExtractJwt.fromAuthHeaderAsBearerToken();

const opts = {
  jwtFromRequest,
  secretOrKey: process.env.JWT_SECRET,
  algorithms: ["HS256"],
  passReqToCallback: true,
};

passport.use(
  new JwtStrategy(opts, async (req, payload, done) => {
    try {
      if (!payload?.sub) {
        return done(null, false);
      }

      const token = jwtFromRequest(req);
      if (await isTokenBlacklisted(token)) {
        return done(null, false);
      }

      const user = await User.findById(payload.sub)
        .select("_id username name")
        .lean();

      if (!user) {
        return done(null, false);
      }

      return done(null, {
        _id: user._id.toString(),
        id: user._id.toString(),
        username: user.username,
        name: user.name,
      });
    } catch (err) {
      return done(err, false);
    }
  })
);

export default passport;
