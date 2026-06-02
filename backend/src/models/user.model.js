import mongoose, { Schema } from "mongoose";

const userScheme = new Schema(
    {
        name: { type: String, required: true, trim: true },
        username: { type: String, required: true, unique: true, trim: true, lowercase: true },
        password: { type: String, required: true },
        email: { type: String, default: null, trim: true, lowercase: true, sparse: true },
        avatar: {
            url: { type: String, default: null },
            publicId: { type: String, default: null },
        },
        bio: { type: String, default: "", maxlength: 280 },
        timezone: { type: String, default: "" },
    },
    { timestamps: true }
);

const User = mongoose.model("UserDb", userScheme);

export { User };