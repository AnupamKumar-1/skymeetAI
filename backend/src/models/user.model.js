import mongoose, { Schema } from "mongoose";

const userScheme = new Schema(
    {
        name: { type: String, required: true, trim: true },
        username: { type: String, required: true, unique: true, trim: true, lowercase: true },
        password: { type: String, required: true },
    },
    { timestamps: true }
);


const User = mongoose.model("UserDb", userScheme);

export { User };