import { v2 as cloudinary } from "cloudinary";

const REQUIRED = ["CLOUD_NAME", "CLOUD_API_KEY", "CLOUD_SECRET"];
const missing = REQUIRED.filter((k) => !process.env[k]);
if (missing.length) {
    console.error(`[cloudinary] FATAL: missing env vars: ${missing.join(", ")}`);
    process.exit(1);
}

cloudinary.config({
    cloud_name: process.env.CLOUD_NAME,
    api_key: process.env.CLOUD_API_KEY,
    api_secret: process.env.CLOUD_SECRET,
    secure: true,
});

const AVATAR_FOLDER = process.env.CLOUDINARY_AVATAR_FOLDER ?? "avatars";
const AVATAR_MAX_BYTES = parseInt(process.env.AVATAR_MAX_BYTES ?? String(2 * 1024 * 1024), 10);
const ALLOWED_FORMATS = ["jpg", "jpeg", "png", "webp", "gif"];

export async function uploadAvatar(fileBuffer, userId) {
    return new Promise((resolve, reject) => {
        const uploadStream = cloudinary.uploader.upload_stream(
            {
                folder: AVATAR_FOLDER,
                public_id: `user_${userId}`,
                overwrite: true,
                resource_type: "image",
                allowed_formats: ALLOWED_FORMATS,
                transformation: [
                    { width: 256, height: 256, crop: "fill", gravity: "face" },
                    { quality: "auto", fetch_format: "auto" },
                ],
            },
            (error, result) => {
                if (error) return reject(error);
                resolve({ url: result.secure_url, publicId: result.public_id });
            }
        );
        uploadStream.end(fileBuffer);
    });
}

export async function destroyAvatar(publicId) {
    if (!publicId) return null;
    return cloudinary.uploader.destroy(publicId, { resource_type: "image" });
}

export { AVATAR_MAX_BYTES, ALLOWED_FORMATS };