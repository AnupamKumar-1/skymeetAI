import React, { useState, useRef, useCallback, useEffect } from "react";
import "../styles/UserProfileModal.css";

const SERVER_BASE = import.meta.env.VITE_SERVER_URL || "http://localhost:8000";
const API_BASE = import.meta.env.VITE_API_URL || `${SERVER_BASE}/api/v1`;

async function apiFetch(path, options = {}) {
    const token = localStorage.getItem("token");
    const headers = {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
        ...options.headers,
    };
    const res = await fetch(`${API_BASE}${path}`, { ...options, headers });
    const data = await res.json();
    return { ok: res.ok, status: res.status, data };
}

async function uploadAvatarToBackend(file) {
    const token = localStorage.getItem("token");
    const fd = new FormData();
    fd.append("avatar", file);
    const res = await fetch(`${API_BASE}/users/profile/avatar`, {
        method: "PUT",
        headers: token ? { Authorization: `Bearer ${token}` } : {},
        body: fd,
    });
    const data = await res.json();
    return { ok: res.ok, status: res.status, data };
}

const TIMEZONES = Intl.supportedValuesOf
    ? Intl.supportedValuesOf("timeZone")
    : ["UTC", "America/New_York", "America/Los_Angeles", "Europe/London", "Asia/Kolkata", "Asia/Tokyo", "Australia/Sydney"];

const AVATAR_MAX_BYTES = 2 * 1024 * 1024;
const ALLOWED_MIME = ["image/jpeg", "image/png", "image/webp", "image/gif"];

function AvatarUploader({ currentUrl, uploading, onUpload, onRemove }) {
    const inputRef = useRef(null);
    const [dragOver, setDragOver] = useState(false);

    const handleFile = useCallback((file) => {
        if (!file) return;
        if (!ALLOWED_MIME.includes(file.type)) {
            alert("Only JPEG, PNG, WebP or GIF images are allowed.");
            return;
        }
        if (file.size > AVATAR_MAX_BYTES) {
            alert(`Image must be under ${AVATAR_MAX_BYTES / (1024 * 1024)} MB.`);
            return;
        }
        onUpload(file);
    }, [onUpload]);

    const onDrop = (e) => {
        e.preventDefault();
        setDragOver(false);
        const file = e.dataTransfer.files?.[0];
        if (file) handleFile(file);
    };

    return (
        <div className="upm-avatar-zone">
            <div
                className={`upm-avatar-dropzone ${dragOver ? "upm-avatar-dropzone-drag" : ""} ${uploading ? "upm-avatar-uploading" : ""}`}
                onClick={() => !uploading && inputRef.current?.click()}
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={onDrop}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => e.key === "Enter" && inputRef.current?.click()}
                aria-label="Upload avatar"
            >
                {currentUrl ? (
                    <img src={currentUrl} alt="Avatar" className="upm-avatar-img" />
                ) : (
                    <div className="upm-avatar-placeholder">
                        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
                            <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
                            <circle cx="12" cy="7" r="4" />
                        </svg>
                    </div>
                )}
                {uploading && (
                    <div className="upm-avatar-uploading-overlay">
                        <div className="upm-avatar-spinner" />
                    </div>
                )}
                {!uploading && (
                    <div className="upm-avatar-overlay">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                            <polyline points="17 8 12 3 7 8" />
                            <line x1="12" y1="3" x2="12" y2="15" />
                        </svg>
                    </div>
                )}
            </div>
            <input
                ref={inputRef}
                type="file"
                accept="image/jpeg,image/png,image/webp,image/gif"
                style={{ display: "none" }}
                onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); e.target.value = ""; }}
            />
            <div className="upm-avatar-actions">
                <button className="upm-avatar-btn" onClick={() => inputRef.current?.click()} disabled={uploading} type="button">
                    {uploading ? "Uploading…" : currentUrl ? "Change photo" : "Upload photo"}
                </button>
                {currentUrl && !uploading && (
                    <button className="upm-avatar-btn upm-avatar-btn-danger" onClick={onRemove} type="button">
                        Remove
                    </button>
                )}
            </div>
            <p className="upm-avatar-hint">JPEG, PNG or WebP · Max 2 MB · Drag &amp; drop or click</p>
        </div>
    );
}

export default function UserProfileModal({ onClose, userData, onProfileUpdate }) {
    const [profile, setProfile] = useState(null);
    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);
    const [uploadingAvatar, setUploadingAvatar] = useState(false);
    const [error, setError] = useState(null);
    const [success, setSuccess] = useState(null);
    const [activeTab, setActiveTab] = useState("profile");

    const [name, setName] = useState("");
    const [bio, setBio] = useState("");
    const [timezone, setTimezone] = useState("");
    const [tzQuery, setTzQuery] = useState("");

    const [email, setEmail] = useState("");

    const [currentPassword, setCurrentPassword] = useState("");
    const [newPassword, setNewPassword] = useState("");
    const [confirmPassword, setConfirmPassword] = useState("");
    const [showCurrentPw, setShowCurrentPw] = useState(false);
    const [showNewPw, setShowNewPw] = useState(false);
    const [showConfirmPw, setShowConfirmPw] = useState(false);
    const [savingPassword, setSavingPassword] = useState(false);

    const overlayRef = useRef(null);

    useEffect(() => {
        let cancelled = false;
        setLoading(true);
        apiFetch("/users/profile")
            .then(({ ok, data }) => {
                if (cancelled) return;
                if (ok && data.profile) {
                    const detectedTz = Intl.DateTimeFormat().resolvedOptions().timeZone;
                    setProfile(data.profile);
                    setName(data.profile.name || "");
                    setBio(data.profile.bio || "");
                    setTimezone(data.profile.timezone || detectedTz || "");
                    setEmail(data.profile.email || "");
                }
            })
            .catch(() => { if (!cancelled) setError("Failed to load profile."); })
            .finally(() => { if (!cancelled) setLoading(false); });
        return () => { cancelled = true; };
    }, []);

    useEffect(() => {
        function onKey(e) { if (e.key === "Escape") onClose(); }
        document.addEventListener("keydown", onKey);
        return () => document.removeEventListener("keydown", onKey);
    }, [onClose]);

    function flash(msg, type = "success") {
        if (type === "success") { setSuccess(msg); setTimeout(() => setSuccess(null), 3000); }
        else { setError(msg); setTimeout(() => setError(null), 4000); }
    }

    async function handleSave(e) {
        e.preventDefault();
        if (saving) return;
        setSaving(true);
        setError(null);
        try {
            const trimmedEmail = email.trim();
            if (trimmedEmail && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(trimmedEmail)) {
                flash("Please enter a valid email address.", "error");
                setSaving(false);
                return;
            }
            const { ok, data } = await apiFetch("/users/profile", {
                method: "PATCH",
                body: JSON.stringify({ name: name.trim(), bio: bio.trim(), timezone, email: trimmedEmail || null }),
            });
            if (ok && data.profile) {
                setProfile(data.profile);
                setEmail(data.profile.email || "");
                if (onProfileUpdate) onProfileUpdate(data.profile);
                flash("Profile updated.");
            } else {
                flash(data.message || "Update failed.", "error");
            }
        } catch {
            flash("Network error.", "error");
        } finally {
            setSaving(false);
        }
    }

    async function handleChangePassword(e) {
        e.preventDefault();
        if (savingPassword) return;
        if (!currentPassword) { flash("Current password is required.", "error"); return; }
        if (newPassword.length < 8) { flash("New password must be at least 8 characters.", "error"); return; }
        if (!/[A-Z]/.test(newPassword)) { flash("New password must contain at least one uppercase letter.", "error"); return; }
        if (!/[0-9]/.test(newPassword)) { flash("New password must contain at least one number.", "error"); return; }
        if (newPassword !== confirmPassword) { flash("Passwords do not match.", "error"); return; }
        setSavingPassword(true);
        setError(null);
        try {
            const { ok, data } = await apiFetch("/users/profile/password", {
                method: "PATCH",
                body: JSON.stringify({ currentPassword, newPassword }),
            });
            if (ok) {
                setCurrentPassword("");
                setNewPassword("");
                setConfirmPassword("");
                flash("Password changed successfully.");
            } else {
                flash(data.message || "Password change failed.", "error");
            }
        } catch {
            flash("Network error.", "error");
        } finally {
            setSavingPassword(false);
        }
    }

    async function handleAvatarUpload(file) {
        setUploadingAvatar(true);
        setError(null);
        try {
            const { ok, data } = await uploadAvatarToBackend(file);
            if (ok && data.profile) {
                setProfile(data.profile);
                if (onProfileUpdate) onProfileUpdate(data.profile);
                flash("Avatar updated.");
            } else {
                flash(data.message || "Avatar update failed.", "error");
            }
        } catch (err) {
            flash(err.message || "Upload failed.", "error");
        } finally {
            setUploadingAvatar(false);
        }
    }

    async function handleAvatarRemove() {
        setUploadingAvatar(true);
        setError(null);
        try {
            const { ok, data } = await apiFetch("/users/profile/avatar", { method: "DELETE" });
            if (ok && data.profile) {
                setProfile(data.profile);
                if (onProfileUpdate) onProfileUpdate(data.profile);
                flash("Avatar removed.");
            } else {
                flash(data.message || "Removal failed.", "error");
            }
        } catch {
            flash("Network error.", "error");
        } finally {
            setUploadingAvatar(false);
        }
    }

    const filteredTz = tzQuery.trim()
        ? TIMEZONES.filter((t) => t.toLowerCase().includes(tzQuery.toLowerCase())).slice(0, 30)
        : TIMEZONES.slice(0, 30);

    const memberSince = profile?.createdAt
        ? new Date(profile.createdAt).toLocaleDateString(undefined, { year: "numeric", month: "long" })
        : null;

    function pwStrength(pw) {
        if (!pw) return { score: 0, label: "", color: "" };
        let score = 0;
        if (pw.length >= 8) score++;
        if (pw.length >= 12) score++;
        if (/[A-Z]/.test(pw)) score++;
        if (/[0-9]/.test(pw)) score++;
        if (/[^A-Za-z0-9]/.test(pw)) score++;
        if (score <= 1) return { score, label: "Weak", color: "#ef4444" };
        if (score <= 3) return { score, label: "Fair", color: "#f59e0b" };
        if (score === 4) return { score, label: "Good", color: "#3b82f6" };
        return { score, label: "Strong", color: "#34d399" };
    }

    const strength = pwStrength(newPassword);

    function handleLogout() {
        localStorage.removeItem("token");
        window.location.href = "/login";
    }

    return (
        <div className="upm-overlay" ref={overlayRef} onClick={(e) => { if (e.target === overlayRef.current) onClose(); }} role="dialog" aria-modal="true" aria-label="User profile">
            <div className="upm-panel">
                <div className="upm-header">
                    <div className="upm-header-info">
                        <div className="upm-header-title">
                            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden>
                                <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
                                <circle cx="12" cy="7" r="4" />
                            </svg>
                            Profile
                        </div>
                        {memberSince && <div className="upm-header-sub">Member since {memberSince}</div>}
                        <div className="upm-header-version-row">
                            <span className="upm-header-version-badge">v1.0.0</span>
                        </div>
                    </div>
                    <button className="upm-close" onClick={onClose} aria-label="Close">
                        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                            <path d="M18 6L6 18M6 6l12 12" />
                        </svg>
                    </button>
                </div>

                <div className="upm-tabs">
                    <button className={`upm-tab ${activeTab === "profile" ? "upm-tab-active" : ""}`} onClick={() => setActiveTab("profile")}>Profile</button>
                    <button className={`upm-tab ${activeTab === "avatar" ? "upm-tab-active" : ""}`} onClick={() => setActiveTab("avatar")}>Avatar</button>
                    <button className={`upm-tab ${activeTab === "security" ? "upm-tab-active" : ""}`} onClick={() => setActiveTab("security")}>Security</button>
                </div>

                {(error || success) && (
                    <div className={`upm-alert ${success ? "upm-alert-success" : "upm-alert-error"}`}>
                        {success ? (
                            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><polyline points="20 6 9 17 4 12" /></svg>
                        ) : (
                            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><path d="M18 6L6 18M6 6l12 12" /></svg>
                        )}
                        {success || error}
                    </div>
                )}

                <div className="upm-body">
                    {loading ? (
                        <div className="upm-loading">
                            <div className="upm-loading-dots"><span /><span /><span /></div>
                        </div>
                    ) : activeTab === "profile" ? (
                        <form className="upm-form" onSubmit={handleSave}>
                            <div className="upm-identity">
                                <div className="upm-identity-avatar">
                                    {profile?.avatar?.url ? (
                                        <img src={profile.avatar.url} alt={profile.name} />
                                    ) : (
                                        <span>{(profile?.name || "?")[0].toUpperCase()}</span>
                                    )}
                                </div>
                                <div className="upm-identity-meta">
                                    <div className="upm-identity-name">{profile?.name}</div>
                                    <div className="upm-identity-username">@{profile?.username}</div>
                                </div>
                            </div>

                            <div className="upm-divider" />

                            <div className="upm-field">
                                <label htmlFor="upm-name">Display name</label>
                                <input
                                    id="upm-name"
                                    type="text"
                                    value={name}
                                    onChange={(e) => setName(e.target.value)}
                                    maxLength={64}
                                    required
                                />
                                <span className="upm-field-hint">{name.length}/64</span>
                            </div>

                            <div className="upm-field">
                                <label htmlFor="upm-bio">Bio</label>
                                <textarea
                                    id="upm-bio"
                                    value={bio}
                                    onChange={(e) => setBio(e.target.value)}
                                    placeholder="Tell others a bit about yourself…"
                                    maxLength={280}
                                    rows={3}
                                />
                                <span className="upm-field-hint">{bio.length}/280</span>
                            </div>

                            <div className="upm-field">
                                <label htmlFor="upm-email">Email address</label>
                                <input
                                    id="upm-email"
                                    type="email"
                                    value={email}
                                    onChange={(e) => setEmail(e.target.value)}
                                    placeholder="you@example.com"
                                    maxLength={254}
                                    autoComplete="email"
                                />
                            </div>

                            <div className="upm-field">
                                <label htmlFor="upm-tz-search">Timezone</label>
                                <input
                                    id="upm-tz-search"
                                    type="text"
                                    placeholder="Search timezone…"
                                    value={tzQuery || timezone}
                                    onChange={(e) => setTzQuery(e.target.value)}
                                    onFocus={() => setTzQuery("")}
                                    autoComplete="off"
                                />
                                {tzQuery.trim() && (
                                    <div className="upm-tz-dropdown">
                                        {filteredTz.length === 0 && <div className="upm-tz-empty">No matches</div>}
                                        {filteredTz.map((tz) => (
                                            <button
                                                key={tz}
                                                type="button"
                                                className={`upm-tz-option ${tz === timezone ? "upm-tz-option-selected" : ""}`}
                                                onClick={() => { setTimezone(tz); setTzQuery(""); }}
                                            >
                                                {tz}
                                            </button>
                                        ))}
                                    </div>
                                )}
                                {timezone && !tzQuery && <span className="upm-field-hint">{timezone}</span>}
                            </div>

                            <div className="upm-form-actions">
                                <button type="button" className="upm-btn-secondary" onClick={onClose}>Cancel</button>
                                <button type="submit" className="upm-btn-primary" disabled={saving}>
                                    {saving ? (
                                        <><span className="upm-btn-spinner" />Saving…</>
                                    ) : "Save changes"}
                                </button>
                            </div>
                        </form>
                    ) : activeTab === "avatar" ? (
                        <div className="upm-avatar-tab">
                            <AvatarUploader
                                currentUrl={profile?.avatar?.url}
                                uploading={uploadingAvatar}
                                onUpload={handleAvatarUpload}
                                onRemove={handleAvatarRemove}
                            />
                            <div className="upm-divider" />
                            <div className="upm-cloudinary-note">
                                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden>
                                    <circle cx="12" cy="12" r="10" /><path d="M12 8v4M12 16h.01" />
                                </svg>
                                Images are stored securely via Cloudinary. Your photo is CDN-delivered and auto-optimised.
                            </div>
                        </div>
                    ) : activeTab === "security" ? (
                        <form className="upm-form" onSubmit={handleChangePassword}>
                            <div className="upm-section-label">
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden>
                                    <rect x="3" y="11" width="18" height="11" rx="2" />
                                    <path d="M7 11V7a5 5 0 0 1 10 0v4" />
                                </svg>
                                Change password
                            </div>
                            <p className="upm-section-desc">Choose a strong password with at least 8 characters, one uppercase letter and one number.</p>

                            <div className="upm-field" style={{ marginTop: "14px" }}>
                                <label htmlFor="upm-cur-pw">Current password</label>
                                <div className="upm-pw-wrap">
                                    <input
                                        id="upm-cur-pw"
                                        type={showCurrentPw ? "text" : "password"}
                                        value={currentPassword}
                                        onChange={(e) => setCurrentPassword(e.target.value)}
                                        placeholder="Enter current password"
                                        autoComplete="current-password"
                                    />
                                    <button type="button" className="upm-pw-eye" onClick={() => setShowCurrentPw((v) => !v)} aria-label={showCurrentPw ? "Hide password" : "Show password"}>
                                        {showCurrentPw ? (
                                            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M17.94 17.94A10.07 10.07 0 0112 20c-7 0-11-8-11-8a18.45 18.45 0 015.06-5.94" /><path d="M9.9 4.24A9.12 9.12 0 0112 4c7 0 11 8 11 8a18.5 18.5 0 01-2.16 3.19" /><line x1="1" y1="1" x2="23" y2="23" /></svg>
                                        ) : (
                                            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" /><circle cx="12" cy="12" r="3" /></svg>
                                        )}
                                    </button>
                                </div>
                            </div>

                            <div className="upm-field">
                                <label htmlFor="upm-new-pw">New password</label>
                                <div className="upm-pw-wrap">
                                    <input
                                        id="upm-new-pw"
                                        type={showNewPw ? "text" : "password"}
                                        value={newPassword}
                                        onChange={(e) => setNewPassword(e.target.value)}
                                        placeholder="At least 8 characters"
                                        autoComplete="new-password"
                                    />
                                    <button type="button" className="upm-pw-eye" onClick={() => setShowNewPw((v) => !v)} aria-label={showNewPw ? "Hide password" : "Show password"}>
                                        {showNewPw ? (
                                            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M17.94 17.94A10.07 10.07 0 0112 20c-7 0-11-8-11-8a18.45 18.45 0 015.06-5.94" /><path d="M9.9 4.24A9.12 9.12 0 0112 4c7 0 11 8 11 8a18.5 18.5 0 01-2.16 3.19" /><line x1="1" y1="1" x2="23" y2="23" /></svg>
                                        ) : (
                                            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" /><circle cx="12" cy="12" r="3" /></svg>
                                        )}
                                    </button>
                                </div>
                                {newPassword && (
                                    <div className="upm-pw-strength">
                                        <div className="upm-pw-strength-bar">
                                            {[1, 2, 3, 4, 5].map((i) => (
                                                <div key={i} className="upm-pw-strength-seg" style={{ background: i <= strength.score ? strength.color : "rgba(255,255,255,0.07)" }} />
                                            ))}
                                        </div>
                                        <span className="upm-pw-strength-label" style={{ color: strength.color }}>{strength.label}</span>
                                    </div>
                                )}
                            </div>

                            <div className="upm-field">
                                <label htmlFor="upm-confirm-pw">Confirm new password</label>
                                <div className="upm-pw-wrap">
                                    <input
                                        id="upm-confirm-pw"
                                        type={showConfirmPw ? "text" : "password"}
                                        value={confirmPassword}
                                        onChange={(e) => setConfirmPassword(e.target.value)}
                                        placeholder="Repeat new password"
                                        autoComplete="new-password"
                                    />
                                    <button type="button" className="upm-pw-eye" onClick={() => setShowConfirmPw((v) => !v)} aria-label={showConfirmPw ? "Hide password" : "Show password"}>
                                        {showConfirmPw ? (
                                            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M17.94 17.94A10.07 10.07 0 0112 20c-7 0-11-8-11-8a18.45 18.45 0 015.06-5.94" /><path d="M9.9 4.24A9.12 9.12 0 0112 4c7 0 11 8 11 8a18.5 18.5 0 01-2.16 3.19" /><line x1="1" y1="1" x2="23" y2="23" /></svg>
                                        ) : (
                                            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" /><circle cx="12" cy="12" r="3" /></svg>
                                        )}
                                    </button>
                                </div>
                                {confirmPassword && newPassword !== confirmPassword && (
                                    <span className="upm-field-hint" style={{ color: "#f87171" }}>Passwords do not match</span>
                                )}
                                {confirmPassword && newPassword === confirmPassword && newPassword && (
                                    <span className="upm-field-hint" style={{ color: "#34d399" }}>Passwords match</span>
                                )}
                            </div>

                            <div className="upm-form-actions">
                                <button type="button" className="upm-btn-secondary" onClick={onClose}>Cancel</button>
                                <button type="submit" className="upm-btn-primary" disabled={savingPassword}>
                                    {savingPassword ? (
                                        <><span className="upm-btn-spinner" />Updating…</>
                                    ) : "Change password"}
                                </button>
                            </div>
                        </form>
                    ) : null}
                </div>

                <div className="upm-logout-section">
                    <button type="button" className="upm-logout-btn" onClick={handleLogout}>
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden>
                            <path d="M9 21H5a2 2 0 01-2-2V5a2 2 0 012-2h4" /><polyline points="16 17 21 12 16 7" /><line x1="21" y1="12" x2="9" y2="12" />
                        </svg>
                        Logout
                    </button>
                </div>
            </div>
        </div>
    );
}