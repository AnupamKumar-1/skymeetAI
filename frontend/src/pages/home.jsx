
import React, { useEffect, useState, useContext } from "react";
import { useNavigate } from "react-router-dom";
import "../styles/home.css";
import { Snackbar, Alert } from "@mui/material";
import { AuthContext } from "../contexts/AuthContext";
import { TRANSCRIPTS_ENABLED } from "../environment";

const SERVER_BASE = process.env.REACT_APP_SERVER_URL || "http://localhost:8000";
const API_BASE = process.env.REACT_APP_API_URL || `${SERVER_BASE}/api/v1`;

async function copyToClipboard(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.style.position = "fixed";
    textarea.style.opacity = 0;
    document.body.appendChild(textarea);
    textarea.focus();
    textarea.select();
    try {
      document.execCommand("copy");
    } catch (err2) {
      console.error("Fallback copy failed", err2);
      document.body.removeChild(textarea);
      return false;
    }
    document.body.removeChild(textarea);
    return true;
  }
}

// --- Helpers for unique keys & dedupe ---
function getTranscriptKey(item, index) {
  const id = item._id || item.id || "";
  const code = (item.meeting_code || item.meetingCode || "local").toString();
  const ts = item.createdAt ? String(new Date(item.createdAt).getTime()) : String(index);
  return `${code}__${id || ts}`;
}

function dedupeByCodeKeepNewest(arr) {
  const map = new Map();
  for (const it of arr) {
    const codeRaw = (it.meeting_code || it.meetingCode || "").toString();
    const code = codeRaw ? codeRaw.toUpperCase() : `__NO_CODE__:${Math.random().toString(36).slice(2,8)}`;
    const existing = map.get(code);
    if (!existing) {
      map.set(code, it);
    } else {
      const existingTs = existing.createdAt || "";
      const itTs = it.createdAt || "";
      if (itTs > existingTs) map.set(code, it);
    }
  }
  return Array.from(map.values());
}

export default function Home() {
  const navigate = useNavigate();
  const { logout } = useContext(AuthContext); // expects AuthProvider to expose logout

  const [name, setName] = useState(localStorage.getItem("displayName") || "");
  const [room, setRoom] = useState("");
  const [recentLocal, setRecentLocal] = useState([]);
  const [expandedTranscripts, setExpandedTranscripts] = useState({});

  // Snackbar state
  const [snackOpen, setSnackOpen] = useState(false);
  const [snackMsg, setSnackMsg] = useState("");
  const [snackSeverity, setSnackSeverity] = useState("success");

  // Info flag when server has transcripts but none match local host keys
  const [serverHadTranscripts, setServerHadTranscripts] = useState(false);

  useEffect(() => {
    // If transcripts are disabled (production Render free), skip network calls and show message only.
    if (!TRANSCRIPTS_ENABLED) {
      setRecentLocal([]);
      setServerHadTranscripts(false);
      return;
    }

    (async () => {
      try {
        const token = localStorage.getItem("token");

        // 1) Attempt authenticated fetch (cross-device owner-based)
        let authItems = [];
        let authHad = false;
        if (token) {
          try {
            const resp = await fetch(`${API_BASE}/transcript?mine=true&limit=200`, {
              headers: {
                "Content-Type": "application/json",
                Authorization: `Bearer ${token}`,
              },
            });
            if (resp.ok) {
              const body = await resp.json();
              if (body && body.success && Array.isArray(body.transcripts)) {
                authHad = body.transcripts.length > 0;
                authItems = body.transcripts.map((t) => ({
                  _id: t._id || t.id || null,
                  meeting_code: (t.meetingCode || t.meeting_code || "").toString().toUpperCase(),
                  transcript: t.transcriptText || t.transcript || "",
                  createdAt: t.createdAt || null,
                  fileName: t.fileName || null,
                  fromServer: true,
                }));
              }
            } else {
              // not authorized or server error — fall through to legacy hostSecret method
              console.warn("Auth transcript request failed:", resp.status);
            }
          } catch (err) {
            console.warn("Auth transcript fetch error:", err);
          }
        }

        // Collect locally-known host entries (keys like `host:ROOMCODE`) for legacy flow
        const hostKeys = Object.keys(localStorage).filter((k) => k.startsWith("host:"));
        const hostEntries = hostKeys
          .map((k) => {
            try {
              const raw = localStorage.getItem(k);
              if (!raw) return null;
              const parsed = JSON.parse(raw);
              const parts = k.split(":");
              const roomCode = (parts[1] || "").toString().toUpperCase();
              return { storageKey: k, roomCode, ...parsed };
            } catch (e) {
              return null;
            }
          })
          .filter(Boolean);

        const hostSecrets = hostEntries
          .map((e) => ({ roomCode: e.roomCode, hostSecret: e.hostSecret || null, hostName: e.hostName || null }))
          .filter((x) => x.hostSecret && typeof x.hostSecret === "string");

        // 2) Legacy hostSecret fetches (only for secrets present locally)
        const allResults = await Promise.all(
          hostSecrets.map(async ({ hostSecret }) => {
            try {
              const resp = await fetch(`${API_BASE}/transcript?limit=200`, {
                headers: {
                  "Content-Type": "application/json",
                  "x-host-secret": hostSecret,
                },
              });
              if (!resp.ok) {
                return { transcripts: [], ok: false, status: resp.status };
              }
              const body = await resp.json();
              if (body && body.success && Array.isArray(body.transcripts)) {
                return { transcripts: body.transcripts, ok: true, status: resp.status };
              }
              return { transcripts: [], ok: true, status: resp.status };
            } catch (err) {
              console.warn("transcript fetch error for hostSecret:", err);
              return { transcripts: [], ok: false, status: 0 };
            }
          })
        );

        // Flatten hostSecret results into serverItems
        const hostServerItems = [];
        let anyHostServerHad = false;
        for (const r of allResults) {
          if (r.ok && r.transcripts.length > 0) anyHostServerHad = true;
          for (const t of r.transcripts || []) {
            hostServerItems.push({
              _id: t._id || t.id || null,
              meeting_code: (t.meetingCode || t.meeting_code || "").toString().toUpperCase(),
              transcript: t.transcriptText || t.transcript || "",
              createdAt: t.createdAt || null,
              fileName: t.fileName || null,
              fromServer: true,
            });
          }
        }

        // Merge authItems + hostServerItems
        const mergedRaw = [...authItems, ...hostServerItems];

        const anyServerHad = authHad || anyHostServerHad;
        setServerHadTranscripts(anyServerHad);

        // If no server items and no host secrets, show empty
        if (mergedRaw.length === 0) {
          setRecentLocal([]);
          return;
        }

        // Dedupe and keep newest per meeting code
        const merged = dedupeByCodeKeepNewest(mergedRaw);
        setRecentLocal(merged);
      } catch (err) {
        console.warn("Could not load server transcripts:", err);
        setRecentLocal([]);
        setServerHadTranscripts(false);
      }
    })();

    return () => {};
  }, []);

  function showSnack(message, severity = "success") {
    setSnackMsg(message);
    setSnackSeverity(severity);
    setSnackOpen(true);
  }

  function extractRoomFromInput(input) {
    let roomId = input.trim();
    try {
      const url = new URL(roomId);
      const segs = url.pathname.split("/").filter(Boolean);
      if (segs.length) roomId = segs.pop();
    } catch {}
    return roomId;
  }

  async function createRoom() {
    if (!name.trim()) {
      showSnack("Please enter your name first.", "error");
      return;
    }
    try {
      const token = localStorage.getItem("token");
      const headers = { "Content-Type": "application/json" };
      if (token) headers.Authorization = `Bearer ${token}`;

      const res = await fetch(`${API_BASE}/rooms`, {
        method: "POST",
        headers,
        body: JSON.stringify({ hostName: name.trim() }),
      });
      if (!res.ok) {
        const txt = await res.text();
        console.error("createRoom failed", res.status, txt);
        throw new Error("Failed to create room");
      }
      const { roomCode, hostSecret } = await res.json();
      const link = `${window.location.origin}/room/${roomCode.toUpperCase()}`;

      setRoom(link);

      localStorage.setItem("displayName", name.trim());
      // store hostSecret so this browser can prove it's the host later
      localStorage.setItem(
        `host:${roomCode.toUpperCase()}`,
        JSON.stringify({ hostName: name.trim(), hostSecret: hostSecret || null, createdAt: new Date().toISOString() })
      );
      navigate(`/room/${roomCode.toUpperCase()}`);
    } catch (err) {
      console.error(err);
      showSnack("Unable to create room.", "error");
    }
  }

  async function copyLink() {
    if (!name.trim()) {
      showSnack("Enter your name before creating a link", "error");
      return;
    }
    try {
      const token = localStorage.getItem("token");
      const headers = { "Content-Type": "application/json" };
      if (token) headers.Authorization = `Bearer ${token}`;

      const res = await fetch(`${API_BASE}/rooms`, {
        method: "POST",
        headers,
        body: JSON.stringify({ hostName: name.trim() }),
      });
      if (!res.ok) {
        const txt = await res.text();
        console.error("create room (copyLink) failed", res.status, txt);
        throw new Error("Failed to create room");
      }
      const { roomCode, hostSecret } = await res.json();
      const link = `${window.location.origin}/room/${roomCode.toUpperCase()}`;

      setRoom(link);

      localStorage.setItem("displayName", name.trim());
      localStorage.setItem(
        `host:${roomCode.toUpperCase()}`,
        JSON.stringify({ hostName: name.trim(), hostSecret: hostSecret || null, createdAt: new Date().toISOString() })
      );

      const copied = await copyToClipboard(link);
      if (copied) {
        showSnack("Link copied to clipboard", "success");
      } else {
        showSnack(`Copy failed — link: ${link}`, "warning");
      }
    } catch (err) {
      console.error(err);
      showSnack("Unable to create room link.", "error");
    }
  }

  async function joinRoom() {
    if (!room.trim()) {
      showSnack("Enter room code or link", "error");
      return;
    }
    const roomId = extractRoomFromInput(room).toUpperCase();
    try {
      const res = await fetch(`${API_BASE}/rooms/${roomId}`);
      const text = await res.text();
      if (!res.ok) {
        console.error("[joinRoom] failed:", res.status, text);
        throw new Error("Room not found");
      }
      localStorage.setItem("displayName", name.trim() || "Guest");
      navigate(`/room/${roomId}`);
    } catch (err) {
      console.error("joinRoom error:", err);
      showSnack("Room does not exist or has expired.", "error");
    }
  }

  // --- Logout handler that uses AuthContext.logout and clears only displayName (preserves host:* local secrets as fallback) ---
  async function handleLogout() {
    try {
      if (logout) {
        await logout(true); // redirect = true
      } else {
        localStorage.removeItem("token");
        try {
          navigate("/login");
        } catch {
          window.location.href = "/login";
        }
      }
    } catch (err) {
      console.warn("logout encountered an error:", err);
    } finally {
      // Keep host:* keys so device-local host sessions still work as a fallback.
      try {
        localStorage.removeItem("displayName");
      } catch (e) {
        console.warn("failed to clear displayName:", e);
      }
    }
  }

  return (
    <div className="home-container">
      {/* Floating history + logout buttons (top-right) */}
      <div style={{ position: "fixed", top: 14, right: 14, display: "flex", gap: 8, zIndex: 50 }}>
        <button
          type="button"
          className="history-btn history-float"
          onClick={(e) => {
            e.preventDefault();
            try {
              navigate("/history");
            } catch (err) {
              console.error("navigate failed:", err);
              window.location.href = "/history";
            }
          }}
          title="History"
          aria-label="Open history"
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" strokeWidth="1.8" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" focusable="false">
            <path d="M3 3v5h5" />
            <path d="M21 12a9 9 0 1 1-9-9" />
            <path d="M12 7v6l4 2" />
          </svg>
        </button>

        <button
          type="button"
          className="history-btn history-float"
          onClick={(e) => {
            e.preventDefault();
            handleLogout();
          }}
          title="Logout"
          aria-label="Logout"
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" strokeWidth="1.8" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" focusable="false">
            <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
            <path d="M16 17l5-5-5-5" />
            <path d="M21 12H9" />
          </svg>
        </button>
      </div>

      <div className="home-card">
        <h1 className="title">Welcome to SkyMeet</h1>
        <p className="subtitle">Create instant video rooms and invite friends in seconds</p>

        <div className="form-group">
          <label>Your name:</label>
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Display name" />
        </div>

        <div className="button-row">
          <button className="btn btn-primary" onClick={createRoom}>🚀 Create New Room</button>
          <button className="btn btn-secondary" onClick={copyLink}>🔗 Create & Copy Link</button>
        </div>

        <div className="form-group">
          <label>Join existing room / paste link:</label>
          <input value={room} onChange={(e) => setRoom(e.target.value)} placeholder="Room code or full link" />
          <button className="btn btn-primary full-width" onClick={joinRoom}>Join Room</button>
        </div>

        <small className="tip">💡 Allow camera & microphone when prompted. Share your link to invite others.</small>
      </div>

      {/* Recent transcripts (server-only via hostSecret or authenticated owner) */}
      <div className="home-card" style={{ marginTop: 16 }}>
        <h3>Your recent transcripts</h3>

        {!TRANSCRIPTS_ENABLED && (
          <div style={{ color: "#a00", marginTop: 8 }}>
            Transcript service is not available on this hosted build due to Render free instance limitations.
            <div style={{ marginTop: 6, fontSize: 12, color: "#666" }}>
              You can still host meetings — local recording will run in your browser but automated transcription is disabled.
            </div>
          </div>
        )}

        {recentLocal.length === 0 && TRANSCRIPTS_ENABLED && (
          <>
            {!serverHadTranscripts ? (
              <div style={{ color: "#666" }}>No transcripts yet — host a meeting and end it to generate one.</div>
            ) : (
              <div style={{ color: "#666" }}>
                No transcripts available on this device. Sign in to view your transcripts across devices (if you were the host).
                <div style={{ marginTop: 8, fontSize: 12, color: "#999" }}>
                  Tip: Transcripts are visible to the meeting owner when they sign in. Local host secrets (host:ROOMCODE) are also supported as a fallback.
                </div>
              </div>
            )}
          </>
        )}

        {recentLocal.map((t, i) => {
          const key = getTranscriptKey(t, i);
          const isExpanded = !!expandedTranscripts[key];

          return (
            <div key={key} className="recent-item" style={{ marginTop: 12 }}>
              <div className="recent-head">
                <div className="recent-code">{t.meeting_code || "UNKNOWN"}</div>
                <div className="recent-date">{t.createdAt ? new Date(t.createdAt).toLocaleString() : "Unknown date"}</div>
              </div>

              <div className={`transcript-preview ${isExpanded ? "expanded" : ""}`}>{t.transcript || "(empty transcript)"}</div>

              <div className="recent-controls">
                <button className="load-more" onClick={() => setExpandedTranscripts((prev) => ({ ...prev, [key]: !prev[key] }))}>
                  {isExpanded ? "Show less" : "Load more"}
                </button>

                <button
                  onClick={() => {
                    const blob = new Blob([t.transcript || ""], { type: "text/plain" });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement("a");
                    a.href = url; a.download = `${(t.meeting_code || "transcript").toString().replace(/[^a-z0-9_\-]/gi, "_")}.txt`; a.click();
                    URL.revokeObjectURL(url);
                  }}
                  className="small-btn"
                >
                  Download .txt
                </button>
              </div>
            </div>
          );
        })}
      </div>

      <Snackbar
        open={snackOpen}
        autoHideDuration={3500}
        onClose={() => setSnackOpen(false)}
        anchorOrigin={{ vertical: "bottom", horizontal: "center" }}
      >
        <Alert onClose={() => setSnackOpen(false)} severity={snackSeverity} sx={{ width: "100%" }}>
          {snackMsg}
        </Alert>
      </Snackbar>
    </div>
  );
}
