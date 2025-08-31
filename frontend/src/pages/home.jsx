import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import "../styles/home.css";
import { Snackbar, Alert } from "@mui/material";

const SERVER_BASE = process.env.REACT_APP_SERVER_URL || "http://localhost:8000";
const API_BASE = process.env.REACT_APP_API_URL || `${SERVER_BASE}/api/v1`;

async function copyToClipboard(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    // fallback for older browsers
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

  const [name, setName] = useState(localStorage.getItem("displayName") || "");
  const [room, setRoom] = useState("");
  const [recentLocal, setRecentLocal] = useState([]);
  const [expandedTranscripts, setExpandedTranscripts] = useState({});

  // Snackbar state
  const [snackOpen, setSnackOpen] = useState(false);
  const [snackMsg, setSnackMsg] = useState("");
  const [snackSeverity, setSnackSeverity] = useState("success");

  useEffect(() => {
    (async () => {
      try {
        const resp = await fetch(`${API_BASE}/transcript`);
        if (!resp.ok) throw new Error(`Server ${resp.status}`);
        const body = await resp.json();
        if (body && body.success && Array.isArray(body.transcripts)) {
          const serverItems = body.transcripts.map((t) => ({
            _id: t._id || t.id || null,
            meeting_code: t.meetingCode,
            transcript: t.transcriptText,
            createdAt: t.createdAt,
            fileName: t.fileName || null,
            fromServer: true,
          }));
          const merged = dedupeByCodeKeepNewest(serverItems);
          setRecentLocal(merged);
        }
      } catch (err) {
        console.warn("Could not load server transcripts:", err);
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
      const res = await fetch(`${API_BASE}/rooms`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ hostName: name.trim() }),
      });
      if (!res.ok) {
        const txt = await res.text();
        console.error("createRoom failed", res.status, txt);
        throw new Error("Failed to create room");
      }
      const { roomCode } = await res.json();
      const link = `${window.location.origin}/room/${roomCode.toUpperCase()}`;

      // fill the join input with the generated link (so user can see/copy it)
      setRoom(link);

      localStorage.setItem("displayName", name.trim());
      localStorage.setItem(
        `host:${roomCode.toUpperCase()}`,
        JSON.stringify({ hostName: name.trim(), createdAt: new Date().toISOString() })
      );
      // Navigate to room immediately (no blocking alert)
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
      const res = await fetch(`${API_BASE}/rooms`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ hostName: name.trim() }),
      });
      if (!res.ok) {
        const txt = await res.text();
        console.error("create room (copyLink) failed", res.status, txt);
        throw new Error("Failed to create room");
      }
      const { roomCode } = await res.json();
      const link = `${window.location.origin}/room/${roomCode.toUpperCase()}`;

      // fill the join input with the link so it's visible / editable
      setRoom(link);

      localStorage.setItem("displayName", name.trim());
      localStorage.setItem(
        `host:${roomCode.toUpperCase()}`,
        JSON.stringify({ hostName: name.trim(), createdAt: new Date().toISOString() })
      );

      const copied = await copyToClipboard(link);
      if (copied) {
        showSnack("Link copied to clipboard", "success");
      } else {
        // show error but include the link so user can copy manually
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

  return (
    <div className="home-container">
      {/* Floating history button (top-right) */}
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

      {/* Recent transcripts (server-only) */}
      <div className="home-card" style={{ marginTop: 16 }}>
        <h3>Your recent transcripts</h3>
        {recentLocal.length === 0 && <div style={{ color: "#666" }}>No transcripts yet — host a meeting and end it to generate one.</div>}
        {recentLocal.map((t, i) => {
          const key = getTranscriptKey(t, i);
          const isExpanded = !!expandedTranscripts[key];

          return (
            <div key={key} className="recent-item" style={{ marginTop: 12 }}>
              <div className="recent-head">
                <div className="recent-code">{t.meeting_code}</div>
                <div className="recent-date">{new Date(t.createdAt).toLocaleString()}</div>
              </div>

              <div className={`transcript-preview ${isExpanded ? "expanded" : ""}`}>{t.transcript}</div>

              <div className="recent-controls">
                <button className="load-more" onClick={() => setExpandedTranscripts((prev) => ({ ...prev, [key]: !prev[key] }))}>
                  {isExpanded ? "Show less" : "Load more"}
                </button>

                <button
                  onClick={() => {
                    const blob = new Blob([t.transcript], { type: "text/plain" });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement("a");
                    a.href = url; a.download = `${t.meeting_code}.txt`; a.click();
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
