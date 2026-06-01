import React, { useContext, useEffect, useRef, useState } from "react";
import { AuthContext } from "../contexts/AuthContext";
import { useNavigate } from "react-router-dom";
import "../styles/history.css";

const _isTrivialName = (n) => {
  if (!n) return true;
  const s = String(n).trim().toLowerCase();
  if (!s || s.length <= 2) return true;
  return ["guest", "participant", "host", "unknown", "user"].includes(s);
};
const _toM = (s) => { if (!s) return 0; const t = new Date(s).getTime(); return Number.isFinite(t) ? t : 0; };
const _formatDate = (d) => {
  if (!d) return "Unknown";
  const dt = new Date(d);
  if (Number.isNaN(dt.getTime())) return "Invalid date";
  return `${String(dt.getDate()).padStart(2, "0")}/${String(dt.getMonth() + 1).padStart(2, "0")}/${dt.getFullYear()} · ${String(dt.getHours()).padStart(2, "0")}:${String(dt.getMinutes()).padStart(2, "0")}`;
};
const _participantName = (p) => { if (!p) return "Guest"; if (typeof p === "string") return p; return p?.name || p?.display || p?.username || "Guest"; };
const _initials = (name) => { if (!name || typeof name !== "string") return "G"; return name.trim().split(/\s+/).slice(0, 2).map(p => p[0]?.toUpperCase() ?? "").join("") || name.slice(0, 1).toUpperCase(); };
const _userMatchesParticipant = (user, participant) => {
  if (!user || !participant) return false;
  const pId = participant?._id || participant?.id || participant?.userId || null;
  const pUsername = participant?.username || participant?.userName || null;
  const pEmail = participant?.email || null;
  const pName = typeof participant === "string" ? participant : participant?.name || participant?.display || null;
  const uId = user?._id || user?.id || null;
  if (uId && pId && String(uId) === String(pId)) return true;
  if (user?.username && pUsername && String(user.username).toLowerCase() === String(pUsername).toLowerCase()) return true;
  if (user?.email && pEmail && String(user.email).toLowerCase() === String(pEmail).toLowerCase()) return true;
  if (user?.name && pName && !_isTrivialName(user.name) && !_isTrivialName(pName) && String(user.name).trim().toLowerCase() === String(pName).trim().toLowerCase()) return true;
  return false;
};

export default function History() {
  const { getHistoryOfUser, userData, authLoading } = useContext(AuthContext);
  const [meetings, setMeetings] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expanded, setExpanded] = useState({});
  const [showAllFor, setShowAllFor] = useState({});
  const [snack, setSnack] = useState({ open: false, msg: '', severity: 'success' });
  const snackTimer = useRef(null);
  const routeTo = useNavigate();

  useEffect(() => {
    if (authLoading) return;
    let mounted = true;
    const toArrayShape = (res) => { if (!res) return []; if (Array.isArray(res)) return res; if (Array.isArray(res.meetings)) return res.meetings; if (Array.isArray(res.data)) return res.data; return []; };
    const readLocal = () => { try { const raw = localStorage.getItem("meeting_history_v1"); if (!raw) return []; const p = JSON.parse(raw); return Array.isArray(p) ? p : []; } catch { return []; } };
    const normParts = (rawParts) => {
      const arr = Array.isArray(rawParts) ? rawParts : [];
      const out = [], seen = new Set();
      for (const p of arr) {
        if (!p && p !== 0) continue;
        const obj = typeof p === "string" ? { name: p } : typeof p === "object" ? p : { name: String(p) };
        const key = obj?._id || obj?.id || obj?.username || obj?.email || (obj?.name ? String(obj.name).trim().toLowerCase() : null) || JSON.stringify(obj);
        if (!key || seen.has(key)) continue;
        seen.add(key); out.push(obj);
      }
      return out;
    };
    const normalize = (m) => {
      const meetingCode = m?.meetingCode || m?.code || m?.room || m?.meeting_code || "";
      const createdAt = m?.createdAt || m?.created_at || m?.date || m?.created || m?.timestamp || "";
      const hostName = m?.hostInfo?.name || m?.hostName || m?.host_name || (m?.host && typeof m.host === "object" && (m.host.name || m.host.username)) || (typeof m?.host === "string" ? m.host : null) || "Unknown";
      const participants = normParts(m?.participants || m?.attendees || m?.people || []);
      const link = m?.link || (meetingCode ? `${window.location.origin}/room/${encodeURIComponent(String(meetingCode).trim().toUpperCase())}` : null);
      const id = m?._id || m?.id || meetingCode || Math.random().toString(36).slice(2, 9);
      const hostId = m?.host?._id || m?.host?.id || m?.host_id || m?.hostId || null;
      return { id, meetingCode, createdAt, hostName, participants, link, raw: m, hostId };
    };
    const merge = (srv, loc) => {
      const out = [], seen = new Set();
      const keyFor = (item) => { const code = item?.meetingCode || item?.meeting_code || item?.code || item?.room; if (code) return String(code).trim().toUpperCase(); if (item?.id) return `ID:${item.id}`; return `RAW:${JSON.stringify(item).slice(0, 100)}`; };
      [...(srv || []), ...(loc || [])].forEach((x) => { const k = keyFor(x); if (!seen.has(k)) { out.push(x); seen.add(k); } });
      return out;
    };
    const fetch_ = async () => {
      setLoading(true); setError(null);
      try {
        const res = await getHistoryOfUser();
        const merged = merge(toArrayShape(res), readLocal());
        const sorted = merged.map(normalize).sort((a, b) => _toM(b.createdAt) - _toM(a.createdAt));
        if (mounted) setMeetings(sorted);
      } catch (err) {
        if (mounted) setError(err?.message || "Failed to load history");
      } finally {
        if (mounted) setLoading(false);
      }
    };
    fetch_();
    const onStorage = (ev) => { if (ev.key === "meeting_history_v1" && mounted) setTimeout(fetch_, 60); };
    const onCustom = () => { if (mounted) fetch_(); };
    window.addEventListener("storage", onStorage);
    window.addEventListener("meeting_history_updated", onCustom);
    return () => { mounted = false; window.removeEventListener("storage", onStorage); window.removeEventListener("meeting_history_updated", onCustom); };
  }, [getHistoryOfUser, userData?._id, authLoading]);

  const buildLink = (m) =>
    m?.link || (m?.meetingCode
      ? `${window.location.origin}/room/${encodeURIComponent(String(m.meetingCode).trim().toUpperCase())}`
      : null);

  const showSnack = (msg, severity = 'success') => {
    if (snackTimer.current) {
      clearTimeout(snackTimer.current);
    }

    setSnack({ open: true, msg, severity });

    snackTimer.current = setTimeout(() => {
      setSnack(s => ({ ...s, open: false }));
    }, 3000);
  };

  const copyLink = async (link) => {
    if (!link) return;
    try {
      if (navigator.clipboard?.writeText) await navigator.clipboard.writeText(link);
      else { const ta = document.createElement("textarea"); ta.value = link; Object.assign(ta.style, { position: "fixed", left: "-9999px" }); document.body.appendChild(ta); ta.select(); document.execCommand("copy"); document.body.removeChild(ta); }
      showSnack("Link copied to clipboard", "success");
    } catch { showSnack_("Failed to copy link", "error"); }
  };
  const toggleExpand = (id) => setExpanded(prev => ({ ...prev, [id]: !prev[id] }));
  const isCardHost = (m) => {
    if (!userData) return false;
    const hostId = m.hostId || (m.raw && (m.raw.host?._id || m.raw.host?.id || m.raw.hostId || m.raw.host_id));
    if (hostId && userData._id && String(hostId) === String(userData._id)) return true;
    const rawHost = m.raw?.host;
    if (rawHost && typeof rawHost === "object") {
      if (rawHost.username && userData.username && String(rawHost.username).toLowerCase() === String(userData.username).toLowerCase()) return true;
      if (rawHost.email && userData.email && String(rawHost.email).toLowerCase() === String(userData.email).toLowerCase()) return true;
    }
    if (m.hostName && userData.name && !_isTrivialName(m.hostName) && !_isTrivialName(userData.name) && String(m.hostName).trim().toLowerCase() === String(userData.name).trim().toLowerCase()) return true;
    return false;
  };
  const isParticipantHost = (pRaw, m) => {
    if (!pRaw || !m) return false;
    const hostId = m.hostId || (m.raw && (m.raw.host?._id || m.raw.host?.id || m.raw.hostId || m.raw.host_id)) || null;
    if (!hostId) return false;
    const pUserId = pRaw?._id || pRaw?.id || pRaw?.userId || pRaw?.user_id || pRaw?.meta?.userId || null;
    if (!pUserId) return false;
    if (!/^[a-f\d]{24}$/i.test(String(hostId)) || !/^[a-f\d]{24}$/i.test(String(pUserId))) return false;
    return String(pUserId) === String(hostId);
  };
  const PREVIEW_LIMIT = 6;

  return (
    <div className="hm-card hm-transcripts" style={{ overflowY: "auto", maxHeight: "calc(100vh - 120px)" }}>
      <div className="hm-card-header">
        <div>
          <div className="hm-card-title">Meeting history</div>
          <div className="hm-card-sub">Meetings you joined or hosted</div>
        </div>
        {!loading && meetings.length > 0 && (
          <span className="hm-tx-badge">{meetings.length} total</span>
        )}
      </div>
      <div className="hm-divider" />

      {(authLoading || loading) && (
        <div className="hm-tx-loading"><div className="hm-tx-loading-dots"><span /><span /><span /></div></div>
      )}

      {error && (
        <div className="hist-alert hist-alert--error" style={{ margin: "14px 16px" }}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden>
            <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
            <line x1="12" y1="9" x2="12" y2="13" /><line x1="12" y1="17" x2="12.01" y2="17" />
          </svg>
          <span>{error}</span>
        </div>
      )}

      {!loading && !userData && (
        <div className="hm-tx-empty">
          <div className="hm-tx-empty-icon" aria-hidden>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#64748b" strokeWidth="1.5" strokeLinecap="round">
              <circle cx="12" cy="8" r="4" /><path d="M4 20c0-4 3.6-7 8-7s8 3 8 7" />
            </svg>
          </div>
          <p>Sign in to view your meeting history.</p>
        </div>
      )}

      {!loading && userData && meetings.length === 0 && !error && (
        <div className="hm-tx-empty">
          <div className="hm-tx-empty-icon" aria-hidden>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#64748b" strokeWidth="1.5" strokeLinecap="round">
              <path d="M3 3v5h5" /><path d="M21 12a9 9 0 1 1-9-9" /><path d="M12 7v6l4 2" />
            </svg>
          </div>
          <p>No meeting history found. Join or host a meeting to see it here.</p>
        </div>
      )}

      <div style={{ padding: "4px 0 8px" }}>
        {!loading && meetings.map((m) => {
          const cardIsHost = isCardHost(m);
          const link = m.link || (m.meetingCode ? `${window.location.origin}/room/${encodeURIComponent(String(m.meetingCode).trim().toUpperCase())}` : null);
          const isExpanded = !!expanded[m.id];
          const showAll = !!showAllFor[m.id];
          const parts = Array.isArray(m.participants) ? m.participants : [];
          const visibleParts = showAll ? parts : parts.slice(0, PREVIEW_LIMIT);

          return (
            <div key={m.id} className="hist-card" style={{ margin: "0 12px 8px", borderRadius: "12px" }}>
              <div className="hist-card-header">
                <div className="hist-card-left">
                  <div className="hist-meta-row">
                    <span className="hist-code-chip">{m.meetingCode ? String(m.meetingCode).trim().toUpperCase() : "—"}</span>
                    <span className="hist-date-text">{_formatDate(m.createdAt)}</span>
                  </div>
                  <div className="hist-host-row">
                    <span className="hist-host-name">{m.hostName || "Unknown"}</span>
                    {cardIsHost && <span className="hist-host-badge">HOST</span>}
                  </div>
                  <div className="hist-participant-count">{parts.length} participant{parts.length !== 1 ? "s" : ""}</div>
                  {link && (
                    <div className="hist-link-row">
                      <a href={link} target="_blank" rel="noreferrer" className="hist-open-link">Open meeting</a>
                      <button className="hist-icon-btn" onClick={() => copyLink(link)} title="Copy link" aria-label="Copy meeting link">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden>
                          <rect x="9" y="9" width="13" height="13" rx="2" ry="2" /><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1" />
                        </svg>
                      </button>
                    </div>
                  )}
                </div>
                {parts.length > 0 && (
                  <button className="hist-expand-btn" onClick={() => toggleExpand(m.id)} aria-expanded={isExpanded}>
                    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden>
                      <path d="M17 21v-2a4 4 0 00-4-4H5a4 4 0 00-4 4v2M9 7a4 4 0 100 8 4 4 0 000-8z" />
                    </svg>
                    Participants
                    <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"
                      className={`hist-expand-chevron ${isExpanded ? "hist-expand-chevron--open" : "hist-expand-chevron--closed"}`} aria-hidden>
                      <path d="M6 9l6 6 6-6" />
                    </svg>
                  </button>
                )}
              </div>
              {isExpanded && (
                <>
                  <div className="hist-divider" />
                  <div className="hist-participants-grid">
                    {parts.length === 0 && <p className="hist-no-participants">No participants recorded.</p>}
                    {visibleParts.map((pRaw, idx) => {
                      const name = _participantName(pRaw);
                      const pIsHost = isParticipantHost(pRaw, m);
                      const pIsYou = _userMatchesParticipant(userData, pRaw);
                      const tileKey = pRaw?._id || pRaw?.id || `${name}-${idx}`;
                      return (
                        <div key={`${m.id}-p-${tileKey}`} className="hist-ptile">
                          <div className="hist-avatar">{_initials(name)}</div>
                          <div className="hist-ptile-info">
                            <div className="hist-ptile-name-row">
                              <span className="hist-p-name">{name}</span>
                              {pIsYou && <span className="hist-you-chip">YOU</span>}
                              {pIsHost && <span className="hist-host-chip">HOST</span>}
                            </div>
                            <div className="hist-role-row">
                              <span className={`hist-role-badge ${pIsHost ? "hist-role-badge--host" : "hist-role-badge--participant"}`}>
                                {pIsHost ? "Host" : "Participant"}
                              </span>
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                  {parts.length > PREVIEW_LIMIT && (
                    <button className="hist-show-more" onClick={() => setShowAllFor(prev => ({ ...prev, [m.id]: !prev[m.id] }))}>
                      {showAll ? "Show less" : `Show all ${parts.length} participants`}
                    </button>
                  )}
                </>
              )}
            </div>
          );
        })}
      </div>

      {snack.open && (
        <div className={`hist-snack hist-snack--${snack.severity}`} role="status" aria-live="polite">
          {snack.severity === "success"
            ? <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" aria-hidden><polyline points="20 6 9 17 4 12" /></svg>
            : <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" aria-hidden><path d="M18 6L6 18M6 6l12 12" /></svg>
          }
          {snack.msg}
        </div>
      )}
    </div>
  );
}