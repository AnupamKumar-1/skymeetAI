import React, { useEffect, useState, useCallback } from "react";
import "../styles/history.css";

function formatDate(dateStr) {
  if (!dateStr) return "—";
  const d = new Date(dateStr);
  if (isNaN(d)) return "—";
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

function formatTime(dateStr) {
  if (!dateStr) return "";
  const d = new Date(dateStr);
  if (isNaN(d)) return "";
  return d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
}

function formatDuration(joinedAt, leftAt) {
  if (!joinedAt || !leftAt) return null;
  const ms = new Date(leftAt) - new Date(joinedAt);
  if (ms <= 0) return null;
  const mins = Math.floor(ms / 60000);
  const secs = Math.floor((ms % 60000) / 1000);
  if (mins === 0) return `${secs}s`;
  return `${mins}m ${secs}s`;
}

function isMeetingActive(meeting) {
  if (meeting.active === false) return false;
  if (meeting.active === true) {
    const lastActivity = meeting.lastActivityAt || meeting.updatedAt || meeting.createdAt;
    if (lastActivity) {
      const idleMs = Date.now() - new Date(lastActivity).getTime();
      if (idleMs > 2 * 60 * 60 * 1000) return false;
    }
    return true;
  }
  const lastActivity = meeting.lastActivityAt || meeting.updatedAt || meeting.createdAt;
  if (!lastActivity) return false;
  const idleMs = Date.now() - new Date(lastActivity).getTime();
  return idleMs < 30 * 60 * 1000;
}

function dedupeParticipants(participants) {
  const byUserId = new Map();
  const byName = new Map();
  const result = [];

  for (const p of participants) {
    const uid = p.userId ? String(p.userId).trim() : null;
    const name = (p.name || "Guest").trim().toLowerCase();
    const isInitEntry = typeof p.socketId === "string" && p.socketId.startsWith("init-");
    const ts = p.joinedAt ? new Date(p.joinedAt).getTime() : 0;

    if (uid && uid !== "null" && uid !== "undefined") {
      if (byUserId.has(uid)) {
        const existing = byUserId.get(uid);
        const existingTs = existing.joinedAt ? new Date(existing.joinedAt).getTime() : 0;
        if (!isInitEntry && ts >= existingTs) {
          byUserId.set(uid, p);
        }
      } else {
        byUserId.set(uid, p);
      }
    } else {
      if (byName.has(name)) {
        const existing = byName.get(name);
        const existingTs = existing.joinedAt ? new Date(existing.joinedAt).getTime() : 0;
        if (!isInitEntry && ts >= existingTs) {
          byName.set(name, p);
        }
      } else {
        byName.set(name, p);
      }
    }
  }

  const seenNames = new Set();
  for (const p of byUserId.values()) {
    result.push(p);
    seenNames.add((p.name || "Guest").trim().toLowerCase());
  }
  for (const [name, p] of byName.entries()) {
    if (!seenNames.has(name)) {
      result.push(p);
    }
  }

  return result;
}

function AvatarPip({ name, index }) {
  const hues = [210, 255, 172, 340, 30, 140, 190, 300];
  const hue = hues[index % hues.length];
  return (
    <div className="hs-avatar-pip" title={name} style={{ "--hue": hue }}>
      {(name || "?")[0].toUpperCase()}
    </div>
  );
}

function ParticipantRow({ participant, index }) {
  const duration = formatDuration(participant.joinedAt, participant.leftAt);
  return (
    <div className="hs-participant-row">
      <AvatarPip name={participant.name} index={index} />
      <div className="hs-participant-info">
        <span className="hs-participant-name">{participant.name || "Guest"}</span>
        {duration && <span className="hs-participant-duration">{duration}</span>}
      </div>
      <div className="hs-participant-times">
        {participant.joinedAt && (
          <span className="hs-participant-time-chip hs-joined">
            <svg width="8" height="8" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
              <path d="M15 3h6v6M14 10l6.1-6.1" />
            </svg>
            {formatTime(participant.joinedAt)}
          </span>
        )}
        {participant.leftAt && (
          <span className="hs-participant-time-chip hs-left">
            <svg width="8" height="8" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
              <path d="M9 21H5a2 2 0 01-2-2V5a2 2 0 012-2h4" />
              <polyline points="16 17 21 12 16 7" />
            </svg>
            {formatTime(participant.leftAt)}
          </span>
        )}
      </div>
    </div>
  );
}

function MeetingCard({ meeting, isExpanded, onToggle }) {
  const dedupedParticipants = dedupeParticipants(meeting.participants || []);
  const totalCount = dedupedParticipants.length;
  const active = isMeetingActive(meeting);

  return (
    <div className={`hs-meeting-card ${isExpanded ? "hs-meeting-card-expanded" : ""}`}>
      <button className="hs-meeting-card-header" onClick={onToggle} aria-expanded={isExpanded}>
        <div className="hs-meeting-card-left">
          <div className="hs-meeting-role-badge">
            {meeting.isHost ? (
              <span className="hs-role-host">
                <svg width="9" height="9" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" />
                </svg>
                Host
              </span>
            ) : (
              <span className="hs-role-participant">
                <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                  <path d="M17 21v-2a4 4 0 00-4-4H5a4 4 0 00-4 4v2" />
                  <circle cx="9" cy="7" r="4" />
                </svg>
                Participant
              </span>
            )}
          </div>
          <div className="hs-meeting-code-block">
            <span className="hs-meeting-code">{meeting.meetingCode || "—"}</span>
          </div>
        </div>

        <div className="hs-meeting-card-meta">
          <div className="hs-meeting-host-row">
            <svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor" className="hs-host-icon">
              <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" />
            </svg>
            <span className="hs-meeting-host-name">{meeting.hostName || "Unknown Host"}</span>
          </div>
          <div className="hs-meeting-date-row">
            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <rect x="3" y="4" width="18" height="18" rx="2" ry="2" />
              <line x1="16" y1="2" x2="16" y2="6" />
              <line x1="8" y1="2" x2="8" y2="6" />
              <line x1="3" y1="10" x2="21" y2="10" />
            </svg>
            <span>{formatDate(meeting.createdAt)}</span>
            {meeting.createdAt && (
              <span className="hs-meeting-time">{formatTime(meeting.createdAt)}</span>
            )}
          </div>
        </div>

        <div className="hs-meeting-card-right">
          <div className="hs-participants-preview">
            {dedupedParticipants.slice(0, 4).map((p, i) => (
              <AvatarPip key={i} name={p.name} index={i} />
            ))}
            {totalCount > 4 && (
              <div className="hs-avatar-overflow">+{totalCount - 4}</div>
            )}
          </div>
          <span className="hs-participant-count">
            {totalCount} {totalCount === 1 ? "person" : "people"}
          </span>
        </div>

        <div className="hs-meeting-card-chevron">
          <svg
            width="14" height="14"
            viewBox="0 0 24 24" fill="none"
            stroke="currentColor" strokeWidth="2.2" strokeLinecap="round"
            className={isExpanded ? "hs-chevron-open" : ""}
          >
            <path d="M6 9l6 6 6-6" />
          </svg>
        </div>
      </button>

      {isExpanded && (
        <div className="hs-meeting-card-body">
          <div className="hs-meeting-detail-grid">
            <div className="hs-detail-section">
              <div className="hs-detail-section-label">
                <svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" />
                </svg>
                Host
              </div>
              <div className="hs-host-card">
                <AvatarPip name={meeting.hostName} index={99} />
                <div>
                  <div className="hs-host-card-name">{meeting.hostName || "Unknown"}</div>
                  <div className="hs-host-card-sub">Meeting organiser</div>
                </div>
              </div>
            </div>

            {meeting.link && (
              <div className="hs-detail-section">
                <div className="hs-detail-section-label">
                  <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round">
                    <path d="M10 13a5 5 0 007.54.54l3-3a5 5 0 00-7.07-7.07l-1.72 1.71" />
                    <path d="M14 11a5 5 0 00-7.54-.54l-3 3a5 5 0 007.07 7.07l1.71-1.71" />
                  </svg>
                  Meeting Link
                  {!active && (
                    <span className="hs-link-ended-badge">
                      <svg width="8" height="8" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                        <circle cx="12" cy="12" r="10" />
                        <line x1="15" y1="9" x2="9" y2="15" />
                        <line x1="9" y1="9" x2="15" y2="15" />
                      </svg>
                      Ended
                    </span>
                  )}
                </div>
                {active ? (
                  <a
                    href={meeting.link}
                    className="hs-meeting-link"
                    target="_blank"
                    rel="noopener noreferrer"
                    onClick={(e) => e.stopPropagation()}
                  >
                    {meeting.link}
                    <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                      <path d="M18 13v6a2 2 0 01-2 2H5a2 2 0 01-2-2V8a2 2 0 012-2h6" />
                      <polyline points="15 3 21 3 21 9" />
                      <line x1="10" y1="14" x2="21" y2="3" />
                    </svg>
                  </a>
                ) : (
                  <div className="hs-meeting-link hs-meeting-link-disabled" aria-disabled="true" title="This meeting has ended">
                    {meeting.link}
                    <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                      <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
                      <path d="M7 11V7a5 5 0 0110 0v4" />
                    </svg>
                  </div>
                )}
              </div>
            )}
          </div>

          {dedupedParticipants.length > 0 && (
            <div className="hs-participants-section">
              <div className="hs-detail-section-label">
                <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round">
                  <path d="M17 21v-2a4 4 0 00-4-4H5a4 4 0 00-4 4v2" />
                  <circle cx="9" cy="7" r="4" />
                  <path d="M23 21v-2a4 4 0 00-3-3.87M16 3.13a4 4 0 010 7.75" />
                </svg>
                Participants ({dedupedParticipants.length})
              </div>
              <div className="hs-participants-list">
                {dedupedParticipants.map((p, i) => (
                  <ParticipantRow key={p.userId || p.socketId || i} participant={p} index={i} />
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default function HistoryPanel({ getHistoryOfUser, userData, authLoading }) {
  const [meetings, setMeetings] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expandedId, setExpandedId] = useState(null);
  const [filter, setFilter] = useState("all");
  const [searchQuery, setSearchQuery] = useState("");

  const loadHistory = useCallback(async () => {
    if (!getHistoryOfUser) return;
    setLoading(true);
    setError(null);
    try {
      const data = await getHistoryOfUser();
      const list = Array.isArray(data) ? data : [];
      list.sort((a, b) => {
        const aTs = a.createdAt ? new Date(a.createdAt).getTime() : 0;
        const bTs = b.createdAt ? new Date(b.createdAt).getTime() : 0;
        return bTs - aTs;
      });
      setMeetings(list);
    } catch {
      setError("Unable to load meeting history.");
    } finally {
      setLoading(false);
    }
  }, [getHistoryOfUser]);

  const silentRefresh = useCallback(async () => {
    if (!getHistoryOfUser) return;
    try {
      const data = await getHistoryOfUser();
      const list = Array.isArray(data) ? data : [];
      list.sort((a, b) => {
        const aTs = a.createdAt ? new Date(a.createdAt).getTime() : 0;
        const bTs = b.createdAt ? new Date(b.createdAt).getTime() : 0;
        return bTs - aTs;
      });
      setMeetings(list);
    } catch {
    }
  }, [getHistoryOfUser]);

  useEffect(() => {
    if (!authLoading) loadHistory();
  }, [authLoading, loadHistory]);

  useEffect(() => {
    if (authLoading) return;
    const id = setInterval(silentRefresh, 30_000);
    return () => clearInterval(id);
  }, [authLoading, silentRefresh]);

  const filtered = meetings.filter((m) => {
    if (filter === "hosted" && !m.isHost) return false;
    if (filter === "joined" && m.isHost) return false;
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      const code = (m.meetingCode || "").toLowerCase();
      const host = (m.hostName || "").toLowerCase();
      const parts = (m.participants || []).map((p) => (p.name || "").toLowerCase()).join(" ");
      if (!code.includes(q) && !host.includes(q) && !parts.includes(q)) return false;
    }
    return true;
  });

  const hostedCount = meetings.filter((m) => m.isHost).length;
  const joinedCount = meetings.filter((m) => !m.isHost).length;

  function toggleExpand(id) {
    setExpandedId((prev) => (prev === id ? null : id));
  }

  return (
    <div className="hs-panel">
      <div className="hs-panel-header">
        <div className="hs-panel-title-group">
          <div className="hs-panel-title">Meeting History</div>
          <div className="hs-panel-sub">
            {loading ? "Loading…" : `${meetings.length} meeting${meetings.length !== 1 ? "s" : ""} in total`}
          </div>
        </div>
        <button
          className="hs-refresh-btn"
          onClick={loadHistory}
          disabled={loading}
          aria-label="Refresh history"
          title="Refresh"
        >
          <svg
            width="13" height="13"
            viewBox="0 0 24 24" fill="none"
            stroke="currentColor" strokeWidth="2.2" strokeLinecap="round"
            className={loading ? "hs-spin" : ""}
          >
            <path d="M1 4v6h6" />
            <path d="M23 20v-6h-6" />
            <path d="M20.49 9A9 9 0 005.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 013.51 15" />
          </svg>
        </button>
      </div>

      <div className="hs-divider" />

      {!loading && meetings.length > 0 && (
        <>
          <div className="hs-stats-row">
            <div className="hs-stat-item">
              <span className="hs-stat-val">{meetings.length}</span>
              <span className="hs-stat-label">Total</span>
            </div>
            <div className="hs-stat-sep" />
            <div className="hs-stat-item">
              <span className="hs-stat-val hs-stat-host">{hostedCount}</span>
              <span className="hs-stat-label">Hosted</span>
            </div>
            <div className="hs-stat-sep" />
            <div className="hs-stat-item">
              <span className="hs-stat-val hs-stat-joined">{joinedCount}</span>
              <span className="hs-stat-label">Joined</span>
            </div>
          </div>

          <div className="hs-controls">
            <div className="hs-search-wrap">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" className="hs-search-icon">
                <circle cx="11" cy="11" r="8" />
                <path d="M21 21l-4.35-4.35" />
              </svg>
              <input
                type="text"
                className="hs-search"
                placeholder="Search code, host, participant…"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
              {searchQuery && (
                <button className="hs-search-clear" onClick={() => setSearchQuery("")} aria-label="Clear search">
                  <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                    <path d="M18 6L6 18M6 6l12 12" />
                  </svg>
                </button>
              )}
            </div>
            <div className="hs-filter-tabs">
              {["all", "hosted", "joined"].map((f) => (
                <button
                  key={f}
                  className={`hs-filter-tab ${filter === f ? "hs-filter-tab-active" : ""}`}
                  onClick={() => setFilter(f)}
                >
                  {f.charAt(0).toUpperCase() + f.slice(1)}
                </button>
              ))}
            </div>
          </div>
        </>
      )}

      <div className="hs-panel-body">
        {loading && (
          <div className="hs-empty-state">
            <div className="hs-loading-dots">
              <span /><span /><span />
            </div>
            <span className="hs-empty-sub">Fetching your meeting history…</span>
          </div>
        )}

        {!loading && error && (
          <div className="hs-empty-state">
            <div className="hs-error-icon">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
                <circle cx="12" cy="12" r="10" />
                <path d="M12 8v4M12 16h.01" />
              </svg>
            </div>
            <p className="hs-empty-title">Something went wrong</p>
            <p className="hs-empty-sub">{error}</p>
            <button className="hs-retry-btn" onClick={loadHistory}>Try again</button>
          </div>
        )}

        {!loading && !error && meetings.length === 0 && (
          <div className="hs-empty-state">
            <div className="hs-empty-orb">
              <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="rgba(59,130,246,0.45)" strokeWidth="1.3" strokeLinecap="round">
                <path d="M3 3v5h5" />
                <path d="M21 12a9 9 0 11-9-9" />
                <path d="M12 7v6l4 2" />
              </svg>
            </div>
            <p className="hs-empty-title">No meeting history yet</p>
            <p className="hs-empty-sub">Create or join a meeting to see your history here.</p>
          </div>
        )}

        {!loading && !error && meetings.length > 0 && filtered.length === 0 && (
          <div className="hs-empty-state">
            <div className="hs-empty-orb">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="rgba(59,130,246,0.4)" strokeWidth="1.4" strokeLinecap="round">
                <circle cx="11" cy="11" r="8" />
                <path d="M21 21l-4.35-4.35" />
              </svg>
            </div>
            <p className="hs-empty-title">No matches</p>
            <p className="hs-empty-sub">Try adjusting your search or filter.</p>
          </div>
        )}

        {!loading && !error && filtered.length > 0 && (
          <div className="hs-meetings-list">
            {filtered.map((meeting, i) => {
              const id = meeting._id || meeting.meetingCode || i;
              return (
                <MeetingCard
                  key={id}
                  meeting={meeting}
                  isExpanded={expandedId === id}
                  onToggle={() => toggleExpand(id)}
                />
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}