import React, { useEffect, useState, useContext, useRef, useCallback } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import io from "socket.io-client";
import "../styles/home.css";
import "../styles/history.css";
import { AuthContext } from "../contexts/AuthContext";
import { TRANSCRIPTS_ENABLED } from "../environment";
import TranscriptViewer from "./TranscriptViewer";
import HistoryPanel from "./history";
import { useRag } from "../hooks/useRag";
import UserProfileModal from "./UserProfileModal";

const SERVER_BASE = import.meta.env.VITE_SERVER_URL || "http://localhost:8000";
const API_BASE = import.meta.env.VITE_API_URL || `${SERVER_BASE}/api/v1`;
const SOCKET_SERVER_URL = import.meta.env.VITE_SOCKET_URL || SERVER_BASE;

const TRANSCRIPT_CACHE_KEY = "tx_cache";
const TRANSCRIPT_CACHE_TTL = 2 * 60 * 1000;
const TRANSCRIPTS_PER_PAGE = 5;
const PENDING_TRANSCRIPT_KEY = "pending_transcript_code";

async function fetchJSON(url, options = {}) {
  const token = localStorage.getItem("token");
  const headers = { "Content-Type": "application/json", ...(token ? { Authorization: `Bearer ${token}` } : {}), ...options.headers };
  const res = await fetch(url, { ...options, headers });
  const data = await res.json();
  return { ok: res.ok, status: res.status, data };
}

async function submitTranscriptRequest(meetingCode) {
  return fetchJSON(`${API_BASE}/transcript-requests`, {
    method: "POST",
    body: JSON.stringify({ meetingCode }),
  });
}

async function resolveTranscriptRequest(requestId, status) {
  return fetchJSON(`${API_BASE}/transcript-requests/${requestId}/resolve`, {
    method: "PATCH",
    body: JSON.stringify({ status }),
  });
}

async function loadHostPendingRequests() {
  return fetchJSON(`${API_BASE}/transcript-requests/host?status=pending`);
}

async function loadMyRequests() {
  return fetchJSON(`${API_BASE}/transcript-requests/mine`);
}

async function loadOwnedRooms() {
  return fetchJSON(`${API_BASE}/rooms/mine`);
}

async function loadParticipatedMeetings() {
  return fetchJSON(`${API_BASE}/users/get_all_activity`);
}

async function copyToClipboard(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    const textarea = document.createElement("textarea");
    textarea.value = text;
    Object.assign(textarea.style, { position: "fixed", opacity: 0 });
    document.body.appendChild(textarea);
    textarea.focus();
    textarea.select();
    try {
      document.execCommand("copy");
      document.body.removeChild(textarea);
      return true;
    } catch {
      document.body.removeChild(textarea);
      return false;
    }
  }
}

async function createRoomAndGetLink(name) {
  const token = localStorage.getItem("token");
  const headers = { "Content-Type": "application/json" };
  if (token) headers.Authorization = `Bearer ${token}`;

  const res = await fetch(`${API_BASE}/rooms`, {
    method: "POST",
    headers,
    body: JSON.stringify({ hostName: name.trim() }),
  });

  if (!res.ok) throw new Error("Failed to create room");

  const { roomCode, hostSecret } = await res.json();
  const code = roomCode.toUpperCase();
  const link = `${window.location.origin}/room/${code}`;

  if (!hostSecret) throw new Error("Server did not return a hostSecret");

  localStorage.setItem("displayName", name.trim());
  localStorage.setItem(
    `host:${code}`,
    JSON.stringify({
      hostName: name.trim(),
      hostSecret,
      meetingCode: code,
      createdAt: new Date().toISOString(),
    })
  );

  return { code, link };
}

function getTranscriptKey(item, index) {
  const id = item._id || item.id || "";
  const code = (item.meetingCode || "local").toString();
  const ts = item.createdAt ? String(new Date(item.createdAt).getTime()) : String(index);
  return `${code}__${id || ts}`;
}

function dedupeByCode(arr) {
  const map = new Map();
  for (const it of arr) {
    const code = (it.meetingCode || "").toUpperCase();
    if (!code) continue;
    const existing = map.get(code);
    if (!existing) {
      map.set(code, it);
    } else {
      const existTs = existing.createdAt ? new Date(existing.createdAt).getTime() : 0;
      const itTs = it.createdAt ? new Date(it.createdAt).getTime() : 0;
      if (itTs > existTs) map.set(code, it);
    }
  }
  return Array.from(map.values()).sort((a, b) => {
    const aTs = a.createdAt ? new Date(a.createdAt).getTime() : 0;
    const bTs = b.createdAt ? new Date(b.createdAt).getTime() : 0;
    return bTs - aTs;
  });
}

function normalizeTranscript(t) {
  const code = (t.meetingCode || t.meeting_code || "").toString().toUpperCase().trim();
  if (!code) return null;
  return {
    _id: t._id || t.id || null,
    meetingCode: code,
    hostId: t.hostId || t.host_id || null,
    transcriptText: t.transcriptText || t.transcript || t.metadata?.transcriptText || "",
    fileName: t.fileName || null,
    metadata: t.metadata || {},
    createdAt: t.createdAt ? new Date(t.createdAt) : null,
    aiSummary: t.aiSummary || null,
  };
}

function getCachedTranscripts() {
  try {
    const raw = sessionStorage.getItem(TRANSCRIPT_CACHE_KEY);
    if (!raw) return null;
    const { ts, data } = JSON.parse(raw);
    if (Date.now() - ts > TRANSCRIPT_CACHE_TTL) return null;
    return data;
  } catch {
    return null;
  }
}

function setCachedTranscripts(data) {
  try {
    sessionStorage.setItem(TRANSCRIPT_CACHE_KEY, JSON.stringify({ ts: Date.now(), data }));
  } catch { }
}

function cleanInvalidHosts() {
  Object.keys(localStorage)
    .filter((k) => k.startsWith("host:"))
    .forEach((k) => {
      try {
        const v = JSON.parse(localStorage.getItem(k));
        if (!v?.hostSecret) localStorage.removeItem(k);
      } catch {
        localStorage.removeItem(k);
      }
    });
}

function Snack({ msg, severity, open }) {
  return (
    <div className={`hm-snack hm-snack-${severity} ${open ? "hm-snack-show" : ""}`}>
      {severity === "success" && (
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
          <polyline points="20 6 9 17 4 12" />
        </svg>
      )}
      {severity === "error" && (
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
          <path d="M18 6L6 18M6 6l12 12" />
        </svg>
      )}
      <span>{msg}</span>
    </div>
  );
}

const EMOTION_COLORS = {
  joy: "#f59e0b", happy: "#f59e0b", sadness: "#60a5fa",
  anger: "#f87171", fear: "#a78bfa", surprise: "#34d399",
  disgust: "#fb923c", neutral: "#64748b",
};

function TranscriptItem({ t, onOpen, requestStatus, onRequest, isOwned: isOwnedProp }) {
  const segments = t.metadata?.segments ?? [];
  const speakers = [...new Set(segments.map((s) => s.speaker).filter(Boolean))];

  const emoCount = {};
  segments.forEach((s) => {
    const e = s.emotion?.toLowerCase() || "neutral";
    emoCount[e] = (emoCount[e] || 0) + 1;
  });
  const dominantEmo = Object.entries(emoCount).sort((a, b) => b[1] - a[1])[0]?.[0] || null;
  const emoColor = dominantEmo ? (EMOTION_COLORS[dominantEmo] || "#64748b") : null;

  const lastSeg = segments.at(-1);
  const duration = lastSeg?.end > 0 ? Math.floor(lastSeg.end) : null;
  function fmtDur(sec) {
    const m = Math.floor(sec / 60), s = sec % 60;
    return m > 0 ? `${m}m ${s}s` : `${s}s`;
  }

  const preview = segments.length > 0
    ? segments.map((s) => s.text).join(" ").slice(0, 140)
    : (t.transcriptText || "").trim().slice(0, 140);

  const dominantSpeaker = speakers[0] || null;
  const isOwned = isOwnedProp !== undefined ? isOwnedProp : !!localStorage.getItem(`host:${t.meetingCode}`);

  return (
    <div
      className="hm-tx-item hm-tx-item-v2"
      role="button"
      tabIndex={0}
      onClick={onOpen}
      onKeyDown={(e) => e.key === "Enter" && onOpen()}
    >
      <div className="hm-tx-v2-bar" style={{ background: emoColor || "rgba(56,189,248,0.4)" }} />
      <div className="hm-tx-v2-content">
        <div className="hm-tx-v2-top">
          <div className="hm-tx-v2-code">{t.meetingCode}</div>
          <div className="hm-tx-v2-meta">
            {duration !== null && <span className="hm-tx-v2-chip">{fmtDur(duration)}</span>}
            {segments.length > 0 && <span className="hm-tx-v2-chip">{segments.length} turns</span>}
            {dominantSpeaker && <span className="hm-tx-v2-chip">{dominantSpeaker}</span>}
          </div>
        </div>
        {preview && (
          <div className="hm-tx-v2-preview">
            {preview}{preview.length >= 140 ? "…" : ""}
          </div>
        )}
        <div className="hm-tx-v2-bottom">
          <span className="hm-tx-v2-date">
            {t.createdAt ? new Date(t.createdAt).toLocaleString(undefined, {
              month: "short", day: "numeric", hour: "2-digit", minute: "2-digit"
            }) : "Unknown date"}
          </span>
          {dominantEmo && (
            <span className="hm-tx-v2-emo" style={{ color: emoColor, borderColor: emoColor + "44" }}>
              {dominantEmo}
            </span>
          )}
          {!isOwned && requestStatus === "approved" && (
            <span className="hm-tx-v2-open">
              <span className="hm-txreq-badge hm-txreq-badge-requested">Requested Transcript</span>
              <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                <path d="M5 12h14M12 5l7 7-7 7" />
              </svg>
            </span>
          )}
          {!isOwned && requestStatus === "pending" && (
            <span className="hm-txreq-badge hm-txreq-badge-pending">Request sent</span>
          )}
          {isOwned && (
            <span className="hm-tx-v2-open">
              View transcript
              <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                <path d="M5 12h14M12 5l7 7-7 7" />
              </svg>
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

function ProcessingTranscriptCard({ meetingCode }) {
  return (
    <div className="hm-tx-item hm-tx-item-v2 hm-tx-processing" aria-live="polite" aria-label="Transcript processing">
      <div className="hm-tx-v2-bar hm-tx-processing-bar" />
      <div className="hm-tx-v2-content">
        <div className="hm-tx-v2-top">
          <div className="hm-tx-v2-code">{meetingCode}</div>
          <div className="hm-tx-v2-meta">
            <span className="hm-tx-v2-chip hm-tx-processing-chip">
              <span className="hm-tx-processing-dot" /><span className="hm-tx-processing-dot" /><span className="hm-tx-processing-dot" />
              Processing
            </span>
          </div>
        </div>
        <div className="hm-tx-processing-message">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden>
            <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
          </svg>
          Transcript is being uploaded and processed — it'll appear here in a moment…
        </div>
        <div className="hm-tx-processing-bar-track">
          <div className="hm-tx-processing-bar-fill" />
        </div>
      </div>
    </div>
  );
}

const EMOTION_COLORS_MAP = {
  joy: "#f59e0b", happy: "#f59e0b", sadness: "#60a5fa",
  anger: "#f87171", fear: "#a78bfa", surprise: "#34d399",
  disgust: "#fb923c", neutral: "#64748b",
};
const EMOTION_ICONS = {
  joy: "✦", happy: "✦", sadness: "◈", anger: "◆",
  fear: "◉", surprise: "◎", disgust: "◇", neutral: "○",
};
function emoColor(e) { return EMOTION_COLORS_MAP[(e || "neutral").toLowerCase()] || "#64748b"; }
function emoIcon(e) { return EMOTION_ICONS[(e || "neutral").toLowerCase()] || "○"; }
function fmtSec(sec = 0) {
  const m = Math.floor(sec / 60), s = Math.floor(sec % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

function ActivityPanel({ transcripts, onShowTranscripts }) {
  const allSegments = transcripts.flatMap(t => t.metadata?.segments ?? []);

  const totalSecs = transcripts.reduce((acc, t) => {
    const segs = t.metadata?.segments ?? [];
    const last = segs.at(-1);
    return acc + (last?.end > 0 ? Math.floor(last.end) : 0);
  }, 0);
  const totalMins = Math.floor(totalSecs / 60);
  const totalHrs = Math.floor(totalMins / 60);
  const displayTime = totalHrs > 0 ? `${totalHrs}h ${totalMins % 60}m` : totalMins > 0 ? `${totalMins}m` : totalSecs > 0 ? `${totalSecs}s` : "—";

  const allSpeakers = new Set(allSegments.map(s => s.speaker).filter(Boolean));

  const summarisedTx = transcripts.filter(t => t.aiSummary?.insights);
  const hasSummaries = summarisedTx.length > 0;

  const aiEmoAgg = {};
  summarisedTx.forEach(t => {
    const dist = t.aiSummary.insights.emotion_distribution || {};
    Object.entries(dist).forEach(([k, v]) => {
      const key = k.toLowerCase();
      aiEmoAgg[key] = (aiEmoAgg[key] || 0) + Number(v);
    });
  });
  const aiEmoTotal = Object.values(aiEmoAgg).reduce((a, b) => a + b, 0);
  const aiEmoEntries = Object.entries(aiEmoAgg).sort((a, b) => b[1] - a[1]).slice(0, 6);

  const liveEmoAgg = {};
  allSegments.forEach(s => {
    const e = (s.emotion || "neutral").toLowerCase();
    liveEmoAgg[e] = (liveEmoAgg[e] || 0) + 1;
  });
  const liveEmoTotal = Object.values(liveEmoAgg).reduce((a, b) => a + b, 0);

  const allDiscrepancies = summarisedTx.flatMap(t =>
    (t.aiSummary.insights.discrepancies || []).map(d => ({
      ...d,
      meetingCode: t.meetingCode,
      meetingDate: t.createdAt,
    }))
  ).slice(0, 6);

  const topicCount = {};
  summarisedTx.forEach(t => {
    (t.aiSummary.insights.top_topics || []).forEach(topic => {
      topicCount[topic] = (topicCount[topic] || 0) + 1;
    });
  });
  const topTopics = Object.entries(topicCount).sort((a, b) => b[1] - a[1]).slice(0, 10).map(([t]) => t);

  const paceValues = summarisedTx.map(t => t.aiSummary.insights.speaking_pace_wpm).filter(Boolean);
  const avgPace = paceValues.length ? Math.round(paceValues.reduce((a, b) => a + b, 0) / paceValues.length) : null;
  const totalWords = summarisedTx.reduce((acc, t) => acc + (t.aiSummary.insights.total_words || 0), 0);

  const allMoments = summarisedTx.flatMap(t =>
    (t.aiSummary.insights.emotional_moments || []).map(m => ({
      ...m,
      meetingCode: t.meetingCode,
    }))
  ).filter(m => m.emotion?.toLowerCase() !== "neutral").slice(0, 4);

  const hasData = transcripts.length > 0;

  return (
    <div className="hm-card hm-activity-panel">
      <div className="hm-card-header">
        <div>
          <div className="hm-card-title">Your activity</div>
          <div className="hm-card-sub">
            {hasSummaries
              ? `AI insights from ${summarisedTx.length} of ${transcripts.length} meeting${transcripts.length !== 1 ? "s" : ""}`
              : "Insights from your meetings"}
          </div>
        </div>
        {hasData && (
          <button className="hm-tx-badge hm-activity-tx-btn" onClick={onShowTranscripts} title="View transcripts">
            {transcripts.length} meeting{transcripts.length !== 1 ? "s" : ""}
          </button>
        )}
      </div>

      <div className="hm-divider" />

      {!hasData ? (
        <div className="hm-activity-empty">
          <div className="hm-activity-empty-orb" aria-hidden>
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="rgba(59,130,246,0.5)" strokeWidth="1.3" strokeLinecap="round">
              <circle cx="12" cy="12" r="10" /><path d="M12 8v4l3 3" />
            </svg>
          </div>
          <p className="hm-activity-empty-title">No activity yet</p>
          <p className="hm-activity-empty-sub">Host your first meeting to see insights here.</p>
        </div>
      ) : (
        <div className="hm-activity-body">

          <div className="hm-activity-stats-row">
            <div className="hm-activity-stat">
              <span className="hm-activity-stat-val">{transcripts.length}</span>
              <span className="hm-activity-stat-label">meetings</span>
            </div>
            <div className="hm-activity-stat-divider" />
            <div className="hm-activity-stat">
              <span className="hm-activity-stat-val">{displayTime}</span>
              <span className="hm-activity-stat-label">talk time</span>
            </div>
            <div className="hm-activity-stat-divider" />
            {totalWords > 0 ? (
              <div className="hm-activity-stat">
                <span className="hm-activity-stat-val">{totalWords >= 1000 ? `${(totalWords / 1000).toFixed(1)}k` : totalWords}</span>
                <span className="hm-activity-stat-label">words spoken</span>
              </div>
            ) : (
              <div className="hm-activity-stat">
                <span className="hm-activity-stat-val">{allSpeakers.size || "—"}</span>
                <span className="hm-activity-stat-label">speakers</span>
              </div>
            )}
            {avgPace && (
              <>
                <div className="hm-activity-stat-divider" />
                <div className="hm-activity-stat">
                  <span className="hm-activity-stat-val">{avgPace}</span>
                  <span className="hm-activity-stat-label">wpm avg</span>
                </div>
              </>
            )}
          </div>

          {hasSummaries && aiEmoEntries.length > 0 && liveEmoTotal > 0 ? (
            <div className="hm-activity-section">
              <div className="hm-activity-section-label-row">
                <span className="hm-activity-section-label">Emotion — NLP vs Live capture</span>
                <span className="hm-activity-emo-source-hint">NLP · Live</span>
              </div>
              <div className="hm-activity-dual-emo">
                {aiEmoEntries.map(([emo, aiVal]) => {
                  const aiPct = Math.round((aiVal / aiEmoTotal) * 100);
                  const livePct = Math.round(((liveEmoAgg[emo] || 0) / liveEmoTotal) * 100);
                  const color = emoColor(emo);
                  const delta = livePct - aiPct;
                  return (
                    <div key={emo} className="hm-activity-dual-row">
                      <div className="hm-activity-dual-label">
                        <span className="hm-activity-dual-icon" style={{ color }}>{emoIcon(emo)}</span>
                        <span style={{ textTransform: "capitalize", color }}>{emo}</span>
                      </div>
                      <div className="hm-activity-dual-bars">
                        <div className="hm-activity-dual-bar-track" title={`NLP: ${aiPct}%`}>
                          <div className="hm-activity-dual-bar-fill hm-activity-dual-bar-ai"
                            style={{ width: `${aiPct}%`, background: color + "99" }} />
                          <span className="hm-activity-dual-bar-val">{aiPct}%</span>
                        </div>
                        <div className="hm-activity-dual-bar-track" title={`Live: ${livePct}%`}>
                          <div className="hm-activity-dual-bar-fill hm-activity-dual-bar-live"
                            style={{ width: `${livePct}%`, background: color }} />
                          <span className="hm-activity-dual-bar-val">{livePct}%</span>
                        </div>
                      </div>
                      {Math.abs(delta) >= 8 && (
                        <span className="hm-activity-dual-delta" style={{ color: delta > 0 ? color : "var(--text-3)" }}>
                          {delta > 0 ? `+${delta}` : delta}
                        </span>
                      )}
                    </div>
                  );
                })}
              </div>
              <div className="hm-activity-emo-key-row">
                <span className="hm-activity-emo-key-item hm-activity-emo-key-ai">
                  <span className="hm-activity-emo-key-swatch hm-activity-emo-key-swatch-ai" />
                  NLP
                </span>
                <span className="hm-activity-emo-key-item">
                  <span className="hm-activity-emo-key-swatch hm-activity-emo-key-swatch-live" />
                  Live camera
                </span>
              </div>
            </div>
          ) : (
            liveEmoTotal > 0 && (() => {
              const entries = Object.entries(liveEmoAgg).sort((a, b) => b[1] - a[1]).slice(0, 5);
              return (
                <div className="hm-activity-section">
                  <div className="hm-activity-section-label">Live mood — across meetings</div>
                  <div className="hm-activity-emo-bar">
                    {entries.map(([emo, count]) => (
                      <div key={emo} className="hm-activity-emo-seg"
                        style={{ width: `${(count / liveEmoTotal) * 100}%`, background: emoColor(emo) }}
                        title={`${emo}: ${Math.round((count / liveEmoTotal) * 100)}%`} />
                    ))}
                  </div>
                  <div className="hm-activity-emo-legend">
                    {entries.map(([emo, count]) => (
                      <div key={emo} className="hm-activity-emo-legend-item">
                        <div className="hm-activity-emo-dot" style={{ background: emoColor(emo) }} />
                        <span style={{ textTransform: "capitalize" }}>{emo}</span>
                        <span className="hm-activity-emo-pct">{Math.round((count / liveEmoTotal) * 100)}%</span>
                      </div>
                    ))}
                  </div>
                  <p className="hm-activity-no-ai-hint">Generate AI summaries to unlock deeper emotion analysis.</p>
                </div>
              );
            })()
          )}

          {allDiscrepancies.length > 0 && (
            <div className="hm-activity-section">
              <div className="hm-activity-section-label-row">
                <span className="hm-activity-section-label">Emotion mismatches</span>
                <span className="hm-activity-section-badge">{allDiscrepancies.length}</span>
              </div>
              <div className="hm-activity-disc-list">
                {allDiscrepancies.map((d, i) => {
                  const nlpColor = emoColor(d.nlp_emotion);
                  const liveColor = emoColor(d.live_emotion);
                  return (
                    <div key={i} className="hm-activity-disc-item">
                      <div className="hm-activity-disc-header">
                        <span className="hm-activity-disc-speaker">{d.participant}</span>
                        <span className="hm-activity-disc-time">{fmtSec(d.at_sec)}</span>
                        <span className="hm-activity-disc-code">{d.meetingCode}</span>
                      </div>
                      <div className="hm-activity-disc-said">"{d.said?.length > 60 ? d.said.slice(0, 60) + "…" : d.said}"</div>
                      <div className="hm-activity-disc-tags">
                        <span className="hm-activity-disc-tag" style={{ color: nlpColor, borderColor: nlpColor + "44", background: nlpColor + "11" }}>
                          {emoIcon(d.nlp_emotion)} speech: {d.nlp_emotion}
                        </span>
                        <span className="hm-activity-disc-arrow">→</span>
                        <span className="hm-activity-disc-tag" style={{ color: liveColor, borderColor: liveColor + "44", background: liveColor + "11" }}>
                          {emoIcon(d.live_emotion)} live: {d.live_emotion}
                        </span>
                      </div>
                      {d.note && <div className="hm-activity-disc-note">{d.note}</div>}
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {allMoments.length > 0 && (
            <div className="hm-activity-section">
              <div className="hm-activity-section-label">Notable moments</div>
              <div className="hm-activity-moments">
                {allMoments.map((m, i) => {
                  const color = emoColor(m.emotion);
                  return (
                    <div key={i} className="hm-activity-moment-item" style={{ borderLeftColor: color }}>
                      <div className="hm-activity-moment-meta">
                        <span style={{ color, fontSize: "0.65rem", fontWeight: 600, textTransform: "capitalize" }}>
                          {emoIcon(m.emotion)} {m.emotion}
                        </span>
                        <span className="hm-activity-moment-code">{m.meetingCode} · {fmtSec(m.start)}</span>
                      </div>
                      <div className="hm-activity-moment-text">
                        "{m.text?.length > 80 ? m.text.slice(0, 80) + "…" : m.text}"
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {topTopics.length > 0 && (
            <div className="hm-activity-section">
              <div className="hm-activity-section-label">Topics discussed</div>
              <div className="hm-activity-topics">
                {topTopics.map((topic, i) => (
                  <span key={i} className="hm-activity-topic-tag" style={{ opacity: 1 - i * 0.07 }}>
                    {topic}
                  </span>
                ))}
              </div>
            </div>
          )}

          <button className="hm-activity-tx-cta" onClick={onShowTranscripts}>
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <path d="M9 12h6M9 16h6M7 4H5a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2V6a2 2 0 00-2-2h-2" />
              <path d="M15 2H9a1 1 0 00-1 1v2a1 1 0 001 1h6a1 1 0 001-1V3a1 1 0 00-1-1z" />
            </svg>
            View transcripts
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
              <path d="M5 12h14M12 5l7 7-7 7" />
            </svg>
          </button>

        </div>
      )}
    </div>
  );
}

function RequestTranscriptPanel({ participatedMeetings, myRequests, onRequestSent }) {
  const [manualCode, setManualCode] = React.useState("");
  const [submitting, setSubmitting] = React.useState({});
  const [submitted, setSubmitted] = React.useState({});

  const participantMeetings = (participatedMeetings || []).filter(
    (m) => !localStorage.getItem(`host:${(m.meetingCode || "").toUpperCase()}`)
  );

  function getStatus(meetingCode) {
    const req = (myRequests || []).find(
      (r) => r.meetingCode?.toUpperCase() === meetingCode?.toUpperCase()
    );
    return req?.status || null;
  }

  async function handleRequest(code) {
    const c = code.trim().toUpperCase();
    if (!c) return;
    setSubmitting((p) => ({ ...p, [c]: true }));
    try {
      const { ok, data } = await submitTranscriptRequest(c);
      if (ok || data?.status === "pending" || data?.status === "approved") {
        setSubmitted((p) => ({ ...p, [c]: data?.status || "pending" }));
        if (typeof onRequestSent === "function") onRequestSent(c, data?.status || "pending");
      } else {
        setSubmitted((p) => ({ ...p, [c]: "error" }));
        setTimeout(() => setSubmitted((p) => { const n = { ...p }; delete n[c]; return n; }), 3000);
      }
    } catch {
      setSubmitted((p) => ({ ...p, [c]: "error" }));
      setTimeout(() => setSubmitted((p) => { const n = { ...p }; delete n[c]; return n; }), 3000);
    } finally {
      setSubmitting((p) => { const n = { ...p }; delete n[c]; return n; });
    }
  }

  async function handleManualSubmit() {
    const c = manualCode.trim().toUpperCase();
    if (!c) return;
    await handleRequest(c);
    setManualCode("");
  }

  function StatusBadge({ code }) {
    const serverStatus = getStatus(code);
    const localStatus = submitted[code];
    const status = localStatus || serverStatus;
    if (!status) return null;
    if (status === "approved") return (
      <span className="hm-txreq-badge hm-txreq-badge-approved">
        <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><polyline points="20 6 9 17 4 12" /></svg>
        Approved
      </span>
    );
    if (status === "pending") return (
      <span className="hm-txreq-badge hm-txreq-badge-pending">
        <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4" /></svg>
        Pending
      </span>
    );
    if (status === "denied") return (
      <span className="hm-txreq-badge hm-txreq-badge-denied">
        <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><path d="M18 6L6 18M6 6l12 12" /></svg>
        Denied
      </span>
    );
    if (status === "error") return <span className="hm-txreq-badge hm-txreq-badge-denied">Failed</span>;
    return null;
  }

  return (
    <div className="hm-card hm-req-transcript-panel">
      <div className="hm-card-header">
        <div>
          <div className="hm-card-title">Request Transcript</div>
          <div className="hm-card-sub">Ask the host for access to a meeting transcript</div>
        </div>
      </div>
      <div className="hm-divider" />

      <div className="hm-req-tx-manual">
        <div className="hm-field">
          <label htmlFor="hm-req-tx-code">Meeting code</label>
          <div className="hm-req-tx-input-row">
            <input
              id="hm-req-tx-code"
              type="text"
              value={manualCode}
              onChange={(e) => setManualCode(e.target.value.toUpperCase())}
              onKeyDown={(e) => e.key === "Enter" && handleManualSubmit()}
              placeholder="e.g. XKCD42"
              maxLength={32}
            />
            <button
              className="hm-txreq-btn"
              onClick={handleManualSubmit}
              disabled={!manualCode.trim() || submitting[manualCode.trim().toUpperCase()]}
            >
              {submitting[manualCode.trim().toUpperCase()] ? "Sending…" : "Request"}
            </button>
          </div>
        </div>
      </div>

      {participantMeetings.length > 0 && (
        <>
          <div className="hm-req-tx-section-label">Meetings you attended</div>
          <div className="hm-txreq-list">
            {participantMeetings.map((m) => {
              const code = (m.meetingCode || "").toUpperCase();
              const status = getStatus(code) || submitted[code];
              const isLoading = !!submitting[code];
              const canRequest = !status || status === "denied";
              return (
                <div key={code} className="hm-txreq-item">
                  <div className="hm-txreq-item-info">
                    <div className="hm-txreq-item-name">{code}</div>
                    <div className="hm-txreq-item-meta">
                      {m.createdAt && (
                        <span className="hm-txreq-item-time">
                          {new Date(m.createdAt).toLocaleString(undefined, { month: "short", day: "numeric" })}
                        </span>
                      )}
                      {m.hostName && (
                        <><span className="hm-txreq-item-dot" /><span className="hm-txreq-item-time">Host: {m.hostName}</span></>
                      )}
                    </div>
                  </div>
                  <div className="hm-txreq-item-actions">
                    {canRequest ? (
                      <button
                        className={`hm-txreq-btn${isLoading ? " hm-txreq-btn-loading" : ""}`}
                        onClick={() => handleRequest(code)}
                        disabled={isLoading}
                      >
                        {isLoading ? "Sending…" : status === "denied" ? "Re-request" : "Request"}
                      </button>
                    ) : (
                      <StatusBadge code={code} />
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </>
      )}

      {participantMeetings.length === 0 && (
        <div className="hm-tx-empty">
          <div className="hm-tx-empty-icon" aria-hidden>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#64748b" strokeWidth="1.5" strokeLinecap="round">
              <path d="M17 21v-2a4 4 0 00-4-4H5a4 4 0 00-4 4v2" /><circle cx="9" cy="7" r="4" /><path d="M23 21v-2a4 4 0 00-3-3.87M16 3.13a4 4 0 010 7.75" />
            </svg>
          </div>
          <p>No participated meetings found. Join a meeting first, or enter a code above.</p>
        </div>
      )}
    </div>
  );
}

function TranscriptRequestBanner({ requests, onClose }) {
  if (!requests || requests.length === 0) return null;
  const req = requests[0];
  return (
    <div className="hm-txreq-banner" role="alert" aria-live="polite">
      <div className="hm-txreq-banner-icon">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden>
          <path d="M9 12h6M9 16h6M7 4H5a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2V6a2 2 0 00-2-2h-2" />
          <path d="M15 2H9a1 1 0 00-1 1v2a1 1 0 001 1h6a1 1 0 001-1V3a1 1 0 00-1-1z" />
        </svg>
      </div>
      <span>
        <strong>{req.requesterName}</strong> requested transcript for <strong>{req.meetingCode}</strong>
        {requests.length > 1 ? ` (+${requests.length - 1} more)` : ""}
      </span>
      <button className="hm-txreq-banner-close" onClick={onClose} aria-label="Dismiss">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
          <path d="M18 6L6 18M6 6l12 12" />
        </svg>
      </button>
    </div>
  );
}

function RagPanel({ transcripts }) {
  const [selectedId, setSelectedId] = React.useState(null);
  const [question, setQuestion] = React.useState('');
  const inputRef = React.useRef(null);
  const chatEndRef = React.useRef(null);

  const selectedTranscript = transcripts.find(
    (t) => (t._id || t.meetingCode) === selectedId
  );
  const ragId = selectedTranscript?._id || selectedTranscript?.meetingCode || null;

  const { index, query, clearSession, history, loading, indexing, error, indexStatus } = useRag(ragId);

  React.useEffect(() => {
    if (ragId) index();
  }, [ragId, index]);

  React.useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [history]);

  async function handleAsk() {
    const q = question.trim();
    if (!q || loading || indexing || indexStatus !== 'ready') return;
    setQuestion('');
    await query(q);
    inputRef.current?.focus();
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleAsk();
    }
  }

  const isReady = indexStatus === 'ready';
  const isIndexing = indexing || indexStatus === 'indexing';
  const isNoContent = indexStatus === 'no_content';
  const hasTranscripts = transcripts.length > 0;

  return (
    <div className="hm-card hm-rag-panel" style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden', minHeight: 480 }}>
      <div className="hm-card-header">
        <div>
          <div className="hm-card-title">Ask your transcripts</div>
          <div className="hm-card-sub">RAG-powered Q&amp;A over your meeting content</div>
        </div>
        {ragId && history.length > 0 && (
          <button
            className="hm-tx-refresh-btn"
            onClick={() => clearSession()}
            title="Clear conversation"
            style={{ fontSize: '0.7rem', padding: '4px 10px', borderRadius: 6 }}
          >
            Clear
          </button>
        )}
      </div>
      <div className="hm-divider" />

      {!hasTranscripts ? (
        <div className="hm-activity-empty" style={{ flex: 1 }}>
          <div className="hm-activity-empty-orb" aria-hidden>
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="rgba(59,130,246,0.5)" strokeWidth="1.3" strokeLinecap="round">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
            </svg>
          </div>
          <p className="hm-activity-empty-title">No transcripts yet</p>
          <p className="hm-activity-empty-sub">Host a meeting to start asking questions about it.</p>
        </div>
      ) : (
        <>
          <div style={{ padding: '12px 16px 0' }}>
            <label style={{ fontSize: '0.7rem', color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.06em', display: 'block', marginBottom: 6 }}>
              Select meeting
            </label>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6, maxHeight: 140, overflowY: 'auto' }}>
              {transcripts.slice(0, 10).map((t) => {
                const id = t._id || t.meetingCode;
                const isSelected = id === selectedId;
                return (
                  <button
                    key={id}
                    onClick={() => {
                      if (!isSelected) {
                        setSelectedId(id);
                        clearSession();
                      }
                    }}
                    style={{
                      textAlign: 'left',
                      padding: '7px 10px',
                      borderRadius: 8,
                      border: isSelected ? '1px solid rgba(56,189,248,0.5)' : '1px solid rgba(255,255,255,0.06)',
                      background: isSelected ? 'rgba(56,189,248,0.08)' : 'rgba(255,255,255,0.03)',
                      color: isSelected ? 'var(--text-1)' : 'var(--text-2)',
                      fontSize: '0.78rem',
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      gap: 8,
                      transition: 'all 0.15s',
                    }}
                  >
                    <span style={{ fontFamily: 'monospace', fontSize: '0.75rem', opacity: 0.7 }}>{t.meetingCode}</span>
                    {t.createdAt && (
                      <span style={{ fontSize: '0.7rem', color: 'var(--text-3)', marginLeft: 'auto' }}>
                        {new Date(t.createdAt).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}
                      </span>
                    )}
                    {isSelected && isIndexing && (
                      <span style={{ fontSize: '0.65rem', color: 'rgba(56,189,248,0.7)', marginLeft: 4 }}>indexing…</span>
                    )}
                    {isSelected && isReady && (
                      <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="rgba(56,189,248,0.8)" strokeWidth="2.5" strokeLinecap="round">
                        <polyline points="20 6 9 17 4 12" />
                      </svg>
                    )}
                  </button>
                );
              })}
            </div>
          </div>

          <div style={{ flex: 1, overflowY: 'auto', padding: '12px 16px', display: 'flex', flexDirection: 'column', gap: 10, minHeight: 160 }}>
            {!selectedId && (
              <div style={{ color: 'var(--text-3)', fontSize: '0.8rem', textAlign: 'center', marginTop: 32 }}>
                Select a meeting above to start asking questions
              </div>
            )}

            {selectedId && isIndexing && history.length === 0 && (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10, marginTop: 32 }}>
                <div className="hm-tx-loading-dots"><span /><span /><span /></div>
                <span style={{ color: 'var(--text-3)', fontSize: '0.8rem' }}>Indexing transcript, please wait…</span>
              </div>
            )}
            {selectedId && isIndexing && history.length > 0 && (
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: '0.75rem', color: 'rgba(56,189,248,0.7)', padding: '6px 10px', borderRadius: 8, background: 'rgba(56,189,248,0.06)', border: '1px solid rgba(56,189,248,0.12)' }}>
                <div className="hm-tx-loading-dots" style={{ margin: 0, transform: 'scale(0.7)' }}><span /><span /><span /></div>
                Re-indexing transcript…
              </div>
            )}

            {selectedId && isNoContent && (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8, marginTop: 32, padding: '0 12px', textAlign: 'center' }}>
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="rgba(100,116,139,0.6)" strokeWidth="1.5" strokeLinecap="round">
                  <path d="M9 12h6M9 16h6M7 4H5a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2V6a2 2 0 00-2-2h-2" />
                  <path d="M15 2H9a1 1 0 00-1 1v2a1 1 0 001 1h6a1 1 0 001-1V3a1 1 0 00-1-1z" />
                </svg>
                <span style={{ color: 'var(--text-2)', fontSize: '0.82rem', fontWeight: 500 }}>No transcript content</span>
                <span style={{ color: 'var(--text-3)', fontSize: '0.76rem', lineHeight: 1.5 }}>
                  This meeting was recorded before transcript indexing was supported. Only meetings with full transcripts can be queried.
                </span>
              </div>
            )}
            {selectedId && isReady && !isIndexing && history.length === 0 && !loading && (
              <div style={{ color: 'var(--text-3)', fontSize: '0.8rem', textAlign: 'center', marginTop: 32 }}>
                Transcript ready — ask anything about this meeting
              </div>
            )}

            {history.map((msg, i) => (
              <div
                key={i}
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: msg.role === 'user' ? 'flex-end' : 'flex-start',
                  gap: 4,
                }}
              >
                <div
                  style={{
                    maxWidth: '88%',
                    padding: '9px 13px',
                    borderRadius: msg.role === 'user' ? '12px 12px 4px 12px' : '12px 12px 12px 4px',
                    background: msg.role === 'user' ? 'rgba(56,189,248,0.12)' : 'rgba(255,255,255,0.05)',
                    border: msg.role === 'user' ? '1px solid rgba(56,189,248,0.2)' : '1px solid rgba(255,255,255,0.07)',
                    fontSize: '0.82rem',
                    lineHeight: 1.55,
                    color: 'var(--text-1)',
                    whiteSpace: 'pre-wrap',
                  }}
                >
                  {msg.content}
                </div>
                {msg.sources && msg.sources.length > 0 && (
                  <div style={{ fontSize: '0.68rem', color: 'var(--text-3)', display: 'flex', flexWrap: 'wrap', gap: 4, maxWidth: '88%' }}>
                    {msg.sources.slice(0, 3).map((s, si) => (
                      <span
                        key={si}
                        style={{ padding: '2px 7px', borderRadius: 4, background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)' }}
                      >
                        {s.speaker ? s.speaker + ': ' : ''}{String(s.text || '').slice(0, 50)}{(s.text || '').length > 50 ? '...' : ''}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))}

            {loading && (
              <div style={{ display: 'flex', alignItems: 'flex-start' }}>
                <div style={{ padding: '9px 14px', borderRadius: '12px 12px 12px 4px', background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.07)' }}>
                  <div className="hm-tx-loading-dots" style={{ margin: 0 }}>
                    <span /><span /><span />
                  </div>
                </div>
              </div>
            )}

            {error && !isIndexing && !isNoContent && (
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: '0.78rem', color: '#f87171', padding: '8px 12px', borderRadius: 8, background: 'rgba(248,113,113,0.08)', border: '1px solid rgba(248,113,113,0.2)' }}>
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                  <circle cx="12" cy="12" r="10" /><path d="M12 8v4M12 16h.01" />
                </svg>
                {error}
                <button
                  onClick={() => index()}
                  style={{ marginLeft: 'auto', fontSize: '0.72rem', padding: '2px 8px', borderRadius: 5, border: '1px solid rgba(248,113,113,0.3)', background: 'transparent', color: '#f87171', cursor: 'pointer' }}
                >
                  Retry
                </button>
              </div>
            )}

            <div ref={chatEndRef} />
          </div>

          {selectedId && (
            <div style={{ padding: '10px 14px 14px', borderTop: '1px solid rgba(255,255,255,0.06)' }}>
              <div style={{ display: 'flex', gap: 8, alignItems: 'flex-end' }}>
                <textarea
                  ref={inputRef}
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder={isIndexing ? (history.length > 0 ? 'Re-indexing, please wait…' : 'Indexing transcript…') : isNoContent ? 'No content available' : isReady ? 'Ask about this meeting…' : 'Preparing…'}
                  disabled={loading || isIndexing || isNoContent || !isReady}
                  rows={2}
                  style={{
                    flex: 1,
                    resize: 'none',
                    padding: '9px 12px',
                    borderRadius: 10,
                    border: '1px solid rgba(255,255,255,0.1)',
                    background: 'rgba(255,255,255,0.04)',
                    color: 'var(--text-1)',
                    fontSize: '0.82rem',
                    lineHeight: 1.5,
                    outline: 'none',
                    fontFamily: 'inherit',
                    opacity: isIndexing ? 0.5 : 1,
                  }}
                />
                <button
                  onClick={handleAsk}
                  disabled={!question.trim() || loading || isIndexing || isNoContent || !isReady}
                  style={{
                    padding: '9px 14px',
                    borderRadius: 10,
                    border: 'none',
                    background: question.trim() && isReady && !loading ? 'rgba(56,189,248,0.18)' : 'rgba(255,255,255,0.05)',
                    color: question.trim() && isReady && !loading ? 'rgba(56,189,248,0.9)' : 'var(--text-3)',
                    cursor: question.trim() && isReady && !loading ? 'pointer' : 'default',
                    fontSize: '0.8rem',
                    transition: 'all 0.15s',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 5,
                    whiteSpace: 'nowrap',
                  }}
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                    <line x1="22" y1="2" x2="11" y2="13" /><polygon points="22 2 15 22 11 13 2 9 22 2" />
                  </svg>
                  Ask
                </button>
              </div>
              <div style={{ fontSize: '0.67rem', color: 'var(--text-3)', marginTop: 5, textAlign: 'center' }}>
                Enter to send · Shift+Enter for new line
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function NotificationDrawer({ open, onClose, requests, onResolve, loading, onResolved }) {
  const [resolving, setResolving] = React.useState({});
  const [resolved, setResolved] = React.useState({});

  React.useEffect(() => {
    if (!open) return;
    function onKey(e) { if (e.key === "Escape") onClose(); }
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  async function handleResolve(requestId, status) {
    setResolving(prev => ({ ...prev, [requestId]: status }));
    try {
      await onResolve(requestId, status);
      setResolved(prev => ({ ...prev, [requestId]: status }));
      setTimeout(() => {
        if (typeof onResolved === "function") onResolved(requestId);
        setResolving(prev => { const n = { ...prev }; delete n[requestId]; return n; });
        setResolved(prev => { const n = { ...prev }; delete n[requestId]; return n; });
      }, 900);
    } catch {
      setResolving(prev => { const n = { ...prev }; delete n[requestId]; return n; });
    }
  }

  const pendingItems = requests.filter(r => !resolved[r._id]);

  return (
    <>
      {open && <div className="hm-drawer-overlay" onClick={onClose} aria-hidden />}
      <div
        className={`hm-drawer ${open ? "hm-drawer-open" : ""}`}
        role="dialog"
        aria-modal="true"
        aria-label="Transcript requests"
      >
        <div className="hm-drawer-header">
          <div className="hm-drawer-header-info">
            <div className="hm-drawer-title">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9" />
                <path d="M13.73 21a2 2 0 0 1-3.46 0" />
              </svg>
              Transcript Requests
            </div>
            <div className="hm-drawer-sub">
              {loading ? "Loading…" : pendingItems.length === 0 ? "All caught up" : `${pendingItems.length} pending approval`}
            </div>
          </div>
          <button className="hm-drawer-close" onClick={onClose} aria-label="Close">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
              <path d="M18 6L6 18M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="hm-drawer-body">
          {loading && pendingItems.length === 0 && (
            <div className="hm-drawer-empty">
              <div className="hm-tx-loading-dots"><span /><span /><span /></div>
            </div>
          )}

          {!loading && pendingItems.length === 0 && (
            <div className="hm-drawer-empty">
              <div className="hm-drawer-empty-icon">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
                  <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9" />
                  <path d="M13.73 21a2 2 0 0 1-3.46 0" />
                </svg>
              </div>
              <p>No pending requests</p>
              <p className="hm-drawer-empty-sub">New requests will appear here</p>
            </div>
          )}

          {pendingItems.map((req) => {
            const isResolving = !!resolving[req._id];
            const resolvedStatus = resolved[req._id];
            return (
              <div key={req._id} className={`hm-drawer-item ${resolvedStatus ? `hm-drawer-item-resolved hm-drawer-item-${resolvedStatus}` : ""}`}>
                <div className="hm-drawer-item-top">
                  <div className="hm-drawer-item-avatar">
                    {(req.requesterName || "?")[0].toUpperCase()}
                  </div>
                  <div className="hm-drawer-item-info">
                    <div className="hm-drawer-item-name">{req.requesterName || "Unknown"}</div>
                    <div className="hm-drawer-item-meta">
                      <span className="hm-drawer-item-code">{req.meetingCode}</span>
                      {req.createdAt && (
                        <>
                          <span className="hm-txreq-item-dot" />
                          <span className="hm-drawer-item-time">
                            {new Date(req.createdAt).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}
                          </span>
                        </>
                      )}
                    </div>
                  </div>
                </div>
                {resolvedStatus ? (
                  <div className={`hm-drawer-resolved-state ${resolvedStatus === "approved" ? "hm-drawer-resolved-approve" : "hm-drawer-resolved-deny"}`}>
                    {resolvedStatus === "approved" ? (
                      <>
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><polyline points="20 6 9 17 4 12" /></svg>
                        Approved
                      </>
                    ) : (
                      <>
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><path d="M18 6L6 18M6 6l12 12" /></svg>
                        Denied
                      </>
                    )}
                  </div>
                ) : (
                  <div className="hm-drawer-item-actions">
                    <button
                      className="hm-drawer-approve-btn"
                      disabled={isResolving}
                      onClick={() => handleResolve(req._id, "approved")}
                    >
                      {resolving[req._id] === "approved" ? "…" : (
                        <>
                          <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><polyline points="20 6 9 17 4 12" /></svg>
                          Approve
                        </>
                      )}
                    </button>
                    <button
                      className="hm-drawer-deny-btn"
                      disabled={isResolving}
                      onClick={() => handleResolve(req._id, "denied")}
                    >
                      {resolving[req._id] === "denied" ? "…" : (
                        <>
                          <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><path d="M18 6L6 18M6 6l12 12" /></svg>
                          Deny
                        </>
                      )}
                    </button>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </>
  );
}

export default function Home() {
  const navigate = useNavigate();
  const location = useLocation();
  const { logout, getHistoryOfUser, userData, setUserData, authLoading } = useContext(AuthContext);

  const [meetingName, setMeetingName] = useState(localStorage.getItem("displayName") || "");
  const [room, setRoom] = useState("");
  const [transcripts, setTranscripts] = useState([]);
  const [txLoading, setTxLoading] = useState(false);
  const [visibleCount, setVisibleCount] = useState(TRANSCRIPTS_PER_PAGE);
  const [snackOpen, setSnackOpen] = useState(false);
  const [snackMsg, setSnackMsg] = useState("");
  const [snackSeverity, setSnackSeverity] = useState("success");
  const [viewingTranscript, setViewingTranscript] = useState(null);
  const [pendingTranscriptCode, setPendingTranscriptCode] = useState(() =>
    localStorage.getItem(PENDING_TRANSCRIPT_KEY) || null
  );
  const [rightPanel, setRightPanel] = useState("activity");
  const [profileOpen, setProfileOpen] = useState(false);
  const [notifOpen, setNotifOpen] = useState(false);
  const [pendingRequests, setPendingRequests] = useState([]);
  const [reqsLoading, setReqsLoading] = useState(false);
  const [bannerRequests, setBannerRequests] = useState([]);
  const [myRequests, setMyRequests] = useState([]);
  const [currentUserId, setCurrentUserId] = useState(null);
  const [participatedMeetings, setParticipatedMeetings] = useState([]);

  const isFetchingRef = useRef(false);
  const prevCountRef = useRef(0);
  const pollTimerRef = useRef(null);
  const pollAttemptsRef = useRef(0);
  const txListRef = useRef(null);
  const txCardRef = useRef(null);

  const loadTranscripts = useCallback(async (bustCache = false) => {
    if (!TRANSCRIPTS_ENABLED) return null;
    if (isFetchingRef.current) return null;

    if (!bustCache) {
      const cached = getCachedTranscripts();
      if (cached) {
        setTranscripts(cached);
        prevCountRef.current = cached.length;
        return cached;
      }
    } else {
      sessionStorage.removeItem(TRANSCRIPT_CACHE_KEY);
    }

    isFetchingRef.current = true;
    if (bustCache) setTxLoading(true);

    try {
      const token = localStorage.getItem("token");
      const res = await fetch(`${API_BASE}/transcripts?limit=200`, {
        headers: token ? { Authorization: `Bearer ${token}` } : {},
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const data = await res.json();

      if (data?.success) {
        const normalized = (data.transcripts || []).map(normalizeTranscript).filter(Boolean);
        const deduped = dedupeByCode(normalized);

        if (deduped.length > 0) setCachedTranscripts(deduped);

        const isNew = deduped.length > prevCountRef.current;
        if (bustCache && isNew) {
          showSnack(
            `${deduped.length - prevCountRef.current} new transcript${deduped.length - prevCountRef.current > 1 ? "s" : ""} available`,
            "success"
          );
          setVisibleCount(TRANSCRIPTS_PER_PAGE);
        }
        prevCountRef.current = deduped.length;
        setTranscripts(deduped);
        return deduped;
      }
    } catch (err) {
      console.error("loadTranscripts error:", err);
    } finally {
      isFetchingRef.current = false;
      setTxLoading(false);
    }
    return null;
  }, []);

  const stopPolling = useCallback(() => {
    if (pollTimerRef.current) {
      clearTimeout(pollTimerRef.current);
      pollTimerRef.current = null;
    }
    pollAttemptsRef.current = 0;
    localStorage.removeItem(PENDING_TRANSCRIPT_KEY);
    setPendingTranscriptCode(null);
  }, []);

  const startPollingForTranscript = useCallback((meetingCode) => {
    if (pollTimerRef.current) return;
    stopPolling();
    pollAttemptsRef.current = 0;
    localStorage.setItem(PENDING_TRANSCRIPT_KEY, meetingCode);
    setPendingTranscriptCode(meetingCode?.toUpperCase());

    const BACKOFFS = [5000, 10000, 20000, 40000];
    const MAX_TOTAL_MS = 10 * 60 * 1000;
    const startTime = Date.now();

    const getDelay = (attempt) => {
      const base = attempt < BACKOFFS.length ? BACKOFFS[attempt] : BACKOFFS[BACKOFFS.length - 1];
      const jitter = base * 0.2 * (Math.random() * 2 - 1);
      return Math.round(base + jitter);
    };

    const poll = async () => {
      const attempt = pollAttemptsRef.current;
      pollAttemptsRef.current++;

      try {
        const fresh = await loadTranscripts(true);
        if (fresh) {
          const found = fresh.find(
            (t) => t.meetingCode?.toUpperCase() === meetingCode?.toUpperCase()
          );
          if (found) {
            stopPolling();
            setViewingTranscript(found);
            showSnack("Transcript ready!", "success");
            return;
          }
        }
      } catch {
        stopPolling();
        return;
      }

      if (Date.now() - startTime < MAX_TOTAL_MS) {
        pollTimerRef.current = setTimeout(poll, getDelay(attempt));
      } else {
        stopPolling();
        showSnack("Transcript unavailable — please contact support if this persists.", "error");
      }
    };

    pollTimerRef.current = setTimeout(poll, getDelay(0));
  }, [loadTranscripts, stopPolling]);

  useEffect(() => { cleanInvalidHosts(); }, []);
  useEffect(() => {
    if (userData?.name && typeof userData.name === "string" && userData.name.trim()) {
      setMeetingName(userData.name.trim());
    }
  }, [userData?.name]);
  useEffect(() => {
    try {
      const token = localStorage.getItem("token");
      if (token) {
        const payload = JSON.parse(atob(token.split(".")[1]));
        setCurrentUserId(payload?.id || payload?.sub || payload?._id || null);
      }
    } catch { }
  }, []);

  useEffect(() => {
    loadOwnedRooms().catch(() => { });
  }, []);

  useEffect(() => {
    loadParticipatedMeetings()
      .then(({ ok, data }) => {
        if (ok && Array.isArray(data?.meetings)) {
          setParticipatedMeetings(data.meetings);
        }
      })
      .catch(() => { });
  }, []);
  useEffect(() => { loadTranscripts(); }, [loadTranscripts]);

  useEffect(() => {
    setReqsLoading(true);
    loadHostPendingRequests()
      .then(({ ok, data }) => {
        if (ok && data?.requests && data.requests.length > 0) {
          setPendingRequests(data.requests);
        }
      })
      .catch(() => { })
      .finally(() => setReqsLoading(false));
  }, []);

  useEffect(() => {
    loadMyRequests()
      .then(({ ok, data }) => {
        if (ok && data?.requests) setMyRequests(data.requests);
      })
      .catch(() => { });
  }, []);
  useEffect(() => {
    const handleFocus = () => loadTranscripts(true);
    window.addEventListener("focus", handleFocus);
    return () => window.removeEventListener("focus", handleFocus);
  }, [loadTranscripts]);
  useEffect(() => {
    const state = location.state;
    if (state?.meetingEnded && state?.meetingCode) {
      startPollingForTranscript(state.meetingCode);
      window.history.replaceState({}, "", window.location.pathname);
    }
  }, [location.state, startPollingForTranscript]);
  useEffect(() => {
    const pending = localStorage.getItem(PENDING_TRANSCRIPT_KEY);
    if (pending && TRANSCRIPTS_ENABLED) startPollingForTranscript(pending);
  }, [startPollingForTranscript]);
  useEffect(() => { return () => stopPolling(); }, [stopPolling]);
  useEffect(() => {
    const el = txListRef.current;
    if (!el || visibleCount <= TRANSCRIPTS_PER_PAGE) return;
    const fourthItem = el.children[3];
    if (fourthItem) el.scrollTop = fourthItem.offsetTop - el.offsetTop;
  }, [visibleCount]);

  const handleTranscriptRequestReceived = React.useCallback((payload) => {
    setPendingRequests((prev) => {
      if (prev.some((r) => r._id === payload.requestId)) return prev;
      return [{ _id: payload.requestId, meetingCode: payload.meetingCode, requesterName: payload.requesterName, requesterId: payload.requesterId, status: "pending", createdAt: new Date().toISOString() }, ...prev];
    });
    setBannerRequests((prev) => {
      if (prev.some((r) => r._id === payload.requestId)) return prev;
      return [{ _id: payload.requestId, meetingCode: payload.meetingCode, requesterName: payload.requesterName }, ...prev];
    });
  }, []);

  const handleTranscriptRequestUpdate = React.useCallback((payload) => {
    setMyRequests((prev) =>
      prev.map((r) => r._id === payload.requestId ? { ...r, status: payload.status } : r)
    );
    if (payload.status === "approved") {
      showSnack(`Transcript access approved for meeting ${payload.meetingCode}!`, "success");
      loadTranscripts(true);
    } else if (payload.status === "denied") {
      showSnack(`Transcript request denied for meeting ${payload.meetingCode}.`, "error");
    }
  }, [loadTranscripts]);

  useEffect(() => {
    const socket = io(SOCKET_SERVER_URL, { autoConnect: false });

    const onConnect = () => {
      try {
        const token = localStorage.getItem("token") || localStorage.getItem("accessToken");
        let userId = null;
        if (token) {
          try {
            const payload = JSON.parse(atob(token.split(".")[1]));
            userId = payload._id || payload.sub || payload.id;
          } catch { }
        }
        if (!userId) userId = localStorage.getItem("userId");
        if (userId) socket.data = { ...socket.data, userId };
        socket.emit("home-presence", { userId });
      } catch { }
    };

    socket.on("connect", onConnect);
    socket.on("transcript-request-received", handleTranscriptRequestReceived);
    socket.on("transcript-request-update", handleTranscriptRequestUpdate);

    socket.connect();

    return () => {
      socket.off("connect", onConnect);
      socket.off("transcript-request-received", handleTranscriptRequestReceived);
      socket.off("transcript-request-update", handleTranscriptRequestUpdate);
      socket.disconnect();
    };
  }, [handleTranscriptRequestReceived, handleTranscriptRequestUpdate]);

  async function handleResolveRequest(requestId, status) {
    const { ok, data } = await resolveTranscriptRequest(requestId, status);
    if (ok) {
      setPendingRequests((prev) => prev.filter((r) => r._id !== requestId));
      setBannerRequests((prev) => prev.filter((r) => r._id !== requestId));
      showSnack(status === "approved" ? "Request approved." : "Request denied.", status === "approved" ? "success" : "error");
    } else {
      showSnack(data?.message || "Failed to resolve request.", "error");
    }
  }

  function showSnack(message, severity = "success") {
    setSnackMsg(message);
    setSnackSeverity(severity);
    setSnackOpen(true);
    setTimeout(() => setSnackOpen(false), 3500);
  }

  function extractRoomCode(input) {
    let roomId = input.trim();
    try {
      const url = new URL(roomId);
      const segs = url.pathname.split("/").filter(Boolean);
      if (segs.length) roomId = segs.pop();
    } catch { }
    return roomId.toUpperCase();
  }

  async function createRoom() {
    if (!meetingName.trim()) { showSnack("Please enter your name first.", "error"); return; }
    try {
      const { code, link } = await createRoomAndGetLink(meetingName);
      await copyToClipboard(link);
      setRoom(link);
      showSnack("Meeting created & link copied", "success");
      navigate(`/room/${code}`);
    } catch {
      showSnack("Unable to create room.", "error");
    }
  }

  async function copyLink() {
    if (!meetingName.trim()) { showSnack("Enter your name before creating a link", "error"); return; }
    try {
      const { link } = await createRoomAndGetLink(meetingName);
      const copied = await copyToClipboard(link);
      setRoom(link);
      showSnack(copied ? "Link copied to clipboard" : `Copy failed — link: ${link}`, copied ? "success" : "error");
    } catch {
      showSnack("Unable to create room link.", "error");
    }
  }

  async function joinRoom() {
    if (!room.trim()) { showSnack("Enter room code or link", "error"); return; }
    const roomId = extractRoomCode(room);
    try {
      const res = await fetch(`${API_BASE}/rooms/${roomId}`);
      if (!res.ok) throw new Error("Room not found");
      localStorage.setItem("displayName", name.trim() || "Guest");
      navigate(`/room/${roomId}`);
    } catch {
      showSnack("Room does not exist or has expired.", "error");
    }
  }

  async function handleLogout() {
    try {
      if (logout) await logout(true);
      else { localStorage.removeItem("token"); navigate("/login"); }
    } catch { }
    try { localStorage.removeItem("displayName"); } catch { }
  }

  const name = typeof userData?.name === "string" && userData.name.trim()
    ? userData.name.trim()
    : (localStorage.getItem("displayName") || "");
  const displayInitial = (name || "?")[0].toUpperCase();
  const dedupedTranscripts = dedupeByCode(transcripts);
  const visibleTranscripts = dedupedTranscripts.slice(0, visibleCount);
  const hasMore = visibleCount < dedupedTranscripts.length;
  const hiddenCount = dedupedTranscripts.length - visibleCount;

  const allSegments = dedupedTranscripts.flatMap(t => t.metadata?.segments ?? []);
  const emoCount = {};
  allSegments.forEach(s => { const e = (s.emotion || "neutral").toLowerCase(); emoCount[e] = (emoCount[e] || 0) + 1; });
  const dominantEmo = Object.entries(emoCount).sort((a, b) => b[1] - a[1])[0]?.[0] || null;
  const EMO_COLORS = { joy: "#f59e0b", happy: "#f59e0b", sadness: "#60a5fa", anger: "#f87171", fear: "#a78bfa", surprise: "#34d399", disgust: "#fb923c", neutral: "#64748b" };
  const dominantEmoColor = dominantEmo ? (EMO_COLORS[dominantEmo] || "#64748b") : null;

  const lastMeeting = dedupedTranscripts[0];
  const lastMeetingDate = lastMeeting?.createdAt
    ? new Date(lastMeeting.createdAt).toLocaleDateString(undefined, { month: "short", day: "numeric" })
    : null;

  return (
    <div className="hm-root">
      <div className="hm-bg" aria-hidden />

      <aside className="hm-sidebar">
        <div className="hm-sidebar-brand" onClick={() => navigate("/")}>
          <img src="/logo.svg" alt="Hoovik" width="24" height="24" />
          <span className="hm-brand-name">Hoovik</span>
          <span className="hm-sidebar-version">v1.0.0</span>
        </div>

        <nav className="hm-sidebar-nav">
          <button className={`hm-sidebar-nav-item ${rightPanel === "activity" ? "hm-sidebar-nav-active" : ""}`} aria-current={rightPanel === "activity" ? "page" : undefined} onClick={() => setRightPanel("activity")}>
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
              <rect x="3" y="3" width="7" height="7" rx="1" /><rect x="14" y="3" width="7" height="7" rx="1" /><rect x="3" y="14" width="7" height="7" rx="1" /><rect x="14" y="14" width="7" height="7" rx="1" />
            </svg>
            Home
          </button>
          <button className={`hm-sidebar-nav-item ${rightPanel === "history" ? "hm-sidebar-nav-active" : ""}`} aria-current={rightPanel === "history" ? "page" : undefined} onClick={() => setRightPanel("history")}>
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
              <path d="M3 3v5h5" /><path d="M21 12a9 9 0 1 1-9-9" /><path d="M12 7v6l4 2" />
            </svg>
            History
          </button>
          <button className={`hm-sidebar-nav-item ${rightPanel === "transcripts" ? "hm-sidebar-nav-active" : ""}`} aria-current={rightPanel === "transcripts" ? "page" : undefined} onClick={() => setRightPanel("transcripts")}>
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
              <path d="M9 12h6M9 16h6M7 4H5a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2V6a2 2 0 00-2-2h-2" />
              <path d="M15 2H9a1 1 0 00-1 1v2a1 1 0 001 1h6a1 1 0 001-1V3a1 1 0 00-1-1z" />
            </svg>
            Transcripts
          </button>
          <button className={`hm-sidebar-nav-item ${rightPanel === "request-transcript" ? "hm-sidebar-nav-active" : ""}`} aria-current={rightPanel === "request-transcript" ? "page" : undefined} onClick={() => setRightPanel("request-transcript")}>
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
              <circle cx="12" cy="12" r="10" /><path d="M12 8v4M12 16h.01" />
            </svg>
            Request Transcript
          </button>
          <button className={`hm-sidebar-nav-item hm-sidebar-nav-rag ${rightPanel === "rag" ? "hm-sidebar-nav-active hm-sidebar-nav-rag-active" : ""}`} aria-current={rightPanel === "rag" ? "page" : undefined} onClick={() => setRightPanel("rag")}>
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
            </svg>
            Ask Transcripts
          </button>
        </nav>

        {(pendingRequests.length > 0 || reqsLoading) && (
          <button className={`hm-sidebar-notif-btn ${pendingRequests.length > 0 ? "hm-sidebar-notif-has-items" : ""}`} onClick={() => setNotifOpen(true)}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
              <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9" />
              <path d="M13.73 21a2 2 0 0 1-3.46 0" />
            </svg>
            <span className="hm-sidebar-notif-label">Requests</span>
            {pendingRequests.length > 0 && (
              <span className="hm-sidebar-notif-count">{pendingRequests.length}</span>
            )}
          </button>
        )}

        {TRANSCRIPTS_ENABLED && (
          <>
            <div className="hm-sidebar-stats-title">Overview</div>
            <div className="hm-sidebar-stats">
              <div className="hm-stat-tile">
                <span className="hm-stat-tile-label">Meetings</span>
                <span className="hm-stat-tile-val">{dedupedTranscripts.length}</span>
              </div>
              {lastMeetingDate && (
                <div className="hm-stat-tile">
                  <span className="hm-stat-tile-label">Last meeting</span>
                  <span className="hm-stat-tile-val">{lastMeetingDate}</span>
                </div>
              )}
              {dominantEmo && (
                <div className="hm-stat-tile">
                  <span className="hm-stat-tile-label">Overall mood</span>
                  <div className="hm-stat-emo-row">
                    <div className="hm-stat-emo-dot" style={{ background: dominantEmoColor }} />
                    <span className="hm-stat-tile-val" style={{ color: dominantEmoColor, textTransform: "capitalize" }}>{dominantEmo}</span>
                  </div>
                </div>
              )}
            </div>
          </>
        )}

        <div className="hm-sidebar-spacer" />

        <button className="hm-profile-btn" onClick={() => setProfileOpen(true)} aria-label="Profile">
          {userData?.avatar?.url ? (
            <img src={userData.avatar.url} alt="Avatar" className="hm-profile-btn-img" />
          ) : (
            <div className="hm-profile-btn-avatar">
              {(typeof userData?.name === "string" && userData.name.trim() ? userData.name.trim() : "?")[0].toUpperCase()}
            </div>
          )}
          <span className="hm-profile-btn-name">
            {typeof userData?.name === "string" && userData.name.trim() ? userData.name.trim() : "Profile"}
          </span>
        </button>

        <button className="hm-logout-btn" onClick={handleLogout} aria-label="Sign out">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden>
            <path d="M9 21H5a2 2 0 01-2-2V5a2 2 0 012-2h4" /><polyline points="16 17 21 12 16 7" /><line x1="21" y1="12" x2="9" y2="12" />
          </svg>
          Sign out
        </button>
      </aside>

      <div className="hm-main-area">
        <div className="hm-welcome">
          <div className="hm-welcome-avatar" aria-hidden>
            {userData?.avatar?.url
              ? <img src={userData.avatar.url} alt="" style={{ width: "100%", height: "100%", objectFit: "cover", borderRadius: "inherit" }} />
              : displayInitial}
          </div>
          <div className="hm-welcome-text">
            <h2>Welcome back{name ? `, ${name.split(" ")[0]}` : ""}!</h2>
            <p>Ready to connect? Create or join a room below.</p>
          </div>
        </div>

        <div className="hm-grid">
          <div className="hm-left">
            <div className="hm-card">
              <div className="hm-card-header">
                <div>
                  <div className="hm-card-title">Create a room</div>
                  <div className="hm-card-sub">Host a new meeting instantly</div>
                </div>
              </div>
              <div className="hm-card-body">
                <div className="hm-field">
                  <label htmlFor="hm-name">Your display name</label>
                  <input id="hm-name" type="text" value={meetingName} onChange={(e) => setMeetingName(e.target.value)} placeholder="e.g. Anupam Kumar" />
                </div>
                <div className="hm-btn-row">
                  <button className="hm-btn-p" onClick={createRoom}>
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden>
                      <path d="M15 10l4.553-2.069A1 1 0 0121 8.82v6.36a1 1 0 01-1.447.89L15 14M3 8a2 2 0 012-2h10a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2V8z" />
                    </svg>
                    Start Meeting
                  </button>
                  <button className="hm-btn-g" onClick={copyLink}>
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden>
                      <path d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3" />
                    </svg>
                    Create &amp; Copy Link
                  </button>
                </div>
                <div className="hm-tip-row">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" strokeWidth="2" strokeLinecap="round" aria-hidden>
                    <circle cx="12" cy="12" r="10" /><path d="M12 16v-4M12 8h.01" />
                  </svg>
                  <span>Allow camera &amp; microphone when prompted. Share your link to invite others.</span>
                </div>
              </div>
            </div>
            <div className="hm-card">
              <div className="hm-card-header">
                <div>
                  <div className="hm-card-title">Join a room</div>
                  <div className="hm-card-sub">Paste a code or full meeting link</div>
                </div>
              </div>
              <div className="hm-card-body">
                <div className="hm-field">
                  <label htmlFor="hm-room">Room code or link</label>
                  <input
                    id="hm-room" type="text" value={room}
                    onChange={(e) => setRoom(e.target.value)}
                    placeholder="e.g. XKCD42 or https://…"
                    onKeyDown={(e) => e.key === "Enter" && joinRoom()}
                  />
                </div>
                <button className="hm-btn-full" onClick={joinRoom}>
                  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden>
                    <path d="M15 3h6v6M14 10l6.1-6.1M9 21H3v-6M10 14l-6.1 6.1" />
                  </svg>
                  Join Room
                </button>
              </div>
            </div>

          </div>

          {rightPanel === "activity" ? (
            <ActivityPanel
              transcripts={dedupedTranscripts}
              onShowTranscripts={() => setRightPanel("transcripts")}
            />
          ) : rightPanel === "history" ? (
            <HistoryPanel
              getHistoryOfUser={getHistoryOfUser}
              userData={userData}
              authLoading={authLoading}
            />
          ) : rightPanel === "rag" ? (
            <RagPanel transcripts={dedupedTranscripts} />
          ) : rightPanel === "request-transcript" ? (
            <RequestTranscriptPanel
              participatedMeetings={participatedMeetings}
              myRequests={myRequests}
              onRequestSent={(code, status) => {
                setMyRequests((prev) => {
                  const existing = prev.find((r) => r.meetingCode?.toUpperCase() === code);
                  if (existing) return prev.map((r) => r.meetingCode?.toUpperCase() === code ? { ...r, status } : r);
                  return [{ meetingCode: code, status, createdAt: new Date().toISOString() }, ...prev];
                });
                showSnack(`Transcript request sent for ${code}.`, "success");
              }}
            />
          ) : (
            <div className="hm-card hm-transcripts" ref={txCardRef}>
              <div className="hm-card-header">
                <div>
                  <div className="hm-card-title">Recent transcripts</div>
                  <div className="hm-card-sub">From your hosted meetings</div>
                </div>
                <div className="hm-tx-header-actions">
                  {dedupedTranscripts.length > 0 && (
                    <span className="hm-tx-badge">{dedupedTranscripts.length} meeting{dedupedTranscripts.length !== 1 ? "s" : ""}</span>
                  )}
                  <button
                    className={`hm-tx-refresh-btn ${txLoading ? "hm-tx-refresh-spinning" : ""}`}
                    onClick={() => loadTranscripts(true)}
                    title="Refresh transcripts"
                    aria-label="Refresh transcripts"
                    disabled={txLoading}
                  >
                    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round">
                      <path d="M1 4v6h6" /><path d="M23 20v-6h-6" />
                      <path d="M20.49 9A9 9 0 0 0 5.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 0 1 3.51 15" />
                    </svg>
                  </button>
                </div>
              </div>

              <div className="hm-divider" />

              {!TRANSCRIPTS_ENABLED && (
                <div className="hm-tx-notice hm-tx-notice-warn">
                  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden>
                    <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
                    <line x1="12" y1="9" x2="12" y2="13" /><line x1="12" y1="17" x2="12.01" y2="17" />
                  </svg>
                  <div>
                    <p>Transcript service unavailable on this build.</p>
                    <p className="hm-tx-notice-sub">Meetings still work — local recording runs in your browser.</p>
                  </div>
                </div>
              )}

              {dedupedTranscripts.length === 0 && TRANSCRIPTS_ENABLED && !txLoading && (
                <div className="hm-tx-empty">
                  <div className="hm-tx-empty-icon" aria-hidden>
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#64748b" strokeWidth="1.5" strokeLinecap="round">
                      <path d="M9 12h6M9 16h6M7 4H5a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2V6a2 2 0 00-2-2h-2" />
                      <path d="M15 2H9a1 1 0 00-1 1v2a1 1 0 001 1h6a1 1 0 001-1V3a1 1 0 00-1-1z" />
                    </svg>
                  </div>
                  <p>No transcripts yet — host a meeting and end it to generate one.</p>
                </div>
              )}

              {txLoading && dedupedTranscripts.length === 0 && (
                <div className="hm-tx-loading">
                  <div className="hm-tx-loading-dots">
                    <span /><span /><span />
                  </div>
                </div>
              )}

              <div className="hm-tx-list" ref={txListRef}>
                {pendingTranscriptCode && TRANSCRIPTS_ENABLED && (
                  <ProcessingTranscriptCard meetingCode={pendingTranscriptCode} />
                )}
                {visibleTranscripts.map((t, i) => {
                  const key = getTranscriptKey(t, i);
                  const isOwned = (currentUserId && t.ownerId && t.ownerId.toString() === currentUserId.toString())
                    || (currentUserId && t.hostId && t.hostId.toString() === currentUserId.toString())
                    || !!localStorage.getItem(`host:${t.meetingCode}`);
                  const myReq = !isOwned ? myRequests.find((r) => r.meetingCode === t.meetingCode) : null;
                  return (
                    <TranscriptItem
                      key={key}
                      t={t}
                      onOpen={() => (isOwned || myReq?.status === "approved") ? setViewingTranscript(t) : undefined}
                      requestStatus={myReq?.status}
                      isOwned={isOwned}
                    />
                  );
                })}
              </div>

              {(hasMore || visibleCount > TRANSCRIPTS_PER_PAGE) && (
                <div className="hm-tx-pagination">
                  {hasMore && (
                    <button
                      className="hm-tx-load-more"
                      onClick={() => setVisibleCount((v) => v + TRANSCRIPTS_PER_PAGE)}
                    >
                      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round">
                        <path d="M6 9l6 6 6-6" />
                      </svg>
                      Show {Math.min(hiddenCount, TRANSCRIPTS_PER_PAGE)} more
                      <span className="hm-tx-remaining">({hiddenCount} remaining)</span>
                    </button>
                  )}
                  {visibleCount > TRANSCRIPTS_PER_PAGE && (
                    <button
                      className="hm-tx-collapse"
                      onClick={() => setVisibleCount(TRANSCRIPTS_PER_PAGE)}
                    >
                      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round">
                        <path d="M18 15l-6-6-6 6" />
                      </svg>
                      Collapse
                    </button>
                  )}
                </div>
              )}
            </div>
          )}
        </div>

        <Snack msg={snackMsg} severity={snackSeverity} open={snackOpen} />

        {viewingTranscript && (
          <TranscriptViewer
            t={viewingTranscript}
            onClose={() => setViewingTranscript(null)}
            onSummaryGenerated={(updated) => {
              setTranscripts((prev) =>
                prev.map((t) => t._id === updated._id ? { ...t, aiSummary: updated.aiSummary } : t)
              );
              setViewingTranscript((prev) => ({ ...prev, aiSummary: updated.aiSummary }));
              sessionStorage.removeItem(TRANSCRIPT_CACHE_KEY);
            }}
          />
        )}
      </div>

      {bannerRequests.length > 0 && (
        <TranscriptRequestBanner
          requests={bannerRequests}
          onClose={() => setBannerRequests([])}
        />
      )}

      <NotificationDrawer
        open={notifOpen}
        onClose={() => setNotifOpen(false)}
        requests={pendingRequests}
        onResolve={handleResolveRequest}
        loading={reqsLoading}
        onResolved={(id) => setPendingRequests(prev => prev.filter(r => r._id !== id))}
      />

      {profileOpen && (
        <UserProfileModal
          onClose={() => setProfileOpen(false)}
          userData={userData}
          onProfileUpdate={(updatedProfile) => setUserData((prev) => ({ ...prev, ...updatedProfile }))}
        />
      )}
    </div>
  );
}