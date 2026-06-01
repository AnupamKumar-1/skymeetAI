import React, { useState, useRef, useEffect } from "react";
import "../styles/transcript-viewer.css";

const EMOTION_META = {
    joy: { color: "#f59e0b", bg: "rgba(245,158,11,0.12)", label: "Joy", icon: "✦" },
    happy: { color: "#f59e0b", bg: "rgba(245,158,11,0.12)", label: "Happy", icon: "✦" },
    sadness: { color: "#60a5fa", bg: "rgba(96,165,250,0.12)", label: "Sadness", icon: "◈" },
    anger: { color: "#f87171", bg: "rgba(248,113,113,0.12)", label: "Anger", icon: "◆" },
    fear: { color: "#a78bfa", bg: "rgba(167,139,250,0.12)", label: "Fear", icon: "◉" },
    surprise: { color: "#34d399", bg: "rgba(52,211,153,0.12)", label: "Surprise", icon: "◎" },
    disgust: { color: "#fb923c", bg: "rgba(251,146,60,0.12)", label: "Disgust", icon: "◇" },
    neutral: { color: "#94a3b8", bg: "rgba(148,163,184,0.08)", label: "Neutral", icon: "○" },
};

function emotionMeta(e) {
    return EMOTION_META[e?.toLowerCase()] ?? EMOTION_META.neutral;
}

const AVATAR_PALETTES = [
    ["#38bdf8", "#0ea5e9"], ["#a78bfa", "#7c3aed"], ["#34d399", "#059669"],
    ["#fb923c", "#ea580c"], ["#f472b6", "#db2777"], ["#facc15", "#ca8a04"],
];

function speakerColor(name = "") {
    let h = 0;
    for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) & 0xffff;
    return AVATAR_PALETTES[h % AVATAR_PALETTES.length];
}

function fmt(sec = 0) {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}:${String(s).padStart(2, "0")}`;
}

function fmtDuration(sec = 0) {
    if (sec < 60) return `${Math.round(sec)}s`;
    const m = Math.floor(sec / 60);
    const s = Math.round(sec % 60);
    return s > 0 ? `${m}m ${s}s` : `${m}m`;
}

function groupSegments(segments = []) {
    if (!segments.length) return [];
    const groups = [];
    let cur = null;
    for (const seg of segments) {
        if (cur && cur.speaker === seg.speaker) {
            cur.lines.push(seg);
            cur.end = seg.end;
        } else {
            if (cur) groups.push(cur);
            cur = { speaker: seg.speaker, start: seg.start, end: seg.end, lines: [seg] };
        }
    }
    if (cur) groups.push(cur);
    return groups;
}

function emotionSummary(segments = []) {
    const counts = {};
    for (const s of segments) {
        const e = (s.emotion || "neutral").toLowerCase();
        counts[e] = (counts[e] || 0) + 1;
    }
    return Object.entries(counts)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 4)
        .map(([e, n]) => ({ emotion: e, count: n, pct: Math.round((n / segments.length) * 100) }));
}

function speakerList(segments = []) {
    const map = {};
    for (const s of segments) {
        if (!map[s.speaker]) map[s.speaker] = 0;
        map[s.speaker]++;
    }
    return Object.entries(map)
        .sort((a, b) => b[1] - a[1])
        .map(([name, turns]) => ({ name, turns, pct: Math.round((turns / segments.length) * 100) }));
}

function SummaryPanel({ analysis }) {
    if (!analysis || !analysis.summary) return (
        <div className="tv-empty">
            <div className="tv-empty-icon">
                <svg width="40" height="40" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <circle cx="24" cy="24" r="24" fill="rgba(245,80,54,0.1)" />
                    <path d="M32 25.5v5.5H21.5C17.36 31 14 27.64 14 23.5v-5C14 14.36 17.36 11 21.5 11H34v4.5H22c-1.1 0-1.8.7-1.8 1.8v5.4c0 1.1.7 1.8 1.8 1.8h2.5v-1.5H22V19.5H32v6z" fill="#F55036" />
                </svg>
            </div>
        </div>
    );

    const { summary, key_points = [], insights = {} } = analysis;
    const {
        dominant_emotion,
        emotion_distribution = {},
        emotional_moments = [],
        top_topics = [],
        speaker_stats = {},
        discrepancies = [],
        total_words,
        speaking_pace_wpm,
        total_duration_sec,
    } = insights;

    const normalizedDist = Object.entries(emotion_distribution).reduce((acc, [k, v]) => {
        const key = k.toLowerCase();
        acc[key] = (acc[key] || 0) + v;
        return acc;
    }, {});

    const dominantMeta = emotionMeta(dominant_emotion);
    const speakerEntries = Object.entries(speaker_stats);

    return (
        <>
            <div className="tv-summary-block">
                <div className="tv-section-label">OVERVIEW</div>
                <p className="tv-summary-text">{summary}</p>
                {discrepancies.length > 0 && (
                    <p className="tv-summary-text" style={{ marginTop: "0.75rem" }}>
                        {discrepancies.map((d, i) => {
                            const liveM = emotionMeta(d.live_emotion);
                            const nlpM = emotionMeta(d.nlp_emotion);
                            return (
                                <span key={i} style={{ display: "block", marginBottom: i < discrepancies.length - 1 ? "0.5rem" : 0 }}>
                                    <span style={{ color: liveM.color, fontWeight: 600 }}>{liveM.icon} {d.participant}</span>
                                    {" "}at {fmt(d.at_sec)} said{" "}
                                    <em>"{d.said}"</em>
                                    {" "}— speech read as{" "}
                                    <span style={{ color: nlpM.color }}>{nlpM.icon} {d.nlp_emotion}</span>
                                    {" "}but live capture showed{" "}
                                    <span style={{ color: liveM.color }}>{liveM.icon} {d.live_emotion}</span>
                                    {d.note ? `. ${d.note}` : "."}
                                </span>
                            );
                        })}
                    </p>
                )}
            </div>

            <div className="tv-meta-strip">
                {total_duration_sec > 0 && (
                    <div className="tv-meta-chip">
                        <span className="tv-meta-icon">⏱</span>
                        <span className="tv-meta-val">{fmtDuration(total_duration_sec)}</span>
                        <span className="tv-meta-lbl">duration</span>
                    </div>
                )}
                {total_words > 0 && (
                    <div className="tv-meta-chip">
                        <span className="tv-meta-icon">💬</span>
                        <span className="tv-meta-val">{total_words.toLocaleString()}</span>
                        <span className="tv-meta-lbl">words spoken</span>
                    </div>
                )}
                {speaking_pace_wpm > 0 && (
                    <div className="tv-meta-chip">
                        <span className="tv-meta-icon">⚡</span>
                        <span className="tv-meta-val">{speaking_pace_wpm}</span>
                        <span className="tv-meta-lbl">wpm pace</span>
                    </div>
                )}
                {dominant_emotion && (
                    <div
                        className="tv-meta-chip"
                        style={{ borderColor: dominantMeta.color + "55", background: dominantMeta.bg }}
                    >
                        <span className="tv-meta-icon">{dominantMeta.icon}</span>
                        <span className="tv-meta-val" style={{ color: dominantMeta.color }}>{dominantMeta.label}</span>
                        <span className="tv-meta-lbl">mood</span>
                    </div>
                )}
            </div>

            <div className="tv-summary-grid">
                {key_points.length > 0 && (
                    <div className="tv-card">
                        <div className="tv-card-label">KEY MOMENTS</div>
                        <ul className="tv-point-list">
                            {key_points.map((p, i) => (
                                <li key={i} className="tv-point-item">
                                    <span className="tv-point-bullet">{String(i + 1).padStart(2, "0")}</span>
                                    <span className="tv-point-text">{p}</span>
                                </li>
                            ))}
                        </ul>
                    </div>
                )}

                <div className="tv-side-col">
                    {Object.keys(emotion_distribution).length > 0 && (
                        <div className="tv-card">
                            <div className="tv-card-label">EMOTIONAL BREAKDOWN</div>
                            <div className="tv-emo-bar-list">
                                {Object.entries(normalizedDist)
                                    .sort((a, b) => b[1] - a[1])
                                    .map(([emo, pct]) => {
                                        const m = emotionMeta(emo);
                                        return (
                                            <div key={emo} className="tv-emo-bar-row">
                                                <span className="tv-emo-bar-label" style={{ color: m.color }}>{m.icon} {m.label}</span>
                                                <div className="tv-emo-bar-track">
                                                    <div className="tv-emo-bar-fill" style={{ width: `${pct}%`, background: m.color }} />
                                                </div>
                                                <span className="tv-emo-bar-pct">{pct}%</span>
                                            </div>
                                        );
                                    })}
                            </div>
                        </div>
                    )}

                    {top_topics.length > 0 && (
                        <div className="tv-card">
                            <div className="tv-card-label">TOP TOPICS</div>
                            <div className="tv-topic-cloud">
                                {top_topics.map((topic, i) => (
                                    <span
                                        key={i}
                                        className="tv-topic-tag"
                                        style={{ fontSize: 11 + Math.max(0, 3 - i), opacity: 1 - i * 0.08 }}
                                    >
                                        {topic}
                                    </span>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            </div>

            {speakerEntries.length > 1 && (
                <div className="tv-card">
                    <div className="tv-card-label">SPEAKER BREAKDOWN</div>
                    <div className="tv-speaker-grid">
                        {speakerEntries.map(([name, stats]) => {
                            const [c1, c2] = speakerColor(name);
                            const m = emotionMeta(stats.dominant_emotion);
                            return (
                                <div key={name} className="tv-speaker-card">
                                    <div className="tv-speaker-avatar" style={{ background: `linear-gradient(135deg,${c1},${c2})` }}>
                                        {name[0]?.toUpperCase()}
                                    </div>
                                    <div className="tv-speaker-info">
                                        <div className="tv-speaker-info-name" style={{ color: c1 }}>{name}</div>
                                        <div className="tv-speaker-info-meta">
                                            <span>{stats.turns} turns</span>
                                            <span>·</span>
                                            <span>{stats.word_count} words</span>
                                            <span>·</span>
                                            <span style={{ color: m.color }}>{m.label}</span>
                                        </div>
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                </div>
            )}

            {emotional_moments.length > 0 && (
                <div className="tv-card">
                    <div className="tv-card-label">NOTABLE MOMENTS</div>
                    <div className="tv-moment-list">
                        {emotional_moments.map((moment, i) => {
                            const m = emotionMeta(moment.emotion);
                            return (
                                <div key={i} className="tv-moment-item" style={{ borderLeftColor: m.color }}>
                                    <div className="tv-moment-meta">
                                        <span style={{ color: m.color }}>{m.icon} {m.label}</span>
                                        <span className="tv-moment-time">{fmt(moment.start)}</span>
                                    </div>
                                    <div className="tv-moment-text">"{moment.text}{moment.text.length >= 80 ? "…" : ""}"</div>
                                </div>
                            );
                        })}
                    </div>
                </div>
            )}
        </>
    );
}

export default function TranscriptViewer({ t, onClose, onSummaryGenerated }) {
    const segments = t?.metadata?.segments ?? [];
    const groups = groupSegments(segments);
    const emoSummary = emotionSummary(segments);
    const speakers = speakerList(segments);

    const [activeTab, setActiveTab] = useState("transcript");
    const [search, setSearch] = useState("");
    const [filterEmo, setFilterEmo] = useState(null);
    const [filterSpk, setFilterSpk] = useState(null);
    const [copied, setCopied] = useState(false);
    const [activeGroup, setActiveGroup] = useState(null);
    const bodyRef = useRef(null);

    const [aiAnalysis, setAiAnalysis] = useState(t?.aiSummary || null);
    const [aiLoading, setAiLoading] = useState(false);
    const [aiError, setAiError] = useState(null);

    useEffect(() => {
        setAiAnalysis(t?.aiSummary || null);
    }, [t?._id, t?.aiSummary]);

    async function handleGenerateSummary() {
        if (!segments.length) return;
        setAiLoading(true);
        setAiError(null);
        try {
            const token = localStorage.getItem("token");
            const idOrCode = t._id || t.meetingCode;
            const hostDataRaw = t.meetingCode
                ? localStorage.getItem(`host:${t.meetingCode.toUpperCase()}`)
                : null;
            const hostSecret = hostDataRaw ? JSON.parse(hostDataRaw)?.hostSecret : null;

            const code = t.meetingCode?.toUpperCase();
            const emotionData = (() => { try { return JSON.parse(localStorage.getItem(`emotions:${code}`) || "{}"); } catch { return {}; } })();
            const emotionNames = (() => { try { return JSON.parse(localStorage.getItem(`emotionNames:${code}`) || "{}"); } catch { return {}; } })();

            const SERVER_BASE = import.meta.env.VITE_SERVER_URL || "http://localhost:8000";
            const res = await fetch(`${SERVER_BASE}/api/v1/transcripts/${idOrCode}/summary`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    ...(token ? { Authorization: `Bearer ${token}` } : {}),
                    ...(hostSecret ? { "x-host-secret": hostSecret } : {}),
                },
                body: JSON.stringify({ emotionData, emotionNames }),
            });

            if (res.status === 429) {
                const data = await res.json().catch(() => ({}));
                throw new Error(data?.message || "Rate limit reached. Max 2 AI summaries per 2 hours.");
            }
            if (!res.ok) {
                const data = await res.json().catch(() => ({}));
                throw new Error(data?.message || `Failed to generate summary (${res.status})`);
            }

            const data = await res.json();
            const result = data.aiSummary;
            setAiAnalysis(result);
            setActiveTab("summary");
            if (data.transcript) {
                onSummaryGenerated?.(data.transcript);
            } else {
                if (t) t.aiSummary = result;
            }
        } catch (err) {
            setAiError(err.message || "Failed to generate summary.");
            console.error("[AI Summary]", err);
        } finally {
            setAiLoading(false);
        }
    }

    useEffect(() => {
        if (!search) return;
        const el = bodyRef.current?.querySelector("mark");
        if (el) el.scrollIntoView({ behavior: "smooth", block: "center" });
    }, [search]);

    useEffect(() => {
        const onKey = (e) => { if (e.key === "Escape") onClose?.(); };
        window.addEventListener("keydown", onKey);
        return () => window.removeEventListener("keydown", onKey);
    }, [onClose]);

    function buildGroupedTranscript(segments = []) {
        if (!segments.length) return "";
        const groups = [];
        let current = null;
        for (const s of segments) {
            const speaker = s.speaker || "Unknown";
            if (current && current.speaker === speaker) {
                current.text += " " + s.text;
            } else {
                if (current) groups.push(current);
                current = { speaker, emotion: s.emotion || "neutral", emoji: s.emoji || "", text: s.text };
            }
        }
        if (current) groups.push(current);
        return groups.map(g => `[${g.speaker}] ${g.emoji} (${g.emotion}) ${g.text}`).join("\n\n");
    }

    function download() {
        const text = buildGroupedTranscript(t?.metadata?.segments || []);
        const blob = new Blob([text], { type: "text/plain" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `${(t.meetingCode || "transcript").replace(/[^a-z0-9_-]/gi, "_")}.txt`;
        a.click();
        URL.revokeObjectURL(url);
    }

    async function copy() {
        try {
            await navigator.clipboard.writeText(buildGroupedTranscript(t?.metadata?.segments || []));
        } catch { }
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    }

    const filteredGroups = groups.filter((g) => {
        if (filterSpk && g.speaker !== filterSpk) return false;
        return g.lines.some((seg) => {
            if (filterEmo && seg.emotion?.toLowerCase() !== filterEmo) return false;
            if (search && !seg.text.toLowerCase().includes(search.toLowerCase())) return false;
            return true;
        });
    });

    function highlight(text) {
        if (!search.trim()) return text;
        const parts = text.split(new RegExp(`(${search.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")})`, "gi"));
        return parts.map((p, i) =>
            p.toLowerCase() === search.toLowerCase()
                ? <mark key={i} className="tv-highlight">{p}</mark>
                : p
        );
    }

    return (
        <div className="tv-overlay" onClick={(e) => e.target === e.currentTarget && onClose?.()}>
            <div className="tv-panel">

                <div className="tv-header">
                    <div className="tv-header-left">
                        <div className="tv-badge">{t.meetingCode}</div>
                        <div>
                            <div className="tv-title">Meeting Transcript</div>
                            <div className="tv-subtitle">
                                {t.createdAt ? new Date(t.createdAt).toLocaleString(undefined, {
                                    month: "long", day: "numeric", year: "numeric",
                                    hour: "2-digit", minute: "2-digit"
                                }) : "—"}
                                {segments.length > 0 && ` · ${segments.length} segments`}
                            </div>
                        </div>
                    </div>
                    <div className="tv-header-right">
                        <button className="tv-icon-btn" onClick={copy} title="Copy transcript">
                            {copied
                                ? <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#34d399" strokeWidth="2.5" strokeLinecap="round"><polyline points="20 6 9 17 4 12" /></svg>
                                : <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><rect x="9" y="9" width="13" height="13" rx="2" /><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1" /></svg>
                            }
                        </button>
                        <button className="tv-icon-btn" onClick={download} title="Download .txt">
                            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                                <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3" />
                            </svg>
                        </button>
                        {onClose && (
                            <button className="tv-close-btn" onClick={onClose}>
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                                    <path d="M18 6L6 18M6 6l12 12" />
                                </svg>
                            </button>
                        )}
                    </div>
                </div>

                <div className="tv-tabs">
                    <button
                        className={`tv-tab ${activeTab === "transcript" ? "tv-tab--active" : ""}`}
                        onClick={() => setActiveTab("transcript")}
                    >
                        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                            <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
                            <polyline points="14 2 14 8 20 8" />
                            <line x1="16" y1="13" x2="8" y2="13" />
                            <line x1="16" y1="17" x2="8" y2="17" />
                        </svg>
                        Transcript
                    </button>
                    <button
                        className={`tv-tab ${activeTab === "summary" ? "tv-tab--active" : ""}`}
                        onClick={() => setActiveTab("summary")}
                    >
                        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                            <circle cx="12" cy="12" r="10" />
                            <line x1="12" y1="8" x2="12" y2="12" />
                            <line x1="12" y1="16" x2="12.01" y2="16" />
                        </svg>
                        AI Summary
                        <span className="tv-tab-pill tv-tab-pill--groq">
                            <svg width="28" height="10" viewBox="0 0 80 28" fill="none" xmlns="http://www.w3.org/2000/svg" style={{ display: "block" }}>
                                {/* G */}
                                <path d="M13.6 14.2v3.6H9.2c-2.6 0-4.4-1.8-4.4-4.4V10c0-2.6 1.8-4.4 4.4-4.4h8v3.2H9.4c-.8 0-1.2.4-1.2 1.2v4c0 .8.4 1.2 1.2 1.2h1.8v-1h-1.4v-3.2h5.6v3.2z" fill="currentColor" />
                                {/* R */}
                                <path d="M17 5.6h7.2c2.4 0 4 1.6 4 3.8 0 1.6-.8 2.8-2 3.4l2.4 5H25l-2.2-4.6H20.2V17.8H17V5.6zm3.2 5.4h3.6c.8 0 1.2-.4 1.2-1.2s-.4-1.2-1.2-1.2h-3.6v2.4z" fill="currentColor" />
                                {/* O */}
                                <path d="M30.8 10c0-2.6 1.8-4.4 4.4-4.4h4c2.6 0 4.4 1.8 4.4 4.4v3.4c0 2.6-1.8 4.4-4.4 4.4h-4c-2.6 0-4.4-1.8-4.4-4.4V10zm3.2.2v3c0 .8.4 1.2 1.2 1.2h3.6c.8 0 1.2-.4 1.2-1.2v-3c0-.8-.4-1.2-1.2-1.2h-3.6c-.8 0-1.2.4-1.2 1.2z" fill="currentColor" />
                                {/* Q */}
                                <path d="M46.4 10c0-2.6 1.8-4.4 4.4-4.4h4c2.6 0 4.4 1.8 4.4 4.4v3.4c0 1.8-.8 3.2-2 3.9l2 2.7h-3.8l-1.4-1.8c-.4.1-.8.1-1.2.1h-4c-2.6 0-4.4-1.8-4.4-4.4V10zm3.2.2v3c0 .8.4 1.2 1.2 1.2h3.6c.8 0 1.2-.4 1.2-1.2v-3c0-.8-.4-1.2-1.2-1.2h-3.6c-.8 0-1.2.4-1.2 1.2z" fill="currentColor" />
                            </svg>
                        </span>
                    </button>
                </div>

                {activeTab === "transcript" && (
                    <>
                        {segments.length > 0 && (
                            <div className="tv-stats-bar">
                                <div className="tv-stat-group">
                                    <span className="tv-stat-label">Speakers</span>
                                    <div className="tv-avatar-row">
                                        {speakers.map(({ name, pct }) => {
                                            const [c1, c2] = speakerColor(name);
                                            const active = filterSpk === name;
                                            return (
                                                <button
                                                    key={name}
                                                    className="tv-avatar-btn"
                                                    style={{
                                                        background: `linear-gradient(135deg,${c1},${c2})`,
                                                        outline: active ? `2px solid ${c1}` : "none",
                                                        outlineOffset: 2
                                                    }}
                                                    title={`${name} (${pct}% of turns)`}
                                                    onClick={() => setFilterSpk(active ? null : name)}
                                                >
                                                    {name[0]?.toUpperCase()}
                                                </button>
                                            );
                                        })}
                                    </div>
                                </div>

                                <div className="tv-stat-divider" />

                                <div className="tv-stat-group">
                                    <span className="tv-stat-label">Emotions</span>
                                    <div className="tv-emo-row">
                                        {emoSummary.map(({ emotion, pct }) => {
                                            const m = emotionMeta(emotion);
                                            const active = filterEmo === emotion;
                                            return (
                                                <button
                                                    key={emotion}
                                                    className="tv-emo-chip"
                                                    style={{
                                                        background: active ? m.bg : "transparent",
                                                        color: active ? m.color : "#64748b",
                                                        border: `1px solid ${active ? m.color : "rgba(255,255,255,0.06)"}`
                                                    }}
                                                    onClick={() => setFilterEmo(active ? null : emotion)}
                                                >
                                                    {m.label} <span style={{ opacity: 0.7 }}>{pct}%</span>
                                                </button>
                                            );
                                        })}
                                    </div>
                                </div>

                                <div className="tv-stat-divider" />

                                {segments.at(-1)?.end > 0 && (
                                    <div className="tv-stat-group">
                                        <span className="tv-stat-label">Duration</span>
                                        <span className="tv-stat-value">{fmt(segments.at(-1).end)}</span>
                                    </div>
                                )}
                            </div>
                        )}

                        <div className="tv-search-wrap">
                            <svg className="tv-search-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#64748b" strokeWidth="2" strokeLinecap="round">
                                <circle cx="11" cy="11" r="8" /><path d="M21 21l-4.35-4.35" />
                            </svg>
                            <input
                                className="tv-search-input"
                                placeholder="Search transcript…"
                                value={search}
                                onChange={(e) => setSearch(e.target.value)}
                            />
                            {(search || filterEmo || filterSpk) && (
                                <button className="tv-clear-btn" onClick={() => { setSearch(""); setFilterEmo(null); setFilterSpk(null); }}>
                                    Clear
                                </button>
                            )}
                        </div>

                        <div className="tv-body" ref={bodyRef}>
                            {filteredGroups.length === 0 ? (
                                <div className="tv-empty">
                                    {segments.length === 0
                                        ? <><div className="tv-empty-icon">📄</div><p className="tv-empty-text">No segments available</p></>
                                        : <><div className="tv-empty-icon">🔍</div><p className="tv-empty-text">No results for your filters</p></>
                                    }
                                </div>
                            ) : (
                                filteredGroups.map((g, gi) => {
                                    const safeSpeaker = g.speaker || "Unknown";
                                    const [c1, c2] = speakerColor(safeSpeaker);
                                    const isActive = activeGroup === gi;

                                    const visibleLines = g.lines.filter((seg) => {
                                        if (filterEmo && seg.emotion?.toLowerCase() !== filterEmo) return false;
                                        if (search && !seg.text.toLowerCase().includes(search.toLowerCase())) return false;
                                        return true;
                                    });

                                    const fullText = visibleLines.map((s) => s.text).join(" ");
                                    const paragraph = fullText.length > 800 ? fullText.slice(0, 800) + "…" : fullText;

                                    const groupEmos = visibleLines.reduce((acc, s) => {
                                        const e = s.emotion?.toLowerCase();
                                        if (e && e !== "neutral") acc[e] = (acc[e] || 0) + 1;
                                        return acc;
                                    }, {});
                                    const topEmo = Object.entries(groupEmos).sort((a, b) => b[1] - a[1])[0]?.[0] || null;
                                    const topEmoMeta = topEmo ? emotionMeta(topEmo) : null;

                                    return (
                                        <div
                                            key={gi}
                                            className="tv-group"
                                            style={{ borderLeftColor: isActive ? c1 : "rgba(255,255,255,0.04)" }}
                                            onMouseEnter={() => setActiveGroup(gi)}
                                            onMouseLeave={() => setActiveGroup(null)}
                                        >
                                            <div className="tv-speaker-row">
                                                <div className="tv-avatar" style={{ background: `linear-gradient(135deg,${c1},${c2})` }}>
                                                    {safeSpeaker[0]?.toUpperCase() || "?"}
                                                </div>
                                                <span className="tv-speaker-name" style={{ color: c1 }}>{safeSpeaker}</span>
                                                <span className="tv-timestamp">{fmt(g.start)}</span>
                                                {topEmoMeta && (
                                                    <span
                                                        className="tv-emotion-tag"
                                                        style={{ color: topEmoMeta.color, background: topEmoMeta.bg }}
                                                    >
                                                        {topEmoMeta.label}
                                                    </span>
                                                )}
                                            </div>
                                            <div className="tv-lines-wrap">
                                                <p className="tv-paragraph">{highlight(paragraph)}</p>
                                            </div>
                                        </div>
                                    );
                                })
                            )}
                        </div>
                    </>
                )}

                {activeTab === "summary" && (
                    <div className="tv-body">
                        {!aiAnalysis && !aiLoading && (
                            <div className="tv-empty">
                                <div className="tv-empty-icon">
                                    <svg width="48" height="48" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
                                        <circle cx="24" cy="24" r="24" fill="rgba(245,80,54,0.1)" />
                                        {/* Groq "G" lettermark */}
                                        <path d="M32 25.5v5.5H21.5C17.36 31 14 27.64 14 23.5v-5C14 14.36 17.36 11 21.5 11H34v4.5H22c-1.1 0-1.8.7-1.8 1.8v5.4c0 1.1.7 1.8 1.8 1.8h2.5v-1.5H22V19.5H32v6z" fill="#F55036" />
                                    </svg>
                                </div>
                                <p className="tv-empty-text">Generate an AI summary of this meeting.</p>
                                {aiError && (
                                    <p style={{ color: "#f87171", fontSize: 12, marginTop: 8 }}>{aiError}</p>
                                )}
                                <button
                                    onClick={handleGenerateSummary}
                                    disabled={!segments.length}
                                    style={{
                                        marginTop: 16,
                                        padding: "8px 20px",
                                        background: "linear-gradient(135deg,#38bdf8,#7c3aed)",
                                        border: "none",
                                        borderRadius: 8,
                                        color: "#fff",
                                        fontSize: 13,
                                        fontWeight: 600,
                                        cursor: "pointer",
                                        opacity: segments.length ? 1 : 0.5,
                                    }}
                                >
                                    Generate AI Summary
                                </button>
                            </div>
                        )}
                        {aiLoading && (
                            <div className="tv-empty">
                                <div style={{ position: "relative", width: 48, height: 48, marginBottom: 8 }}>
                                    <svg width="48" height="48" viewBox="0 0 48 48" fill="none" style={{ position: "absolute", inset: 0 }}>
                                        <circle cx="24" cy="24" r="20" stroke="rgba(245,80,54,0.15)" strokeWidth="3" />
                                        <circle cx="24" cy="24" r="20" stroke="#F55036" strokeWidth="3" strokeLinecap="round"
                                            strokeDasharray="30 96" style={{ transformOrigin: "center", animation: "tv-spin 0.9s linear infinite" }} />
                                    </svg>
                                    <svg width="48" height="48" viewBox="0 0 48 48" fill="none" style={{ position: "absolute", inset: 0 }}>
                                        <path d="M32 25.5v5.5H21.5C17.36 31 14 27.64 14 23.5v-5C14 14.36 17.36 11 21.5 11H34v4.5H22c-1.1 0-1.8.7-1.8 1.8v5.4c0 1.1.7 1.8 1.8 1.8h2.5v-1.5H22V19.5H32v6z" fill="#F55036" />
                                    </svg>
                                </div>
                                <p className="tv-empty-text" style={{ color: "#F55036", fontWeight: 500 }}>Groq is analyzing…</p>
                            </div>
                        )}
                        {aiAnalysis && !aiLoading && (
                            <>
                                <SummaryPanel analysis={aiAnalysis} />
                                {aiError && (
                                    <p style={{ color: "#f87171", fontSize: 12, margin: "4px 0 0", textAlign: "right" }}>{aiError}</p>
                                )}
                                <div style={{ textAlign: "right", padding: "8px 0" }}>
                                    <button
                                        onClick={handleGenerateSummary}
                                        style={{ fontSize: 11, color: "#64748b", background: "none", border: "none", cursor: "pointer", textDecoration: "underline" }}
                                    >
                                        Regenerate
                                    </button>
                                </div>
                            </>
                        )}
                    </div>
                )}

            </div>
        </div>
    );
}