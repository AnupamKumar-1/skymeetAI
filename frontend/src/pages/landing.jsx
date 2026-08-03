import React from "react";
import { useNavigate } from "react-router-dom";
import "../styles/landing.css";

export default function LandingPage() {
  const navigate = useNavigate();

  return (
    <div className="lp-root">

      <div className="lp-bg-mesh" aria-hidden />
      <header className="lp-nav" role="navigation" aria-label="Main navigation">
        <div className="lp-brand" onClick={() => navigate("/")}>
          <img src="/logo.svg" alt="Hoovik" width="36" height="36" />
          <span className="lp-brand-name">Hoovik</span>
        </div>

        <div className="lp-nav-right">
          <button className="lp-nav-link" onClick={() => navigate("/auth")}>
            Login
          </button>
          <button
            className="lp-nav-cta"
            onClick={() => navigate("/auth")}
            aria-label="Get started with Hoovik"
          >
            Get Started →
          </button>
        </div>
      </header>

      <main className="lp-hero">

        <section className="lp-hero-left">
          <div className="lp-badge" aria-label="AI features">
            <div className="lp-badge-dot" />
            <span className="lp-badge-text">
              Real-time Emotion AI + Smart Transcription
            </span>
          </div>

          <h1>
            Intelligent meetings,
            <span className="lp-h1-line2">
              {" beyond "}
              <span className="lp-accent">video calls.</span>
            </span>
          </h1>

          <p className="lp-lead">
            Understand how people feel, not just what they say. Analyze
            emotions in real-time, generate speaker-aware transcripts, and gain
            deeper insights from every conversation.
          </p>

          <div className="lp-cta-row">
            <button
              className="lp-btn-primary"
              onClick={() => navigate("/auth")}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden>
                <path
                  d="M15 10l4.553-2.069A1 1 0 0121 8.82v6.36a1 1 0 01-1.447.89L15 14M3 8a2 2 0 012-2h10a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2V8z"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
              Start Smart Meeting
            </button>

            <button
              className="lp-btn-ghost"
              onClick={() => navigate("/auth")}
            >
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" aria-hidden>
                <path
                  d="M15 3h6v6M14 10l6.1-6.1M9 21H3v-6M10 14l-6.1 6.1"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  strokeLinecap="round"
                />
              </svg>
              Join with Room Code
            </button>
          </div>

          <div className="lp-stats-row">
            <div className="lp-stat">
              <span className="lp-stat-val">Instant</span>
              <span className="lp-stat-label">WebRTC real-time streaming</span>
            </div>

            <div className="lp-stat">
              <span className="lp-stat-val">Live AI</span>
              <span className="lp-stat-label"> Multimodal emotion inference with sub-second latency</span>
            </div>

            <div className="lp-stat">
              <span className="lp-stat-val">Whisper</span>
              <span className="lp-stat-label">Accurate speaker-aware transcripts</span>
            </div>

            <div className="lp-stat">
              <span className="lp-stat-val">RAG Q&A</span>
              <span className="lp-stat-label">Ask questions about your meeting transcript</span>
            </div>
          </div>
        </section>


        <aside className="lp-hero-right" aria-label="Live meeting preview">
          <div className="lp-glass-card">
            <div className="lp-card-top">
              <span className="lp-card-title">AI POWERED SESSION</span>
              <div className="lp-card-dots" aria-hidden>
                <div className="lp-dot lp-dot-r" />
                <div className="lp-dot lp-dot-y" />
                <div className="lp-dot lp-dot-g" />
              </div>
            </div>

            <div className="lp-video-grid" aria-hidden>
              <div className="lp-vid-tile lp-vid-main">
                <div className="lp-speaking-ring" />
                <div className="lp-vid-avatar">
                  <div className="lp-avatar-ring lp-av1">A</div>
                  <span className="lp-vid-name">Anupam K.</span>
                </div>
                <div className="lp-vid-bar">
                  <div className="lp-live-dot" />
                  <span className="lp-vid-bar-text">Active Speaker</span>
                  <div className="lp-wave-bars">
                    {[0.6, 0.45, 0.75, 0.55, 0.65].map((d, i) => (
                      <div
                        key={i}
                        className="lp-wave-bar"
                        style={{ "--d": `${d}s` }}
                      />
                    ))}
                  </div>
                </div>
              </div>

              <div className="lp-vid-tile">
                <div className="lp-vid-avatar">
                  <div className="lp-avatar-ring lp-av2">K</div>
                  <span className="lp-vid-name">Kumar</span>
                </div>
              </div>

              <div className="lp-vid-tile">
                <div className="lp-vid-avatar">
                  <div className="lp-avatar-ring lp-av3">A</div>
                  <span className="lp-vid-name">Anup K.</span>
                </div>
              </div>
            </div>

            <div className="lp-card-bottom">
              <span className="lp-chip lp-chip-sky">● Live Emotion AI</span>
              <span className="lp-chip lp-chip-gold">✦ Smart Transcript</span>
              <span className="lp-chip lp-chip-green">⬤ Real-time Meeting</span>
            </div>
          </div>
        </aside>
      </main>

      <footer className="lp-foot">
        <p>
          Made with ❤️ at <strong>Hoovik</strong> — where meetings become smarter in real time.
        </p>
        <span className="lp-foot-status">
          Emotion & transcription services active
        </span>
      </footer>
    </div>
  );
}