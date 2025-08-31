import React from "react";
import { useNavigate } from "react-router-dom";
import "../App.css";
import "../styles/landing.css";

export default function LandingPage() {
  const router = useNavigate();

  return (
    <div className="lp-root">
      <header className="lp-nav" role="navigation" aria-label="Main navigation">
        <div className="lp-brand" onClick={() => router("/")}>
          <svg width="34" height="34" viewBox="0 0 24 24" fill="none" aria-hidden>
            <rect x="2" y="2" width="20" height="20" rx="6" fill="#05233a"/>
            <path d="M7 12h10M7 8h10M7 16h10" stroke="white" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" opacity="0.92"/>
          </svg>
          <h2>SkyMeet</h2>
        </div>

        <div className="nav-actions" role="menubar" aria-label="Auth">
          <p role="menuitem" onClick={() => router("/auth")}>Login</p>
          <button
            className="nav-cta"
            onClick={() => router("/auth")}
            aria-label="Get started"
          >
            Get Started
          </button>
        </div>
      </header>

      <main className="lp-hero">
        <section className="lp-card lp-hero-left">
          <h1>
            <span className="accent">Connect</span> with your loved ones
          </h1>
          <p className="lead">
            Close the distance with secure, instant video rooms. Easy to create,
            simple to share — built for moments that matter.
          </p>

          <div style={{ marginTop: 18, color: "var(--muted)", fontSize: 13 }}>
            <strong>Tip:</strong> After creating an account you can host meetings,
            save transcripts and access recordings from anywhere.
          </div>
        </section>

        <aside className="lp-card lp-hero-right" aria-hidden>
          <div className="lp-illustration">
            <img src="/mobile.png" alt="SkyMeet preview on mobile" />
          </div>
        </aside>
      </main>

      <div className="lp-foot">
        Built with ❤️ — SkyMeet. Privacy-first video rooms. 
        <span style={{ color: "var(--muted)" }}> sign up to get full access.</span>
      </div>
    </div>
  );
}
