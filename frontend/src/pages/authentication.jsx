import * as React from "react";
import { useNavigate } from "react-router-dom";
import { AuthContext } from "../contexts/AuthContext";
import "../styles/authentication.css";

const FIELD_RULES = {
  name: (v) => {
    if (!v.trim()) return "Name is required.";
    if (v.trim().length > 64) return "Name must be under 64 characters.";
    return "";
  },
  username: (v) => {
    if (v.length < 3) return "Username must be at least 3 characters.";
    if (v.length > 32) return "Username must be under 32 characters.";
    if (!/^[a-z0-9_.-]+$/.test(v)) return "Lowercase letters, numbers, _, ., - only.";
    return "";
  },
  password: (v, isLogin) => {
    if (v.length < 8) return "Password must be at least 8 characters.";
    if (v.length > 128) return "Password is too long.";
    if (!isLogin && !/[A-Z]/.test(v)) return "Include at least one uppercase letter.";
    if (!isLogin && !/[0-9]/.test(v)) return "Include at least one number.";
    return "";
  },
  email: (v) => {
    if (!v.trim()) return "Email is required.";
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(v.trim())) return "Enter a valid email address.";
    if (v.trim().length > 254) return "Email is too long.";
    return "";
  },
};

function useFieldState(initial = "") {
  const [value, setValue] = React.useState(initial);
  const [error, setError] = React.useState("");
  return { value, setValue, error, setError };
}

export default function Authentication() {
  const navigate = useNavigate();
  const { handleRegister, handleLogin } = React.useContext(AuthContext);

  const [formState, setFormState] = React.useState(0);
  const [loading, setLoading] = React.useState(false);
  const [apiError, setApiError] = React.useState("");
  const [snack, setSnack] = React.useState({ open: false, message: "" });

  const name = useFieldState();
  const username = useFieldState();
  const password = useFieldState();
  const email = useFieldState();

  const isLogin = formState === 0;

  React.useEffect(() => {
    name.setValue(""); name.setError("");
    username.setValue(""); username.setError("");
    password.setValue(""); password.setError("");
    email.setValue(""); email.setError("");
    setApiError("");
  }, [formState]);

  React.useEffect(() => {
    if (!snack.open) return;
    const t = setTimeout(() => setSnack((s) => ({ ...s, open: false })), 4000);
    return () => clearTimeout(t);
  }, [snack.open]);

  function validateAll() {
    let valid = true;

    if (!isLogin) {
      const e = FIELD_RULES.name(name.value);
      name.setError(e);
      if (e) valid = false;

      const ee = FIELD_RULES.email(email.value);
      email.setError(ee);
      if (ee) valid = false;
    }

    const eu = FIELD_RULES.username(username.value);
    username.setError(eu);
    if (eu) valid = false;

    const ep = FIELD_RULES.password(password.value, isLogin);
    password.setError(ep);
    if (ep) valid = false;

    return valid;
  }

  const handleAuth = async (e) => {
    e.preventDefault();
    setApiError("");

    if (!validateAll()) return;

    setLoading(true);
    try {
      if (isLogin) {
        await handleLogin(username.value.toLowerCase().trim(), password.value);
        navigate("/home", { replace: true });
      } else {
        const result = await handleRegister(
          name.value.trim(),
          username.value.toLowerCase().trim(),
          password.value,
          email.value.toLowerCase().trim()
        );
        setSnack({ open: true, message: result || "Account created — please sign in." });
        setFormState(0);
      }
    } catch (err) {
      const msg = err?.response?.data?.message ?? "Something went wrong. Please try again.";
      setApiError(msg);
    } finally {
      setLoading(false);
    }
  };

  function handleFieldChange(field, value) {
    field.setValue(value);
    if (field.error) field.setError("");
    if (apiError) setApiError("");
  }

  return (
    <div className="au-root">
      <div className="au-bg" aria-hidden="true" />

      <aside className="au-hero" aria-hidden="true">
        <div className="au-hero-inner">
          <div className="au-hero-logo">
            <img src="/logo.svg" alt="Hoovik" width="32" height="32" />
            <span>Hoovik</span>
          </div>

          <div className="au-hero-content">
            <div className="au-hero-badge">
              <span className="au-hero-badge-dot" />
              Real-time AI Meeting Platform
            </div>
            <h2 className="au-hero-heading">
              Meetings that<br />
              <span className="au-hero-accent">understand you.</span>
            </h2>
            <p className="au-hero-lead">
              Emotion-aware transcription, live AI insights,
              and crystal-clear WebRTC — all in one place.
            </p>

            <div className="au-hero-stats">
              <div className="au-hero-stat">
                <span className="au-stat-val">Instant</span>
                <span className="au-stat-label">WebRTC real-time streaming</span>
              </div>
              <div className="au-hero-stat">
                <span className="au-stat-val">Live AI</span>
                <span className="au-stat-label">Emotion insights</span>
              </div>
              <div className="au-hero-stat">
                <span className="au-stat-val">Whisper</span>
                <span className="au-stat-label">Accurate transcripts</span>
              </div>
            </div>
          </div>

          <div className="au-hero-card">
            <div className="au-hero-card-top">
              <span className="au-hero-card-label">AI POWERED SESSION</span>
              <div className="au-hero-card-dots">
                <span className="au-dot au-dot-r" />
                <span className="au-dot au-dot-y" />
                <span className="au-dot au-dot-g" />
              </div>
            </div>
            <div className="au-hero-avatars">
              {["A", "P", "R"].map((l, i) => (
                <div key={i} className={`au-av au-av-${i + 1}`}>{l}</div>
              ))}
            </div>
            <div className="au-hero-chips">
              <span className="au-chip au-chip-sky">● Live Emotion AI</span>
              <span className="au-chip au-chip-gold">✦ Smart Transcript</span>
            </div>
          </div>
        </div>
      </aside>

      <main className="au-panel">
        <div className="au-form-card">
          <div className="au-form-brand">
            <img src="/logo.svg" alt="Hoovik" width="32" height="32" />
            <span className="au-form-brand-name">Hoovik</span>
          </div>

          <h1 className="au-form-title">
            {isLogin ? "Welcome back" : "Create account"}
          </h1>
          <p className="au-form-sub">
            {isLogin
              ? "Sign in to your Hoovik account"
              : "Join the intelligent meeting platform"}
          </p>

          <div className="au-tabs" role="tablist">
            <button
              role="tab"
              aria-selected={isLogin}
              className={`au-tab${isLogin ? " au-tab-active" : ""}`}
              onClick={() => setFormState(0)}
              type="button"
            >
              Sign In
            </button>
            <button
              role="tab"
              aria-selected={!isLogin}
              className={`au-tab${!isLogin ? " au-tab-active" : ""}`}
              onClick={() => setFormState(1)}
              type="button"
            >
              Sign Up
            </button>
          </div>

          <form className="au-form" onSubmit={handleAuth} noValidate>
            {!isLogin && (
              <div className="au-field au-field-animate">
                <label className="au-label" htmlFor="au-name">Full Name</label>
                <input
                  id="au-name"
                  className={`au-input${name.error ? " au-input-invalid" : ""}`}
                  type="text"
                  placeholder="e.g. Anupam Kumar"
                  value={name.value}
                  onChange={(e) => handleFieldChange(name, e.target.value)}
                  autoComplete="name"
                  maxLength={64}
                  required
                  aria-invalid={!!name.error}
                  aria-describedby={name.error ? "au-name-err" : undefined}
                />
                {name.error && (
                  <span id="au-name-err" className="au-field-error" role="alert">
                    {name.error}
                  </span>
                )}
              </div>
            )}

            {!isLogin && (
              <div className="au-field au-field-animate">
                <label className="au-label" htmlFor="au-email">Email address</label>
                <input
                  id="au-email"
                  className={`au-input${email.error ? " au-input-invalid" : ""}`}
                  type="email"
                  placeholder="you@example.com"
                  value={email.value}
                  onChange={(e) => handleFieldChange(email, e.target.value)}
                  autoComplete="email"
                  maxLength={254}
                  required
                  aria-invalid={!!email.error}
                  aria-describedby={email.error ? "au-email-err" : undefined}
                />
                {email.error && (
                  <span id="au-email-err" className="au-field-error" role="alert">
                    {email.error}
                  </span>
                )}
              </div>
            )}

            <div className="au-field">
              <label className="au-label" htmlFor="au-username">Username</label>
              <input
                id="au-username"
                className={`au-input${username.error ? " au-input-invalid" : ""}`}
                type="text"
                placeholder="your_username"
                value={username.value}
                onChange={(e) => handleFieldChange(username, e.target.value.toLowerCase())}
                autoComplete="username"
                maxLength={32}
                required
                aria-invalid={!!username.error}
                aria-describedby={username.error ? "au-username-err" : undefined}
              />
              {username.error && (
                <span id="au-username-err" className="au-field-error" role="alert">
                  {username.error}
                </span>
              )}
            </div>

            <div className="au-field">
              <label className="au-label" htmlFor="au-password">Password</label>
              <input
                id="au-password"
                className={`au-input${password.error ? " au-input-invalid" : ""}`}
                type="password"
                placeholder="••••••••"
                value={password.value}
                onChange={(e) => handleFieldChange(password, e.target.value)}
                autoComplete={isLogin ? "current-password" : "new-password"}
                maxLength={128}
                required
                aria-invalid={!!password.error}
                aria-describedby={password.error ? "au-password-err" : undefined}
              />
              {password.error && (
                <span id="au-password-err" className="au-field-error" role="alert">
                  {password.error}
                </span>
              )}
            </div>

            {apiError && (
              <div className="au-error" role="alert">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" aria-hidden="true">
                  <circle cx="12" cy="12" r="10" />
                  <line x1="12" y1="8" x2="12" y2="12" />
                  <line x1="12" y1="16" x2="12.01" y2="16" />
                </svg>
                <span>{apiError}</span>
              </div>
            )}

            <button
              type="submit"
              className="au-submit"
              disabled={loading}
              aria-busy={loading}
            >
              {loading ? (
                <span className="au-spinner" aria-label="Loading" />
              ) : (
                <>
                  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    {isLogin
                      ? <><path d="M15 3h6v6M14 10l6.1-6.1M9 21H3v-6M10 14l-6.1 6.1" /></>
                      : <><path d="M16 21v-2a4 4 0 00-4-4H6a4 4 0 00-4 4v2" /><circle cx="9" cy="7" r="4" /><line x1="19" y1="8" x2="19" y2="14" /><line x1="22" y1="11" x2="16" y2="11" /></>
                    }
                  </svg>
                  {isLogin ? "Sign In" : "Create Account"}
                </>
              )}
            </button>
          </form>

          <p className="au-form-foot">
            {isLogin ? "Don't have an account? " : "Already have an account? "}
            <button
              type="button"
              className="au-form-foot-link"
              onClick={() => setFormState(isLogin ? 1 : 0)}
            >
              {isLogin ? "Sign up" : "Sign in"}
            </button>
          </p>
        </div>
      </main>

      {snack.open && (
        <div className="au-snack" role="status" aria-live="polite">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" aria-hidden="true">
            <polyline points="20 6 9 17 4 12" />
          </svg>
          {snack.message}
        </div>
      )}
    </div>
  );
}