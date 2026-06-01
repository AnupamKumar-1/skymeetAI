import { TRANSCRIPTS_ENABLED, EMOTIONS_ENABLED } from "../environment";

export const SOCKET_SERVER_URL =
  import.meta.env.VITE_SIGNALING_URL || "http://localhost:8000";

export const TRANSCRIPT_ENDPOINT = (() => {
  if (!TRANSCRIPTS_ENABLED) return null;

  const env =
    import.meta.env.VITE_TRANSCRIPT_URL ||
    import.meta.env.VITE_AI_URL;

  if (!env) return "http://localhost:5001/process_meeting";

  const trimmed = env.replace(/\/+$/, "");
  return trimmed.endsWith("/process_meeting")
    ? trimmed
    : `${trimmed}/process_meeting`;
})();

export const API_BASE =
  import.meta.env.VITE_API_URL ||
  "http://localhost:8000/api/v1";

export const ICE_CONFIG = {
  iceServers: [
    {
      urls: ["stun:stun.l.google.com:19302"],
    },
    {
      urls: [
        import.meta.env.VITE_TURN_URL_UDP,
        import.meta.env.VITE_TURN_URL_80,
        import.meta.env.VITE_TURN_URL_443,
        import.meta.env.VITE_TURN_URL_443_TCP,
        import.meta.env.VITE_TURN_URL_TLS,
      ].filter(Boolean),
      username: import.meta.env.VITE_TURN_USERNAME,
      credential: import.meta.env.VITE_TURN_CREDENTIAL,
    },
  ],
};

export const EMO_CONFIG = {
  captureIntervalMs: 3000,
};

export { EMOTIONS_ENABLED };