# Project README

> **VideoMeet Frontend** — A React-based in-browser meeting app with WebRTC, Socket.IO signalling, host-side recording & transcription, and optional emotion analysis.

---

## Table of contents

- [Project README](#project-readme)
  - [Table of contents](#table-of-contents)
  - [What this repo contains](#what-this-repo-contains)
  - [Features](#features)
  - [Prerequisites](#prerequisites)
  - [Environment variables](#environment-variables)
  - [Install \& run (development)](#install--run-development)
  - [Build \& run (production)](#build--run-production)
  - [Important routes \& components](#important-routes--components)
    - [`VideoMeet` responsibilities (quick summary)](#videomeet-responsibilities-quick-summary)
  - [Signalling / WebRTC flow](#signalling--webrtc-flow)
  - [API contracts (expected request / response shapes)](#api-contracts-expected-request--response-shapes)
    - [Transcript service (`REACT_APP_TRANSCRIPT_URL`)](#transcript-service-react_app_transcript_url)
    - [Emotion analysis service (`REACT_APP_EMOTION_URL`)](#emotion-analysis-service-react_app_emotion_url)
    - [Backend API (`REACT_APP_API_URL` / `REACT_APP_SERVER_URL`)](#backend-api-react_app_api_url--react_app_server_url)
  - [Auth \& user state](#auth--user-state)
  - [Media \& recordings](#media--recordings)
  - [Troubleshooting \& FAQ](#troubleshooting--faq)
  - [Runbook / Deployment checklist](#runbook--deployment-checklist)
  - [License](#license)

---

## What this repo contains

This is the frontend application for the VideoMeet project (React). Key files/components you should know:

- `src/components/VideoMeet.jsx` — Main meeting view: local media capture, Socket.IO signalling, per-peer `RTCPeerConnection`, active-speaker detection, recording uploads, and optional emotion capture.
- `src/components/home.jsx` — Landing / dashboard: create room, copy link, join room, and show recent transcripts.
- `src/contexts/AuthContext.jsx` — Authentication provider: login/register logic, axios clients, meeting history persistence, and helper functions used across the app.
- `src/components/authentication.jsx` — MUI-based sign in / sign up UI.
- `src/mediaController.js` — Media utility helpers (present in the repo and referenced by `VideoMeet`).

> Note: file paths above are shown as examples. Adjust path prefixes to match where files are located in your repository.

---

## Features

- Camera & microphone capture via `getUserMedia`.
- Socket.IO signalling server integration for room coordination.
- One `RTCPeerConnection` per remote participant with stable offer/answer/ICE handling.
- Host-side per-participant audio recording and upload for transcription.
- Periodic short clip capture (host) for emotion analysis (multiple delivery fallbacks).
- Active-speaker detection using WebAudio `AnalyserNode`.
- Basic authentication flow (register / login) and persistence of tokens to `localStorage` via `AuthContext`.

---

## Prerequisites

- Node.js (v16+ recommended)
- npm or yarn
- A signalling server (Socket.IO compatible) reachable from the frontend
- A backend API to persist rooms, transcripts, and user data (optional for local-only testing)
- (Optional) Transcript / Emotion analysis services if you want recordings to be analyzed

---

## Environment variables

Place these in a `.env` file at the project root (create if missing). The frontend reads variables with the `REACT_APP_` prefix.

```env
REACT_APP_SIGNALING_URL=http://localhost:8000    # Socket.IO signaling server
REACT_APP_API_URL=http://localhost:8000/api/v1  # Backend API base
REACT_APP_SERVER_URL=http://localhost:8000      # Backend server (non-API root)
REACT_APP_TRANSCRIPT_URL=http://localhost:5001/process_meeting  # Transcript service endpoint
REACT_APP_EMOTION_URL=http://localhost:5002/analyze           # Emotion analysis endpoint
REACT_APP_SUPPORTS_GLOBAL_MEETINGS=true  # If your backend exposes global /meetings endpoints
```

Defaults are used in many files when env vars are not provided. Adjust for production before building.

---

## Install & run (development)

```bash
# Install
npm install

# Run dev server
npm start
```

This starts the frontend on the default React port (typically `http://localhost:3000`). Ensure the signalling server and backend API are accessible from the browser.

---

## Build & run (production)

```bash
npm run build
# Serve the build directory with your preferred static host (nginx, surge, Netlify, etc.).
```

When deploying, set the `REACT_APP_*` environment variables at build time so the compiled bundle contains the correct endpoints.

---

## Important routes & components

- `/home` — Home dashboard (`home.jsx`) with room creation and transcript listing.
- `/room/:roomId` — Meeting room (`VideoMeet.jsx`) where all call features are executed.
- `/login` — Authentication page (component `authentication.jsx`) that consumes `AuthContext`.


### `VideoMeet` responsibilities (quick summary)

- Acquire local media (`getUserMedia`).
- Connect to Socket.IO signalling server and emit `join-call`.
- Create and maintain `RTCPeerConnection` objects for remote participants.
- Start/stop host recordings and upload audio chunks to the transcript service.
- Run periodic short clip capture for emotion analysis (host-only by default).
- Cleanup and leave-handling when the meeting ends.

---

## Signalling / WebRTC flow

This project expects a Socket.IO-style signalling server. Typical events used by the frontend include (but are not limited to):

- `join-call` — client emits on joining a room; payload typically `{ roomId, displayName, isHost }`.
- `existing-participants` — server -> newly joined client: list of participant IDs already in the room.
- `user-joined` / `user-left` — server broadcasts changes in participants.
- `signal` — generic container for SDP offers/answers and ICE candidates: `{ to, from, type, data }`.
- `chat-message`, `participant-meta-updated`, `end-meeting`, `emotion.frame` — optional application events used in the UI.

Adapt your signalling server to accept and relay these events. The frontend creates an offer per new remote participant and exchanges SDP/ICE via `signal` events.

---

## API contracts (expected request / response shapes)

Below are example minimal contracts inferred from the frontend code so backend/AI services can be implemented. These are intentionally light — adapt as needed.

### Transcript service (`REACT_APP_TRANSCRIPT_URL`)

**Endpoint (example)**
```
POST /process_meeting
Content-Type: multipart/form-data
```

**Form fields**
- `file` — audio file (webm/ogg/wav) or `Blob` chunk recorded by `MediaRecorder`.
- `roomCode` — string; meeting room code.
- `participantId` or `participantName` — who the recording belongs to.
- `startTime` / `endTime` — optional ISO timestamps.

**Response (example)**
```json
{
  "success": true,
  "transcript": "... full transcript text ...",
  "segments": [
    {"start": "2025-09-03T10:00:00Z", "end": "2025-09-03T10:00:05Z", "text": "Hi everyone"}
  ]
}
```

> Note: The frontend expects a JSON response containing at least a transcript string. If the transcript service returns chunks/segments, the frontend may send those to the backend API for persistence.

---

### Emotion analysis service (`REACT_APP_EMOTION_URL`)

The frontend uses multiple fallbacks to deliver short clips (binary via socket, base64 payload via socket, or REST multipart). Support any one of these.

**Endpoint (example)**
```
POST /analyze
Content-Type: multipart/form-data
```

**Form fields**
- `clip` — short video/audio clip (webm) or single-frame image
- `participantId` — string
- `roomId` — string

**Response (example)**
```json
{
  "success": true,
  "emotions": { "happy": 0.72, "neutral": 0.20, "sad": 0.08 }
}
```

If using socket events, the frontend will send binary payloads under event names like `emotion.frame` and may expect an ACK or `emotion.result` event with analysis.

---

### Backend API (`REACT_APP_API_URL` / `REACT_APP_SERVER_URL`)

The frontend contacts several endpoints to create rooms, fetch transcripts, and manage user history. Example endpoints used (implement any subset supported by your backend):

- `POST /api/v1/rooms` — create a room (body `{ hostName }`) -> returns `{ code: "ROOM123", roomId: "..." }`.
- `GET /api/v1/rooms/:roomId` — validate / fetch room metadata.
- `GET /api/v1/transcript` — list recent transcripts.
- `POST /api/v1/transcript` — persist transcripts (used after transcription service returns text).
- Auth endpoints under `/api/v1/users`: `POST /login`, `POST /register`, etc., returning tokens.
- Meeting history endpoints (used by `AuthContext`): `/meetings`, `/users/meetings`, `/get_all_activity`, `/meetings/:code/participants`.

The frontend is defensive and tries multiple endpoints in order to maximize compatibility with different backend shapes — see the `AuthContext` provider for the exact order and fallback logic.

---

## Auth & user state

`AuthContext.jsx` exposes the app-wide auth helpers and two axios clients:

- `client` — axios instance targeting user routes (`/api/v1/users`).
- `apiClient` — general API axios instance (`/api/v1`) with an interceptor that attaches `localStorage.token` on each request.

Key functions exported by the context:

- `handleLogin(username, password)` — logs in, stores token in `localStorage`, and sets default headers.
- `handleRegister(name, username, password)` — creates account and returns a success message.
- `getHistoryOfUser()` / `addToUserHistory(payload)` — helper methods to read / write meeting history using best-effort fallbacks.

`authentication.jsx` shows how the UI consumes `AuthContext` for login & register flows.

---

## Media & recordings

- Recordings are handled via the DOM `MediaRecorder` API. The host records per participant audio (and local audio) and uploads audio chunks to the configured `REACT_APP_TRANSCRIPT_URL`.
- The app also optionally captures short video/audio clips periodically for emotion analysis (host-only by default).
- Video streams are rendered into `<video>` elements. Active-speaker detection uses Web Audio API analysis.

---

## Troubleshooting & FAQ

**No camera or mic available**
- Ensure the app is served over HTTPS (or using `localhost` during development). Browser restrictions block `getUserMedia` on insecure contexts.
- Check OS-level permissions for camera/microphone.

**Frozen video frames**
- Video tiles may show a blank/placeholder while `VideoMeet` detects and replaces frozen frames. Inspect the senders' bandwidth and network conditions.

**Recordings not uploaded**
- Confirm `REACT_APP_TRANSCRIPT_URL` is reachable and CORS allows the frontend origin.
- Check console/network for errors during the multipart upload.

**Emotion analysis not working**
- The code uses multiple delivery methods. Ensure the server accepts at least one: socket binary frames, socket base64 payloads, or REST multipart POST.

**401 / token issues**
- `AuthContext` will clear tokens and redirect to `/login` on 401 responses. Ensure tokens are refreshed or reissued by login when expired.

---

## Runbook / Deployment checklist

1. Host your signalling server (Socket.IO) and ensure `REACT_APP_SIGNALING_URL` points to it.
2. Host backend API and set `REACT_APP_API_URL` and `REACT_APP_SERVER_URL` accordingly.
3. If you need transcription or emotion analysis, deploy those services and set `REACT_APP_TRANSCRIPT_URL` and `REACT_APP_EMOTION_URL`.
4. Configure CORS to allow requests from your frontend origin.
5. Build the frontend with `NODE_ENV=production` and serve static assets via CDN or web server.
6. Monitor `socket` events, ICE negotiation logs, and media upload errors for next-level debugging.

---

## License

This project does not include a license file by default. Add one (e.g. `MIT`) if you plan to open-source the repo.

---
