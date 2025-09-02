# Backend — Meeting / Transcript / Emotion Service

> Node.js + Express backend for a simple meeting/transcript/emotion pipeline with Socket.IO realtime features.  
> Supports user auth (JWT), meeting rooms, transcripts, basic emotion-service integration and realtime signalling/participant events.

---

## Quick summary / features

- REST API for users, rooms, meetings and transcripts (`/api/v1/*`)
- Simple user registration & JWT login
- Room creation endpoint returning a room code
- Transcript storage (DB + optional file metadata)
- Socket.IO-based call/join/leave signalling and basic chat/emotion events
- Integration helper to forward audio/video/frame files to an external emotion analysis service
- Body-size / upload-friendly defaults and some sensible environment-configurable limits

---

## Table of contents

- [Requirements](#requirements)  
- [Install & run](#install--run)  
- [Environment variables](#environment-variables)  
- [API overview & examples](#api-overview--examples)  
  - Authentication (register / login)  
  - Rooms (create / get)  
  - Transcripts (create / list / download)  
  - Emotion upload endpoint  
  - Meetings endpoints  
- [Socket.IO events (client ↔ server)](#socketio-events-client--server)  
- [Project structure](#project-structure)  
- [Notes / tips](#notes--tips)  
- [License](#license)

---

## Requirements

- Node.js (v16+ recommended)
- npm
- MongoDB instance (cloud or local)
- Optional: PM2 for the `npm run prod` script

---

## Install & run

```bash
# from project root (where package.json is)
cd backend

# install dependencies
npm install

# run in dev (uses nodemon)
npm run dev
# or start normally
npm start
# production via pm2
npm run prod
```

The app listens on `process.env.PORT` (default `8000`) and mounts the API under `/api/v1`.

---

## Environment variables

Create a `.env` file in the `backend` folder (the repo includes a sample/empty `.env`). Important variables used by the app:

```env
# Required
MONGO_URI=    # MongoDB connection string, e.g. mongodb://user:pass@host:27017/dbname
JWT_SECRET=   # Secret used to sign JWT tokens (required for auth-protected routes)

# Server
PORT=8000
NODE_ENV=development

# CORS / client
CLIENT_ORIGIN=http://localhost:3000
CLIENT_PORT=3000

# Request body size limits
REQUEST_JSON_LIMIT=4mb
REQUEST_URLENCODED_LIMIT=4mb

# Socket settings (optional)
SOCKET_MAX_HTTP_BUFFER=100000000  # maximum socket upload size (bytes)
SOCKET_PING_INTERVAL=
SOCKET_PING_TIMEOUT=

# Emotion service integration
EMOTION_SERVICE_URL=http://localhost:5002/analyze
EMOTION_UPLOAD_TMP_DIR=/tmp/emotion_uploads
EMOTION_MAX_FILE_BYTES=
PARTIAL_UPLOAD_MAX_BYTES=
PARTIAL_UPLOAD_TTL_MS=
```

Defaults are used in code where a variable is not set (e.g. `PORT` defaults to `8000`, `CLIENT_ORIGIN` defaults to `http://localhost:3000`, `EMOTION_SERVICE_URL` defaults to `http://localhost:5002/analyze`).

---

## API overview & examples

All endpoints are mounted under `/api/v1`. Below are the most important endpoints and example requests.

### Authentication

#### Register
`POST /api/v1/users/register`

Body (JSON):
```json
{
  "name": "Alice",
  "username": "alice@example.com",
  "password": "strongpassword"
}
```

Response: `201 Created` (on success)

#### Login
`POST /api/v1/users/login`

Body (JSON):
```json
{
  "username": "alice@example.com",
  "password": "strongpassword"
}
```

Response (200):
```json
{
  "success": true,
  "accessToken": "<JWT>",
  "expiresIn": "1h",
  "user": {
    "_id": "...",
    "username": "...",
    "name": "..."
  }
}
```

Use the `accessToken` as a Bearer token for protected routes (e.g. `Authorization: Bearer <token>`).

---

### Rooms

#### Create room
`POST /api/v1/rooms`

Body (JSON):
```json
{
  "hostName": "Alice"
}
```

Response:
```json
{
  "roomCode": "ABCD1234",
  "hostName": "Alice",
  "createdAt": "...",
  "participantsCount": 0
}
```

#### Get room
`GET /api/v1/rooms/:code`
- Returns basic room info and participants count.

---

### Transcripts

#### Create a transcript
`POST /api/v1/transcript`

Body (JSON):
```json
{
  "meetingCode": "ABCD1234",
  "transcriptText": "Hello everyone...",
  "fileName": "meeting-2025-09-01.txt",
  "metadata": { "language": "en" }
}
```
Creates/updates a transcript record for the meeting code.

#### List transcripts
`GET /api/v1/transcript?meeting_code=ABCD1234&limit=20`
- Returns transcripts (most recent first).

#### Get/download transcript
`GET /api/v1/transcript/:id`
- Download or fetch the transcript by DB id (or by meeting code depending on implementation).

---

### Emotion service uploads

Endpoint: `POST /api/v1/emotion`

- Uses `multer` to accept multipart `file` uploads. Required fields:
  - `meeting_id` (or `meetingId`)
  - `participant_id` (or `participantId`)
  - `type` (optional: `frame|audio|video`; default: `frame`)
  - `file` (multipart file)

Example (curl):
```bash
curl -X POST "http://localhost:8000/api/v1/emotion"   -F "meeting_id=ABCD1234"   -F "participant_id=participant1"   -F "type=audio"   -F "file=@/path/to/audio.webm"
```

This endpoint forwards the uploaded file (or buffer) to the configured `EMOTION_SERVICE_URL` (defaults to `http://localhost:5002/analyze`) and returns the response from that service.

---

### Meetings (higher-level)

Routes under `/api/v1/meetings` provide meeting listing and upsert behavior (protected for writes). Example exports include:
- `GET /api/v1/meetings` (optionally authenticated)
- `POST /api/v1/meetings` (JWT protected) — create/update meeting metadata
- Participant management endpoints: `POST /api/v1/meetings/:code/participants`, `POST /api/v1/meetings/add_participant` (JWT protected)

Refer to the controllers for exact request body fields, but typical fields include: `meetingCode`, `hostName`, `hostInfo`, `participants`, and metadata.

---

## Socket.IO events (client ↔ server)

The server initializes a Socket.IO server and exposes the following common events for real-time interactions:

### From client → server (`socket.emit`)

- `join-call` — join a meeting room  
  payload: `meetingCode`, optional metadata `{ name, userId, muted, video, screen, ... }`

- `leave-call` — leave the meeting (server will mark participant left)

- `signal` — WebRTC signaling messages (server relays to other participants)

- `chat-message` — send a chat message to the meeting

- `update-meta` — update participant metadata (name, muted, video, etc.)

- `emotion.frame`, `emotion.chunk`, `emotion.chunk.complete`, `emotion.chunk.abort` — emotion upload related events (server supports partial uploads & forwards to emotion service)

- `transcription-update` — send incremental transcript updates to the room

- `keywords-update` — update or share keywords detected

### From server → client (`socket.on` client side to listen)

- `existing-participants` — list of participants already in room (after join)
- `assigned-role` — server assigns roles/permissions
- `signal` — relayed signaling messages
- `chat-history` / `chat-ack` — chat acknowledgements & history
- `emotion-update` / `emotion.update` / `emotion.ack` — emotion analysis updates
- `transcription-update` — broadcast transcription segments
- `error` — server-side errors

> Implementation detail: the server keeps per-meeting in-memory state maps and attempts to persist/reflect participants in the Meeting model. Reconnects and polite role semantics are handled in `src/controllers/socketManager.js`.

---

## Project structure (important files)

```
backend/
├─ package.json
├─ .env
├─ src/
│  ├─ app.js                  # Express app bootstrap & server start
│  ├─ config/
│  │  └─ passport.js          # JWT passport strategy
│  ├─ controllers/
│  │  ├─ user.controller.js
│  │  ├─ transcript.controller.js
│  │  ├─ emotion.controller.js
│  │  └─ socketManager.js
│  ├─ routes/
│  │  ├─ users.routes.js
│  │  ├─ rooms.js
│  │  ├─ transcripts.js
│  │  └─ emotion.routes.js
│  ├─ models/
│  │  ├─ user.model.js
│  │  ├─ meeting.model.js
│  │  └─ transcript.model.js
│  └─ utils/ ...
```

---

## Notes / tips

- **JWT_SECRET**: set this strong in production — login issues happen when JWT secret is missing.
- **MONGO_URI**: ensure Mongo connection string includes credentials and network access for your environment.
- **Emotion service**: the backend expects a separate emotion analysis service at `EMOTION_SERVICE_URL`. If you don't have that, set up a mock or change the value to a local stub.
- **File uploads**: emotion endpoints write temporary files to `EMOTION_UPLOAD_TMP_DIR` — ensure the server has write permission to the directory.
- **CORS**: `CLIENT_ORIGIN` is honored; change for production and restrict origins.
- **Scaling**: currently Socket.IO and meeting state are in-memory — for horizontal scaling use a shared adapter (Redis) or persist participant state externally.

---

## Contributing

- Follow the code layout in `src/`.
- Keep controller functions small and testable.
- Add route tests if you add critical behavior (no test suite included here by default).

---

## License

This project uses the license specified in `package.json`: **ISC**.

---
