# SkymeetAI Backend Documentation

## Introduction

MeetSync is a backend service for a real-time meeting and collaboration platform. It supports features such as user authentication, meeting creation and management, real-time chat and participant tracking via WebSockets, transcript storage, and integration with an external emotion analysis service. The system is built using Node.js, Express.js for HTTP APIs, Socket.io for real-time communication, Mongoose for MongoDB interactions, and various libraries for security and utilities.

Key features:
- User registration and login with JWT authentication.
- Meeting rooms with participant tracking, chat history, and analytics (e.g., emotion scores, keywords).
- Real-time events for joining/leaving calls, chat, screen sharing, and emotion updates.
- Transcript management for meetings.
- Integration with an external emotion analysis service (via HTTP POST to a configurable URL).
- File uploads for audio/video frames with partial upload support and cleanup.

## Architecture Overview

### High-Level Architecture
MeetSync follows a monolithic backend architecture with real-time extensions. The core components are:

1. **Web Server (Express.js)**: Handles HTTP requests for APIs (e.g., user auth, meeting history, transcripts). It uses middleware like CORS, Passport.js for authentication, and error handlers.
   
2. **Real-Time Layer (Socket.io)**: Attached to the HTTP server for WebSocket-based communication. Manages meeting rooms, participant states, chat, and emotion/frame uploads.

3. **Database (MongoDB via Mongoose)**: Stores users, meetings (with participants, chat, analytics), and transcripts. Schemas are defined with methods for common operations (e.g., adding participants).

4. **External Services**:
   - **Emotion Analysis Service**: An external API (default: `http://localhost:5002/analyze`) that processes audio/video/frame data via multipart/form-data POST requests. Results are stored in meeting analytics and emitted to hosts.
   
5. **File System**: Temporary storage for partial uploads (e.g., audio chunks) in `os.tmpdir()/meet_uploads`. Cleaned up via intervals.

6. **Security**:
   - JWT for API auth.
   - Password hashing with bcrypt.
   - Input sanitization (sanitize-html).
   - CORS restrictions via env vars.
   - Limits on request sizes and upload TTLs.

### Data Flow
- **User Authentication**: Client sends username/password → Server validates → Issues JWT.
- **Meeting Join**: Client connects via Socket.io → Emits "join-call" with meeting code → Server validates, adds participant to DB and in-memory state → Broadcasts updates.
- **Real-Time Interactions**: Chat messages, screen shares, emotion frames are handled via socket events → Updated in DB (e.g., chat history) and broadcasted.
- **Emotion Analysis**: Client uploads frame/audio → Socket.io handles (with partial support for large files) → Forwards to external service → Stores results in meeting analytics → Emits to host.
- **Transcripts**: POST to API → Upserted in DB by meeting code.
- **History/Analytics**: Authenticated API calls fetch user-specific or all meetings.

### In-Memory State
- `meetingState`: Array of socket IDs per meeting (for roles like "polite").
- `meetingParticipants`: Map of userId → {socketId, meta} per meeting.
- `PARTIAL_UPLOADS`: Map for tracking chunked uploads with TTL cleanup.

### Scalability Notes
- Stateful (in-memory meeting state), so not horizontally scalable without shared state (e.g., Redis).
- MongoDB for persistence; cleanup job runs every hour for old inactive meetings.
- Uploads are disk-based temporaries; configure limits via env vars.

### Dependencies
- Core: express, socket.io, mongoose, axios, form-data.
- Security: bcrypt, jwt, passport, cors, sanitize-html.
- Utils: dotenv, crypto, fs/promises.

## Components

### Server Setup (app.js)
- Initializes Express app and HTTP server.
- Loads env vars via dotenv.
- Connects to MongoDB.
- Sets up middleware: CORS, JSON/URL-encoded parsers (with limits), Passport.
- Mounts routes: /api/v1/users, /rooms, /transcript, /emotion, /meetings.
- Error handling: Global JSON error responder.
- Socket.io initialization via `connectToSocket`.
- Dev helper: Logs registered routes on startup.

### Real-Time Communication (socketManager.js)
- Creates Socket.io server with configurable CORS, buffers, pings.
- Manages temporary upload dirs with cleanup intervals.
- Exposes globals: `io`, `meetingState`, `meetingParticipants`.
- Socket Events:
  - "connection": Logs client connect.
  - "join-call": Validates code, adds/restores participant, joins room, emits existing participants and role.
  - "existing-participants": Broadcasts current participants.
  - "assigned-role": Assigns "polite" role based on join order.
  - "chat": Sanitizes and broadcasts message, saves to DB.
  - "keywords-update": Updates/saves keywords in analytics, broadcasts.
  - "start-share", "stop-share": Broadcasts screen share events.
  - "signal": Relays WebRTC signals.
  - "upload-chunk": Handles partial file uploads (e.g., audio), assembles on complete.
  - "emotion.frame": Processes frame/audio for emotion service (async), emits ACK and updates to host.
  - "emotion-update": Updates analytics, emits to host or room.
  - "leave-call"/"disconnect": Handles leave, updates DB, broadcasts.

### Models

#### User Model (user.model.js)
- Schema: name (required), username (required, unique), password (required), token.
- Model: "UserDb".

#### Meeting Model (meeting.model.js)
- Sub-Schemas:
  - Participant: socketId (required), userId (Mixed), name, meta (Object), joinedAt, leftAt.
  - Chat: id, userId, fromSocketId, name, text (maxlength 2000), meta, ts.
  - Analytics: transcription (String), emotionScores (Object), keywords (Array).
- Main Schema: meetingCode (unique, uppercase), host (ref User), participants (Array), chat (Array), analytics, active (Boolean), lastActivityAt, timestamps.
- Methods:
  - addParticipant: Upserts by socketId/userId, normalizes userId.
  - updateParticipantMeta: Merges meta.
  - restoreParticipant: Rejoins left participant by userId/name.
  - removeParticipant: Pulls by socketId, deactivates if empty.
  - markParticipantLeft: Sets leftAt, deactivates if all left.
  - addChatMessage: Pushes message, limits to 500.
  - updateAnalytics: Merges data.
  - Statics: upsertByMeetingCode, cleanupOldMeetings (interval job).
- Indexes: meetingCode.

#### Transcript Model (transcript.model.js)
- Schema: meetingCode (required, unique), transcriptText, fileName, metadata (Mixed), createdAt/updatedAt.
- Pre-save: Updates updatedAt.
- Model: "Transcript".

### Controllers

#### User Controller (user.controller.js)
- login: Validates credentials, issues JWT.
- register: Hashes password, creates user.
- getUserHistory: Fetches meetings where user is host/participant.
- addToHistory: Creates meeting if not exists, adds as participant.
- addParticipant: Adds/updates participant in meeting.
- getMeetings: Lists active or user-related meetings.
- upsertMeeting: Upserts meeting by code with payload.

#### Emotion Controller (emotion.controller.js)
- sendToEmotionService: Builds FormData, posts to external service (with retry on transients), supports Buffer/path/stream.
- uploadEmotionFileHandler: Multer handler, forwards to service, emits results to host socket, cleans up temp file.

#### Transcript Controller (transcript.controller.js)
- createTranscript/saveTranscript: Upserts by meetingCode.
- getTranscript/getTranscriptByCode: Fetches by _id or code.
- listTranscripts: Lists recent, optional filter by code.

## API Endpoints
Mounted under /api/v1/ (from app.js routes imports; assume standard CRUD from controllers).

- **Users** (/users):
  - POST /login
  - POST /register
  - GET /get_all_activity (auth)
  - POST /add_to_activity (auth)

- **Rooms** (/rooms): Assumed meeting-related (not detailed in provided code).

- **Transcripts** (/transcript):
  - POST / (create)
  - GET /:id (get by id/code)
  - GET / (list)

- **Emotions** (/emotion): Assumed upload handler (e.g., POST /upload).

- **Meetings** (/meetings):
  - POST / (upsert)
  - GET / (list)
  - POST /:code/participants (add)

All APIs return JSON {success, message/data/error}.

## Socket Events
See socketManager.js for details. Key ones: join-call, chat, signal, upload-chunk, emotion.frame, leave-call, etc. Emits include error, existing-participants, user-joined/left, message, emotion.update.

## Environment Variables
- CLIENT_ORIGIN: CORS origin (default: http://localhost:3000)
- REQUEST_JSON_LIMIT/REQUEST_URLENCODED_LIMIT: Body limits (default: 4mb)
- PORT: Server port (default: 8000)
- MONGO_URI: MongoDB connection string
- JWT_SECRET: For token signing
- EMOTION_SERVICE_URL: External service (default: http://localhost:5002/analyze)
- PARTIAL_UPLOAD_MAX_BYTES: Upload size (default: 200MB)
- PARTIAL_UPLOAD_TTL_MS: TTL (default: 10min)
- SOCKET_MAX_HTTP_BUFFER: Socket buffer (default: 100MB)
- SOCKET_PING_INTERVAL/TIMEOUT: Ping settings

## Setup and Deployment
1. Install dependencies: `npm install`.
2. Set env vars in .env file.
3. Run: `node src/app.js` (or via PM2/Nodemon).
4. MongoDB: Ensure running; auto-connects.
5. Emotion Service: Deploy separately; configure URL.
6. Production: Restrict CORS, use HTTPS, monitor uploads/DB size.
7. Testing: Use Postman for APIs, Socket.io client for events.
