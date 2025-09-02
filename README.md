# SkymeetAI

---

SkymeetAI is a modular meeting platform designed to meet high-scale production requirements. It provides:

- Robust, low-latency WebRTC meetings powered by Socket.IO signalling.
- Host-side audio capture with deterministic chunking for reliable ASR ingestion.
- Optional per-clip emotion-analysis pipeline for ML-enabled UX features.
- An AI microservice integrating Whisper ASR + text-level emotion classification for fast, auditable transcripts.

---


## High-level architecture

```
+----------------+      +---------------------+      +----------------+
|   Frontend     | <--> |  Signalling Server   | <--> |  Media Broker   |
| (React / WebRTC)|      |   (Socket.IO w/     |      | (S3 / Object)   |
|                 |      | Redis adapter)      |      +----------------+
+----------------+      +---------------------+             |
       |  ^                      | |                         |
       |  |                      | |                         v
       |  +----> Rest API <-------+ +----------------->  Transcription Service (Whisper)
       |                       (JWT / REST)               Emotion Service (optional)
       |                                                  ML Pipeline / Batch
       v
  Observability
  (Tracing / Logs / Metrics)
```

**Notes:**
- The signalling server is horizontally scalable behind a Load Balancer and uses Redis Pub/Sub (Socket.IO adapter) for cross-node message routing.
- Media blobs are uploaded to an object store (S3 or equivalent) for durable processing; the backend forwards references to ML microservices.
- Whisper / AI services are independent microservices and should be pinned to specific model versions for reproducibility.

---

## Component responsibilities

**Frontend (React — `VideoMeet`)**
- Acquire local media (`getUserMedia`), render participants as `<video>` tiles.
- Manage one `RTCPeerConnection` per remote participant with explicit offer/answer lifecycle.
- Host-record per-participant audio streams using `MediaRecorder` with deterministic chunk sizes and monotonic timestamps.
- Provide three emotion upload fallbacks: socket-binary, socket-base64, or REST multipart.
- Authenticate via JWT, attach tokens to API calls.

**Signalling Server (Node.js + Socket.IO)**
- Authenticate socket connects via JWT policy.
- Relay `signal` events (SDP, ICE) between participants.
- Broadcast durable room events (`user-joined`, `user-left`, `transcription-update`, `emotion.update`).
- Accept partial emotion uploads, stream or persist to object store, forward to ML service.

**Backend REST API (Express)**
- User & room management (create rooms, list transcripts, persist meeting metadata).
- Transcript persistence endpoint for final transcripts and segment metadata.
- Auth endpoints (JWT) and meeting history APIs consumed by `AuthContext`.

**Transcription Service (ai_service)**
- Deterministic audio preprocessing: convert to mono, 16 kHz WAV via `ffmpeg`.
- Run Whisper (pinned model) to produce time-stamped segments.
- Run text-level emotion classifier per segment and produce two artifacts: human `.txt` and structured `.json`.
- Schedule ephemeral outputs for deletion and return stable download URLs.

**Emotion/ML Pipeline (emotion_service)**
- Preprocess audio/images → HDF5 → embeddings.
- Train multimodal classifiers & anomaly detectors.
- Provide an inference REST endpoint (`/analyze`) for short clip scoring. Model artifacts are versioned via git tags or artifact registry.

---

## API contracts (stable surface)

Design principles: keep payloads small, idempotent where possible, and explicit in time/range semantics.

### `POST /api/v1/rooms`  — create room
**Request**
```json
{ "hostName": "Alice" }
```
**Response**
```json
{ "roomCode": "ABCD1234", "roomId": "<uuid>", "createdAt": "2025-09-03T10:00:00Z" }
```

### `POST /process_meeting` (ai_service)
**Form (multipart)**: `audio_files[]`, `meeting_code`, `speaker_map` (JSON string)
**Response**
```json
{
  "success": true,
  "transcript_text": "...",
  "txt_filename": "MEETING_<uuid>.txt",
  "json_filename": "MEETING_<uuid>.json",
  "files_will_be_deleted_in_sec": 120
}
```
**Guarantee:** transient outputs are retained for the configured TTL; clients must download or the backend persist to storage if the transcript is required permanently.

### `POST /api/v1/emotion` — upload clip
**Form**: `meeting_id`, `participant_id`, `type` (`frame|audio|video`), `file` (multipart)
**Response**
```json
{ "success": true, "emotions": { "happy": 0.72, "neutral": 0.20 } }
```

---

## Signalling & WebRTC flow (deterministic)

1. Client requests local media and connects to Socket.IO with `Authorization: Bearer <JWT>`.
2. Client emits `join-call` with `{ meetingCode, displayName, isHost }`.
3. Server responds with `existing-participants` (array of participant metadata).
4. For each remote peer, clients create `RTCPeerConnection`, add local tracks, createOffer(), setLocalDescription(), and send SDP via `signal` event to the remote peer.
5. Remote side setRemoteDescription(), createAnswer(), and exchange ICE candidates via `signal` events.
6. On negotiationcomplete + stable ICE, media flows P2P (or via TURN when NATed).

**Operational check**: record and emit metrics for `time_to_connected` per peer (offer → connected) and `ice_restart_count` to detect flaky networks.

---

## Security, privacy & compliance

- **Authentication:** JWT for REST and sockets. Short lived tokens recommended (rotate with refresh tokens or an SSO provider).
- **Authorization:** Enforce room-level ACLs server-side. Hosts can mark rooms as private/public.
- **Transport security:** TLS everywhere (LB → services → clients). No mixed content.
- **Data at rest:** use SSE-KMS or server-side encryption for object store (S3) and encrypt DB secrets at rest.
- **PII minimization:** store only metadata and transcript URIs by default. Raw audio should be deleted when not needed or moved to customer-controlled buckets.
- **Compliance:** redact or ignore any compliance-sensitive features (e.g., call recording) unless explicit consent is captured.

---

## Scalability & resiliency strategy

**Signalling / Real-time**
- Socket.IO with Redis adapter for horizontal scaling.
- Autoscale signalling nodes based on open socket count and event rate.
- Use sticky sessions only if you don't have Redis adapter (not recommended).

**Media uploads & ML**
- Offload media uploads to S3 using presigned URLs from the backend to avoid node memory pressure.
- Process media asynchronously using workers (e.g., K8s Jobs / AWS Batch). Use message queue (RabbitMQ / SQS) for task dispatch.

**Data storage**
- MongoDB for user/meeting metadata (replica set + monitored backups).
- Object store (S3) for blobs.
- Use lifecycle rules on S3 for auto-expiry of ephemeral files.

**Availability plan**
- Multi-AZ deployment for DB and services.
- Deploy critical services behind health-checked LBs with readiness/liveness probes.

---

## Observability & SLOs

**Metrics (minimum)**
- Signalling request latency (95p & 99p)
- Socket connect success rate & authentications errors
- Media upload success rate and average duration
- ASR pipeline end-to-end latency (upload → transcript available)
- Error rate per service (4xx/5xx)

**Tracing & logs**
- Distributed tracing integration (OpenTelemetry). Include `trace_id` propagated across REST and socket events.
- Structured logs (JSON) with correlation IDs.

**SLOs (suggested)**
- Socket connect success rate: 99.9% per week
- Offer → connected latency: p95 < 2s under normal conditions
- ASR transcript return time: p95 < 30s for < 60s clips (depends on model)
- Media upload success rate: 99.5%

**Alerting**
- Page on sustained drop below SLOs or spike in 5xx errors.
- Alert on task queue length > threshold (indicates backlog to ML workers).

---

## Operational runbook (incidents & routine ops)

**P1 — Signalling down / clients cannot join**
1. Check LB health and signalling node instance health.
2. Inspect Redis adapter connectivity (or sticky session / LB misconfiguration).
3. Restart signalling nodes in a controlled rolling manner.
4. Post-mortem: check autoscaling and throttling metrics.

**P1 — ASR pipeline slow / backlog increasing**
1. Inspect queue length and worker pod CPU/GPU utilization.
2. Scale workers (increase replicas / add GPU instances if model uses GPU).
3. If urgent, return degraded response to frontend indicating transcript pending.

**Routine ops**
- Daily: check error budget and SLO burn rates.
- Weekly: rotate JWT signing keys if using self-managed secrets, or verify rotation for managed KMS.
- Monthly: run canary deployment for new Whisper model versions and validate latency & transcript quality.

---

## Deployment & CI/CD (recommended)

**CI**
- Lint, unit tests, and contract tests for all services.
- Integration test that spins a minimal stack via `docker-compose` (frontend, signalling, backend, ai_service mock).

**CD**
- Blue/green or canary deployments in k8s (Argo Rollouts or similar).
- Model deployments for emotion/ASR are separate artifacts; tag with semantic versions and model hash.

**K8s recommendations**
- Readiness/liveness probes for all containers.
- Resource requests/limits per container (based on profiled CPU/GPU usage).
- Horizontal Pod Autoscaler based on custom metrics (open socket count, queue length).

---

## Developer quickstart & local dev matrix

**Local matrix (fast path)**
1. `yarn` / `npm install` in `frontend/` → `npm start`.
2. `cd backend` → `npm install` → `npm run dev` (requires `MONGO_URI` env pointing to local Mongo).
3. `cd ai_service` → `pip install -r requirements.txt` → `python app.py` (requires ffmpeg installed locally).
4. Optional: `emotion_service` tools installed for ML work.

**Env examples**
- `REACT_APP_SIGNALING_URL=http://localhost:8000`
- `REACT_APP_TRANSCRIPT_URL=http://localhost:5001/process_meeting`
- `PORT=8000`, `MONGO_URI=mongodb://localhost:27017/skymeet`, `JWT_SECRET=devsecret`

**Acceptance tests**
- End-to-end smoke test that opens two headless browser clients, joins room, and verifies offer/answer exchange and transcript round-trip for a short recorded clip.

---

## Troubleshooting & FAQ (engineered checks)

**WebRTC: no video / black screen**
- Confirm `getUserMedia` succeeded (browser console), retry permissions.
- Check ICE candidate exchange logs; missing candidates often imply STUN/TURN misconfiguration.

**Users unable to join intermittently**
- Verify Redis adapter connectivity and open file/socket limits on signalling nodes.
- Check LB connection draining or health-check misconfiguration.

**Transcripts not generated**
- Inspect object store to confirm audio blob uploaded successfully.
- Check worker logs for ASR processing errors (model OOM, ffmpeg errors due to codecs).

---

## Appendix: CLI flags, service envs, and contacts

### Key CLI flags (selected)
- `preprocessing_audio.py --audio_dir --out_h5 --fixed_duration --workers`
- `train_multimodal.py --train_h5 --val_h5 --epochs --batch_size --lr`
- `train_anomaly.py --train_h5 --n_estimators --contamination`

### Must-set service envs (core)
- Frontend: `REACT_APP_SIGNALING_URL`, `REACT_APP_API_URL`, `REACT_APP_TRANSCRIPT_URL`, `REACT_APP_EMOTION_URL`
- Backend: `PORT`, `MONGO_URI`, `JWT_SECRET`, `CLIENT_ORIGIN`, `EMOTION_SERVICE_URL`
- AI Service: ensure `ffmpeg` on PATH; `CLEANUP_DELAY_SEC` configurable (default 120s)

---