# Contributing to Hoovik

If Hoovik has been useful to you, consider giving it a ⭐ on [GitHub](https://github.com/AnupamKumar-1/Hoovik) — it helps the project grow!

Hoovik is a distributed real-time communication platform composed of four independent services across Node.js and Python ecosystems. Before contributing, please read the setup and documentation for the subsystem you plan to modify, and keep changes scoped where possible.

---

## Table of Contents

- [Contributing to Hoovik](#contributing-to-hoovik)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
  - [Local Setup](#local-setup)
    - [1. Clone the repository](#1-clone-the-repository)
    - [2. MongoDB and Redis](#2-mongodb-and-redis)
    - [3. Backend](#3-backend)
    - [4. Emotion Service](#4-emotion-service)
    - [5. Transcript Service](#5-transcript-service)
    - [6. Frontend](#6-frontend)
  - [Dataset (Emotion Service only)](#dataset-emotion-service-only)
  - [Starting all services](#starting-all-services)
  - [Verifying your setup](#verifying-your-setup)
  - [Load Testing (Emotion Service)](#load-testing-emotion-service)
  - [Contribution guidelines](#contribution-guidelines)
  - [PR checklist](#pr-checklist)

---

## Prerequisites

| Tool | Minimum version | Notes |
|---|---|---|
| Node.js | 20.x | Backend and frontend |
| npm | 9+ | Comes with Node |
| Python | 3.12.x (emotion service), 3.13.x (transcript service) | Version mismatch may cause dependency issues |
| pip | 23+ | |
| MongoDB | 6.x | Local or Atlas |
| Redis | 7+ | Local instance |
| ffmpeg | any recent | Required by transcript service; must be in `PATH`. Install: `brew install ffmpeg` (macOS) / `sudo apt install ffmpeg` (Ubuntu) |
| pm2 | 5+ | `npm install -g pm2` — for multi-process backend |

---

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/AnupamKumar-1/Hoovik.git
cd Hoovik
```

---

### 2. MongoDB and Redis

Both must be running before any other service starts. The backend exits immediately if either is unreachable.

**Install Redis:**

macOS:
```bash
brew install redis
```

Ubuntu:
```bash
sudo apt install redis-server
```

Windows — Use [Upstash](https://upstash.com) (recommended) — free tier, no install required:

1. Create a free account at [upstash.com](https://upstash.com)
2. Create a new Redis database
3. In the database dashboard, select **TCP** (not REST) and copy the connection URL
4. Add it to `backend/.env`:

```dotenv
REDIS_URL=rediss://<your-upstash-url>
```

> TLS is enabled automatically when the URL starts with `rediss://` — no code changes needed.

**Start both services (macOS / Linux):**

```bash
mongod        # local MongoDB
redis-server  # local Redis (default port 6379)
```

**Verify Redis is running** (macOS / Linux, should return `PONG`):

```bash
redis-cli ping
```

---

### 3. Backend

```bash
cd backend
npm install
cp .env.example .env
```

Edit `.env` and fill in the required values:

```dotenv
# Required
JWT_SECRET=<generate a 64-char random hex string — see below>
MONGO_URI=mongodb://localhost:27017/hoovik
REDIS_URL=redis://localhost:6379

# Service URLs
Ts_SERVICE_URL=http://localhost:5001/process_meeting

# CORS
CLIENT_ORIGIN=http://localhost:3000

NODE_ENV=development

# Redis lock
REDIS_LOCK_TTL_MS=15000
REDIS_LOCK_MAX_WAIT_MS=5000

# Transcript
TRANSCRIPT_MAX_TEXT_LENGTH=500000
TRANSCRIPT_CACHE_TTL_SEC=300
TRANSCRIPT_RATE_LIMIT_MAX=30
TRANSCRIPT_RATE_LIMIT_WIN_SEC=60

# AI Summary (Groq)
GROQ_API_KEY=<your-groq-api-key>
AI_SUMMARY_RATE_LIMIT_MAX=2
AI_SUMMARY_RATE_LIMIT_WIN_SEC=7200

# RAG pipeline (Nomic embeddings + Groq LLM)
NOMIC_API_KEY=<your-nomic-api-key>

# Cache TTLs
HISTORY_CACHE_TTL_SEC=120
MEETINGS_CACHE_TTL_SEC=60
USER_CACHE_TTL_SEC=300

# Rate limits
LOGIN_RATE_MAX=8
LOGIN_RATE_WIN_SEC=60
REGISTER_RATE_MAX=4
REGISTER_RATE_WIN_SEC=60

# Validation
MAX_NAME_LEN=100
MAX_USERNAME_LEN=50
MAX_MEETINGCODE_LEN=32
# cloudinary
CLOUD_NAME=
CLOUD_API_KEY=
CLOUD_SECRET=
```

**Generate a JWT secret:**

macOS / Linux:
```bash
openssl rand -hex 32
```

Windows (PowerShell):
```powershell
node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"
```

Paste the output as your `JWT_SECRET` value.

**Run (single process for development):**

```bash
npm run dev        # nodemon, auto-restarts on changes
```

**Run (multi-process with pm2):**

Install pm2 globally if not already installed:

```bash
npm install -g pm2
```

Start all three backend instances:

```bash
npm run prod       # pm2 start ecosystem.config.cjs
```

This starts three processes on ports 8000, 8001, and 8002 as defined in `ecosystem.config.cjs`:

| Name | Port | Memory limit |
|---|---|---|
| `hoovik-backend-8000` | 8000 | 512 MiB |
| `hoovik-backend-8001` | 8001 | 512 MiB |
| `hoovik-backend-8002` | 8002 | 512 MiB |

Each process reads `.env` via `env_file` and restarts automatically with exponential backoff on failure.

Useful pm2 commands:

```bash
pm2 list                        # check status of all processes
pm2 logs                        # stream logs from all processes
pm2 restart ecosystem.config.cjs   # restart all three instances
pm2 delete all                  # stop and remove all processes
```

> For local development a single process (`npm run dev`) is sufficient. pm2 is only needed for production or multi-process testing.

---

### 4. Emotion Service

```bash
cd emotion_service
python3.12 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`:

```dotenv
EMOTION_SERVER_URL=http://localhost:5002
```

> The emotion service is primarily configured via `config/config.json` (model paths, EMA alpha, sequence length). Environment variables are supplementary.

**Model files required before starting:**

The following files must be present before the server will start. The server refuses to start if any model fails to load.

```
emotion_service/
├── embeddings/
│   └── face_landmarker.task
└── models/
    ├── anomaly/
    │   ├── iso_audio_only.joblib
    │   ├── iso_both.joblib
    │   ├── iso_global_fallback.joblib
    │   ├── iso_video_only.joblib
    │   └── meta.json
    ├── ensemble/
    │   └── weights.json
    ├── modal/
    │   ├── best_modal.pt
    │   ├── best_modal_a.pt
    │   ├── best_modal_b.pt
    │   └── temperature.pt
    └── xgb/
        ├── col_medians.npy
        ├── pca.joblib
        └── xgb_model.joblib
```

`extracted_dataset/` is only required if you are retraining the model. It is not needed to run the inference server.

**Run:**

```bash
uvicorn app:app --host 0.0.0.0 --port 5002 --reload
```

---

### 5. Transcript Service

```bash
cd transcript_service
python3.13 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`:

```dotenv
HF_ASR_MODEL=openai/whisper-small
HF_EMOTION_MODEL=j-hartmann/emotion-english-distilroberta-base
CLEANUP_DELAY_SEC=120
NODE_API=http://localhost:8000/api/v1/transcripts
```

> Whisper and DistilRoBERTa are downloaded from HuggingFace on first run if not already cached locally. This may take a few minutes.

**Run:**

```bash
uvicorn app:app --host 0.0.0.0 --port 5001 --reload
```

> Do not use `python app.py` — invoke via `uvicorn app:app` directly. The `--reload` flag is used in development; omit it in production.

---

### 6. Frontend

```bash
cd frontend
npm install
cp .env.example .env
```

Edit `.env`:

```dotenv
VITE_SERVER_URL=http://localhost:8000
VITE_API_URL=http://localhost:8000/api/v1
VITE_SOCKET_URL=http://localhost:8000
VITE_SIGNALING_URL=http://localhost:8000
VITE_AI_URL=http://localhost:8000
VITE_EMOTION_SOCKET_URL=http://localhost:5002
VITE_TRANSCRIPT_URL=http://localhost:8000/api/v1/transcripts/proxy
VITE_TURN_URL_UDP=turn:in.relay.metered.ca:3478?transport=udp
VITE_TURN_URL_80=turn:in.relay.metered.ca:80
VITE_TURN_URL_443=turn:in.relay.metered.ca:443
VITE_TURN_URL_443_TCP=turn:in.relay.metered.ca:443?transport=tcp
VITE_TURN_URL_TLS=turns:in.relay.metered.ca:443
VITE_TURN_USERNAME=openrelayproject
VITE_TURN_CREDENTIAL=openrelayproject
VITE_SUPPORTS_GLOBAL_MEETINGS=true
VITE_NOISE_GATE_RMS=0.008
VITE_NOISE_GATE_HOLD_MS=1500
VITE_NOISE_GATE_SMOOTHING=0.8
VITE_SPEECH_MIN_ACTIVE_MS=800
```

**Run:**

```bash
npm start       # development server on localhost:3000
npm run build   # production build
```

---

## Dataset (Emotion Service only)

The `EmotionTransformer` and XGBoost ensemble require a pre-built dataset to train.

**Download:** [dataset.npz — Google Drive](https://drive.google.com/file/d/135wYH7DB8_10Jc8g08MfC6Poews_Lkgp/view?usp=sharing)

**Placement:**

```
emotion_service/
└── extracted_dataset/
    ├── dataset.npz
    ├── norm_stats.npz
    └── splits.json
```

See [docs/realTimeEmotionService.md](docs/realTimeEmotionService.md) for the full training procedure. If you are not modifying the model, you only need the pre-trained files under `models/` — not the dataset.

---

## Starting all services

Once all `.env` files are configured and MongoDB + Redis are running, start everything from the repository root:

```bash
chmod +x dev.sh   # one-time
./dev.sh
```

This starts all four services in parallel with colour-coded prefixed output:

| Prefix | Service | Port |
|---|---|---|
| `FRONTEND` | React SPA | 3000 |
| `BACKEND` | Node.js / Express | 8000 |
| `EMOTION` | FastAPI emotion inference | 5002 |
| `TRANSCRIPT` | FastAPI transcription | 5001 |

Ctrl+C cleanly shuts down all four services at once.

> Python virtual environments must already be set up at `emotion_service/venv` and `transcript_service/venv` before running this command — the script invokes them directly via `./emotion_service/venv/bin/python` and `./transcript_service/venv/bin/python`.

**Windows users:** `dev.sh` is a bash script and does not run in Command Prompt or PowerShell. Choose one of the following approaches:

- **WSL2 (recommended):** Run the entire project inside WSL2 — `./dev.sh` works as-is in the WSL2 terminal.
- **Git Bash:** Open Git Bash from the repo root and run `./dev.sh` directly. Git Bash ships with most Git for Windows installations.
- **Manual start (no bash):** Open four separate terminals from the repo root and start each service individually:

```powershell
# Terminal 1 — Frontend
cd frontend; npm start

# Terminal 2 — Backend
cd backend; npm run dev

# Terminal 3 — Emotion Service
.\emotion_service\venv\Scripts\python.exe -m uvicorn app:app --app-dir emotion_service --host 0.0.0.0 --port 5002 --reload

# Terminal 4 — Transcript Service
.\transcript_service\venv\Scripts\python.exe -m uvicorn app:app --app-dir transcript_service --host 0.0.0.0 --port 5001 --reload
```

**Start order:**

```
1. MongoDB + Redis  ← start manually before anything else
2. ./dev.sh         ← starts Backend, Emotion Service, Transcript Service, and Frontend together
```

---

## Verifying your setup

Once all services are running:

```bash
curl http://localhost:8000/api/v1/rooms/TEST  # → 401 Unauthorized (backend is up)
curl http://localhost:5002/health             # → {"status": "ok"}
curl http://localhost:5002/ready             # → {"status": "ready"} (only after models load)
curl http://localhost:5002/stats/json        # → latency snapshot
```

Open the observability dashboard in your browser:

http://localhost:5002/stats

Open the frontend in your browser:

http://localhost:3000

---

## Load Testing (Emotion Service)

The `load_testing/` directory contains a Locust WebSocket stress test for the emotion service.

**Install Locust** (inside the emotion service venv or globally):

```bash
pip install locust
```

**Add participant face images:**

Place at least one `.jpg` image inside `load_testing/src/`. These are used as fake video frames during the test. The script will raise an error if the folder is empty.

```
load_testing/
└── src/
    ├── participant1.jpg
    └── participant2.jpg
```

**Run the emotion service first**, then start Locust:

```bash
# from the repo root
EMOTION_SERVER_URL=http://localhost:5002 locust -f load_testing/locustfile.py
```

Open the Locust dashboard at `http://localhost:8089`, set the number of users and spawn rate, and start the test.

**What it tests:**

| Task | Weight | Description |
|---|---|---|
| `send_audio` | 5 | Emits a fake PCM audio chunk per cycle |
| `send_frame` | 3 | Emits a random JPEG frame per cycle |
| `toggle_mic` | 1 | Toggles mic state via `participant.media_state` |
| `toggle_camera` | 1 | Toggles camera state via `participant.media_state` |
| `random_pause` | 1 | Simulates network jitter |

Inference latency per participant is tracked via `requestId` round-trip and reported in the Locust UI under `emotion_inference`.

---

## Contribution guidelines

- **Read the subsystem README first.** Each service has detailed implementation docs under `docs/`. Read the relevant one before writing code.
- **Keep changes scoped to a single subsystem** where possible. Cross-service changes require updating both the implementation and the relevant doc.
- **If you change a Socket.IO event name, payload shape, or HTTP contract**, update `docs/` to match.
- **Open an issue before tackling large changes** — especially anything touching the emotion service inference pipeline, distributed state, or cross-service contracts.
- **Coding style:** Node.js backend uses ES Modules (`"type": "module"`); Python services follow PEP 8.

---

## PR checklist

- [ ] Changes are confined to the intended subsystem
- [ ] Relevant subsystem README updated if behaviour changed
- [ ] `.env.example` updated if a new env variable was added
- [ ] `docs/` updated if an API contract or event shape changed
- [ ] Screenshot or `curl` output included if the change affects an observable endpoint (e.g. `/stats`, `/health`)