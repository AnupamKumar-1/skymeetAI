# SkymeetAI

## Project Overview

SkymeetAI is an intelligent, all-in-one platform for real-time meetings and collaboration. It seamlessly integrates video conferencing, live chat, participant tracking, automatic speech recognition (ASR), emotion analysis, and anomaly detection into a single powerful system.
Built on WebRTC for smooth multi-user video calls and Socket.io for real-time signaling, SkymeetAI ensures effortless connectivity and communication. Every conversation is enriched with AI-generated transcripts and deep participant insights, powered by advanced text-based and multimodal (audio-visual) models.
With its ability to not only connect people but also understand them—through emotion recognition and intelligent anomaly detection—SkymeetAI transforms meetings into smarter, more human-centered experiences.

The platform is composed of four main components:
- **Frontend**: A React-based single-page application (SPA) for the user interface, handling video meetings, authentication, and interactions.
- **Backend**: A Node.js service using Express.js for APIs and Socket.io for real-time communication, managing user authentication, meetings, transcripts, and integration with AI services.
- **Transcription Service**: A Flask-based web service for processing meeting audio, performing ASR with Whisper, classifying text-based emotions, and generating transcripts.
- **Emotion Service**: A Flask-based multimodal emotion recognition system using deep learning for audio (mel spectrograms) and visual (facial) inputs, with anomaly detection via Isolation Forest.

Key features across the system:
- Real-time video/audio streaming, screen sharing, and chat.
- User authentication with JWT.
- Meeting creation, history, and analytics (e.g., emotion scores, keywords).
- ASR and transcript generation in text/JSON formats.
- Emotion classification (e.g., anger, joy, neutral) and anomaly detection for affective computing.
- Integration between services for seamless data flow (e.g., backend forwards audio to emotion/transcription services).

The system is designed for applications like virtual meetings, sentiment monitoring, or collaborative tools, with a focus on scalability, security, and ethical AI use.

## Datasets (For Emotion Service)

The Emotion Service uses publicly available datasets:
- **CREMA-D**: Audio recordings of emotions (anger, disgust, fear, happy, neutral, sad). Source: [Kaggle - CREMA-D](https://www.kaggle.com/datasets/ejlok1/cremad). License: CC BY 4.0.
- **AffectNet Aligned**: Facial images labeled with emotions. Source: [Kaggle - AffectNet Aligned](https://www.kaggle.com/datasets/yakhyokhuja/affectnetaligned). License: Research/non-commercial use.

**Note**: Download datasets manually and place in the appropriate directories (e.g., `data/audio/`, `data/images/`). Ensure ethical use, respecting privacy and avoiding biased applications.

## Architecture Overview

SkymeetAI follows a microservices architecture with real-time extensions:

### High-Level Components
1. **Frontend (React)**: Handles UI, WebRTC media, and client-side logic.
2. **Backend (Node.js/Express/Socket.io)**: Manages APIs, real-time events, database (MongoDB), and integrations with AI services.
3. **Transcription Service (Flask)**: Processes audio for ASR and text-based emotion classification.
4. **Emotion Service (Flask)**: Analyzes audio/video for multimodal emotions and anomalies.

### Data Flow
- **User Interaction**: Frontend connects to Backend via APIs and Socket.io.
- **Meeting Setup**: Backend creates rooms, tracks participants, and handles real-time events (e.g., join, chat, signals).
- **Media Processing**: Frontend captures streams; Backend forwards audio/video to Transcription/Emotion Services.
- **AI Analysis**: Transcription Service generates transcripts with text emotions; Emotion Service processes frames for multimodal insights. Results are stored in Backend DB and emitted via Socket.io.
- **Persistence**: MongoDB stores users, meetings, chats, analytics, and transcripts.

### Diagram (Text-Based Representation)
```
+-------------+
|   Frontend  | <--> Socket.io / APIs <--> +-------------+
| (React App) |                              |   Backend   |
+-------------+                              | (Node.js)   |
                                             +-------------+
                                                   |
                                                   v
+-------------+                              +-------------+
| Transcription| <--- Audio Uploads/Requests |   MongoDB   |
| Service     |                              | (Database)  |
| (Flask)     |                              +-------------+
+-------------+                                      |
                                                   |
+-------------+                                      v
|   Emotion   | <--- Frame/Audio Uploads ----> +-------------+
| Service     |                                | Analytics & |
| (Flask)     |                                | Transcripts |
+-------------+                                +-------------+
```

### Scalability Notes
- Backend: Stateful (in-memory meeting state); use Redis for horizontal scaling.
- Services: Models loaded at startup; GPU recommended for AI inference.
- Potential Bottlenecks: Whisper transcription, FFmpeg conversions, WebRTC bandwidth.

## Installation and Setup

### Prerequisites
- Python 3.8+ (for Flask services).
- Node.js 18+ and npm (for Backend and Frontend).
- MongoDB (for Backend).
- FFmpeg (system-level, for audio/video processing in services).
- GPU (optional, for faster AI inference via CUDA/Torch).
- Git (for cloning the repo).

### Steps
1. Clone the repository:
   ```
   git clone <repo-url>
   cd skymeetai
   ```

2. Install dependencies for each component (detailed below).

3. Download datasets for Emotion Service (if using).

4. Set up environment variables in `.env` files (see each component's section).

### Component-Specific Installation

#### Frontend (React)
- Directory: `frontend/`
- Install: `npm install`
- Environment: Set `REACT_APP_API_URL`, `REACT_APP_SIGNALING_URL`, `REACT_APP_TRANSCRIPT_URL`, `REACT_APP_EMOTION_URL` in `.env`.

#### Backend (Node.js)
- Directory: `backend/`
- Install: `npm install`
- Environment Variables (in `.env`):
  - `CLIENT_ORIGIN`: CORS origin (e.g., `http://localhost:3000`).
  - `PORT`: Server port (default: 8000).
  - `MONGO_URI`: MongoDB connection string.
  - `JWT_SECRET`: For token signing.
  - `EMOTION_SERVICE_URL`: Emotion Service endpoint (default: `http://localhost:5002/analyze`).
  - `PARTIAL_UPLOAD_MAX_BYTES`: Upload size limit (default: 200MB).
- Connect to MongoDB.

#### Transcription Service (Flask)
- Directory: `transcription_service/`
- Install: `pip install -r requirements.txt` (includes Flask, Whisper from GitHub, transformers, torch, etc.).
- Folders: Ensure `uploads/` and `outputs/` exist (auto-created on startup).
- Configuration: Modify globals in `app.py` (e.g., `MIN_DURATION_SEC=0.30`, `CLEANUP_DELAY_SEC=120`).

#### Emotion Service (Flask)
- Directory: `emotion_service/`
- Install: `pip install -r requirements.txt` (includes Torch, librosa, MTCNN, Flask, etc.).
- Environment Variables (in `.env` or shell):
  - `FLASK_CORS_ORIGINS`: CORS origins (e.g., `http://localhost:3000`).
  - `BACKEND_URL`: Backend API for forwarding results.
  - `LOG_LEVEL`: Logging level (e.g., `DEBUG`).
- Preprocess data: Run scripts like `preprocess_images.py`, `preprocessing_audio.py`, etc.

## Running the Application

1. **Backend**: `node src/app.js` (or `pm2 start src/app.js` for production).
2. **Transcription Service**: Development: `python app.py` (runs on `http://0.0.0.0:5001`). Production: `gunicorn -w 4 app:app -b 0.0.0.0:5001`.
3. **Emotion Service**: Development: `FLASK_ENV=development python app.py` (runs on port 5002). Production: `gunicorn -w 4 app:app`.
4. **Frontend**: `npm start` (runs on `http://localhost:3000`).

Access the app at `http://localhost:3000`. Ensure services are running and URLs match environment configs.

For testing:
- Use curl/Postman for APIs.
- Socket.io client for real-time events.

## Key Pages and Usage (Frontend)

- **Landing (`/`)**: Entry page.
- **Authentication (`/auth`)**: Sign-in/register.
- **Home (`/home`)**: Create/join rooms, view transcripts.
- **History (`/history`)**: Past meetings.
- **Video Meeting (`/room/:roomId`)**: Core call interface.

Example Client Request (Transcription Service via curl):
```
curl -X POST http://localhost:5001/process_meeting \
  -F "audio_files=@speaker1.webm" \
  -F "speaker_map={\"speaker1\": \"Alice\"}"
```

API Endpoints (Backend):
- `/api/v1/users`: Login/register.
- `/api/v1/meetings`: List/upsert meetings.
- `/api/v1/transcript`: Manage transcripts.

Socket Events (Backend): `join-call`, `chat`, `signal`, `emotion.frame`, etc.

## API Endpoints (Emotion Service)
- POST `/analyze`: Analyze audio/video file.

## Components Documentation

### Frontend
- **Technology Stack**: React 18, react-router-dom, socket.io-client, axios, @mui/material.
- **Key Modules**:
  - `VideoMeet.jsx`: Manages WebRTC, signaling, chat, screen sharing, emotion analysis.
  - `AuthContext.jsx`: Handles auth, JWT, history.
  - `mediaController.js`: Abstracts media streams and track management.
  - `home.jsx`: Room creation/join, transcript display.
- **Scripts**: `npm start`, `npm build`.

### Backend
- **Models**: User, Meeting (with participants, chat, analytics), Transcript.
- **Controllers**: User (auth/history), Emotion (forwarding), Transcript (CRUD).
- **Socket Events**: Join/leave, chat, signals, uploads, emotions.

### Transcription Service
- **Endpoints**: POST `/process_meeting`, GET `/outputs/<filename>`.
- **Dependencies**: Flask, Whisper, transformers, FFmpeg.
- **Configuration**: Globals in `app.py`.

### Emotion Service
- **Preprocessing/Training Scripts**: `preprocess_*.py`, `extract_embeddings.py`, `train_multimodal.py`, `train_anomaly.py`.
- **Inference**: `predict.py` (CLI), `app.py` (API).
- **Models**: ResNet18 + MLP for fusion, Isolation Forest for anomalies.

## Troubleshooting
- **CORS Issues**: Verify origins in `.env` and service configs.
- **Media Access**: Ensure HTTPS/localhost and browser permissions.
- **Socket Failures**: Check URLs and backend logs.
- **AI Latency**: Use GPU; monitor FFmpeg/Torch.
- **Resource Leaks**: Ensure cleanup timers/jobs run.
- **Errors**: Check console/server logs; fallback to localStorage for history.

For detailed module docs, refer to sub-directories or original READMEs.

## License
This project is for educational/research purposes. Respect dataset licenses and ethical guidelines. No explicit license.