# SkymeetAI - Frontend Documentation

This document provides a comprehensive overview of the frontend application for a video meeting platform. It details the key pages, technology stack, setup instructions, core components, and specific module documentation for `VideoMeet.jsx`, `AuthContext.jsx`, `mediaController.js`, and `home.jsx`. The application is built using React 18 and integrates WebRTC for peer-to-peer video/audio communication, Socket.io for real-time signaling, and Material-UI for the user interface.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Pages](#key-pages)
3. [Technology Stack & Dependencies](#technology-stack--dependencies)
4. [Important Scripts](#important-scripts)
5. [Environment & Runtime Configuration](#environment--runtime-configuration)
6. [Developer Quick Start](#developer-quick-start)
7. [Module Documentation](#module-documentation)
   - [VideoMeet.jsx](#videomeetjsx)
   - [AuthContext.jsx](#authcontextjsx)
   - [mediaController.js](#mediacontrollerjs)
   - [home.jsx](#homejsx)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The frontend application is a React-based single-page application (SPA) that provides a video conferencing platform with features like room creation, real-time video/audio streaming, chat, screen sharing, active speaker detection, and emotion analysis (host-only). It integrates with a backend API and Socket.io server for signaling and data persistence. The application is bootstrapped with Create React App and uses Material-UI for consistent, modern styling.

---

## Key Pages

- **Landing (`/`)**: A marketing-focused entry page with navigation to other app sections.
- **Authentication (`/auth`)**: Handles user sign-in and registration using Material-UI components.
- **Home (`/home`)**: Allows users to create/join rooms, copy room links, and view recent transcripts.
- **History (`/history`)**: Displays a list of past meetings for the authenticated user.
- **Video Meeting (`/room/:roomId`)**: The core video call interface, leveraging Socket.io for signaling and WebRTC for peer-to-peer media streaming.

Authentication and API interactions are managed centrally via `src/contexts/AuthContext.jsx`, which provides a context for user state and helper functions.

---

## Technology Stack & Dependencies

- **Core Framework**: React 18 (via Create React App)
- **Routing**: `react-router-dom`
- **Real-time Signaling**: `socket.io-client`
- **HTTP Client**: `axios`
- **UI Components**: `@mui/material`, `@mui/icons-material`
- **Animations**: `framer-motion`
- **HTTP Status Helper**: `http-status`

For a complete list, refer to `package.json`.

---

## Important Scripts

From `package.json`:

- **`start`**: `react-scripts start` - Launches the development server.
- **`build`**: `react-scripts build` - Creates a production-optimized build.
- **`test`**: `react-scripts test` - Runs tests.
- **`eject`**: `react-scripts eject` - Ejects from Create React App for full configuration control.

---

## Environment & Runtime Configuration

The application uses `src/environment.js` to define the `server` constant, defaulting to `http://localhost:8000` when `IS_PROD` is `false`. Key environment variables include:

- **`REACT_APP_SERVER_URL`**: Fallback server URL for some pages.
- **`REACT_APP_API_URL`**: Base URL for API calls (e.g., rooms, meetings).
- **`REACT_APP_SIGNALING_URL`**: Socket.io server URL (default: `http://localhost:8000`).
- **`REACT_APP_TRANSCRIPT_URL` or `REACT_APP_AI_URL`**: Transcript service (default: `http://localhost:5001/process_meeting`).
- **`REACT_APP_EMOTION_URL`**: Emotion analysis service (default: `http://localhost:5002/analyze`).
- **`REACT_APP_SUPPORTS_GLOBAL_MEETINGS`**: Boolean (default: `true`) for global `/meetings` endpoint.

To configure for a remote backend:
- Set `IS_PROD = true` in `src/environment.js`, or
- Provide `REACT_APP_API_URL` / `REACT_APP_SERVER_URL` during build.

---

## Developer Quick Start

1. **Install Dependencies**:
   ```bash
   cd frontend
   npm install
   ```

2. **Start Development Server**:
   ```bash
   npm start
   ```

3. **Access the App**:
   - Open `http://localhost:3000`.
   - Ensure the backend (API + Socket.io) is running at the configured URL (default: `http://localhost:8000`).

4. **Build for Production**:
   ```bash
   npm run build
   ```

---

## Module Documentation

### VideoMeet.jsx

#### Overview
`VideoMeet.jsx` is the core React component for the video call interface, accessible at `/room/:roomId`. It manages WebRTC peer connections, Socket.io signaling, and UI elements for a video meeting, including participant cards, a spotlight view, chat, and emotion analysis.

#### Key Features
- **Media Streaming**: Handles local/remote streams via WebRTC.
- **Signaling**: Uses Socket.io for SDP/ICE candidate exchange and participant coordination.
- **Active Speaker Detection**: Analyzes audio streams to highlight the active speaker.
- **Chat**: Supports in-call messaging with optimistic updates.
- **Screen Sharing**: Replaces video feed with screen media.
- **Emotion Analysis**: Captures and analyzes audio/video clips (host-only or debug mode).
- **Error Recovery**: Handles SDP mismatches, video stalls, and connection issues.

#### Dependencies
- React 18, `react-router-dom`, `socket.io-client`, `@mui/material`, `@mui/icons-material`, `framer-motion`, `AuthContext`, `mediaController.js`.

#### Key Functions
- **Initialization (`start`)**: Requests media, initializes Socket.io, and sets up `mediaController`.
- **Socket Handlers**: Manages events like `join-call`, `signal`, `chat-message`, `emotion.update`.
- **WebRTC**: Creates `RTCPeerConnection` with STUN servers, handles tracks and ICE candidates.
- **Active Speaker**: Uses `AudioContext` and `AnalyserNode` for RMS-based detection.
- **Screen Sharing**: Replaces video tracks with `getDisplayMedia` stream.
- **Emotion Analysis**: Sends periodic clips to `REACT_APP_EMOTION_URL`.
- **Cleanup**: Stops tracks, closes connections, and navigates to `/home`.

#### UI Components
- `ParticipantCard`: Shows participant video, name, and emotion badge.
- `SpotlightCard`: Displays active speaker in a larger view.
- `EmotionServicePanel`: Host-only panel for real-time emotion data.
- `ChatInput`: Input for sending chat messages.

#### State
- `remoteStreams`, `connecting`, `muted`, `videoOff`, `chatOpen`, `chatMessages`, `participantsMeta`, `myId`, `activeSpeakerId`, `shareEmotion`, `emotionsMap`.

#### API Endpoints
- **Socket.io**: `join-call`, `signal`, `chat-message`, `emotion.frame`, `end-meeting`, etc.
- **REST**: `POST ${API_BASE}/transcript`, `POST ${EMOTION_ENDPOINT}`.

#### Error Handling
- Alerts for media access failures.
- Timeout for Socket.io connection issues.
- Recovers from SDP mismatches and video stalls.

#### Accessibility
- ARIA attributes for buttons and chat.
- Keyboard shortcuts (`M`, `V`, `C`).

#### Styling
- Uses `videoComponent.module.css` and `framer-motion` for animations.

#### Troubleshooting
- **Media Blocked**: Ensure HTTPS/localhost and permissions.
- **Socket Issues**: Verify `REACT_APP_SIGNALING_URL`.
- **CORS**: Check backend origin settings.
- **Video Stalls**: Monitor bandwidth and WebRTC stats.

---

### AuthContext.jsx

#### Overview
`AuthContext.jsx` provides a React context for managing user authentication and meeting history. It centralizes login, registration, logout, and history operations, using Axios for API calls and localStorage for token/history persistence.

#### Key Responsibilities
- Manages user state and token storage.
- Provides registration, login, and logout functions.
- Handles meeting history with server/localStorage fallbacks.
- Adds participants to meetings via API.

#### Dependencies
- React 18, `axios`, `http-status`, `react-router-dom`, `environment.js`.

#### Context Value
- `userData`, `setUserData`, `logout`, `handleRegister`, `handleLogin`, `getHistoryOfUser`, `addToUserHistory`, `addParticipant`.

#### Axios Clients
- `client`: `${server}/api/v1/users` (register, login, activity).
- `apiClient`: `${server}/api/v1` (meetings, participants).

#### Key Functions
- **Interceptors**: Attach `Authorization: Bearer <token>`; handle 401 errors.
- **handleRegister**: `POST /register` for new users.
- **handleLogin**: `POST /login`, stores token, navigates to `/home`.
- **getHistoryOfUser**: Fetches history from `/meetings`, `/users/meetings`, or localStorage.
- **addToUserHistory**: Saves meetings to server or localStorage.
- **addParticipant**: Adds participants via `/meetings/:code/participants`.

#### Storage
- `localStorage`: Stores `token` and `meeting_history_v1`.

#### API Endpoints
- `POST /api/v1/users/register`, `POST /api/v1/users/login`.
- `GET /meetings`, `GET /users/meetings`, `POST /meetings/:code/participants`.

#### Error Handling
- Handles 401 errors with logout and redirect.
- Falls back to alternative endpoints or localStorage.

#### Troubleshooting
- **401 Errors**: Verify token and backend logic.
- **History Issues**: Check `SUPPORTS_GLOBAL_MEETINGS` and endpoints.
- **CORS**: Ensure backend allows frontend origin.

---

### mediaController.js

#### Overview
`mediaController.js` abstracts WebRTC media stream management, handling track toggling, placeholders, and cleanup. It ensures robust track replacement and cross-browser compatibility, especially for Safari.

#### Key Features
- Initializes media with streams, sockets, and peer connections.
- Toggles audio/video with placeholder tracks.
- Creates low-bandwidth canvas-based placeholders.
- Replaces tracks across peers with fallbacks.
- Cleans up resources to prevent leaks.

#### Exported Functions
- `initMediaController`, `setLocalStream`, `setVideoElement`, `setPeerConnections`, `setSocketRef`, `toggleVideo`, `toggleAudio`, `stopAllVideoAndCleanup`, `forceReleaseEverything`, `setPreferPeerPlaceholder`, `setExternalCleaners`, `registerRemoteVideoElement`, `unregisterRemoteVideoElement`, `attachRemoteStream`, `replaceTrackInPeers`, `replaceLocalTrack`, `stopAndRemoveTracks`, `restoreOutgoingVideoToPeers`, `stopTransceivers`.

#### Internal State
- `localStream`, `socketRef`, `pcsRef`, `localVideoEl`, `togglingAudio`, `togglingVideo`, `remoteVideoEls`, `_placeholderTrack`, `_placeholderStream`, `preferPeerPlaceholder`, `externalCleaners`.

#### Error Handling
- Multi-level track replacement fallbacks.
- Safari-specific preview handling.
- Prevents resource leaks with aggressive cleanup.

#### Troubleshooting
- **Black Screens (Safari)**: Ensure `localVideoEl` and call `refreshSafariPreview`.
- **Track Issues**: Check peer connection state.
- **Resource Leaks**: Register external cleaners; use `forceReleaseEverything`.

---

### home.jsx

#### Overview
`home.jsx` renders the home page, allowing users to create/join rooms, copy links, and view recent transcripts. It integrates with the backend for room creation and transcript fetching.

#### Key Features
- Creates rooms via `POST /rooms`.
- Joins rooms by validating codes/URLs.
- Fetches and displays deduplicated transcripts.
- Copies links to clipboard.
- Shows feedback via Material-UI Snackbars.

#### Dependencies
- React 18, `react-router-dom`, `@mui/material`, `home.css`.

#### Key Functions
- **createRoom**: Creates room and navigates.
- **copyLink**: Creates room and copies link.
- **joinRoom**: Validates and joins room.
- **copyToClipboard**: Copies text with fallback.
- **dedupeByCodeKeepNewest**: Deduplicates transcripts.

#### UI Components
- Form inputs for name and room code.
- Buttons for room actions and history.
- Transcript list with expand/download options.
- Snackbar for feedback.

#### State
- `name`, `room`, `recentLocal`, `expandedTranscripts`, `snackOpen`, `snackMsg`, `snackSeverity`.

#### API Endpoints
- `POST /rooms`, `GET /rooms/:roomId`, `GET /transcript`.

#### Error Handling
- Shows Snackbars for API/input errors.
- Falls back to `execCommand` for clipboard.

#### Accessibility
- ARIA labels for buttons.
- Focusable inputs.

#### Styling
- Uses `home.css` and Material-UI.

#### Troubleshooting
- **Room Creation Fails**: Check `REACT_APP_API_URL`.
- **Transcripts Not Loading**: Verify endpoint and fetch errors.
- **Clipboard Issues**: Test fallback in older browsers.

---

## Troubleshooting

- **Media Devices Blocked**: Ensure HTTPS/localhost and browser permissions.
- **Socket.io Failures**: Verify `REACT_APP_SIGNALING_URL` and backend availability.
- **CORS Issues**: Configure backend to allow frontend origin.
- **API Errors**: Check `REACT_APP_API_URL` and server logs.
- **Resource Leaks**: Ensure cleanup functions are called on component unmount.
- **Video Stalls**: Monitor bandwidth and WebRTC connection stats.

--- 
