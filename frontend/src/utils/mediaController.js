// mediaController.js (updated — robust toggles, better placeholder handling, explicit transceiver restores)

/**
 * Public surface:
 * - initMediaController(stream, socket, peerConnections, videoElement)
 * - setLocalStream(stream)
 * - setVideoElement(el)
 * - setPeerConnections(pcMap)
 * - setSocketRef(socket)
 * - toggleVideo(currentVideoOff) -> returns updated boolean (true if video is OFF)
 * - toggleAudio(currentMuted) -> returns updated boolean (true if muted)
 * - stopAllVideoAndCleanup()
 * - forceReleaseEverything()
 * - setPreferPeerPlaceholder(enabled)
 * - setExternalCleaners(refs)
 * - registerRemoteVideoElement(peerId, el)
 * - unregisterRemoteVideoElement(peerId)
 * - attachRemoteStream(peerId, stream)
 * - replaceTrackInPeers(track, kind)
 * - replaceLocalTrack(newTrack, kind)
 * - stopAndRemoveTracks(kind)
 * - restoreOutgoingVideoToPeers(realTrack)
 */

let localStream = null;
let socketRef = null;
let pcsRef = {}; // peerId -> RTCPeerConnection
let localVideoEl = null; // optional local preview element (Safari needs this)

let togglingAudio = false;
let togglingVideo = false;

let localMirrorEnabled = false; // default: NO mirror (match Meet/Teams remote behavior)

const remoteVideoEls = new Map(); // peerId -> HTMLVideoElement

// placeholder track currently in use (if any)
let _placeholderTrack = null;
// placeholder stream (for local preview only)
let _placeholderStream = null;

/**
 * Controls whether controller injects placeholder track into peers.
 * Default: false (do NOT inject into peers). Use setPreferPeerPlaceholder(true) to opt in.
 */
let preferPeerPlaceholder = false;
export function setPreferPeerPlaceholder(enabled = false) {
  preferPeerPlaceholder = !!enabled;
}

/**
 * External cleaners:
 * - recordersRef: expected shape { current: { [key]: { recorder } } } (best-effort)
 * - audioContextRef: expected shape { current: AudioContext }
 * - removeAnalyzerFn: expected function(kind) to remove/disconnect analyzers
 * - prevLocalStreamRef: expected shape { current: MediaStream } (optional)
 *
 * Register from your app (VideoMeet.jsx) with setExternalCleaners(...)
 */
let externalCleaners = {
  recordersRef: null,
  audioContextRef: null,
  removeAnalyzerFn: null,
  prevLocalStreamRef: null,
};

export function setExternalCleaners(refs = {}) {
  externalCleaners.recordersRef = refs.recordersRef ?? null;
  externalCleaners.audioContextRef = refs.audioContextRef ?? null;
  externalCleaners.removeAnalyzerFn = refs.removeAnalyzerFn ?? null;
  externalCleaners.prevLocalStreamRef = refs.prevLocalStreamRef ?? null;
}

/* ---------------- utility helpers ---------------- */

function isSafari() {
  return typeof navigator === "object" && /^((?!chrome|android).)*safari/i.test(navigator.userAgent);
}

function enforceVideoMirrorBehavior(videoEl, { mirror = false } = {}) {
  if (!videoEl || typeof videoEl.style === "undefined") return;
  try {
    if (mirror) {
      videoEl.style.transform = "scaleX(-1)";
      videoEl.style.webkitTransform = "scaleX(-1)";
    } else {
      videoEl.style.transform = "none";
      videoEl.style.webkitTransform = "none";
    }
  } catch (err) {
    console.warn("[mediaController] enforceVideoMirrorBehavior failed:", err);
  }
}

function _safeEmit(event, payload) {
  try {
    if (!socketRef) {
      console.error("[mediaController] No signaling socket connected. Did you pass a socket to initMediaController? event=", event);
      return;
    }
    socketRef.emit?.(event, payload);
  } catch (e) {
    console.warn("[mediaController] socket emit failed:", e);
  }
}

function safePlay(videoEl) {
  if (!videoEl) return;
  try {
    const p = videoEl.play?.();
    if (p && typeof p.then === "function") {
      p.catch(() => { /* ignore autoplay block */ });
    }
  } catch (e) { /* ignore */ }
}

/* ---------------- Helpers: placeholder & cleanup ---------------- */

/**
 * Create a very small, low-FPS canvas-capture video track to use as a placeholder.
 * Mark track with __isPlaceholder so we can identify and stop it later.
 *
 * NOTE: the returned track will have a __placeholderStream reference to the underlying stream,
 * so callers can use that stream for local preview without adding the placeholder to the canonical localStream.
 */
function createPlaceholderVideoTrack({ width = 16, height = 12, fps = 1 } = {}) {
  try {
    const canvas = Object.assign(document.createElement("canvas"), { width, height });
    const ctx = canvas.getContext("2d");
    // draw a single black frame (you can draw branding / blur etc)
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, width, height);

    // capture a very low-framerate stream
    const stream = canvas.captureStream(fps);
    const [track] = stream.getVideoTracks();
    if (!track) return null;
    track.__isPlaceholder = true;
    track.__placeholderCanvas = canvas;
    // keep a reference to the stream for local preview usage
    try { track.__placeholderStream = stream; } catch (e) { /* ignore */ }
    return track;
  } catch (e) {
    console.warn("[mediaController] createPlaceholderVideoTrack failed:", e);
    return null;
  }
}

function stopAndCleanupPlaceholder(track) {
  if (!track) return;
  try {
    // stop track (this will stop the underlying captureStream)
    try { track.stop(); } catch (e) { console.warn("[mediaController] stopAndCleanupPlaceholder: track.stop failed", e); }
    if (track.__placeholderCanvas) {
      // allow GC
      try { track.__placeholderCanvas.width = 0; } catch (e) { console.warn("[mediaController] stopAndCleanupPlaceholder: clearing canvas width failed", e); }
      track.__placeholderCanvas = null;
    }
    if (track.__placeholderStream) {
      try {
        // stop all tracks of the preview stream and clear reference
        (track.__placeholderStream.getTracks() || []).forEach(t => { try { t.stop(); } catch(e){} });
      } catch(e) {}
      track.__placeholderStream = null;
    }
  } catch (e) {
    console.warn("[mediaController] stopAndCleanupPlaceholder error:", e);
  }
}

/* ---------------- Robust cleanup helper ---------------- */

// Defensive _runExternalCleaners: only stop tracks of `kind`, and DO NOT stop tracks
// if the prevLocalStreamRef points at the active localStream (safety guard for Chrome).
function _runExternalCleaners(kind = "video") {
  try {
    // 1) stop recorders kept by component-level ref (e.g. recordersRef.current)
    try {
      const rr = externalCleaners.recordersRef && externalCleaners.recordersRef.current;
      if (rr && typeof rr === "object") {
        Object.keys(rr).forEach((k) => {
          try {
            const entry = rr[k];
            if (!entry) return;
            const recorder = entry.recorder ?? (entry?.recorderRef?.current) ?? null;
            if (recorder && recorder.state && recorder.state !== "inactive") {
              try { recorder.stop(); } catch (e) { console.warn("[mediaController] external recorder.stop() failed for key=" + k, e); }
            }
            try { delete rr[k]; } catch (e) {}
          } catch (e) {}
        });
      }
    } catch (e) {
      console.warn("[mediaController] _runExternalCleaners: stopping recorders failed", e);
    }

    // 2) close audio context if provided (best-effort)
    try {
      const acr = externalCleaners.audioContextRef && externalCleaners.audioContextRef.current;
      if (acr && typeof acr.close === "function") {
        try { acr.close().catch(()=>{}); } catch (e) { console.warn("[mediaController] external AudioContext.close() failed", e); }
        try { externalCleaners.audioContextRef.current = null; } catch (e) {}
      }
    } catch (e) {
      console.warn("[mediaController] _runExternalCleaners: closing audio context failed", e);
    }

    // 3) call removeAnalyzerFn if present
    try {
      if (typeof externalCleaners.removeAnalyzerFn === "function") {
        try { externalCleaners.removeAnalyzerFn(kind); } catch (e) { console.warn("[mediaController] external removeAnalyzerFn failed", e); }
      }
    } catch (e) {}

    // 4) stop any prevLocalStreamRef if present — STOP ONLY TRACKS MATCHING THE REQUESTED 'kind'
    try {
      const prev = externalCleaners.prevLocalStreamRef && externalCleaners.prevLocalStreamRef.current;
      if (prev && typeof prev.getTracks === "function") {
        // SAFETY GUARD: do not stop tracks if prev is the same object as the active localStream.
        if (prev === localStream) {
          console.warn("[mediaController] _runExternalCleaners: prevLocalStreamRef === localStream -> skipping stop to avoid killing live mic/video");
        } else {
          try {
            // stop only tracks whose kind matches the requested kind (defensive: skip placeholder tracks)
            (prev.getTracks() || []).forEach((t) => {
              try {
                if (t.__isPlaceholder) return;
                if (typeof t.kind === "string" && t.kind === kind) {
                  if (typeof t.stop === "function") {
                    t.stop();
                  }
                }
              } catch (e) {}
            });
          } catch (e) {}
        }
        // clear the prev ref (we don't want to keep stale refs around)
        try { externalCleaners.prevLocalStreamRef.current = null; } catch (e) {}
      }
    } catch (e) {
      console.warn("[mediaController] _runExternalCleaners: stopping prevLocalStreamRef failed", e);
    }
  } catch (e) {
    console.warn("[mediaController] _runExternalCleaners top-level error", e);
  }
}



function _stopAllVideoAndCleanup() {
  try {
    const stopped = [];

    function safeStopTrack(t, owner) {
      if (!t) return;
      if (t.__isPlaceholder) return; // don’t touch placeholder tracks
      stopped.push({ owner, id: t.id, label: t.label || "(no label)", state: t.readyState });
      try { t.stop(); } catch (e) {
        console.warn(`[mediaController] safeStopTrack stop failed for owner=${owner} id=${t?.id}`, e);
      }
    }

    // 1) Stop tracks on canonical localStream (but DON’T remove them)
    try {
      if (localStream?.getVideoTracks) {
        (localStream.getVideoTracks() || []).forEach(t => safeStopTrack(t, 'module.localStream'));
        // ⚠️ Unlike before: we do NOT call localStream.removeTrack(t)
        // This keeps the structure intact for smooth re-enable later.
      }
    } catch (e) {
      console.warn("[mediaController] error stopping localStream video tracks:", e);
    }

    // 2) Defensive: stop tracks in window.localStream if it’s a different object
    try {
      if (window?.localStream && window.localStream !== localStream) {
        (window.localStream.getVideoTracks?.() || []).forEach(
          t => safeStopTrack(t, 'window.localStream')
        );
        try { window.localStream = null; } catch (e) {}
      }
    } catch (e) {
      console.warn("[mediaController] error checking window.localStream:", e);
    }

    // 3) Stop video tracks in <video> elements (but don’t nuke srcObject if it’s the canonical stream)
    try {
      document.querySelectorAll('video').forEach((el, i) => {
        const s = el.srcObject;
        if (s?.getVideoTracks) {
          (s.getVideoTracks() || []).forEach(t => safeStopTrack(t, `video#${i}.srcObject`));
          if (s !== localStream && s !== _placeholderStream) {
            try { el.srcObject = null; } catch {}
          }
        }
      });
    } catch (e) {
      console.warn("[mediaController] error iterating video elements:", e);
    }

    // 4) Clear sender tracks on peer connections
    try {
      const pcList = Object.values(pcsRef || {});
      pcList.forEach(pc => {
        (pc.getSenders?.() || []).forEach((s, idx) => {
          if (s?.track?.kind === 'video') {
            safeStopTrack(s.track, `pc.sender[${idx}]`);
            try { s.replaceTrack(null); } catch (e) {
              console.warn("[mediaController] sender.replaceTrack(null) failed", e);
            }
          }
        });
      });
    } catch (e) {
      console.warn("[mediaController] error processing peer connections:", e);
    }

    // 5) Run external cleaners (recorders, analyzers, etc.)
    try { _runExternalCleaners("video"); } catch (e) {}

    if (stopped.length) {
      console.info("[mediaController] stopped video tracks:", stopped);
    } else {
      console.info("[mediaController] stopAllVideoAndCleanup: no tracks stopped");
    }
  } catch (e) {
    console.warn("[mediaController] _stopAllVideoAndCleanup error", e);
  }
}


function _forceReleaseEverything() {
  try {
    // 1) stop tracks from known peer connections
    try {
      const pcs = Object.values(pcsRef || {});
      pcs.forEach((pc) => {
        try {
          (pc.getSenders()||[]).forEach((s) => {
            try {
              if (s && s.track && s.track.kind === 'video') {
                try { s.track.stop(); } catch(e) { console.warn("[mediaController] forcing stop on sender.track failed", e); }
                try { if (typeof s.replaceTrack === 'function') s.replaceTrack(null); } catch(e) { console.warn("[mediaController] forcing replaceTrack(null) failed", e); }
              }
            } catch(e){ console.warn("[mediaController] error stopping sender track", e); }
          });
          try { if (typeof pc.close === 'function') pc.close(); } catch(e) { console.warn("[mediaController] pc.close() failed", e); }
        } catch(e){
          console.warn("[mediaController] error iterating pc senders/closing pc:", e);
        }
      });
    } catch(e){
      console.warn("[mediaController] error enumerating pcsRef in _forceReleaseEverything:", e);
    }

    // 2) stop any MediaRecorder-like objects reachable from window
    try {
      Object.keys(window).forEach(k => {
        try {
          const v = window[k];
          if (!v) return;
          if (v && (v.constructor && v.constructor.name === 'MediaRecorder')) {
            try { v.stop(); } catch(e) { console.warn("[mediaController] stopping MediaRecorder window property failed for key=" + k, e); }
          }
        } catch(e){ /* noise - ignore */ }
      });
    } catch(e){
      console.warn("[mediaController] error scanning window keys for MediaRecorder instances:", e);
    }

    // 3) stop and null app-level recorders/audio contexts/analyzers/prevLocalStream if provided
    try {
      _runExternalCleaners("video");
    } catch (e) {
      console.warn("[mediaController] _forceReleaseEverything: external cleaners failed", e);
    }

    // 4) null and stop any stream refs on window we can reach (localStream only)
    try {
      ['localStream'].forEach(name => {
        try {
          const s = window[name];
          if (s && typeof s.getVideoTracks === 'function') {
            (s.getVideoTracks()||[]).forEach(t => { try { t.stop(); } catch(e){ console.warn("[mediaController] stopping window stream track failed", e); } });
          }
          try { window[name] = null; } catch(e){ console.warn("[mediaController] clearing window stream reference failed", e); }
        } catch(e){ console.warn("[mediaController] error clearing window stream " + name + ":", e); }
      });
    } catch(e){
      console.warn("[mediaController] error nulling window stream refs:", e);
    }

    // 5) clear video element srcObjects
    try {
      document.querySelectorAll('video').forEach(el => {
        try {
          const s = el.srcObject;
          if (s && (s.getVideoTracks || s.getTracks)) {
            (s.getVideoTracks ? s.getVideoTracks() : s.getTracks()).forEach(t => { try { if (t && t.kind === 'video') t.stop(); } catch(e){ console.warn("[mediaController] stopping element-attached track failed", e); } });
          }
          try { el.srcObject = null; } catch(e){ console.warn("[mediaController] clearing element.srcObject failed", e); }
        } catch(e){
          console.warn("[mediaController] error processing a video element in _forceReleaseEverything:", e);
        }
      });
    } catch(e){
      console.warn("[mediaController] error iterating video elements in _forceReleaseEverything:", e);
    }

    // 6) UA nudge: brief getUserMedia then stop (helps some stubborn drivers)
    try {
      if (navigator && navigator.mediaDevices && typeof navigator.mediaDevices.getUserMedia === 'function') {
        navigator.mediaDevices.getUserMedia({ video: true }).then(s => {
          try { (s.getTracks()||[]).forEach(t => { try { t.stop(); } catch(e){ console.warn("[mediaController] stopping nudge track failed", e); } }); } catch(e){ console.warn("[mediaController] error stopping nudge tracks", e); }
        }).catch((err) => {
          // non-fatal but can help with debugging stubborn driver problems
          console.debug("[mediaController] prewarm getUserMedia (nudge) failed (non-fatal):", err);
        });
      }
    } catch(e){
      console.warn("[mediaController] error attempting UA nudge getUserMedia:", e);
    }

    console.info('[mediaController] _forceReleaseEverything attempted');
  } catch (e) {
    console.warn('[mediaController] _forceReleaseEverything error', e);
  }
}

/* ---------------- Initialization / small setters ---------------- */

export function initMediaController(stream, socket, peerConnections = {}, videoElement = null) {
  localStream = stream ?? null;
  socketRef = socket ?? null;
  pcsRef = peerConnections ?? {};
  localVideoEl = videoElement ?? null;

  // debug hook (safe to remove in prod)
  try {
    window.__SKYMEET_MEDIA_CTRL = {
      getLocalStream: () => localStream,
      stopAndClear: _stopAllVideoAndCleanup,
      stopAndForce: _forceReleaseEverything
    };
  } catch (e) {
    console.warn("[mediaController] initMediaController: failed to set debug hook on window", e);
  }

  if (localVideoEl) {
    try {
      localVideoEl.autoplay = true;
      localVideoEl.playsInline = true;
      localVideoEl.muted = true;
      // local preview element should show the canonical localStream by default
      // unless a placeholder preview is active (we prefer the placeholder for local-only UI)
      if (!_placeholderStream) localVideoEl.srcObject = localStream ?? null;
      enforceVideoMirrorBehavior(localVideoEl, { mirror: !!localMirrorEnabled });
      safePlay(localVideoEl);
    } catch (err) {
      console.warn("[mediaController] initMediaController: localVideoEl attach failed:", err);
    }
    if (isSafari()) refreshSafariPreview();
  }
}

export function setLocalStream(stream) {
  localStream = stream ?? null;
  if (localVideoEl) {
    try {
      // only set localVideoEl.srcObject to the canonical localStream if there is no active placeholder preview
      // placeholder preview is intentionally local-only; prefer it for visual parity with Zoom/Teams.
      if (!_placeholderStream) {
        localVideoEl.srcObject = localStream ?? null;
        safePlay(localVideoEl);
      }
      enforceVideoMirrorBehavior(localVideoEl, { mirror: !!localMirrorEnabled });
    } catch (err) {
      console.warn("[mediaController] setLocalStream: failed to update localVideoEl.srcObject:", err);
    }
    if (isSafari()) refreshSafariPreview();
  }
}

export function setLocalMirrorEnabled(enabled) {
  localMirrorEnabled = !!enabled;
  if (localVideoEl) {
    enforceVideoMirrorBehavior(localVideoEl, { mirror: localMirrorEnabled });
  }
}

export function registerRemoteVideoElement(peerId, videoEl) {
  if (!peerId || !videoEl) return;
  remoteVideoEls.set(peerId, videoEl);
  try {
    videoEl.autoplay = true;
    videoEl.playsInline = true;
    enforceVideoMirrorBehavior(videoEl, { mirror: false });
  } catch (err) {
    console.warn("[mediaController] registerRemoteVideoElement: failed to configure videoEl:", err);
  }
}

export function unregisterRemoteVideoElement(peerId) {
  if (!peerId) return;
  try {
    const el = remoteVideoEls.get(peerId);
    if (el) {
      try { el.srcObject = null; } catch (e) { console.warn("[mediaController] unregisterRemoteVideoElement clearing srcObject failed", e); }
    }
  } catch (err) { console.warn("[mediaController] unregisterRemoteVideoElement error:", err); }
  remoteVideoEls.delete(peerId);
}

export function attachRemoteStream(peerId, stream) {
  const el = remoteVideoEls.get(peerId);
  if (!el) return;

  try {
    if (stream && stream.getVideoTracks && stream.getVideoTracks().length > 0) {
      // Has video → show it
      el.srcObject = stream;
      el.style.display = "block";
    } else {
      // No video tracks → clear stream and show camera-off state
      el.srcObject = null;
      el.style.display = "none"; // hide <video>
      // Optional: add a CSS class or sibling <div> with avatar/placeholder
      const placeholder = document.querySelector(`#placeholder-${peerId}`);
      if (placeholder) {
        placeholder.style.display = "flex";
      }
    }

    enforceVideoMirrorBehavior(el, { mirror: false });
  } catch (err) {
    console.warn("[attachRemoteStream] failed:", err);
  }
}


export function setPeerConnections(peerConnections) {
  if (!peerConnections || typeof peerConnections !== 'object') {
    console.warn("[mediaController] setPeerConnections: invalid value provided", peerConnections);
    pcsRef = {};
    return;
  }
  pcsRef = peerConnections ?? {};
}

export function setSocketRef(socket) {
  if (!socket) {
    socketRef = null;
    console.debug("[mediaController] setSocketRef: clearing socketRef");
    return;
  }
  socketRef = socket ?? null;
}

export function setVideoElement(videoEl) {
  localVideoEl = videoEl ?? null;
  if (localVideoEl) {
    try {
      localVideoEl.autoplay = true;
      localVideoEl.playsInline = true;
      localVideoEl.muted = true;
      // only set to canonical localStream if no placeholder preview is currently active
      if (!_placeholderStream) localVideoEl.srcObject = localStream ?? null;
      enforceVideoMirrorBehavior(localVideoEl, { mirror: !!localMirrorEnabled });
      safePlay(localVideoEl);
    } catch (err) {
      console.warn("[mediaController] setVideoElement: attach failed:", err);
    }
    if (isSafari()) refreshSafariPreview();
  }
}

/* ---------------- Toggle functions ---------------- */

export async function toggleAudio(currentMuted) {
  if (togglingAudio) return currentMuted;
  togglingAudio = true;
  const newMuted = !currentMuted;

  try {
    if (!navigator?.mediaDevices || !localStream) {
      _safeEmit("update-participant-state", { muted: newMuted });
      return newMuted;
    }

    if (newMuted) {
      // Mute: stop audio tracks and inform peers
      stopAndRemoveTracks("audio");
      try { _runExternalCleaners("audio"); } catch {}
      _safeEmit("update-participant-state", { muted: true });
      if (isSafari()) refreshSafariPreview();
      return true;
    }

    // Unmute: get a fresh mic track and replace in peers
    let acquired;
    try {
      acquired = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
      console.warn("[mediaController] getUserMedia(audio) failed:", err);
      return currentMuted; // rollback
    }

    const newTrack = acquired.getAudioTracks()[0];
    if (!newTrack) {
      acquired.getTracks().forEach(t => { try { t.stop(); } catch {} });
      return currentMuted;
    }

    try {
      // attach to peers and localStream
      await replaceTrackInPeers(newTrack, "audio");
      replaceLocalTrack(newTrack, "audio");
      _safeEmit("update-participant-state", { muted: false });
      if (isSafari()) refreshSafariPreview();
      return false;
    } catch (err) {
      console.warn("[mediaController] toggleAudio attach failed:", err);
      try { newTrack.stop(); } catch {}
      return currentMuted;
    } finally {
      // stop any acquired tracks that did not become part of localStream
      acquired?.getTracks()?.forEach(t => {
        const present = localStream?.getTracks()?.some(lt => lt.id === t.id);
        if (!present) try { t.stop(); } catch {}
      });
    }
  } finally {
    togglingAudio = false;
  }
}


export async function toggleVideo(currentVideoOff) {
  if (togglingVideo) return currentVideoOff;
  togglingVideo = true;
  const newVideoOff = !currentVideoOff;
  let acquired = null;

  try {
    if (!navigator?.mediaDevices || !localStream) {
      _safeEmit("update-participant-state", { video: !newVideoOff });
      return newVideoOff;
    }

    // --- TURN VIDEO OFF ---
    if (newVideoOff) {
      console.log("[mediaController] toggleVideo -> OFF");

      // cleanup any existing placeholder first
      if (_placeholderTrack) {
        try { stopAndCleanupPlaceholder(_placeholderTrack); } catch (e) { console.warn("[mediaController] placeholder cleanup failed", e); }
        _placeholderTrack = null;
        _placeholderStream = null;
      }

      // create and use a placeholder preview locally (does not keep camera)
      _placeholderTrack = createPlaceholderVideoTrack({ width: 16, height: 12, fps: 1 });
      _placeholderStream = _placeholderTrack?.__placeholderStream ?? null;

      if (_placeholderStream && localVideoEl) {
        try {
          localVideoEl.srcObject = _placeholderStream;
          enforceVideoMirrorBehavior(localVideoEl, { mirror: !!localMirrorEnabled });
          safePlay(localVideoEl);
        } catch (e) { console.warn("[mediaController] attaching placeholder preview failed", e); }
      }

      // If we want peers to receive a placeholder, try that; otherwise stop outgoing real video
      if (preferPeerPlaceholder && _placeholderTrack) {
        try {
          // attempt to replace outgoing with the placeholder track
          await replaceTrackInPeers(_placeholderTrack, "video");
        } catch (e) {
          console.warn("[mediaController] preferPeerPlaceholder: replace failed, falling back to stopOutgoingVideoToPeers", e);
          try { await stopOutgoingVideoToPeers(); } catch (ee) {}
        }
      } else {
        try {
          // aggressively stop outgoing video on peers and set transceivers to recvonly
          await stopOutgoingVideoToPeers();
        } catch (e) {
          console.warn("[mediaController] stopOutgoingVideoToPeers failed", e);
        }
      }

      // ── Chrome-safe mic preservation: try to acquire an audio-only track and swap it in
      // This prevents Chrome from tearing down the mic when we stop video tracks that came from a combined AV capture.
      let __micSwapAcquired = null;
      try {
        if (localStream?.getAudioTracks?.().length) {
          try {
            __micSwapAcquired = await navigator.mediaDevices.getUserMedia({ audio: true });
            const __micSwapTrack = __micSwapAcquired.getAudioTracks()[0];
            if (__micSwapTrack) {
              // Attach new audio to peers and canonical localStream (match toggleAudio flow)
              try {
                await replaceTrackInPeers(__micSwapTrack, "audio");
                replaceLocalTrack(__micSwapTrack, "audio");
              } catch (e) {
                console.warn("[mediaController] mic-swap replace failed:", e);
                // if replace fails we'll clean up below and proceed
              }
            }
          } catch (e) {
            console.warn("[mediaController] mic-swap getUserMedia(audio) failed, proceeding without swap:", e);
            try { __micSwapAcquired?.getTracks()?.forEach(t => { try { t.stop(); } catch {} }); } catch {}
            __micSwapAcquired = null;
          }
        }
      } catch (e) {
        console.warn("[mediaController] mic-swap unexpected error, proceeding:", e);
        try { __micSwapAcquired?.getTracks()?.forEach(t => { try { t.stop(); } catch {} }); } catch {}
        __micSwapAcquired = null;
      }

      // Ensure any local video tracks are stopped and removed from canonical localStream
      try {
        const vids = (localStream.getVideoTracks?.() || []);
        vids.forEach((t) => {
          try { t.stop(); } catch (e) { /* ignore */ }
          try { localStream.removeTrack(t); } catch (e) { /* ignore */ }
        });
      } catch (e) {
        console.warn("[mediaController] stopping/removing localStream video tracks failed", e);
      }

      // Clean up any acquired mic-swap tracks that did not become part of localStream
      try {
        __micSwapAcquired?.getTracks?.().forEach(t => {
          const present = localStream?.getTracks?.().some(lt => lt.id === t.id);
          if (!present) try { t.stop(); } catch {}
        });
      } catch (e) {
        // non-fatal
      }

      // Run app-level cleaners (e.g. MediaRecorder, analyzers)
      try { _runExternalCleaners("video"); } catch (e) {}

      // Safari: ensure preview cleared if no canonical live tracks remain
      if (isSafari()) { clearPreviewIfNoTracks(); refreshSafariPreview(); }

      _safeEmit("update-participant-state", { video: false });
      console.log("[mediaController] toggleVideo -> OFF complete");
      return true;
    }

    // --- TURN VIDEO ON ---
    console.log("[mediaController] toggleVideo -> ON");
    try {
      acquired = await navigator.mediaDevices.getUserMedia({ video: true });
    } catch (err) {
      console.warn("[mediaController] getUserMedia(video) failed:", err);
      return currentVideoOff;
    }

    const newTrack = acquired.getVideoTracks()[0];
    if (!newTrack) {
      acquired.getTracks().forEach(t => { try { t.stop(); } catch {} });
      return currentVideoOff;
    }

    // keep a reference to originating stream for hygiene later
    try { newTrack._sourceStream = acquired; } catch (e) {}

    try {
      // remove placeholder preview (we're about to show the real stream)
      if (_placeholderTrack) {
        try { stopAndCleanupPlaceholder(_placeholderTrack); } catch (e) { console.warn("[mediaController] placeholder cleanup failed", e); }
        _placeholderTrack = null;
      }
      if (_placeholderStream && localVideoEl) {
        try { localVideoEl.srcObject = null; } catch (e) { /* ignore */ }
      }
      _placeholderStream = null;

      // Replace track for peers (preferred method). Also ensure transceivers set to sendrecv after.
      await replaceTrackInPeers(newTrack, "video");

      // Ensure transceivers / senders are set to sendrecv for video for all peer connections
      try {
        const pcs = Object.values(pcsRef || {});
        for (const pc of pcs) {
          try {
            const txs = pc.getTransceivers?.() || [];
            for (const tx of txs) {
              try {
                if (!tx) continue;
                if (tx.kind === "video" || (tx.sender && tx.sender.track && tx.sender.track.kind === "video")) {
                  try { tx.direction = "sendrecv"; } catch (e) { /* ignore */ }
                }
              } catch (e) { /* ignore per-transceiver */ }
            }
          } catch (e) { /* ignore per-pc */ }
        }
      } catch (e) { /* ignore */ }

      // Attach to canonical localStream
      try {
        // If canonical stream already has audio tracks, merge the new video into it; otherwise create a new stream
        if (localStream?.getAudioTracks?.().length > 0) {
          try {
            localStream.addTrack(newTrack);
            attachTrackEndHandler(newTrack, "video");
            setLocalStream(localStream);
          } catch (e) {
            console.warn("[mediaController] adding newVideoTrack to existing localStream failed, creating merged stream", e);
            const merged = new MediaStream([...localStream.getAudioTracks(), newTrack]);
            attachTrackEndHandler(newTrack, "video");
            setLocalStream(merged);
            localStream = merged;
          }
        } else {
          const merged = new MediaStream([newTrack]);
          attachTrackEndHandler(newTrack, "video");
          setLocalStream(merged);
          localStream = merged;
        }
      } catch (e) {
        console.warn("[mediaController] attach newTrack to localStream failed:", e);
        try { newTrack.stop(); } catch (ee) {}
        return currentVideoOff;
      }

      // Safari preview helpers
      if (isSafari()) refreshSafariPreview();

      // ensure preview plays
      if (localVideoEl) {
        try {
          // force reattach cycle in case some UAs cache rendering
          localVideoEl.srcObject = null;
          localVideoEl.srcObject = localStream;
          enforceVideoMirrorBehavior(localVideoEl, { mirror: !!localMirrorEnabled });
          safePlay(localVideoEl);
        } catch (e) {
          console.warn("[mediaController] setting localVideoEl preview failed:", e);
        }
      }

      _safeEmit("update-participant-state", { video: true });
      console.log("[mediaController] toggleVideo -> ON complete");
      return false;
    } catch (err) {
      console.warn("[mediaController] toggleVideo attach failed:", err);
      try { newTrack.stop(); } catch {}
      return currentVideoOff;
    } finally {
      // stop any acquired tracks that were not integrated into localStream
      acquired?.getTracks()?.forEach(t => {
        const present = localStream?.getTracks()?.some(lt => lt.id === t.id);
        if (!present) try { t.stop(); } catch {}
      });
    }
  } finally {
    togglingVideo = false;
  }
}



/* ---------------- Helpers ---------------- */

function attachTrackEndHandler(track, kind) {
  if (!track) return;
  // prefer addEventListener to allow multiple listeners safely
  try {
    track.addEventListener?.('ended', () => {
      try {
        const present = localStream && localStream.getTracks().some((t) => t.id === track.id);
        if (!present) return;
      } catch (err) {
        // ignore localStream reading issues
      }

      if (kind === "audio") {
        _safeEmit("update-participant-state", { muted: true });
      } else if (kind === "video") {
        _safeEmit("update-participant-state", { video: false });
        if (isSafari()) {
          clearPreviewIfNoTracks();
          refreshSafariPreview();
        }
        if (localVideoEl) enforceVideoMirrorBehavior(localVideoEl, { mirror: !!localMirrorEnabled });
      }
    });
  } catch (e) {
    // fallback to setting onended if addEventListener isn't available
    try {
      track.onended = () => {
        try {
          const present = localStream && localStream.getTracks().some((t) => t.id === track.id);
          if (!present) return;
        } catch (err) {
          // ignore
        }

        if (kind === "audio") {
          _safeEmit("update-participant-state", { muted: true });
        } else if (kind === "video") {
          _safeEmit("update-participant-state", { video: false });
          if (isSafari()) {
            clearPreviewIfNoTracks();
            refreshSafariPreview();
          }
          if (localVideoEl) enforceVideoMirrorBehavior(localVideoEl, { mirror: !!localMirrorEnabled });
        }
      };
    } catch (e2) {
      console.warn("[mediaController] attachTrackEndHandler failed to attach handlers:", e2);
    }
  }
}

function ensureDefaultTransceivers(pc, preferredSendKind = null) {
  if (!pc || typeof pc.getTransceivers !== "function" || typeof pc.addTransceiver !== "function") return;

  try {
    const existing = pc.getTransceivers() || [];
    if (existing.length > 0) return;

    const kinds = ["audio", "video"];
    for (const kind of kinds) {
      const wantSend = preferredSendKind === kind;
      try {
        pc.addTransceiver(kind, { direction: wantSend ? "sendrecv" : "recvonly" });
      } catch (e) {
        console.warn("[mediaController] ensureDefaultTransceivers: addTransceiver failed for", kind, e);
      }
    }
  } catch (err) {
    console.warn("[mediaController] ensureDefaultTransceivers top-level error:", err);
  }
}

/**
 * Stop outgoing video to peers (replaceTrack(null) + transceiver.direction fallback).
 * Exported helper so UI layer can call it at precise timing if desired.
 */
export async function stopOutgoingVideoToPeers() {
  const pcs = Object.values(pcsRef || {});
  for (const pc of pcs) {
    try {
      // first attempt: replace existing video sender with null
      try {
        const senders = pc.getSenders?.() || [];
        for (const s of senders) {
          try {
            if (s && s.track && s.track.kind === "video" && typeof s.replaceTrack === "function") {
              try { await s.replaceTrack(null); } catch (e) { console.debug("[mediaController] stopOutgoingVideoToPeers sender.replaceTrack(null) failed", e); }
            }
            // also try to stop any sender.track we can to force hardware release
            try {
              if (s && s.track && !s.track.__isPlaceholder && typeof s.track.stop === "function") {
                try { s.track.stop(); } catch (e) { /* ignore */ }
              }
            } catch (e) { /* ignore */ }
          } catch (e) { /* ignore per-sender */ }
        }
      } catch (e) { console.debug("[mediaController] stopOutgoingVideoToPeers: getSenders threw", e); }

      // second attempt: set transceivers to recvonly / replace sender track if available
      try {
        const txs = pc.getTransceivers?.() || [];
        for (const tx of txs) {
          try {
            if (!tx) continue;
            if (tx.kind === "video" || (tx.sender && tx.sender.track && tx.sender.track.kind === "video")) {
              try { tx.direction = "recvonly"; } catch (e) { /* ignore */ }
              try { if (tx.sender && typeof tx.sender.replaceTrack === "function") tx.sender.replaceTrack(null); } catch(e){/*ignore*/ }
              try {
                if (tx.sender && tx.sender.track && !tx.sender.track.__isPlaceholder && typeof tx.sender.track.stop === 'function') {
                  tx.sender.track.stop();
                }
              } catch (e) { /* ignore */ }
            }
          } catch (e) { /* ignore per-transceiver */ }
        }
      } catch (e) { /* ignore */ }
    } catch (err) {
      console.warn("[mediaController] stopOutgoingVideoToPeers: pc-level error", err);
    }
  }
}

/**
 * Restore outgoing video to peers by replacing with realTrack (if provided).
 * Exposed helper for UI to call when it wants to restore sending.
 */
export async function restoreOutgoingVideoToPeers(realTrack) {
  if (!realTrack) {
    // nothing to restore
    return;
  }
  try {
    await replaceTrackInPeers(realTrack, "video");
    // also attempt to set transceivers direction back to sendrecv
    const pcs = Object.values(pcsRef || {});
    for (const pc of pcs) {
      try {
        const txs = pc.getTransceivers?.() || [];
        for (const tx of txs) {
          try {
            if (!tx) continue;
            if (tx.kind === "video" || (tx.sender && tx.sender.track && tx.sender.track.kind === "video")) {
              try { tx.direction = "sendrecv"; } catch (e) { /* ignore */ }
            }
          } catch (e) { /* ignore per-transceiver */ }
        }
      } catch (e) { /* ignore per-pc */ }
    }
  } catch (e) {
    console.warn("[mediaController] restoreOutgoingVideoToPeers failed:", e);
  }
}

/**
 * Replace track in peers. Best-effort sequence of strategies.
 * Exposed as named export for advanced usage / testing.
 */
export async function replaceTrackInPeers(track, kind) {
  const pcs = Object.values(pcsRef || {});
  for (const pc of pcs) {
    try {
      // SPECIAL CASE: if track === null, attempt to stop outgoing video on this pc
      if (track === null) {
        try {
          const senders = pc.getSenders?.() || [];
          for (const s of senders) {
            try {
              if (s && s.track && s.track.kind === kind && typeof s.replaceTrack === "function") {
                try { await s.replaceTrack(null); } catch (e) { console.debug("[mediaController] replaceTrackInPeers(null) sender.replaceTrack failed", e); }
                try { if (s.track && !s.track.__isPlaceholder && typeof s.track.stop === 'function') s.track.stop(); } catch(e){/*ignore*/ }
              }
            } catch (e) {}
          }
        } catch (e) { /* ignore */ }
        try {
          const txs = pc.getTransceivers?.() || [];
          for (const tx of txs) {
            try {
              if (!tx) continue;
              if (tx.kind === kind || (tx.sender && tx.sender.track && tx.sender.track.kind === kind)) {
                try { tx.direction = "recvonly"; } catch (e) { /* ignore */ }
              }
            } catch (e) {}
          }
        } catch (e) {}
        continue;
      }

      // 1) prefer existing sender
      let sender = null;
      try {
        sender = pc.getSenders?.().find((s) => s.track && s.track.kind === kind) ?? null;
      } catch (e) {
        sender = null;
        console.debug("[mediaController] replaceTrackInPeers: pc.getSenders threw:", e);
      }

      if (sender) {
        try {
          const res = sender.replaceTrack(track);
          if (res && typeof res.then === "function") await res;
          continue;
        } catch (err) {
          console.warn(`[mediaController] sender.replaceTrack failed for ${kind}:`, err);
          // try to stop the old sender track as a fallback
          try { if (sender.track && !sender.track.__isPlaceholder && typeof sender.track.stop === 'function') sender.track.stop(); } catch (e) {}
        }
      }

      // 2) find a transceiver matching kind
      let tx = null;
      try {
        const txs = pc.getTransceivers?.() || [];
        tx = txs.find((t) => {
          try {
            const sKind = t.sender && t.sender.track && t.sender.track.kind;
            const rKind = t.receiver && t.receiver.track && t.receiver.track.kind;
            return sKind === kind || rKind === kind || t.kind === kind;
          } catch (e) {
            return false;
          }
        }) ?? null;
      } catch (e) {
        tx = null;
        console.debug("[mediaController] replaceTrackInPeers: pc.getTransceivers threw:", e);
      }

      if (tx) {
        try {
          if (tx.direction !== "sendrecv" && typeof tx.direction !== "undefined") {
            try { tx.direction = "sendrecv"; } catch (e) { console.debug("[mediaController] could not set tx.direction to sendrecv", e); }
          }
          if (tx.sender && typeof tx.sender.replaceTrack === "function") {
            const res = tx.sender.replaceTrack(track);
            if (res && typeof res.then === "function") await res;
            continue;
          }
        } catch (err) {
          console.warn(`[mediaController] transceiver.replaceTrack failed for ${kind}:`, err);
        }
      }

      // 3) if there are zero transceivers, add deterministic defaults
      try {
        const existing = pc.getTransceivers?.() || [];
        if (existing.length === 0) {
          ensureDefaultTransceivers(pc, kind);
          const txs2 = pc.getTransceivers?.() || [];
          const tx2 = txs2.find((t) => {
            try {
              const sKind = t.sender && t.sender.track && t.sender.track.kind;
              const rKind = t.receiver && t.receiver.track && t.receiver.track.kind;
              return sKind === kind || rKind === kind || t.kind === kind;
            } catch (e) {
              return false;
            }
          }) ?? null;
          if (tx2 && tx2.sender && typeof tx2.sender.replaceTrack === "function") {
            try {
              const res = tx2.sender.replaceTrack(track);
              if (res && typeof res.then === "function") await res;
              continue;
            } catch (err) {
              console.warn(`[mediaController] post-default transceiver.replaceTrack failed for ${kind}:`, err);
            }
          }
        }
      } catch (e) {
        console.warn("[mediaController] replaceTrackInPeers: post-default transceiver handling failed:", e);
      }

      // 4) try addTransceiver(kind, {direction: 'sendrecv'})
      try {
        const newTx = pc.addTransceiver?.(kind, { direction: "sendrecv" });
        if (newTx && newTx.sender && typeof newTx.sender.replaceTrack === "function") {
          try {
            const res = newTx.sender.replaceTrack(track);
            if (res && typeof res.then === "function") await res;
            continue;
          } catch (err) {
            console.warn(`[mediaController] new transceiver.replaceTrack failed for ${kind}:`, err);
          }
        }
      } catch (err) {
        console.warn(`[mediaController] addTransceiver failed for ${kind}:`, err);
      }

      // 5) final fallback: addTrack
      if (track && localStream) {
        try {
          const newSender = pc.addTrack(track, localStream);
          // Defensive hygiene: remove/stop other stale senders of same kind to avoid accumulating senders.
          try {
            const allSenders = pc.getSenders?.() || [];
            allSenders.forEach((s) => {
              try {
                if (!s || s === newSender) return;
                if (s.track && s.track.kind === kind) {
                  try { if (typeof s.replaceTrack === 'function') s.replaceTrack(null); } catch(e){/*ignore*/ }
                  try {
                    if (!s.track.__isPlaceholder && typeof s.track.stop === 'function') s.track.stop();
                  } catch(e){/*ignore*/}
                }
              } catch(e){/*ignore*/}
            });
          } catch(e){
            console.debug("[mediaController] post-addTrack sender cleanup threw:", e);
          }
        } catch (err) {
          console.warn("[mediaController] pc.addTrack fallback failed:", err);
        }
      }
    } catch (err) {
      console.warn("[mediaController] replaceTrackInPeers error for pc:", err);
    }
  }
}

export function replaceLocalTrack(newTrack, kind) {
  if (!localStream) {
    console.warn("[mediaController] replaceLocalTrack: no localStream available");
    try { newTrack?.stop?.(); } catch (e) { console.warn("[mediaController] replaceLocalTrack: failed to stop newTrack when no localStream", e); }
    return;
  }

  try {
    const toRemove = localStream.getTracks().filter((t) => t.kind === kind);
    toRemove.forEach((t) => {
      try { t.stop(); } catch (err) { console.warn("[mediaController] replaceLocalTrack: stop old track failed", err); }
      try { localStream.removeTrack(t); } catch (err) { console.warn("[mediaController] replaceLocalTrack: removeTrack failed", err); }
    });
  } catch (err) {
    console.warn("[mediaController] replaceLocalTrack: error enumerating/removing existing tracks:", err);
  }

  try {
    // ensure newTrack has reference to its origin stream if possible
    try { newTrack._sourceStream = newTrack._sourceStream ?? null; } catch (e) {}
    localStream.addTrack(newTrack);
    attachTrackEndHandler(newTrack, kind);
  } catch (err) {
    console.warn("[mediaController] localStream.addTrack failed:", err);
    try { newTrack.stop(); } catch (e) { console.warn("[mediaController] failed to stop newTrack after addTrack error", e); }
  }

  if (localVideoEl && kind === "video") {
    try {
      // force reattach to ensure preview updates
      localVideoEl.srcObject = null;
      localVideoEl.srcObject = localStream;
      enforceVideoMirrorBehavior(localVideoEl, { mirror: !!localMirrorEnabled });
      safePlay(localVideoEl);
    } catch (err) {
      console.warn("[mediaController] replaceLocalTrack: failed to update localVideoEl.srcObject:", err);
    }
  }

  if (isSafari()) refreshSafariPreview();
}

export function stopAndRemoveTracks(kind) {
  if (!localStream) return;

  try {
    const toRemove = localStream.getTracks().filter((t) => t.kind === kind);
    toRemove.forEach((t) => {
      try { t.stop(); } catch (err) { console.warn("[mediaController] stopAndRemoveTracks: stop failed", err); }
      try { localStream.removeTrack(t); } catch (err) { console.warn("[mediaController] stopAndRemoveTracks: removeTrack failed", err); }
    });
  } catch (err) {
    console.warn("[mediaController] stopAndRemoveTracks top-level error:", err);
  }

  const pcs = Object.values(pcsRef || {});
  for (const pc of pcs) {
    try {
      const txs = (pc.getTransceivers && pc.getTransceivers()) || [];
      const matched = txs.filter((t) => {
        try {
          const sKind = t.sender && t.sender.track && t.sender.track.kind;
          const rKind = t.receiver && t.receiver.track && t.receiver.track.kind;
          return sKind === kind || rKind === kind || t.kind === kind;
        } catch (e) {
          return false;
        }
      });

      if (matched.length > 0) {
        matched.forEach((t) => {
          try {
            if (typeof t.direction !== "undefined") {
              try { t.direction = "recvonly"; } catch (e) { console.warn("[mediaController] stopAndRemoveTracks: setting t.direction failed", e); }
            }
            if (t.sender && typeof t.sender.replaceTrack === "function") {
              try { t.sender.replaceTrack(null); } catch (e) { console.warn("[mediaController] stopAndRemoveTracks: transceiver.replaceTrack(null) failed", e); }
            }
            // additional aggressive cleanup: stop sender.track if present and not placeholder
            try {
              if (t.sender && t.sender.track && !t.sender.track.__isPlaceholder && typeof t.sender.track.stop === 'function') {
                try { t.sender.track.stop(); } catch (e) { console.warn("[mediaController] stopAndRemoveTracks: stopping sender.track failed", e); }
              }
            } catch (e) { /* ignore */ }
          } catch (err) {
            console.warn("[mediaController] stopAndRemoveTracks transceiver adjustment failed:", err);
          }
        });
        continue;
      }

      try {
        const senders = (pc.getSenders && pc.getSenders()) || [];
        const sendersToReplace = senders.filter((s) => s.track && s.track.kind === kind);
        sendersToReplace.forEach((s) => {
          try {
            try { s.replaceTrack(null); } catch (err) { console.warn("[mediaController] stopAndRemoveTracks: sender.replaceTrack(null) failed", err); }
            try {
              if (s.track && !s.track.__isPlaceholder && typeof s.track.stop === 'function') {
                s.track.stop();
              }
            } catch (e) { console.warn("[mediaController] stopAndRemoveTracks: stopping sender.track failed", e); }
          } catch (err) {
            /* ignore per-sender error */
          }
        });
      } catch (err) {
        console.warn("[mediaController] stopAndRemoveTracks (pc senders) error:", err);
      }
    } catch (err) {
      console.warn("stopAndRemoveTracks (pc loop) error:", err);
    }
  }

  // run external cleaners as well (best-effort)
  try { _runExternalCleaners(kind); } catch (e) { console.warn("[mediaController] stopAndRemoveTracks: external cleaners threw", e); }

  if (isSafari()) {
    clearPreviewIfNoTracks();
  }

  if (localVideoEl) enforceVideoMirrorBehavior(localVideoEl, { mirror: !!localMirrorEnabled });
}

export function stopTransceivers(kind) {
  const pcs = Object.values(pcsRef || {});
  for (const pc of pcs) {
    try {
      const txs = (pc.getTransceivers && pc.getTransceivers()) || [];
      const matched = txs.filter((tx) => {
        try {
          return (tx.sender && tx.sender.track && tx.sender.track.kind === kind)
            || (tx.receiver && tx.receiver.track && tx.receiver.track.kind === kind)
            || tx.kind === kind;
        } catch (e) {
          return false;
        }
      });

      matched.forEach((tx) => {
        try {
          if (typeof tx.direction !== "undefined") {
            try { tx.direction = "recvonly"; } catch (e) { console.warn("[mediaController] stopTransceivers: setting direction failed", e); }
          }
          try { if (tx.sender && typeof tx.sender.replaceTrack === "function") tx.sender.replaceTrack(null); } catch (e) { console.warn("[mediaController] stopTransceivers: sender.replaceTrack(null) failed", e); }
          // Do not call tx.stop() by default — it's destructive on some UA implementations
        } catch (err) {
          console.warn("stopTransceivers error on a transceiver:", err);
        }
      });
    } catch (err) {
      console.warn("stopTransceivers error for pc:", err);
    }
  }
}

function clearPreviewIfNoTracks() {
  if (!isSafari() || !localStream || !localVideoEl) return;

  try {
    const activeTracks = localStream.getTracks().filter((t) => t.readyState === "live");
    if (activeTracks.length === 0) {
      try { localVideoEl.srcObject = null; } catch (err) {
        console.warn("Failed to clear Safari video preview:", err);
      }
    }
  } catch (err) {
    console.warn("clearPreviewIfNoTracks error:", err);
  }
}

function refreshSafariPreview() {
  if (!isSafari() || !localVideoEl) return;
  if (!localStream) {
    try { localVideoEl.srcObject = null; } catch (err) { console.warn("[mediaController] refreshSafariPreview: clearing srcObject failed", err); }
    return;
  }

  try {
    localVideoEl.srcObject = null;
    setTimeout(() => {
      try {
        // only set canonical localStream (not preview placeholder)
        if (!_placeholderStream) {
          localVideoEl.srcObject = localStream;
        }
        enforceVideoMirrorBehavior(localVideoEl, { mirror: !!localMirrorEnabled });
        safePlay(localVideoEl);
      } catch (err) {
        console.warn("Safari preview refresh failed:", err);
      }
    }, 0);
  } catch (err) {
    console.warn("refreshSafariPreview error:", err);
  }
}

/* ---------------- Exported wrappers for convenience ---------------- */

/**
 * Public wrappers so consumers can explicitly call controller cleanup flows.
 */
export function stopAllVideoAndCleanup() {
  _stopAllVideoAndCleanup();
}
export function forceReleaseEverything() {
  _forceReleaseEverything();
}
