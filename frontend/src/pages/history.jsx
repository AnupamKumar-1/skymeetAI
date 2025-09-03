import React, { useContext, useEffect, useState } from 'react';
import { AuthContext } from '../contexts/AuthContext';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Card,
  CardContent,
  Typography,
  IconButton,
  CircularProgress,
  Button,
  Link as MuiLink,
  Collapse,
  Tooltip,
  Avatar,
  Divider,
  Snackbar,
  Alert,
  Chip,
  Stack,
  Grid
} from '@mui/material';
import HomeIcon from '@mui/icons-material/Home';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import OpenInNewIcon from '@mui/icons-material/OpenInNew';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import PersonIcon from '@mui/icons-material/Person';

export default function History() {
  const { getHistoryOfUser, userData } = useContext(AuthContext);
  const [meetings, setMeetings] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expanded, setExpanded] = useState({});
  const routeTo = useNavigate();

  const [snackOpen, setSnackOpen] = useState(false);
  const [snackMsg, setSnackMsg] = useState('');
  const [snackSeverity, setSnackSeverity] = useState('success');

  // UI state to expand a meeting's participant list beyond collapse for very large lists
  const [showAllParticipantsFor, setShowAllParticipantsFor] = useState({});

  // helper: safe parse date to millis (invalid -> 0)
  const toMillis = (s) => {
    if (!s) return 0;
    const t = new Date(s).getTime();
    return Number.isFinite(t) ? t : 0;
  };

  // helper: consider names trivial/generic — moved to component scope so both useEffect and render can use it
  const isTrivialName = (n) => {
    if (!n) return true;
    const s = String(n).trim().toLowerCase();
    if (!s) return true;
    const trivial = ['guest', 'participant', 'host', 'unknown', 'user'];
    if (trivial.includes(s)) return true;
    if (s.length <= 2) return true;
    return false;
  };

  useEffect(() => {
    let mounted = true;

    const toArrayShape = (res) => {
      if (!res) return [];
      if (Array.isArray(res)) return res;
      if (Array.isArray(res.meetings)) return res.meetings;
      if (Array.isArray(res.data)) return res.data;
      return [];
    };

    const readLocalFallback = () => {
      try {
        const raw = localStorage.getItem('meeting_history_v1');
        if (!raw) return [];
        const parsed = JSON.parse(raw);
        return Array.isArray(parsed) ? parsed : [];
      } catch (err) {
        console.warn('readLocalFallback failed', err);
        if (mounted) setError('Failed to read local meeting fallback (localStorage).');
        return [];
      }
    };

    // Normalize participant entries: turn strings into objects, dedupe participants by id/name/email
    const normalizeParticipants = (rawParts) => {
      const arr = Array.isArray(rawParts) ? rawParts : [];
      const out = [];
      const seen = new Set();
      for (let p of arr) {
        if (!p && p !== 0) continue;
        let obj;
        if (typeof p === 'string') {
          obj = { name: p };
        } else if (typeof p === 'object') {
          obj = p;
        } else {
          obj = { name: String(p) };
        }

        const key = obj?._id || obj?.id || obj?.username || obj?.email || (obj?.name ? String(obj.name).trim().toLowerCase() : null) || JSON.stringify(obj);
        if (!key) continue;
        if (seen.has(key)) continue;
        seen.add(key);
        out.push(obj);
      }
      return out;
    };

    const normalize = (m) => {
      const meetingCode = m?.meetingCode || m?.code || m?.room || m?.meeting_code || '';
      const createdAt = m?.createdAt || m?.created_at || m?.date || m?.created || m?.timestamp || '';
      const hostName =
        (m?.host && (m.host.name || m.host.display)) ||
        m?.hostName ||
        m?.host_name ||
        (typeof m?.host === 'string' ? m.host : null) ||
        (m?.host && typeof m.host === 'object' && (m.host.username || m.host.email)) ||
        (typeof m === 'string' ? m : 'Unknown');

      const rawParticipants = m?.participants || m?.attendees || m?.people || [];
      const participants = normalizeParticipants(rawParticipants);

      const link = m?.link || (meetingCode ? `${window.location.origin}/room/${encodeURIComponent(String(meetingCode).trim().toUpperCase())}` : null);
      const id = m?._id || m?.id || meetingCode || Math.random().toString(36).slice(2, 9);
      const hostId = m?.host?._id || m?.host?.id || m?.host_id || m?.hostId || null;

      return { id, meetingCode, createdAt, hostName, participants, link, raw: m, hostId };
    };

    // Dedupe + merge server + local fallback (server items preferred)
    const mergeServerAndLocal = (serverArr, localArr) => {
      const out = [];
      const seen = new Set();

      const keyFor = (item) => {
        const code = item?.meetingCode || item?.meeting_code || item?.code || item?.room;
        if (code) return String(code).trim().toUpperCase();
        if (item?.id) return `ID:${String(item.id).trim()}`;
        return `RAW:${JSON.stringify(item).slice(0, 100)}`;
      };

      (serverArr || []).forEach((s) => {
        const k = keyFor(s);
        if (!seen.has(k)) {
          out.push(s);
          seen.add(k);
        }
      });

      (localArr || []).forEach((l) => {
        const k = keyFor(l);
        if (!seen.has(k)) {
          out.push(l);
          seen.add(k);
        }
      });

      return out;
    };

    // Returns true if current user participates or hosts the meeting.
    const userMatchesParticipant = (user, participant) => {
      if (!user || !participant) return false;

      const pId = participant?._id || participant?.id || participant?.userId || participant?.user_id || null;
      const pUsername = participant?.username || participant?.userName || participant?.user || null;
      const pEmail = participant?.email || participant?.mail || null;
      const pName = typeof participant === 'string' ? participant : participant?.name || participant?.display || participant?.fullName || null;

      const uId = user?._id || user?.id || null;
      const uUsername = user?.username || user?.userName || null;
      const uEmail = user?.email || null;
      const uName = user?.name || user?.display || null;

      if (uId && pId && String(uId) === String(pId)) return true;
      if (uUsername && pUsername && String(uUsername).toLowerCase() === String(pUsername).toLowerCase()) return true;
      if (uEmail && pEmail && String(uEmail).toLowerCase() === String(pEmail).toLowerCase()) return true;

      // name-based match only as a last resort and require non-trivial names
      if (uName && pName && !isTrivialName(uName) && !isTrivialName(pName)) {
        if (String(uName).trim().toLowerCase() === String(pName).trim().toLowerCase()) return true;
      }

      return false;
    };

    const isUserInMeeting = (meet, user) => {
      if (!user) return false;

      // prefer host id checks
      if (meet.hostId && (user?._id && String(user._id) === String(meet.hostId))) return true;

      // raw host object checks (id, username, email)
      const raw = meet.raw || {};
      const rawHost = raw.host || raw.host_info || raw.hostId || raw.host_id || null;
      if (rawHost && typeof rawHost === 'object') {
        if (user._id && String(rawHost._id || rawHost.id) === String(user._id)) return true;
        if (user.username && rawHost.username && String(user.username).toLowerCase() === String(rawHost.username).toLowerCase()) return true;
        if (user.email && rawHost.email && String(user.email).toLowerCase() === String(rawHost.email).toLowerCase()) return true;
      }

      // fallback: participant matching using several heuristics
      const parts = meet.participants || [];
      for (let i = 0; i < parts.length; i++) {
        if (userMatchesParticipant(user, parts[i])) return true;
      }
      return false;
    };

    const fetchHistory = async () => {
      setLoading(true);
      setError(null);

      try {
        const res = await getHistoryOfUser();
        const serverArr = toArrayShape(res);
        const localArr = readLocalFallback();
        const merged = mergeServerAndLocal(serverArr, localArr);
        const normalized = (merged || []).map(normalize);

        // filter: only meetings where the current user is a participant/host.
        const filtered = userData ? normalized.filter((m) => isUserInMeeting(m, userData)) : [];

        // sort by createdAt robustly (malformed dates will be pushed to end)
        const mapped = (filtered || []).sort((a, b) => {
          const ta = toMillis(a.createdAt);
          const tb = toMillis(b.createdAt);
          return tb - ta;
        });

        if (mounted) setMeetings(mapped);

        // small debug hint if nothing found while userData exists
        if (mounted && userData && mapped.length === 0) {
          console.debug('History: no meetings matched current user after filtering. Normalized items:', normalized.length, 'Server+Local merged count:', merged.length);
        }
      } catch (err) {
        console.error('fetchHistory error:', err);
        if (mounted) setError(err?.message || 'Failed to load history');
      } finally {
        if (mounted) setLoading(false);
      }
    };

    fetchHistory();

    // refresh when other tabs modify meeting_history_v1
    const onStorage = (ev) => {
      if (ev.key && ev.key === 'meeting_history_v1') {
        setTimeout(() => {
          if (mounted) fetchHistory();
        }, 60);
      }
    };

    // custom event hook: other parts of the app can dispatch `window.dispatchEvent(new Event('meeting_history_updated'))`
    const onCustomUpdate = () => {
      if (mounted) fetchHistory();
    };

    window.addEventListener('storage', onStorage);
    window.addEventListener('meeting_history_updated', onCustomUpdate);

    return () => {
      mounted = false;
      window.removeEventListener('storage', onStorage);
      window.removeEventListener('meeting_history_updated', onCustomUpdate);
    };
  }, [getHistoryOfUser, userData]);

  const formatDate = (dateString) => {
    if (!dateString) return 'Unknown';
    const date = new Date(dateString);
    if (Number.isNaN(date.getTime())) return 'Invalid date';
    const day = date.getDate().toString().padStart(2, '0');
    const month = (date.getMonth() + 1).toString().padStart(2, '0');
    const year = date.getFullYear();
    const hours = date.getHours().toString().padStart(2, '0');
    const mins = date.getMinutes().toString().padStart(2, '0');
    return `${day}/${month}/${year} ${hours}:${mins}`;
  };

  const participantName = (p) => {
    if (!p) return 'Guest';
    if (typeof p === 'string') return p;
    return p?.name || p?.display || p?.username || 'Guest';
  };

  const participantRole = (p) => {
    if (!p) return null;
    if (typeof p === 'string') return null;
    const r = p.role || p.roleName || p.meta?.role || p.meta?.roles?.[0] || null;
    if (!r) return null;
    return String(r).toLowerCase();
  };

  const initials = (name) => {
    if (!name || typeof name !== 'string') return 'G';
    const parts = name.trim().split(/\s+/).slice(0, 2);
    const i = parts.map(p => p[0]?.toUpperCase() ?? '').join('');
    return i || name.slice(0, 1).toUpperCase();
  };

  const copyLink = async (link) => {
    if (!link) return;
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(link);
      } else {
        const ta = document.createElement('textarea');
        ta.value = link;
        ta.style.position = 'fixed';
        ta.style.left = '-9999px';
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
      }
      setSnackMsg('Link copied to clipboard');
      setSnackSeverity('success');
      setSnackOpen(true);
    } catch (err) {
      console.error('copy failed', err);
      setSnackMsg('Failed to copy link');
      setSnackSeverity('error');
      setSnackOpen(true);
    }
  };

  const toggleExpand = (key) => {
    setExpanded(prev => ({ ...prev, [key]: !prev[key] }));
  };

  const buildLink = (m) => m?.link || (m?.meetingCode ? `${window.location.origin}/room/${encodeURIComponent(String(m.meetingCode).trim().toUpperCase())}` : null);

  const BG = '#020617';
  const CARD_BG = '#ffffff';
  const ACCENT = '#1976d2';
  const MUTED = '#6b7280';

  const roleChipProps = (role) => {
    if (!role) {
      return {
        label: 'Participant',
        sx: { bgcolor: 'transparent', color: MUTED, border: '1px solid rgba(0,0,0,0.06)', fontWeight: 600, fontSize: 12 }
      };
    }
    if (role === 'host') {
      return {
        label: 'Host',
        sx: { bgcolor: ACCENT, color: '#fff', fontWeight: 700, fontSize: 12 }
      };
    }
    if (role === 'speaker' || role === 'presenter') {
      return {
        label: 'Speaker',
        sx: { bgcolor: 'rgba(25,118,210,0.12)', color: ACCENT, border: `1px solid rgba(25,118,210,0.18)`, fontWeight: 700, fontSize: 12 }
      };
    }
    return {
      label: String(role).charAt(0).toUpperCase() + String(role).slice(1),
      sx: { bgcolor: 'rgba(0,0,0,0.04)', color: MUTED, fontWeight: 600, fontSize: 12 }
    };
  };

  const renderLoggedOutMessage = () => (
    <Box sx={{ mt: 3 }}>
      <Alert severity="info">
        You are not signed in. Sign in to view meetings you participated in.
      </Alert>
      <Box sx={{ mt: 2 }}>
        <Button variant="contained" onClick={() => routeTo('/login')}>Sign in</Button>
      </Box>
    </Box>
  );

  const renderNoParticipantHistory = () => (
    <Box sx={{ mt: 3 }}>
      <Alert severity="info">
        We couldn't find any meeting history where you participated or hosted.
      </Alert>
      <Typography sx={{ color: MUTED, mt: 1 }}>
        If you recently left a meeting, try refreshing this page. If you used another device or browser, check there as well.
      </Typography>
    </Box>
  );

  return (
    <Box sx={{ minHeight: '100vh', backgroundColor: BG, pb: 6 }}>
      <Box sx={{ maxWidth: 1100, mx: 'auto', px: 2, pt: 18 }}>
        <Box sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: 2,
          mb: 3
        }}>
          <Stack direction="row" spacing={1} alignItems="center">
            <IconButton
              aria-label="home"
              onClick={() => routeTo('/home')}
              sx={{
                color: CARD_BG,
                backgroundColor: 'transparent',
                border: '1px solid rgba(255,255,255,0.06)',
                '&:hover': { backgroundColor: 'rgba(255,255,255,0.02)' },
                width: 44,
                height: 44,
              }}
            >
              <HomeIcon />
            </IconButton>

            <Box>
              <Typography variant="h5" sx={{ color: '#e6eefc', fontWeight: 700, letterSpacing: '-0.2px' }}>
                Meeting history
              </Typography>
              <Typography variant="body2" sx={{ color: '#bbd6ff', mt: 0.25 }}>
                Meetings you joined or hosted
              </Typography>
            </Box>
          </Stack>

          <Chip
            label={`Total ${meetings.length}`}
            sx={{
              bgcolor: 'rgba(255,255,255,0.03)',
              color: '#e6eefc',
              border: '1px solid rgba(255,255,255,0.04)',
              fontWeight: 600
            }}
          />
        </Box>

        {loading && (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, color: '#cfe3ff', mb: 2 }}>
            <CircularProgress size={22} sx={{ color: ACCENT }} />
            <Typography sx={{ color: '#cfe3ff' }}>Loading history…</Typography>
          </Box>
        )}

        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        {!loading && !userData && renderLoggedOutMessage()}

        {!loading && userData && meetings.length === 0 && renderNoParticipantHistory()}

        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          {!loading && meetings.map((m) => {
            const key = m.id;
            const link = buildLink(m);
            const participantsRaw = Array.isArray(m.participants) ? m.participants : [];
            const hostName = m?.hostName ?? 'Unknown';

            // Card-level host detection: strict (prefer hostId, then raw.host username/email)
            const isHost = (() => {
              if (!userData) return false;

              const hostId = m.hostId || (m.raw && (m.raw.host?._id || m.raw.host?.id || m.raw.hostId || m.raw.host_id));
              if (hostId && userData._id && String(hostId) === String(userData._id)) return true;

              const rawHost = m.raw && m.raw.host;
              if (typeof rawHost === 'object') {
                if (rawHost.username && userData.username && String(rawHost.username).toLowerCase() === String(userData.username).toLowerCase()) return true;
                if (rawHost.email && userData.email && String(rawHost.email).toLowerCase() === String(userData.email).toLowerCase()) return true;
              }

              // last resort: exact display-name match but only if both names are non-trivial length (>2) and not generic
              if (m.hostName && userData.name && !isTrivialName(m.hostName) && !isTrivialName(userData.name)) {
                const hn = String(m.hostName).trim().toLowerCase();
                const un = String(userData.name).trim().toLowerCase();
                if (hn && un && hn === un && un.length > 2) return true;
              }

              return false;
            })();

            const PARTICIPANT_PREVIEW_LIMIT = 6;
            const showAll = !!showAllParticipantsFor[key];
            const visibleParticipants = showAll ? participantsRaw : participantsRaw.slice(0, PARTICIPANT_PREVIEW_LIMIT);

            return (
              <Card
                key={key}
                variant="outlined"
                sx={{
                  bgcolor: CARD_BG,
                  borderRadius: 2,
                  boxShadow: '0 8px 28px rgba(2,6,23,0.36)',
                  border: '1px solid rgba(255,255,255,0.02)',
                  overflow: 'visible',
                  transition: 'transform 180ms ease, box-shadow 180ms ease',
                  '&:hover': { transform: 'translateY(-6px)', boxShadow: '0 18px 40px rgba(2,6,23,0.46)' }
                }}
              >
                <CardContent sx={{ p: 2.5 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', gap: 2, alignItems: 'flex-start' }}>
                    <Box sx={{ minWidth: 0 }}>
                      <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 0.5 }}>
                        <Chip
                          label={m.meetingCode ? String(m.meetingCode).trim().toUpperCase() : '—'}
                          size="small"
                          sx={{
                            fontWeight: 700,
                            bgcolor: 'rgba(25,118,210,0.06)',
                            color: ACCENT,
                            border: `1px solid rgba(25,118,210,0.12)`
                          }}
                        />
                        <Typography variant="body2" sx={{ color: MUTED }}>
                          {formatDate(m.createdAt)}
                        </Typography>
                      </Stack>

                      <Stack direction="row" spacing={1} alignItems="center">
                        <Typography variant="subtitle1" sx={{ fontWeight: 700, color: '#111827' }}>
                          {hostName}
                        </Typography>
                        {isHost && (
                          <Chip
                            label="Host"
                            size="small"
                            sx={{ ml: 0.5, bgcolor: ACCENT, color: '#fff', fontWeight: 700, fontSize: 12 }}
                          />
                        )}
                      </Stack>

                      <Typography variant="body2" sx={{ color: MUTED, mt: 0.5 }}>
                        {participantsRaw.length} participant{participantsRaw.length !== 1 ? 's' : ''}
                      </Typography>

                      {link && (
                        <Stack direction="row" spacing={1} alignItems="center" sx={{ mt: 1 }}>
                          <MuiLink href={link} target="_blank" rel="noreferrer" underline="hover" sx={{ color: ACCENT, fontWeight: 600 }}>
                            Open meeting
                          </MuiLink>

                          <Tooltip title="Open in new tab">
                            <IconButton size="small" onClick={() => window.open(link, '_blank')} aria-label="open-new" sx={{ color: MUTED }}>
                              <OpenInNewIcon fontSize="small" />
                            </IconButton>
                          </Tooltip>

                          <Tooltip title="Copy link">
                            <IconButton size="small" onClick={() => copyLink(link)} aria-label="copy-link" sx={{ color: MUTED }}>
                              <ContentCopyIcon fontSize="small" />
                            </IconButton>
                          </Tooltip>
                        </Stack>
                      )}
                    </Box>

                    <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end' }}>
                      <Button
                        size="small"
                        startIcon={<PersonIcon />}
                        onClick={() => toggleExpand(key)}
                        endIcon={
                          <ExpandMoreIcon
                            sx={{
                              transform: expanded[key] ? 'rotate(180deg)' : 'rotate(0deg)',
                              transition: 'transform 200ms ease'
                            }}
                          />
                        }
                        sx={{
                          textTransform: 'none',
                          color: ACCENT,
                          borderRadius: 2,
                          fontWeight: 700,
                          px: 1.5,
                          '&:hover': { backgroundColor: 'rgba(25,118,210,0.06)' }
                        }}
                      >
                        Participants
                      </Button>

                      <Typography variant="caption" sx={{ color: MUTED, mt: 1 }}>
                        {formatDate(m.createdAt)}
                      </Typography>
                    </Box>
                  </Box>

                  <Collapse in={!!expanded[key]} timeout="auto" unmountOnExit>
                    <Divider sx={{ my: 1 }} />

                    <Box sx={{ mt: 1 }}>
                      <Grid container spacing={2}>
                        {participantsRaw.length === 0 && (
                          <Grid item xs={12}>
                            <Typography sx={{ color: MUTED }}>No participants recorded</Typography>
                          </Grid>
                        )}

                        {visibleParticipants.map((pRaw, idx) => {
                          const name = participantName(pRaw);
                          const role = participantRole(pRaw);
                          const chip = roleChipProps(role);

                          // Detect if this participant is host — strict checks first
                          const isParticipantHost = (() => {
                            // explicit role / flag first
                            if (role === 'host') return true;
                            if (pRaw && typeof pRaw === 'object' && (pRaw.isHost === true || pRaw.host === true)) return true;

                            // explicit id match against meeting hostId
                            const pId = pRaw?._id || pRaw?.id || pRaw?.userId || pRaw?.user_id || null;
                            const hostId = m.hostId || (m.raw && (m.raw.host?._id || m.raw.host?.id || m.raw.hostId || m.raw.host_id)) || null;
                            if (pId && hostId && String(pId) === String(hostId)) return true;

                            // username/email matching if raw host object present
                            const pUsername = pRaw?.username || pRaw?.user || null;
                            const pEmail = pRaw?.email || null;
                            if (pUsername && m.raw && typeof m.raw.host === 'object' && m.raw.host.username && String(pUsername).toLowerCase() === String(m.raw.host.username).toLowerCase()) return true;
                            if (pEmail && m.raw && typeof m.raw.host === 'object' && m.raw.host.email && String(pEmail).toLowerCase() === String(m.raw.host.email).toLowerCase()) return true;

                            // last resort: exact display-name match (only if hostName present and not trivial)
                            const pName = pRaw && (typeof pRaw === 'string' ? pRaw : (pRaw.name || pRaw.display));
                            if (pName && m.hostName && !isTrivialName(pName) && !isTrivialName(m.hostName)) {
                              const pn = String(pName).trim().toLowerCase();
                              const hn = String(m.hostName).trim().toLowerCase();
                              if (pn && hn && pn === hn) return true;
                            }

                            return false;
                          })();

                          const participantIsYou = (() => {
                            if (!userData) return false;

                            // id match
                            const pId = pRaw?._id || pRaw?.id || pRaw?.userId || pRaw?.user_id || null;
                            if (pId && userData._id && String(pId) === String(userData._id)) return true;

                            // username/email match
                            const pUsername = pRaw?.username || pRaw?.user || null;
                            if (pUsername && userData.username && String(pUsername).toLowerCase() === String(userData.username).toLowerCase()) return true;
                            const pEmail = pRaw?.email || null;
                            if (pEmail && userData.email && String(pEmail).toLowerCase() === String(userData.email).toLowerCase()) return true;

                            // Name-only match: very last resort and require that both names are not trivial
                            const pName = pRaw?.name || pRaw?.display || (typeof pRaw === 'string' ? pRaw : null);
                            if (pName && userData.name && !isTrivialName(pName) && !isTrivialName(userData.name)) {
                              const pn = String(pName).trim().toLowerCase();
                              const un = String(userData.name).trim().toLowerCase();
                              if (pn && un && pn === un) return true;
                            }

                            return false;
                          })();

                          const tileKey = pRaw?._id || pRaw?.id || `${(name || 'p')}-${idx}`;

                          return (
                            <Grid key={`${key}-p-${tileKey}`} item xs={12} sm={6}>
                              <Box sx={{
                                display: 'flex',
                                gap: 1.25,
                                alignItems: 'center',
                                bgcolor: 'rgba(0,0,0,0.02)',
                                p: 1,
                                borderRadius: 1,
                                border: '1px solid rgba(0,0,0,0.04)'
                              }}>
                                <Avatar sx={{ bgcolor: ACCENT, width: 40, height: 40, fontSize: 14 }}>
                                  {initials(name)}
                                </Avatar>

                                <Box sx={{ flex: 1, minWidth: 0 }}>
                                  <Stack direction="row" spacing={1} alignItems="center">
                                    <Typography sx={{ fontWeight: 700, fontSize: 14, color: '#111827' }} noWrap>
                                      {name}
                                    </Typography>

                                    {participantIsYou && (
                                      <Chip size="small" label="You" sx={{ fontSize: 11, ml: 0.5 }} />
                                    )}

                                    {isParticipantHost && (
                                      <Chip size="small" label="Host" sx={{ bgcolor: ACCENT, color: '#fff', fontWeight: 700, fontSize: 11, ml: 0.5 }} />
                                    )}
                                  </Stack>

                                  <Stack direction="row" spacing={1} alignItems="center" sx={{ mt: 0.5 }}>
                                    <Chip
                                      size="small"
                                      {...chip}
                                      sx={{ borderRadius: 1.25, height: 26, ...chip.sx }}
                                    />
                                    {typeof pRaw !== 'string' && pRaw?.meta?.title && (
                                      <Typography variant="caption" sx={{ color: MUTED }}>
                                        {pRaw.meta.title}
                                      </Typography>
                                    )}
                                  </Stack>
                                </Box>
                              </Box>
                            </Grid>
                          );
                        })}

                        {participantsRaw.length > PARTICIPANT_PREVIEW_LIMIT && (
                          <Grid item xs={12}>
                            <Box sx={{ display: 'flex', justifyContent: 'center', mt: 1 }}>
                              <Button
                                size="small"
                                variant="outlined"
                                onClick={() => setShowAllParticipantsFor(prev => ({ ...prev, [m.id]: !prev[m.id] }))}
                              >
                                {showAllParticipantsFor[m.id] ? `Show less` : `Show all ${participantsRaw.length} participants`}
                              </Button>
                            </Box>
                          </Grid>
                        )}
                      </Grid>
                    </Box>
                  </Collapse>
                </CardContent>
              </Card>
            );
          })}
        </Box>
      </Box>

      <Snackbar open={snackOpen} autoHideDuration={3000} onClose={() => setSnackOpen(false)} anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}>
        <Alert onClose={() => setSnackOpen(false)} severity={snackSeverity} sx={{ width: '100%' }}>
          {snackMsg}
        </Alert>
      </Snackbar>
    </Box>
  );
}


