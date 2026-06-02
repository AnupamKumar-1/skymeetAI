import axios from "axios";
import httpStatus from "http-status";
import { createContext, useState, useEffect, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import server from "../environment";

export const AuthContext = createContext({});

const client = axios.create({
  baseURL: `${server}/api/v1/users`,
});

const apiClient = axios.create({
  baseURL: `${server}/api/v1`,
  timeout: 10_000,
});

const SUPPORTS_GLOBAL_MEETINGS =
  import.meta.env.VITE_SUPPORTS_GLOBAL_MEETINGS === "false" ? false : true;

function readToken() {
  const t = localStorage.getItem("token")?.trim();
  return t && t !== "undefined" && t !== "null" ? t : null;
}

function authHeader() {
  const t = readToken();
  return t ? { Authorization: `Bearer ${t}` } : {};
}

const attachToken = (config) => {
  const token = readToken();
  if (token) {
    config.headers = config.headers || {};
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
};

client.interceptors.request.use(attachToken);
apiClient.interceptors.request.use(attachToken);

function extractArray(body) {
  if (!body) return null;
  if (Array.isArray(body)) return body;
  if (Array.isArray(body.meetings)) return body.meetings;
  if (Array.isArray(body.data)) return body.data;
  const found = Object.values(body).find((v) => Array.isArray(v));
  return found ?? null;
}

function decodeTokenUser(token) {
  try {
    const payload = JSON.parse(atob(token.split(".")[1]));
    if (payload.exp && payload.exp * 1000 < Date.now()) return null;
    return {
      _id: payload.sub ?? payload.id ?? payload._id ?? null,
      username: payload.username ?? payload.user ?? null,
      email: payload.email ?? null,
      name: payload.name ?? payload.display ?? null,
    };
  } catch {
    return null;
  }
}

export const AuthProvider = ({ children }) => {
  const [userData, setUserData] = useState(null);
  const [authLoading, setAuthLoading] = useState(true);
  const router = useNavigate();

  const logout = useCallback(
    async (redirect = true, callServer = true) => {
      if (callServer) {
        try {
          await apiClient.post(
            "/auth/logout",
            {},
            { withCredentials: true, headers: { "Content-Type": "application/json" } }
          );
        } catch (err) {
          console.warn("logout: server-side logout failed, clearing client anyway", err?.response?.status ?? err.message);
        }
      }

      try {
        localStorage.removeItem("token");
      } catch (err) {
        console.warn("logout: failed to remove token from localStorage", err.message);
      }

      setUserData(null);

      if (redirect) {
        try {
          router("/auth");
        } catch {
          window.location.href = "/auth";
        }
      }
    },
    [router]
  );

  useEffect(() => {
    const token = readToken();

    if (!token) {
      setAuthLoading(false);
      return;
    }

    const decoded = decodeTokenUser(token);
    if (decoded) setUserData(decoded);

    apiClient
      .get("/users/me")
      .then((resp) => {
        const user = resp.data?.user ?? resp.data ?? null;
        if (user && (user._id || user.id || user.username)) {
          setUserData(user);
        }
      })
      .catch((err) => {
        const status = err?.response?.status;
        if (status === 401) {
          localStorage.removeItem("token");
          setUserData(null);
        }
      })
      .finally(() => {
        setAuthLoading(false);
      });
  }, []);

  useEffect(() => {
    const clientResId = client.interceptors.response.use(
      (resp) => resp,
      (err) => {
        if (err?.response?.status === 401) {
          logout(true, false);
        }
        return Promise.reject(err);
      }
    );

    const apiResId = apiClient.interceptors.response.use(
      (resp) => resp,
      (err) => {
        if (err?.response?.status === 401) {
          logout(true, false);
        }
        return Promise.reject(err);
      }
    );

    return () => {
      client.interceptors.response.eject(clientResId);
      apiClient.interceptors.response.eject(apiResId);
    };
  }, [logout]);

  const handleRegister = async (name, username, password, email) => {
    const resp = await client.post("/register", { name, username, password, email });
    if (resp.status === httpStatus.CREATED) return resp.data.message;
    return null;
  };

  const handleLogin = async (username, password) => {
    const resp = await client.post(
      "/login",
      { username, password },
      { withCredentials: true }
    );

    if (resp.status === httpStatus.OK) {
      const token =
        resp.data?.accessToken ??
        resp.data?.token ??
        resp.data?.data?.token ??
        resp.data?.access_token ??
        null;

      if (token) {
        localStorage.setItem("token", token);
      } else {
        console.warn("handleLogin: response did not include a token");
      }

      if (resp.data?.user) {
        setUserData(resp.data.user);
      } else if (token) {
        const decoded = decodeTokenUser(token);
        if (decoded) setUserData(decoded);
      }

      if (token) {
        apiClient
          .get("/users/me", { headers: { Authorization: `Bearer ${token}` } })
          .then((r) => {
            const user = r.data?.user ?? r.data ?? null;
            if (user && (user._id || user.id || user.username)) setUserData(user);
          })
          .catch(() => { });
      }

      router("/home");
      return resp.data;
    }

    return null;
  };

  const getHistoryOfUser = async () => {
    const isAuth = !!readToken();
    const headers = { "Content-Type": "application/json", ...authHeader() };

    // if (SUPPORTS_GLOBAL_MEETINGS) {
    //   try {
    //     const resp = await apiClient.get(
    //       isAuth ? "/meetings?mine=true" : "/meetings",
    //       { headers }
    //     );
    //     const items = extractArray(resp?.data);
    //     if (items) return items;
    //   } catch (err) {
    //     const status = err?.response?.status;
    //     if (status !== 404) {
    //       console.warn("getHistoryOfUser: /meetings failed", status ?? err.message);
    //     }
    //   }
    // }

    // try {
    //   const resp = await apiClient.get(
    //     isAuth ? "/users/meetings?mine=true" : "/users/meetings",
    //     { headers }
    //   );
    //   const items = extractArray(resp?.data);
    //   if (items) return items;
    // } catch (err) {
    //   console.warn("getHistoryOfUser: /users/meetings failed", err?.response?.status ?? err.message);
    // }

    try {
      const resp = await client.get("/get_all_activity");
      const items = extractArray(resp?.data);
      const serverItems = items ?? [];
      try {
        const raw = localStorage.getItem("meeting_history_v1");
        const localItems = raw ? JSON.parse(raw) : [];
        const merged = [...serverItems];
        const seenCodes = new Set(merged.map(m => (m?.meetingCode || m?.meeting_code || m?.code || "").toUpperCase()).filter(Boolean));
        for (const l of (Array.isArray(localItems) ? localItems : [])) {
          const code = (l?.meetingCode || l?.meeting_code || l?.code || "").toUpperCase();
          if (!code || !seenCodes.has(code)) merged.push(l);
        }
        return merged;
      } catch {
        return serverItems;
      }
    } catch (err) {
      console.warn("getHistoryOfUser: get_all_activity failed", err?.response?.status ?? err.message);
    }

    try {
      const raw = localStorage.getItem("meeting_history_v1");
      if (!raw) return [];
      const parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? parsed : [];
    } catch (err) {
      console.warn("getHistoryOfUser: localStorage fallback failed", err.message);
      return [];
    }
  };

  const addToUserHistory = async (meetingPayload) => {
    const payloadObj =
      typeof meetingPayload === "string"
        ? { meetingCode: meetingPayload, createdAt: new Date().toISOString() }
        : { ...meetingPayload };

    if (!payloadObj.meetingCode && payloadObj.meeting_code) {
      payloadObj.meetingCode = payloadObj.meeting_code;
    }

    const body = {
      meetingCode: payloadObj.meetingCode,
      hostName: payloadObj.hostName || payloadObj.host || payloadObj.host_name || null,
      participants: payloadObj.participants || payloadObj.attendees || [],
      createdAt: payloadObj.createdAt || new Date().toISOString(),
      link:
        payloadObj.link ||
        (payloadObj.meetingCode
          ? `${window.location.origin}/room/${payloadObj.meetingCode}`
          : null),
    };

    const headers = { "Content-Type": "application/json", ...authHeader() };

    if (SUPPORTS_GLOBAL_MEETINGS) {
      try {
        const resp = await apiClient.post("/meetings", body, { headers });
        if (resp?.status === 200 || resp?.status === 201) return resp.data ?? body;
        console.warn("addToUserHistory: POST /meetings returned non-OK status", resp?.status);
      } catch (err) {
        const status = err?.response?.status;
        if (status !== 404) {
          console.warn("addToUserHistory: POST /meetings failed", status ?? err.message);
        }
      }
    }

    try {
      const resp = await apiClient.post("/users/meetings", body, { headers });
      if (resp?.status === 200 || resp?.status === 201) return resp.data ?? body;
      console.warn("addToUserHistory: POST /users/meetings returned non-OK status", resp?.status);
    } catch (err) {
      console.warn("addToUserHistory: POST /users/meetings failed", err?.response?.status ?? err.message);
    }

    try {
      const resp = await client.post("/add_to_activity", {
        meeting_code: payloadObj.meetingCode,
      });
      return resp.data ?? resp;
    } catch (err) {
      console.warn("addToUserHistory: add_to_activity failed, falling back to localStorage", err?.response?.status ?? err.message);
    }

    try {
      const key = "meeting_history_v1";
      const raw = localStorage.getItem(key);
      const arr = raw ? JSON.parse(raw) : [];

      const newEntry = {
        meetingCode: payloadObj.meetingCode || `misc-${Date.now()}`,
        hostName: payloadObj.hostName || payloadObj.host || "Host",
        participants: payloadObj.participants || [],
        createdAt: payloadObj.createdAt || new Date().toISOString(),
        link:
          payloadObj.link ||
          (payloadObj.meetingCode
            ? `${window.location.origin}/room/${payloadObj.meetingCode}`
            : null),
      };

      if (payloadObj.meetingCode) {
        const idx = arr.findIndex((m) => m.meetingCode === payloadObj.meetingCode);
        if (idx >= 0) arr[idx] = newEntry;
        else arr.unshift(newEntry);
      } else {
        arr.unshift(newEntry);
      }

      localStorage.setItem(key, JSON.stringify(arr.slice(0, 200)));
      return { success: true, source: "localStorage" };
    } catch (err) {
      console.error("addToUserHistory: localStorage fallback failed", err.message);
      return { success: false, error: err?.message ?? "unknown" };
    }
  };

  const addParticipant = async (meetingCode, participant) => {
    if (!meetingCode) throw new Error("meetingCode required");

    const headers = { "Content-Type": "application/json", ...authHeader() };
    const payload = typeof participant === "object" ? participant : { participant };

    try {
      const resp = await apiClient.post(
        `/meetings/${encodeURIComponent(meetingCode)}/participants`,
        payload,
        { headers }
      );
      if (resp?.status === 200 || resp?.status === 201) return resp.data ?? resp;
      console.warn("addParticipant: /meetings/:code/participants returned non-OK status", resp?.status);
    } catch (err) {
      console.warn("addParticipant: /meetings/:code/participants failed", err?.response?.status ?? err.message);
    }

    try {
      const resp = await apiClient.post(
        "/meetings/add_participant",
        { meetingCode, ...payload },
        { headers }
      );
      if (resp?.status === 200 || resp?.status === 201) return resp.data ?? resp;
      console.warn("addParticipant: /meetings/add_participant returned non-OK status", resp?.status);
    } catch (err) {
      console.warn("addParticipant: /meetings/add_participant failed", err?.response?.status ?? err.message);
    }

    throw new Error("addParticipant: all endpoints failed");
  };

  return (
    <AuthContext.Provider
      value={{
        userData,
        setUserData,
        authLoading,
        logout,
        addToUserHistory,
        getHistoryOfUser,
        handleRegister,
        handleLogin,
        addParticipant,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export { apiClient };