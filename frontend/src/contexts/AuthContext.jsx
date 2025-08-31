
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
  process.env.REACT_APP_SUPPORTS_GLOBAL_MEETINGS === "false" ? false : true;

export const AuthProvider = ({ children }) => {
  const [userData, setUserData] = useState(null);
  const router = useNavigate();

  const logout = useCallback(
    (redirect = true) => {
      localStorage.removeItem("token");
      setUserData(null);
      delete client.defaults.headers.common["Authorization"];
      delete apiClient.defaults.headers.common["Authorization"];
      if (redirect) {
        try {
          router("/login");
        } catch (e) {

        }
      }
    },
    [router]
  );

  useEffect(() => {

    const reqInterceptor = client.interceptors.request.use(
      (config) => {
        const token = localStorage.getItem("token");
        config.headers = config.headers || {};
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;

          apiClient.defaults.headers.common["Authorization"] = `Bearer ${token}`;
        }
        console.debug(
          `[Axios Request] ${config.method?.toUpperCase()} ${config.url}`,
          { Authorization: config.headers.Authorization }
        );
        return config;
      },
      (err) => Promise.reject(err)
    );

    const resInterceptor = client.interceptors.response.use(
      (resp) => resp,
      (err) => {
        const status = err?.response?.status;
        if (status === 401) {
          console.warn("Received 401 — clearing token and redirecting to /login");
          logout(true);
        }
        return Promise.reject(err);
      }
    );

    return () => {
      client.interceptors.request.eject(reqInterceptor);
      client.interceptors.response.eject(resInterceptor);
    };
  }, [logout]);

  const handleRegister = async (name, username, password) => {
    try {
      const request = await client.post("/register", { name, username, password });
      if (request.status === httpStatus.CREATED) return request.data.message;
      return null;
    } catch (err) {
      throw err;
    }
  };

  const handleLogin = async (username, password) => {
    try {
      const request = await client.post("/login", { username, password });
      console.log("login response:", request.data);

      if (request.status === httpStatus.OK) {
        const token =
          request.data?.accessToken ??
          request.data?.token ??
          request.data?.data?.token ??
          request.data?.access_token;

        if (token) {
          localStorage.setItem("token", token);
          client.defaults.headers.common["Authorization"] = `Bearer ${token}`;
          apiClient.defaults.headers.common["Authorization"] = `Bearer ${token}`;
        } else {
          console.warn("Login response did not include a token:", request.data);
        }

        if (request.data.user) {
          setUserData(request.data.user);
        }

        router("/home");
        return request.data;
      }
      return null;
    } catch (err) {
      throw err;
    }
  };

  const getAuthHeaders = () => {
    const token = localStorage.getItem("token");
    return token ? { Authorization: `Bearer ${token}` } : {};
  };

  const getHistoryOfUser = async () => {
    try {
      const isAuth = !!localStorage.getItem("token");

      if (SUPPORTS_GLOBAL_MEETINGS) {
        try {
          const query = isAuth ? "/meetings?mine=true" : "/meetings";
          const resp = await apiClient.get(query, {
            headers: { "Content-Type": "application/json", ...(isAuth ? getAuthHeaders() : {}) },
          });
          if (resp && resp.data) {
            const body = resp.data;
            if (Array.isArray(body)) return body;
            if (Array.isArray(body.meetings)) return body.meetings;
            if (Array.isArray(body.data)) return body.data;
            const foundArray = Object.values(body).find((v) => Array.isArray(v));
            if (foundArray) return foundArray;
            return [];
          }
        } catch (err) {
          const status = err?.response?.status;
          if (status === 404) {
            console.debug('/meetings not found (404) — will try users-scoped endpoints');
          } else {
            console.debug("/meetings attempt failed or empty:", err?.response?.status ?? err);
          }
        }
      } else {
        console.debug("SUPPORTS_GLOBAL_MEETINGS is false — skipping /meetings attempt");
      }

      try {
        const usersQuery = isAuth ? "/users/meetings?mine=true" : "/users/meetings";
        const respUsers = await apiClient.get(usersQuery, {
          headers: { "Content-Type": "application/json", ...(isAuth ? getAuthHeaders() : {}) },
        });
        if (respUsers && respUsers.data) {
          const body = respUsers.data;
          if (Array.isArray(body)) return body;
          if (Array.isArray(body.meetings)) return body.meetings;
          if (Array.isArray(body.data)) return body.data;
          const found = Object.values(body).find((v) => Array.isArray(v));
          if (found) return found;
        }
      } catch (err) {
        console.debug("/users/meetings attempt failed or empty:", err?.response?.status ?? err);
      }
      try {
        const request = await client.get("/get_all_activity");
        const payload = request?.data ?? {};
        const items = Array.isArray(payload)
          ? payload
          : Array.isArray(payload.data)
          ? payload.data
          : Array.isArray(payload.history)
          ? payload.history
          : payload.meetings ?? [];
        if (Array.isArray(items) && items.length > 0) {
          return items;
        }
      } catch (err) {
        console.debug("get_all_activity failed or empty:", err?.response?.status ?? err);
      }

      try {
        const raw = localStorage.getItem("meeting_history_v1");
        if (!raw) return [];
        const parsed = JSON.parse(raw);
        return Array.isArray(parsed) ? parsed : [];
      } catch (err) {
        console.warn("localStorage fallback read failed", err);
        return [];
      }
    } catch (outerErr) {
      console.error("getHistoryOfUser unexpected error:", outerErr);
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
        (payloadObj.meetingCode ? `${window.location.origin}/room/${payloadObj.meetingCode}` : null),
    };

    const isAuth = !!localStorage.getItem("token");

    if (SUPPORTS_GLOBAL_MEETINGS) {
      try {
        const resp = await apiClient.post("/meetings", body, {
          headers: { "Content-Type": "application/json", ...(isAuth ? getAuthHeaders() : {}) },
        });
        if (resp && (resp.status === 200 || resp.status === 201)) {
          return resp.data ?? body;
        }
        console.warn("POST /meetings returned non-OK:", resp.status, resp.data);
      } catch (err) {
        const status = err?.response?.status;
        if (status === 404) {
          console.debug("POST /meetings not found (404) — will try /users/meetings");
        } else {
          console.warn("POST /meetings failed — will try users or user-scoped fallback", status ?? err);
        }
      }
    } else {
      console.debug("SUPPORTS_GLOBAL_MEETINGS is false — skipping POST /meetings attempt");
    }

    try {
      const respUsers = await apiClient.post("/users/meetings", body, {
        headers: { "Content-Type": "application/json", ...(isAuth ? getAuthHeaders() : {}) },
      });
      if (respUsers && (respUsers.status === 200 || respUsers.status === 201)) {
        return respUsers.data ?? body;
      }
      console.warn("POST /users/meetings returned non-OK:", respUsers.status, respUsers.data);
    } catch (err) {
      console.debug("POST /users/meetings failed or not present:", err?.response?.status ?? err);
    }

    try {
      const request = await client.post("/add_to_activity", {
        meeting_code: payloadObj.meetingCode,
      });
      return request.data ?? request;
    } catch (err) {
      console.warn("user-scoped add_to_activity failed — falling back to localStorage", err?.response?.status ?? err);
    }

    try {
      const key = "meeting_history_v1";
      const raw = localStorage.getItem(key);
      const arr = raw ? JSON.parse(raw) : [];
      if (payloadObj.meetingCode) {
        const idx = arr.findIndex((m) => m.meetingCode === payloadObj.meetingCode);
        const newEntry = {
          meetingCode: payloadObj.meetingCode,
          hostName: payloadObj.hostName || payloadObj.host || "Host",
          participants: payloadObj.participants || [],
          createdAt: payloadObj.createdAt || new Date().toISOString(),
          link:
            payloadObj.link ||
            (payloadObj.meetingCode ? `${window.location.origin}/room/${payloadObj.meetingCode}` : null),
        };
        if (idx >= 0) arr[idx] = newEntry;
        else arr.unshift(newEntry);
      } else {
        arr.unshift({
          meetingCode: payloadObj.meetingCode || `misc-${Date.now()}`,
          hostName: payloadObj.hostName || "Host",
          participants: payloadObj.participants || [],
          createdAt: payloadObj.createdAt || new Date().toISOString(),
          link: payloadObj.link || null,
        });
      }
      localStorage.setItem(key, JSON.stringify(arr.slice(0, 200)));
      return { success: true, source: "localStorage" };
    } catch (lsErr) {
      console.error("addToUserHistory fallback localStorage failed", lsErr);
      return { success: false, error: lsErr?.message ?? "unknown" };
    }
  };


  const addParticipant = async (meetingCode, participant) => {
    if (!meetingCode) throw new Error("meetingCode required");
    const isAuth = !!localStorage.getItem("token");
    const payload = typeof participant === "object" ? participant : { participant };


    try {
      const resp = await apiClient.post(
        `/meetings/${encodeURIComponent(meetingCode)}/participants`,
        payload,
        {
          headers: { "Content-Type": "application/json", ...(isAuth ? getAuthHeaders() : {}) },
        }
      );
      if (resp && (resp.status === 200 || resp.status === 201)) {
        return resp.data ?? resp;
      }
      console.warn("POST /meetings/:code/participants returned non-OK", resp.status, resp.data);
    } catch (err) {
      console.debug("POST /meetings/:code/participants failed:", err?.response?.status ?? err);
    }

    try {
      const resp = await apiClient.post(
        "/meetings/add_participant",
        { meetingCode, ...payload },
        {
          headers: { "Content-Type": "application/json", ...(isAuth ? getAuthHeaders() : {}) },
        }
      );
      if (resp && (resp.status === 200 || resp.status === 201)) {
        return resp.data ?? resp;
      }
      console.warn("POST /meetings/add_participant returned non-OK", resp.status, resp.data);
    } catch (err) {
      console.debug("POST /meetings/add_participant failed:", err?.response?.status ?? err);
    }

    throw new Error("Unable to add participant: server endpoints failed or not available");
  };

  const data = {
    userData,
    setUserData,
    logout,
    addToUserHistory,
    getHistoryOfUser,
    handleRegister,
    handleLogin,
    addParticipant,
  };

  return <AuthContext.Provider value={data}>{children}</AuthContext.Provider>;
};