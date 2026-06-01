import { useState, useCallback, useRef, useEffect } from "react";
import SERVER from "../environment";

const SERVER_BASE = import.meta.env.VITE_SERVER_URL || SERVER || "http://localhost:8000";
const API = `${SERVER_BASE}/api/v1/rag`;

export function useRag(transcriptIdOrCode) {
    const [indexStatus, setIndexStatus] = useState("unknown");
    const [history, setHistory] = useState([]);
    const [loading, setLoading] = useState(false);
    const [indexing, setIndexing] = useState(false);
    const [error, setError] = useState(null);
    const sessionIdRef = useRef(null);
    const pollTimerRef = useRef(null);
    const mountedRef = useRef(true);

    useEffect(() => {
        mountedRef.current = true;
        return () => {
            mountedRef.current = false;
            if (pollTimerRef.current) clearTimeout(pollTimerRef.current);
        };
    }, []);

    useEffect(() => {
        if (pollTimerRef.current) clearTimeout(pollTimerRef.current);
        setIndexStatus("unknown");
        setHistory([]);
        setError(null);
        setIndexing(false);
        setLoading(false);
        sessionIdRef.current = null;
    }, [transcriptIdOrCode]);

    const getHeaders = useCallback(() => {
        const token = localStorage.getItem("token");
        return {
            "Content-Type": "application/json",
            ...(token ? { Authorization: `Bearer ${token}` } : {}),
        };
    }, []);

    const pollUntilReady = useCallback((id, attempt = 0) => {
        if (attempt >= 25) {
            if (mountedRef.current) {
                setIndexing(false);
                setIndexStatus("error");
                setError("Indexing timed out");
            }
            return;
        }

        pollTimerRef.current = setTimeout(async () => {
            if (!mountedRef.current) return;
            try {
                const res = await fetch(`${API}/${id}/index`, {
                    headers: getHeaders(),
                    credentials: "include",
                });
                if (res.status === 403 || res.status === 401) {
                    if (mountedRef.current) {
                        setIndexing(false);
                        setIndexStatus("error");
                        setError("Not authorized to access this transcript");
                    }
                    return;
                }
                const data = await res.json();
                if (!mountedRef.current) return;

                if (data?.indexStatus === "ready") {
                    setIndexStatus("ready");
                    setIndexing(false);
                    setError(null);
                } else if (data?.indexStatus === "no_content") {
                    setIndexStatus("no_content");
                    setIndexing(false);
                } else if (data?.indexStatus === "failed" || data?.indexStatus === "error") {
                    setIndexStatus("failed");
                    setIndexing(false);
                    setError("Indexing failed on server. Please try again.");
                } else {
                    pollUntilReady(id, attempt + 1);
                }
            } catch {
                if (mountedRef.current) pollUntilReady(id, attempt + 1);
            }
        }, 5000);
    }, [getHeaders]);

    const index = useCallback(async () => {
        if (!transcriptIdOrCode) return;
        if (pollTimerRef.current) clearTimeout(pollTimerRef.current);
        setIndexing(true);
        setIndexStatus("indexing");
        setError(null);

        try {
            const res = await fetch(`${API}/${transcriptIdOrCode}/index`, {
                method: "POST",
                headers: getHeaders(),
                credentials: "include",
            });
            const data = await res.json();
            if (!mountedRef.current) return data;

            if (data?.indexStatus === "ready") {
                setIndexStatus("ready");
                setIndexing(false);
            } else if (data?.indexStatus === "no_content") {
                setIndexStatus("no_content");
                setIndexing(false);
            } else {
                setIndexStatus("indexing");
                pollUntilReady(transcriptIdOrCode, 0);
            }
            return data;
        } catch (err) {
            if (mountedRef.current) {
                setError(err.message);
                setIndexing(false);
                setIndexStatus("error");
            }
        }
    }, [transcriptIdOrCode, getHeaders, pollUntilReady]);

    const query = useCallback(async (question) => {
        if (!transcriptIdOrCode || !question?.trim()) return;
        setLoading(true);
        setError(null);

        setHistory((prev) => [...prev, { role: "user", content: question }, { role: "assistant", content: "", sources: [] }]);

        try {
            const res = await fetch(`${API}/${transcriptIdOrCode}/query`, {
                method: "POST",
                headers: getHeaders(),
                credentials: "include",
                body: JSON.stringify({ question }),
            });

            if (!res.ok) {
                const data = await res.json().catch(() => ({}));
                throw new Error(data.message || "Query failed");
            }

            let sseBuffer = "";
            const reader = res.body.getReader();
            const decoder = new TextDecoder();

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                sseBuffer += decoder.decode(value, { stream: true });
                const lines = sseBuffer.split("\n");
                sseBuffer = lines.pop() ?? "";

                for (const line of lines) {
                    const trimmed = line.trim();
                    if (!trimmed.startsWith("data: ")) continue;
                    const raw = trimmed.slice(6).trim();
                    if (raw === "[DONE]") continue;
                    try {
                        const data = JSON.parse(raw);
                        if (data.sources !== undefined) {
                            sessionIdRef.current = data.sessionId || null;
                            setHistory(prev => {
                                const newH = [...prev];
                                newH[newH.length - 1].sources = data.sources || [];
                                return newH;
                            });
                        }
                        if (data.token) {
                            setHistory(prev => {
                                const newH = [...prev];
                                newH[newH.length - 1].content += data.token;
                                return newH;
                            });
                        }
                    } catch (_) { }
                }
            }
        } catch (err) {
            setError(err.message);
            setHistory((prev) => prev.slice(0, -2));
        } finally {
            setLoading(false);
        }
    }, [transcriptIdOrCode, getHeaders]);

    const clearSession = useCallback(async () => {
        if (!transcriptIdOrCode) return;
        try {
            await fetch(`${API}/${transcriptIdOrCode}/session`, {
                method: "DELETE",
                headers: getHeaders(),
                credentials: "include",
            });
        } catch { }
        setHistory([]);
        setError(null);
        sessionIdRef.current = null;
    }, [transcriptIdOrCode, getHeaders]);

    return {
        indexStatus,
        indexing,
        history,
        loading,
        error,
        index,
        query,
        clearSession,
    };
}