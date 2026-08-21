export function isValidToken(token) {
  return Boolean(token) && token !== "undefined" && token !== "null";
}

export function getStoredToken() {
  const token = localStorage.getItem("token");
  const trimmed = token ? token.trim() : token;
  return isValidToken(trimmed) ? trimmed : null;
}
