// In a production build (e.g. Replit), the API is served from the same
// origin as the SPA, so the base URL is empty. In local dev the Vite
// server runs on a different port, so fall back to localhost:8000.
const API_BASE =
  import.meta.env.VITE_API_BASE ?? (import.meta.env.DEV ? "http://localhost:8000" : "");

async function request(path, { method = "GET", body, adminKey } = {}) {
  const headers = { "Content-Type": "application/json" };
  if (adminKey) headers["X-Admin-Key"] = adminKey;
  const res = await fetch(`${API_BASE}${path}`, {
    method,
    headers,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status}: ${text}`);
  }
  if (res.headers.get("content-type")?.includes("application/json")) {
    return res.json();
  }
  return res;
}

export const api = {
  courseByToken: (token) => request(`/api/public/courses/${token}`),
  teeTimes: (token, date) =>
    request(`/api/public/courses/${token}/tee-times${date ? `?date=${date}` : ""}`),
  register: (payload) => request(`/api/public/register`, { method: "POST", body: payload }),

  gallery: (token) => request(`/api/gallery/${token}`),
  flagClip: (token, clipId, note) =>
    request(`/api/gallery/${token}/clips/${clipId}/flag`, { method: "POST", body: { note } }),

  listCourses: (key) => request(`/api/admin/courses`, { adminKey: key }),
  createCourse: (key, payload) => request(`/api/admin/courses`, { method: "POST", body: payload, adminKey: key }),
  stats: (key) => request(`/api/admin/stats`, { adminKey: key }),
  flaggedClips: (key) => request(`/api/admin/flagged-clips`, { adminKey: key }),
  listHIO: (key, status) =>
    request(`/api/admin/hio${status ? `?status=${status}` : ""}`, { adminKey: key }),
  hioDetail: (key, id) => request(`/api/admin/hio/${id}`, { adminKey: key }),
  hioDecide: (key, id, action, reviewer, note) =>
    request(`/api/admin/hio/${id}/decision`, {
      method: "POST",
      body: { action, reviewer, note },
      adminKey: key,
    }),
  courseQrUrl: (token) => `${API_BASE}/api/public/courses/${token}/qr.png`,
  simulateRound: (key, teeTimeId, includeHio) =>
    request(`/api/webhooks/debug/simulate-round`, {
      method: "POST",
      body: { tee_time_id: teeTimeId, include_hio: includeHio },
      adminKey: key,
    }),
};

export { API_BASE };
