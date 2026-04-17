// In a production build (e.g. Replit), the API is served from the same
// origin as the SPA, so the base URL is empty. In local dev the Vite
// server runs on a different port, so fall back to localhost:8000.
const API_BASE =
  import.meta.env.VITE_API_BASE ?? (import.meta.env.DEV ? "http://localhost:8000" : "");

async function request(path, { method = "GET", body, adminPassword } = {}) {
  const headers = { "Content-Type": "application/json" };
  if (adminPassword) headers["X-Admin-Password"] = adminPassword;
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
  register: async (formData) => {
    const res = await fetch(`${API_BASE}/api/public/register`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`${res.status}: ${text}`);
    }
    return res.json();
  },
  selfieUrl: (path) => `${API_BASE}/uploads/${path}`,

  gallery: (token) => request(`/api/gallery/${token}`),
  flagClip: (token, clipId, note) =>
    request(`/api/gallery/${token}/clips/${clipId}/flag`, { method: "POST", body: { note } }),

  listCourses: (key) => request(`/api/admin/courses`, { adminPassword: key }),
  createCourse: (key, payload) => request(`/api/admin/courses`, { method: "POST", body: payload, adminPassword: key }),
  updateCourse: (key, id, payload) => request(`/api/admin/courses/${id}`, { method: "PATCH", body: payload, adminPassword: key }),
  stats: (key) => request(`/api/admin/stats`, { adminPassword: key }),
  flaggedClips: (key) => request(`/api/admin/flagged-clips`, { adminPassword: key }),
  listParticipants: (key, params = {}) => {
    const qs = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => {
      if (v !== undefined && v !== null && v !== "") qs.set(k, v);
    });
    return request(`/api/admin/participants${qs.toString() ? `?${qs}` : ""}`, { adminPassword: key });
  },
  participantClips: (key, id) => request(`/api/admin/participants/${id}/clips`, { adminPassword: key }),
  resendGallery: (key, id) =>
    request(`/api/admin/participants/${id}/resend-gallery`, { method: "POST", adminPassword: key }),
  listHIO: (key, status) =>
    request(`/api/admin/hio${status ? `?status=${status}` : ""}`, { adminPassword: key }),
  hioDetail: (key, id) => request(`/api/admin/hio/${id}`, { adminPassword: key }),
  hioDecide: (key, id, action, reviewer, note) =>
    request(`/api/admin/hio/${id}/decision`, {
      method: "POST",
      body: { action, reviewer, note },
      adminPassword: key,
    }),
  courseQrUrl: (token) => `${API_BASE}/api/public/courses/${token}/qr.png`,
  simulateRound: (key, teeTimeId, includeHio) =>
    request(`/api/webhooks/debug/simulate-round`, {
      method: "POST",
      body: { tee_time_id: teeTimeId, include_hio: includeHio },
      adminPassword: key,
    }),
};

export { API_BASE };
