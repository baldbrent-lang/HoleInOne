// In a production build (e.g. Replit), the API is served from the same
// origin as the SPA, so the base URL is empty. In local dev the Vite
// server runs on a different port, so fall back to localhost:8000.
const API_BASE =
  import.meta.env.VITE_API_BASE ?? (import.meta.env.DEV ? "http://localhost:8000" : "");

const USER_TOKEN_STORAGE = "golfreelz.userToken";
const OPERATOR_TOKEN_STORAGE = "golfreelz.operatorToken";

export function getUserToken() {
  return localStorage.getItem(USER_TOKEN_STORAGE) || "";
}
export function setUserToken(token) {
  if (token) localStorage.setItem(USER_TOKEN_STORAGE, token);
  else localStorage.removeItem(USER_TOKEN_STORAGE);
}

export function getOperatorToken() {
  return localStorage.getItem(OPERATOR_TOKEN_STORAGE) || "";
}
export function setOperatorToken(token) {
  if (token) localStorage.setItem(OPERATOR_TOKEN_STORAGE, token);
  else localStorage.removeItem(OPERATOR_TOKEN_STORAGE);
}

async function operatorRequest(path) {
  const token = getOperatorToken();
  if (!token) throw new Error("operator login required");
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
  return res.json();
}

async function request(path, { method = "GET", body, adminPassword, auth = true, timeoutMs } = {}) {
  const headers = { "Content-Type": "application/json" };
  if (adminPassword) headers["X-Admin-Password"] = adminPassword;
  if (auth) {
    const t = getUserToken();
    if (t) headers["Authorization"] = `Bearer ${t}`;
  }
  const controller = timeoutMs ? new AbortController() : null;
  const timer = controller ? setTimeout(() => controller.abort(), timeoutMs) : null;
  let res;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      method,
      headers,
      body: body ? JSON.stringify(body) : undefined,
      signal: controller?.signal,
    });
  } catch (e) {
    if (e.name === "AbortError") {
      throw new Error(`request timed out after ${Math.round(timeoutMs / 1000)}s`);
    }
    throw e;
  } finally {
    if (timer) clearTimeout(timer);
  }
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
  listPublicCourses: () => request(`/api/public/courses`),
  stripeConfig: () => request(`/api/public/stripe-config`, { auth: false }),
  inviteInfo: (token) => request(`/api/public/invite/${token}`, { auth: false }),
  inviteSelfie: async (token, file) => {
    const fd = new FormData();
    fd.append("selfie", file, file.name || "selfie.jpg");
    const res = await fetch(`${API_BASE}/api/public/invite/${token}/selfie`, {
      method: "POST", body: fd,
    });
    if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
    return res.json();
  },
  listShowcase: () => request(`/api/public/showcase`),
  publicStats: () => request(`/api/public/stats`, { auth: false }),
  leaderboards: (limit) => request(`/api/public/leaderboards${limit ? `?limit=${limit}` : ""}`, { auth: false }),
  contests: () => request(`/api/public/contests`, { auth: false }),
  broadcastNext: (viewerId, courseId) => {
    const qs = new URLSearchParams({ viewer_id: viewerId });
    if (courseId) qs.set("course_id", courseId);
    return request(`/api/broadcast/next?${qs}`, { auth: false });
  },
  tagHighlight: (key, clipId, tag) =>
    request(`/api/broadcast/admin/clips/${clipId}/highlight${tag ? `?tag=${encodeURIComponent(tag)}` : ""}`, {
      method: "POST", adminPassword: key,
    }),
  untagHighlight: (key, clipId) =>
    request(`/api/broadcast/admin/clips/${clipId}/highlight`, {
      method: "DELETE", adminPassword: key,
    }),
  autoTagHighlights: (key) =>
    request(`/api/broadcast/admin/auto-tag`, { method: "POST", adminPassword: key }),
  publicProfile: (userId) => request(`/api/public/profile/${userId}`, { auth: false }),
  setOperatorPassword: (key, courseId, password) =>
    request(`/api/admin/courses/${courseId}/operator-password`, {
      method: "POST", body: { password }, adminPassword: key,
    }),
  operatorLogin: ({ course_token, password }) =>
    fetch(`${API_BASE}/api/operator/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ course_token, password }),
    }).then(async (r) => {
      if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
      return r.json();
    }),
  operatorMe: () => operatorRequest(`/api/operator/me`),
  operatorDashboard: () => operatorRequest(`/api/operator/dashboard`),
  operatorParticipants: () => operatorRequest(`/api/operator/participants`),
  adminListShowcase: (key) => request(`/api/admin/showcase`, { adminPassword: key }),
  updateShowcase: (key, position, payload) =>
    request(`/api/admin/showcase/${position}`, {
      method: "PATCH", body: payload, adminPassword: key,
    }),
  clearShowcase: (key, position) =>
    request(`/api/admin/showcase/${position}`, { method: "DELETE", adminPassword: key }),
  uploadShowcase: (key, position, formData, onProgress) =>
    new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${API_BASE}/api/admin/showcase/${position}/upload`);
      xhr.setRequestHeader("X-Admin-Password", key);
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable && onProgress) onProgress(Math.round((e.loaded / e.total) * 100));
      };
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try { resolve(JSON.parse(xhr.responseText)); } catch (e) { reject(e); }
        } else {
          reject(new Error(`${xhr.status}: ${xhr.responseText}`));
        }
      };
      xhr.onerror = () => reject(new Error("network error"));
      xhr.send(formData);
    }),
  courseByToken: (token) => request(`/api/public/courses/${token}`),
  teeTimes: (token, date) =>
    request(`/api/public/courses/${token}/tee-times${date ? `?date=${date}` : ""}`),
  register: async (formData) => {
    const headers = {};
    const t = getUserToken();
    if (t) headers["Authorization"] = `Bearer ${t}`;
    const res = await fetch(`${API_BASE}/api/public/register`, {
      method: "POST",
      headers,
      body: formData,
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`${res.status}: ${text}`);
    }
    return res.json();
  },

  // ---- User auth ----
  signup: (payload) => request(`/api/auth/signup`, { method: "POST", body: payload, auth: false }),
  login: (payload) => request(`/api/auth/login`, { method: "POST", body: payload, auth: false }),
  me: () => request(`/api/auth/me`),
  myRounds: () => request(`/api/auth/me/rounds`),
  myRoundClips: (participantId) => request(`/api/auth/me/rounds/${participantId}/clips`),
  selfieUrl: (path) => `${API_BASE}/uploads/${path}`,

  gallery: (token) => request(`/api/gallery/${token}`),
  flagClip: (token, clipId, note) =>
    request(`/api/gallery/${token}/clips/${clipId}/flag`, { method: "POST", body: { note } }),

  listCourses: (key) => request(`/api/admin/courses`, { adminPassword: key }),
  createCourse: (key, payload) => request(`/api/admin/courses`, { method: "POST", body: payload, adminPassword: key }),
  updateCourse: (key, id, payload) => request(`/api/admin/courses/${id}`, { method: "PATCH", body: payload, adminPassword: key }),
  stats: (key) => request(`/api/admin/stats`, { adminPassword: key }),
  flaggedClips: (key) => request(`/api/admin/flagged-clips`, { adminPassword: key }),
  listAllClips: (key, limit = 100) =>
    request(`/api/admin/clips?limit=${limit}`, { adminPassword: key }),
  listBroadcastClips: (key, limit = 100) =>
    request(`/api/admin/broadcast-clips?limit=${limit}`, { adminPassword: key }),
  deleteClip: (key, clipId) =>
    request(`/api/admin/clips/${clipId}`, {
      method: "DELETE", adminPassword: key,
    }),
  listLongUploads: (key, limit = 100) =>
    request(`/api/admin/long-uploads?limit=${limit}`, { adminPassword: key }),
  reprocessLongUpload: (key, uploadId, formData) =>
    // Re-runs the segmenter + AI tracer + composite on a stored long
    // upload. XHR-based so we get FormData support without rewriting
    // the request helper.
    new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${API_BASE}/api/admin/long-uploads/${uploadId}/reprocess`);
      xhr.setRequestHeader("X-Admin-Password", key);
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try { resolve(JSON.parse(xhr.responseText)); } catch (e) { reject(e); }
        } else {
          reject(new Error(`${xhr.status}: ${xhr.responseText}`));
        }
      };
      xhr.onerror = () => reject(new Error("network error"));
      xhr.send(formData);
    }),
  deleteLongUpload: (key, uploadId) =>
    request(`/api/admin/long-uploads/${uploadId}`, {
      method: "DELETE", adminPassword: key,
    }),
  processLongUploadSegment: (key, uploadId, { holeNumber, startSec, endSec, aiTracerModel }) =>
    // Synchronous endpoint that runs the full per-segment pipeline
    // (real cut + AI tracer + composite + VideoClip row) on ONE
    // detected window. Typically 30-90 s; allow up to 5 min before
    // timing out so the request doesn't fall over on slow encoders.
    new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${API_BASE}/api/admin/long-uploads/${uploadId}/process-segment`);
      xhr.setRequestHeader("X-Admin-Password", key);
      xhr.timeout = 5 * 60 * 1000;
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try { resolve(JSON.parse(xhr.responseText)); } catch (e) { reject(e); }
        } else {
          reject(new Error(`${xhr.status}: ${xhr.responseText}`));
        }
      };
      xhr.onerror = () => reject(new Error("network error"));
      xhr.ontimeout = () => reject(new Error("timed out after 5 min"));
      const fd = new FormData();
      fd.append("hole_number", String(holeNumber));
      fd.append("start_sec", String(startSec));
      fd.append("end_sec", String(endSec));
      if (aiTracerModel) fd.append("ai_tracer_model", aiTracerModel);
      xhr.send(fd);
    }),
  testCutLongUpload: (key, uploadId, detector = "motion", opts = {}) =>
    // Form-encoded POST so we get FastAPI's Form(...) parsing without
    // needing the JSON path in request().
    new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${API_BASE}/api/admin/long-uploads/${uploadId}/test-cut`);
      xhr.setRequestHeader("X-Admin-Password", key);
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try { resolve(JSON.parse(xhr.responseText)); } catch (e) { reject(e); }
        } else {
          reject(new Error(`${xhr.status}: ${xhr.responseText}`));
        }
      };
      xhr.onerror = () => reject(new Error("network error"));
      const fd = new FormData();
      fd.append("detector", detector);
      if (opts.audioMinPeakRatio != null) {
        fd.append("audio_min_peak_ratio", String(opts.audioMinPeakRatio));
      }
      if (opts.motionRatio != null) {
        fd.append("motion_ratio", String(opts.motionRatio));
      }
      if (opts.combinedPairWindowSec != null) {
        fd.append("combined_pair_window_sec", String(opts.combinedPairWindowSec));
      }
      if (opts.cutClips === false) {
        fd.append("cut_clips", "false");
      }
      xhr.send(fd);
    }),
  retryTracer: (key, clipId, { sensitivity } = {}) =>
    // Tracer can run ~1-3 min on long clips. Time out at 4 min so the
    // UI doesn't spin forever if the server hangs or gets killed.
    // Optional `sensitivity` multiplier (>1 = looser, <1 = stricter)
    // gets sent as a form param.
    new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${API_BASE}/api/admin/clips/${clipId}/retry-tracer`);
      xhr.setRequestHeader("X-Admin-Password", key);
      xhr.timeout = 240_000;
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try { resolve(JSON.parse(xhr.responseText)); } catch (e) { reject(e); }
        } else {
          reject(new Error(`${xhr.status}: ${xhr.responseText}`));
        }
      };
      xhr.onerror = () => reject(new Error("network error"));
      xhr.ontimeout = () => reject(new Error("timed out after 4 min"));
      const fd = new FormData();
      if (sensitivity != null) fd.append("sensitivity", String(sensitivity));
      xhr.send(fd);
    }),
  aiTrace: (key, clipId, model) => {
    // Five Claude steps: address, handedness, rough impact, refined
    // impact, ball-track. The track step is up to 60 parallel calls
    // and dominates wall time (~30-60 s on a typical swing). Cap at
    // 5 min so we don't time out the UI mid-track. Optional `model`
    // overrides the backend default for per-clip A/B testing.
    const qs = model ? `?model=${encodeURIComponent(model)}` : "";
    return request(`/api/admin/clips/${clipId}/ai-trace${qs}`, {
      method: "POST", adminPassword: key, timeoutMs: 300_000,
    });
  },
  listParticipants: (key, params = {}) => {
    const qs = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => {
      if (v !== undefined && v !== null && v !== "") qs.set(k, v);
    });
    return request(`/api/admin/participants${qs.toString() ? `?${qs}` : ""}`, { adminPassword: key });
  },
  participantClips: (key, id) => request(`/api/admin/participants/${id}/clips`, { adminPassword: key }),
  assignClip: (key, clipId, participantId) =>
    request(`/api/admin/clips/${clipId}/assign?participant_id=${participantId}`, {
      method: "POST",
      adminPassword: key,
    }),
  uploadClip: (key, formData, onProgress) =>
    new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${API_BASE}/api/admin/clips/upload`);
      xhr.setRequestHeader("X-Admin-Password", key);
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable && onProgress) onProgress(Math.round((e.loaded / e.total) * 100));
      };
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try { resolve(JSON.parse(xhr.responseText)); } catch (e) { reject(e); }
        } else {
          reject(new Error(`${xhr.status}: ${xhr.responseText}`));
        }
      };
      xhr.onerror = () => reject(new Error("network error"));
      xhr.send(formData);
    }),
  longUploadClips: (key, formData, onProgress) =>
    new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${API_BASE}/api/admin/clips/long-upload`);
      xhr.setRequestHeader("X-Admin-Password", key);
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable && onProgress) onProgress(Math.round((e.loaded / e.total) * 100));
      };
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try { resolve(JSON.parse(xhr.responseText)); } catch (e) { reject(e); }
        } else {
          reject(new Error(`${xhr.status}: ${xhr.responseText}`));
        }
      };
      xhr.onerror = () => reject(new Error("network error"));
      xhr.send(formData);
    }),
  resendGallery: (key, id) =>
    request(`/api/admin/participants/${id}/resend-gallery`, { method: "POST", adminPassword: key }),
  refundParticipant: (key, id) =>
    request(`/api/admin/participants/${id}/refund`, { method: "POST", adminPassword: key }),
  sendRoundSummary: (key, id, force = false) =>
    request(`/api/admin/participants/${id}/send-summary${force ? "?force=true" : ""}`, {
      method: "POST", adminPassword: key,
    }),
  sendTestEmail: (key, payload) =>
    request(`/api/admin/test-email`, { method: "POST", body: payload, adminPassword: key }),
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
