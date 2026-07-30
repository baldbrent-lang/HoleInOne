// In a production build (e.g. Replit), the API is served from the same
// origin as the SPA, so the base URL is empty. In local dev the Vite
// server runs on a different port, so fall back to localhost:8000.
const API_BASE =
  import.meta.env.VITE_API_BASE ??
  (import.meta.env.DEV ? "http://localhost:8000" : "");

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

async function request(
  path,
  { method = "GET", body, adminPassword, auth = true, timeoutMs } = {},
) {
  // FormData bodies get sent as multipart/form-data — the browser
  // sets the Content-Type (incl. the boundary) automatically when we
  // *don't* set it ourselves. JSON-style bodies keep the explicit
  // application/json header.
  const isFormData =
    typeof FormData !== "undefined" && body instanceof FormData;
  const headers = {};
  if (!isFormData) headers["Content-Type"] = "application/json";
  if (adminPassword) headers["X-Admin-Password"] = adminPassword;
  if (auth) {
    const t = getUserToken();
    if (t) headers["Authorization"] = `Bearer ${t}`;
  }
  const controller = timeoutMs ? new AbortController() : null;
  const timer = controller
    ? setTimeout(() => controller.abort(), timeoutMs)
    : null;
  let res;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      method,
      headers,
      body: !body ? undefined : isFormData ? body : JSON.stringify(body),
      signal: controller?.signal,
    });
  } catch (e) {
    if (e.name === "AbortError") {
      throw new Error(
        `request timed out after ${Math.round(timeoutMs / 1000)}s`,
      );
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
  startWatchingCamera: (key, id) =>
    request(`/api/admin/cameras/${id}/watch`, {
      method: "POST",
      adminPassword: key,
    }),
  stopWatchingCamera: (key, id) =>
    request(`/api/admin/cameras/${id}/watch`, {
      method: "DELETE",
      adminPassword: key,
    }),
  cameraLiveFrameUrl: (id) => `${API_BASE}/api/admin/cameras/${id}/live-frame`,
  listPublicCourses: () => request(`/api/public/courses`),
  stripeConfig: () => request(`/api/public/stripe-config`, { auth: false }),
  inviteInfo: (token) =>
    request(`/api/public/invite/${token}`, { auth: false }),
  inviteSelfie: async (token, file) => {
    const fd = new FormData();
    fd.append("selfie", file, file.name || "selfie.jpg");
    const res = await fetch(`${API_BASE}/api/public/invite/${token}/selfie`, {
      method: "POST",
      body: fd,
    });
    if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
    return res.json();
  },
  listShowcase: () => request(`/api/public/showcase`),
  publicStats: () => request(`/api/public/stats`, { auth: false }),
  leaderboards: (limit) =>
    request(`/api/public/leaderboards${limit ? `?limit=${limit}` : ""}`, {
      auth: false,
    }),
  contests: () => request(`/api/public/contests`, { auth: false }),
  broadcastChannels: () => request(`/api/broadcast/channels`, { auth: false }),
  channelShareLink: (key, channelKey) =>
    request(
      `/api/broadcast/admin/channel-share/${encodeURIComponent(channelKey)}`,
      { adminPassword: key },
    ),
  sharedChannel: (token) =>
    request(`/api/broadcast/shared/${encodeURIComponent(token)}`, {
      auth: false,
    }),
  sharedChannelPlaylist: (token, limit = 200) =>
    request(
      `/api/broadcast/shared/${encodeURIComponent(token)}/playlist?limit=${limit}`,
      { auth: false },
    ),
  broadcastChannelPlaylist: (key, limit = 200) =>
    request(`/api/broadcast/channels/${encodeURIComponent(key)}/playlist?limit=${limit}`, {
      auth: false,
    }),
  broadcastNext: (viewerId, courseId) => {
    const qs = new URLSearchParams({ viewer_id: viewerId });
    if (courseId) qs.set("course_id", courseId);
    return request(`/api/broadcast/next?${qs}`, { auth: false });
  },
  tagHighlight: (key, clipId, tag) =>
    request(
      `/api/broadcast/admin/clips/${clipId}/highlight${tag ? `?tag=${encodeURIComponent(tag)}` : ""}`,
      {
        method: "POST",
        adminPassword: key,
      },
    ),
  untagHighlight: (key, clipId) =>
    request(`/api/broadcast/admin/clips/${clipId}/highlight`, {
      method: "DELETE",
      adminPassword: key,
    }),
  autoTagHighlights: (key) =>
    request(`/api/broadcast/admin/auto-tag`, {
      method: "POST",
      adminPassword: key,
    }),
  publicProfile: (userId) =>
    request(`/api/public/profile/${userId}`, { auth: false }),
  setOperatorPassword: (key, courseId, password) =>
    request(`/api/admin/courses/${courseId}/operator-password`, {
      method: "POST",
      body: { password },
      adminPassword: key,
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
  adminListShowcase: (key) =>
    request(`/api/admin/showcase`, { adminPassword: key }),
  updateShowcase: (key, position, payload) =>
    request(`/api/admin/showcase/${position}`, {
      method: "PATCH",
      body: payload,
      adminPassword: key,
    }),
  clearShowcase: (key, position) =>
    request(`/api/admin/showcase/${position}`, {
      method: "DELETE",
      adminPassword: key,
    }),
  uploadShowcase: (key, position, formData, onProgress) =>
    new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${API_BASE}/api/admin/showcase/${position}/upload`);
      xhr.setRequestHeader("X-Admin-Password", key);
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable && onProgress)
          onProgress(Math.round((e.loaded / e.total) * 100));
      };
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            resolve(JSON.parse(xhr.responseText));
          } catch (e) {
            reject(e);
          }
        } else {
          reject(new Error(`${xhr.status}: ${xhr.responseText}`));
        }
      };
      xhr.onerror = () => reject(new Error("network error"));
      xhr.send(formData);
    }),
  courseByToken: (token) => request(`/api/public/courses/${token}`),
  teeTimes: (token, date) =>
    request(
      `/api/public/courses/${token}/tee-times${date ? `?date=${date}` : ""}`,
    ),
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
  signup: (payload) =>
    request(`/api/auth/signup`, { method: "POST", body: payload, auth: false }),
  login: (payload) =>
    request(`/api/auth/login`, { method: "POST", body: payload, auth: false }),
  me: () => request(`/api/auth/me`),
  myRounds: () => request(`/api/auth/me/rounds`),
  myRoundClips: (participantId) =>
    request(`/api/auth/me/rounds/${participantId}/clips`),
  selfieUrl: (path) => `${API_BASE}/uploads/${path}`,

  gallery: (token) => request(`/api/gallery/${token}`),
  flagClip: (token, clipId, note) =>
    request(`/api/gallery/${token}/clips/${clipId}/flag`, {
      method: "POST",
      body: { note },
    }),

  listCourses: (key) => request(`/api/admin/courses`, { adminPassword: key }),
  createCourse: (key, payload) =>
    request(`/api/admin/courses`, {
      method: "POST",
      body: payload,
      adminPassword: key,
    }),
  updateCourse: (key, id, payload) =>
    request(`/api/admin/courses/${id}`, {
      method: "PATCH",
      body: payload,
      adminPassword: key,
    }),
  stats: (key) => request(`/api/admin/stats`, { adminPassword: key }),
  flaggedClips: (key) =>
    request(`/api/admin/flagged-clips`, { adminPassword: key }),
  listAllClips: (key, limit = 100, offset = 0) =>
    request(
      `/api/admin/clips?limit=${limit}&offset=${offset}`,
      { adminPassword: key },
    ),
  listBroadcastClips: (key, limit = 100, offset = 0) =>
    request(
      `/api/admin/broadcast-clips?limit=${limit}&offset=${offset}`,
      { adminPassword: key },
    ),
  makeClipVertical: (key, clipId, force = false) =>
    request(`/api/admin/clips/${clipId}/vertical${force ? "?force=1" : ""}`, {
      method: "POST",
      adminPassword: key,
      timeoutMs: 180000,
    }),
  setClipBroadcast: (key, clipId, broadcast) =>
    request(`/api/admin/clips/${clipId}/broadcast`, {
      method: "POST",
      body: broadcast == null ? {} : { broadcast: !!broadcast },
      adminPassword: key,
    }),
  deleteClip: (key, clipId) =>
    request(`/api/admin/clips/${clipId}`, {
      method: "DELETE",
      adminPassword: key,
    }),
  listLongUploads: (key, limit = 100, offset = 0, opts = {}) => {
    const params = new URLSearchParams({ limit, offset });
    if (opts.course) params.set("course", opts.course);
    if (opts.sort) params.set("sort", opts.sort);
    if (opts.order) params.set("order", opts.order);
    return request(
      `/api/admin/long-uploads?${params.toString()}`,
      { adminPassword: key },
    );
  },

  // ---- Camera-event production queue ----
  listCameraEvents: (key, limit = 100, offset = 0) =>
    request(
      `/api/admin/camera-events?limit=${limit}&offset=${offset}`,
      { adminPassword: key },
    ),
  reprocessCameraEvent: (key, eventId) =>
    request(`/api/admin/camera-events/${eventId}/reprocess`, {
      method: "POST",
      adminPassword: key,
    }),
  deleteCameraEvent: (key, eventId) =>
    request(`/api/admin/camera-events/${eventId}`, {
      method: "DELETE",
      adminPassword: key,
    }),

  // ---- Cameras (always-on capture devices) ----
  listCameras: (key) => request(`/api/admin/cameras`, { adminPassword: key }),
  createCamera: (key, { courseId, assignedHole, assignedRole, name }) => {
    const fd = new FormData();
    fd.append("course_id", String(courseId));
    fd.append("assigned_hole", String(assignedHole));
    fd.append("assigned_role", assignedRole);
    fd.append("name", name || "");
    return request(`/api/admin/cameras`, {
      method: "POST",
      adminPassword: key,
      body: fd,
    });
  },
  pairCameras: (key, cameraId, partnerId) => {
    const fd = new FormData();
    fd.append("partner_id", String(partnerId));
    return request(`/api/admin/cameras/${cameraId}/pair`, {
      method: "POST",
      adminPassword: key,
      body: fd,
    });
  },
  unpairCamera: (key, cameraId) =>
    request(`/api/admin/cameras/${cameraId}/unpair`, {
      method: "POST",
      adminPassword: key,
    }),
  rotateCameraToken: (key, cameraId) =>
    request(`/api/admin/cameras/${cameraId}/rotate-token`, {
      method: "POST",
      adminPassword: key,
    }),
  updateCamera: (key, cameraId, patch) => {
    const fd = new FormData();
    if (patch.name !== undefined) fd.append("name", patch.name || "");
    if (patch.enabled !== undefined)
      fd.append("enabled", patch.enabled ? "true" : "false");
    if (patch.triggeringEnabled !== undefined)
      fd.append("triggering_enabled", patch.triggeringEnabled ? "true" : "false");
    if (patch.note !== undefined) fd.append("note", patch.note || "");
    if (patch.teeBoxRoi !== undefined) {
      fd.append("tee_box_roi", JSON.stringify(patch.teeBoxRoi));
    }
    if (patch.courseId !== undefined)
      fd.append("course_id", String(patch.courseId));
    if (patch.assignedHole !== undefined)
      fd.append("assigned_hole", String(patch.assignedHole));
    if (patch.assignedRole !== undefined)
      fd.append("assigned_role", patch.assignedRole);
    return request(`/api/admin/cameras/${cameraId}/update`, {
      method: "POST",
      adminPassword: key,
      body: fd,
    });
  },
  deleteCamera: (key, cameraId) =>
    request(`/api/admin/cameras/${cameraId}`, {
      method: "DELETE",
      adminPassword: key,
    }),
  reprocessLongUpload: (key, uploadId, formData) =>
    // Re-runs the segmenter + AI tracer + composite on a stored long
    // upload. XHR-based so we get FormData support without rewriting
    // the request helper.
    new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open(
        "POST",
        `${API_BASE}/api/admin/long-uploads/${uploadId}/reprocess`,
      );
      xhr.setRequestHeader("X-Admin-Password", key);
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            resolve(JSON.parse(xhr.responseText));
          } catch (e) {
            reject(e);
          }
        } else {
          reject(new Error(`${xhr.status}: ${xhr.responseText}`));
        }
      };
      xhr.onerror = () => reject(new Error("network error"));
      xhr.send(formData);
    }),
  deleteLongUpload: (key, uploadId) =>
    request(`/api/admin/long-uploads/${uploadId}`, {
      method: "DELETE",
      adminPassword: key,
    }),
  autoDetectLongUpload: (key, uploadId) =>
    // Cheap detection (audio impact + one Claude handedness call).
    // Typically returns in 5-10s; bump the helper's default timeout
    // so the Edit-wizard spinner doesn't fall over on cold-start.
    request(`/api/admin/long-uploads/${uploadId}/auto-detect`, {
      method: "POST",
      adminPassword: key,
      timeoutMs: 90_000,
    }),
  detectSwingsForUpload: (key, uploadId) =>
    // Multi-swing wizard: audio + motion swing detection only — no
    // Claude calls. Returns the list of swing windows (start_frame
    // / end_frame / address_frame / impact_frame per swing).
    request(`/api/admin/long-uploads/${uploadId}/detect-swings`, {
      method: "POST",
      adminPassword: key,
      timeoutMs: 90_000,
    }),
  getLongUploadFrame: (key, uploadId, frame) =>
    request(`/api/admin/long-uploads/${uploadId}/frame?frame=${frame}`, {
      adminPassword: key,
      timeoutMs: 20_000,
    }),
  saveEditMetrics: (key, uploadId, patch) =>
    request(`/api/admin/long-uploads/${uploadId}/edit-metrics`, {
      method: "POST",
      body: patch,
      adminPassword: key,
    }),
  renderWizardTracer: (key, uploadId, overrides = {}) =>
    // Heavy: runs the full ai-trace pipeline (address + handedness +
    // impact + ball-track + tracer render). Bump timeout to a few
    // minutes so we don't fall over on cold starts.
    request(`/api/admin/long-uploads/${uploadId}/render-tracer`, {
      method: "POST",
      body: overrides,
      adminPassword: key,
      timeoutMs: 5 * 60_000,
    }),
  renderWizardTracerFast: (key, uploadId, payload = {}) =>
    // cv2-only: merges manual_positions into the cached ball_track
    // and re-renders the tracer overlay. No Claude calls. Timeout is
    // generous because the overlay is re-rendered across the whole source
    // clip — a long mirrored clip (2+ min) can take a few minutes.
    request(`/api/admin/long-uploads/${uploadId}/render-tracer-fast`, {
      method: "POST",
      body: payload,
      adminPassword: key,
      timeoutMs: 5 * 60_000,
    }),
  finalizeWizardVideo: (key, uploadId, payload = {}) =>
    request(`/api/admin/long-uploads/${uploadId}/finalize`, {
      method: "POST",
      body: payload,
      adminPassword: key,
      timeoutMs: 2 * 60_000,
    }),
  scanPlotRegion: (key, uploadId, payload = {}) =>
    // Frame-diff deep scan of a zoomed region — returns every motion
    // blob as a plottable dot. Decodes real video, so give it time.
    request(`/api/admin/long-uploads/${uploadId}/scan-region`, {
      method: "POST",
      body: payload,
      adminPassword: key,
      timeoutMs: 3 * 60_000,
    }),
  commitWizardClip: (key, uploadId, payload = {}) =>
    // payload.clip_id targets a specific produced clip — required on
    // multi-swing uploads or the backend updates the most recent clip.
    request(`/api/admin/long-uploads/${uploadId}/commit`, {
      method: "POST",
      body: payload,
      adminPassword: key,
    }),
  mirrorFromProd: (key) =>
    request(`/api/admin/mirror-from-prod`, { method: "POST", adminPassword: key }),
  mirrorFromProdStatus: (key) =>
    request(`/api/admin/mirror-from-prod/status`, { adminPassword: key }),
  produceDebug: (key, uploadId, analyzeOnly = false) =>
    request(
      `/api/admin/long-uploads/${uploadId}/produce-debug${
        analyzeOnly ? "?analyze_only=true" : ""
      }`,
      { method: "POST", adminPassword: key },
    ),
  debug2: (key, uploadId) =>
    // Runs the five-stage pipeline synchronously and returns the whole
    // report — pose passes, club-arc ball, AI judge, windowed heat and
    // the chain all happen in the request, so give it room.
    request(`/api/admin/long-uploads/${uploadId}/debug2`, {
      method: "POST",
      adminPassword: key,
      timeoutMs: 10 * 60_000,
    }),

  debug3: (key, uploadId) =>
    // The blob-and-track method. Per-frame MOG2 over the whole flight
    // window for every candidate, so if anything it is slower than debug2.
    request(`/api/admin/long-uploads/${uploadId}/debug3`, {
      method: "POST",
      adminPassword: key,
      timeoutMs: 10 * 60_000,
    }),
  emailStatus: (key) =>
    request("/api/admin/email-status", { adminPassword: key }),
  emailSendTemplates: (key, to) =>
    // Sends one of each template. Real clip attachments make this slow.
    request("/api/admin/email-send-templates", {
      method: "POST",
      body: { to },
      adminPassword: key,
      timeoutMs: 3 * 60_000,
    }),
  produceDebugStatus: (key, uploadId) =>
    request(`/api/admin/long-uploads/${uploadId}/produce-debug/status`, {
      adminPassword: key,
    }),
  setBallRoi: (key, courseId, roi) =>
    request(`/api/admin/courses/${courseId}/ball-roi`, {
      method: "POST",
      adminPassword: key,
      body: { roi },
    }),
  rescanBall: (key, uploadId) =>
    request(`/api/admin/long-uploads/${uploadId}/rescan-ball`, {
      method: "POST",
      adminPassword: key,
      timeoutMs: 120000,
    }),
  processLongUploadSegment: (
    key,
    uploadId,
    { holeNumber, startSec, endSec, aiTracerModel },
  ) =>
    // Synchronous endpoint that runs the full per-segment pipeline
    // (real cut + AI tracer + composite + VideoClip row) on ONE
    // detected window. Typically 30-90 s; allow up to 5 min before
    // timing out so the request doesn't fall over on slow encoders.
    new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open(
        "POST",
        `${API_BASE}/api/admin/long-uploads/${uploadId}/process-segment`,
      );
      xhr.setRequestHeader("X-Admin-Password", key);
      xhr.timeout = 5 * 60 * 1000;
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            resolve(JSON.parse(xhr.responseText));
          } catch (e) {
            reject(e);
          }
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
      xhr.open(
        "POST",
        `${API_BASE}/api/admin/long-uploads/${uploadId}/test-cut`,
      );
      xhr.setRequestHeader("X-Admin-Password", key);
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            resolve(JSON.parse(xhr.responseText));
          } catch (e) {
            reject(e);
          }
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
        fd.append(
          "combined_pair_window_sec",
          String(opts.combinedPairWindowSec),
        );
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
          try {
            resolve(JSON.parse(xhr.responseText));
          } catch (e) {
            reject(e);
          }
        } else {
          reject(new Error(`${xhr.status}: ${xhr.responseText}`));
        }
      };
      xhr.onerror = () => reject(new Error("network error"));
      xhr.ontimeout = () => reject(new Error("timed out after 4 min"));
      // Always send sensitivity (default 1.0) — FastAPI's multipart
      // parser rejects an empty body with "There was an error parsing
      // the body".
      const fd = new FormData();
      fd.append("sensitivity", String(sensitivity ?? 1.0));
      xhr.send(fd);
    }),
  audioImpactFrame: (key, clipId, { minRatio = 25 } = {}) =>
    // Synchronous, cheap (~1-2 s): runs only the audio impact detector
    // and grabs the matching frame as a JPG. Test harness for swapping
    // the AI impact-pick / refine steps out of the production tracer.
    new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open(
        "POST",
        `${API_BASE}/api/admin/clips/${clipId}/audio-impact-frame`,
      );
      xhr.setRequestHeader("X-Admin-Password", key);
      xhr.timeout = 30_000;
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            resolve(JSON.parse(xhr.responseText));
          } catch (e) {
            reject(e);
          }
        } else {
          reject(new Error(`${xhr.status}: ${xhr.responseText}`));
        }
      };
      xhr.onerror = () => reject(new Error("network error"));
      xhr.ontimeout = () => reject(new Error("timed out after 30 s"));
      const fd = new FormData();
      fd.append("min_ratio", String(minRatio));
      xhr.send(fd);
    }),
  aiTrace: (key, clipId, modelOrOpts) => {
    // Five Claude steps: address, handedness, rough impact, refined
    // impact, ball-track. The track step is up to 60 parallel calls
    // and dominates wall time (~30-60 s on a typical swing). Cap at
    // 5 min so we don't time out the UI mid-track. Optional `model`
    // overrides the backend default for per-clip A/B testing.
    //
    // Backwards-compatible signature: pass a string for just the
    // model, or an object for model + manual-override params:
    //   { model, impactFrameOverride, ballTrackMaxFrames,
    //     ballAtRestX, ballAtRestY, manualBallPositions }
    // `manualBallPositions` is an array of {frame, x, y} in NATIVE
    // pixel coords of the source video.
    const opts =
      typeof modelOrOpts === "string"
        ? { model: modelOrOpts }
        : modelOrOpts || {};
    const qs = opts.model ? `?model=${encodeURIComponent(opts.model)}` : "";
    const hasOverrides =
      opts.impactFrameOverride != null ||
      opts.ballTrackMaxFrames != null ||
      opts.ballAtRestX != null ||
      opts.ballAtRestY != null ||
      (opts.manualBallPositions && opts.manualBallPositions.length > 0) ||
      (opts.handednessOverride && opts.handednessOverride !== "auto");

    // Fast path: no overrides → use the existing JSON request helper.
    if (!hasOverrides) {
      return request(`/api/admin/clips/${clipId}/ai-trace${qs}`, {
        method: "POST",
        adminPassword: key,
        timeoutMs: 300_000,
      });
    }

    // Override path: hand-roll XHR with multipart FormData so the
    // backend's Form(...) params parse cleanly.
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${API_BASE}/api/admin/clips/${clipId}/ai-trace${qs}`);
      xhr.setRequestHeader("X-Admin-Password", key);
      xhr.timeout = 300_000;
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            resolve(JSON.parse(xhr.responseText));
          } catch (e) {
            reject(e);
          }
        } else {
          reject(new Error(`${xhr.status}: ${xhr.responseText}`));
        }
      };
      xhr.onerror = () => reject(new Error("network error"));
      xhr.ontimeout = () => reject(new Error("timed out after 5 min"));
      const fd = new FormData();
      if (opts.impactFrameOverride != null) {
        fd.append("impact_frame_override", String(opts.impactFrameOverride));
      }
      if (opts.ballTrackMaxFrames != null) {
        fd.append("ball_track_max_frames", String(opts.ballTrackMaxFrames));
      }
      if (opts.ballAtRestX != null) {
        fd.append("ball_at_rest_x", String(opts.ballAtRestX));
      }
      if (opts.ballAtRestY != null) {
        fd.append("ball_at_rest_y", String(opts.ballAtRestY));
      }
      if (opts.manualBallPositions && opts.manualBallPositions.length > 0) {
        fd.append(
          "manual_ball_positions_json",
          JSON.stringify(opts.manualBallPositions),
        );
      }
      if (opts.handednessOverride && opts.handednessOverride !== "auto") {
        fd.append("handedness_override", opts.handednessOverride);
      }
      xhr.send(fd);
    });
  },
  listParticipants: (key, params = {}) => {
    const qs = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => {
      if (v !== undefined && v !== null && v !== "") qs.set(k, v);
    });
    return request(`/api/admin/participants${qs.toString() ? `?${qs}` : ""}`, {
      adminPassword: key,
    });
  },
  participantClips: (key, id) =>
    request(`/api/admin/participants/${id}/clips`, { adminPassword: key }),
  assignClip: (key, clipId, participantId) =>
    request(
      `/api/admin/clips/${clipId}/assign?participant_id=${participantId}`,
      {
        method: "POST",
        adminPassword: key,
      },
    ),
  uploadClip: (key, formData, onProgress) =>
    new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${API_BASE}/api/admin/clips/upload`);
      xhr.setRequestHeader("X-Admin-Password", key);
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable && onProgress)
          onProgress(Math.round((e.loaded / e.total) * 100));
      };
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            resolve(JSON.parse(xhr.responseText));
          } catch (e) {
            reject(e);
          }
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
        if (e.lengthComputable && onProgress)
          onProgress(Math.round((e.loaded / e.total) * 100));
      };
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            resolve(JSON.parse(xhr.responseText));
          } catch (e) {
            reject(e);
          }
        } else {
          reject(new Error(`${xhr.status}: ${xhr.responseText}`));
        }
      };
      xhr.onerror = () => reject(new Error("network error"));
      xhr.send(formData);
    }),
  // Simple upload backing /admin/upload — saves files, creates a
  // queued LongVideoUpload row, does NOT start processing.
  quickUploadVideos: (key, formData, onProgress) =>
    new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${API_BASE}/api/admin/clips/quick-upload`);
      xhr.setRequestHeader("X-Admin-Password", key);
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable && onProgress)
          onProgress(Math.round((e.loaded / e.total) * 100));
      };
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            resolve(JSON.parse(xhr.responseText));
          } catch (e) {
            reject(e);
          }
        } else {
          reject(new Error(`${xhr.status}: ${xhr.responseText}`));
        }
      };
      xhr.onerror = () => reject(new Error("network error"));
      xhr.send(formData);
    }),
  resendGallery: (key, id) =>
    request(`/api/admin/participants/${id}/resend-gallery`, {
      method: "POST",
      adminPassword: key,
    }),
  refundParticipant: (key, id) =>
    request(`/api/admin/participants/${id}/refund`, {
      method: "POST",
      adminPassword: key,
    }),
  sendRoundSummary: (key, id, force = false) =>
    request(
      `/api/admin/participants/${id}/send-summary${force ? "?force=true" : ""}`,
      {
        method: "POST",
        adminPassword: key,
      },
    ),
  sendTestEmail: (key, payload) =>
    request(`/api/admin/test-email`, {
      method: "POST",
      body: payload,
      adminPassword: key,
    }),
  listHIO: (key, status) =>
    request(`/api/admin/hio${status ? `?status=${status}` : ""}`, {
      adminPassword: key,
    }),
  hioDetail: (key, id) =>
    request(`/api/admin/hio/${id}`, { adminPassword: key }),
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
