// Same-origin in the built app (the backend serves the SPA); the Vite dev
// server proxies /api to localhost:8100 (see vite.config.js).
async function request(path, { method = "GET", body, timeoutMs = 30000 } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  let res;
  try {
    res = await fetch(path, {
      method,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    });
  } catch (e) {
    if (e.name === "AbortError") {
      throw new Error(`request timed out after ${Math.round(timeoutMs / 1000)}s`);
    }
    throw e;
  } finally {
    clearTimeout(timer);
  }
  if (!res.ok) {
    throw new Error(`${res.status}: ${await res.text()}`);
  }
  if (res.headers.get("content-type")?.includes("application/json")) {
    return res.json();
  }
  return res;
}

export const api = {
  news: () => request("/api/news"),
  quotes: () => request("/api/quotes"),
  todos: () => request("/api/todos"),
  addTodo: (text) => request("/api/todos", { method: "POST", body: { text } }),
  updateTodo: (id, patch) => request(`/api/todos/${id}`, { method: "PATCH", body: patch }),
  deleteTodo: (id) => request(`/api/todos/${id}`, { method: "DELETE" }),
};
