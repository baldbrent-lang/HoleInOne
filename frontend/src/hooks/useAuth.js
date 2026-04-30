import { useCallback, useEffect, useState } from "react";
import { api, getUserToken, setUserToken } from "../api.js";

let cachedUser = null;

export default function useAuth() {
  const [user, setUser] = useState(cachedUser);
  const [loading, setLoading] = useState(!cachedUser && !!getUserToken());

  const refresh = useCallback(async () => {
    if (!getUserToken()) {
      setUser(null);
      cachedUser = null;
      setLoading(false);
      return null;
    }
    try {
      const u = await api.me();
      setUser(u);
      cachedUser = u;
      setLoading(false);
      return u;
    } catch {
      setUserToken("");
      setUser(null);
      cachedUser = null;
      setLoading(false);
      return null;
    }
  }, []);

  useEffect(() => {
    if (!cachedUser && getUserToken()) refresh();
  }, [refresh]);

  const logout = useCallback(() => {
    setUserToken("");
    setUser(null);
    cachedUser = null;
  }, []);

  const adopt = useCallback((token, u) => {
    setUserToken(token);
    setUser(u);
    cachedUser = u;
  }, []);

  return { user, loading, refresh, logout, adopt };
}
