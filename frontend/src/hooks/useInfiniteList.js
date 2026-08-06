/**
 * useInfiniteList: paginated fetch + scroll-to-load-more for long lists.
 *
 * Pages are off the server in fixed-size batches (default 25). The
 * caller passes a `fetcher(limit, offset) => Promise<items[]>` that
 * wraps the relevant api.js method, and we hand back:
 *
 *   - `items`           : the full concatenated list so far (or null
 *                         while the first page is still in flight)
 *   - `hasMore`         : whether the last batch was a full page —
 *                         heuristic for "there are probably more"
 *   - `loadingMore`     : true while a non-initial page is loading
 *   - `error`           : last error message string, or null
 *   - `sentinelRef`     : attach to a div at the bottom; entering
 *                         the viewport triggers the next page load
 *   - `reload()`        : drop everything and re-fetch page 1
 *   - `refresh()`       : re-fetch the currently-loaded range (for
 *                         poll loops that want fresh status without
 *                         dropping the user's scroll position)
 *   - `setItems(fn)`    : mutate in place (e.g. flip a flag on one
 *                         row after a server action — saves the
 *                         round-trip cost of refresh())
 *
 * Concurrent-load protection lives in a ref so React Strict-Mode's
 * double-effect doesn't fire two parallel page-1 fetches.
 */
import { useCallback, useEffect, useRef, useState } from "react";

export function useInfiniteList(fetcher, { pageSize = 25, deps = [] } = {}) {
  const [items, setItems] = useState(null);
  const [hasMore, setHasMore] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState(null);
  const pagesLoadedRef = useRef(0);
  const loadingRef = useRef(false);
  // The refresh currently in flight, so a caller that needs post-change
  // data can queue behind it instead of being turned away.
  const inflightRef = useRef(null);
  const pendingRef = useRef(null);
  const sentinelRef = useRef(null);
  const fetcherRef = useRef(fetcher);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => { fetcherRef.current = fetcher; }, [fetcher]);

  const reload = useCallback(async () => {
    if (loadingRef.current) return;
    loadingRef.current = true;
    setError(null);
    pagesLoadedRef.current = 0;
    try {
      const batch = await fetcherRef.current(pageSize, 0);
      setItems(batch);
      pagesLoadedRef.current = 1;
      setHasMore(batch.length === pageSize);
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      loadingRef.current = false;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pageSize]);

  const loadMore = useCallback(async () => {
    if (loadingRef.current || !hasMore) return;
    loadingRef.current = true;
    setLoadingMore(true);
    try {
      const offset = pagesLoadedRef.current * pageSize;
      const batch = await fetcherRef.current(pageSize, offset);
      setItems((prev) => [...(prev || []), ...batch]);
      pagesLoadedRef.current += 1;
      setHasMore(batch.length === pageSize);
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      setLoadingMore(false);
      loadingRef.current = false;
    }
  }, [hasMore, pageSize]);

  // The fetch itself. Wrapped below so callers can never stack more
  // than one of these behind the one that is running.
  const doRefresh = useCallback(async () => {
    const totalLoaded = pagesLoadedRef.current * pageSize;
    if (totalLoaded === 0) return reload();
    loadingRef.current = true;
    setError(null);
    try {
      const fresh = await fetcherRef.current(totalLoaded, 0);
      setItems(fresh);
      // hasMore stays sticky-true unless the refresh came up short,
      // which means rows were deleted and there might be fewer pages
      // than we thought.
      setHasMore(fresh.length === totalLoaded);
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      loadingRef.current = false;
    }
    return undefined;
  }, [pageSize, reload]);

  const refresh = useCallback(() => {
    // AT MOST ONE RUNNING AND ONE WAITING.
    //
    // This used to return immediately whenever a load was in flight,
    // which quietly broke every caller that awaits it to learn the
    // result of something it just changed: a delete landing during the
    // poll's request resolved against data fetched BEFORE the delete,
    // so the caller un-greyed the card and the row caught up a poll
    // later.
    //
    // Making it always fetch fixed that and introduced something worse.
    // The page polls every 3-8s, and this endpoint probes file metadata
    // for every row -- on production that takes longer than the
    // interval, so each tick queued another full fetch behind the last
    // and they piled up without ever draining. The server spent all its
    // time answering a backlog of its own most expensive query.
    //
    // So: one queued fetch, shared. A caller arriving now joins the
    // pending one, whose request starts AFTER this call -- which is all
    // "data from after my change" ever required.
    if (pendingRef.current) return pendingRef.current;

    const ahead = inflightRef.current;
    if (!ahead) {
      const p = doRefresh().finally(() => {
        if (inflightRef.current === p) inflightRef.current = null;
      });
      inflightRef.current = p;
      return p;
    }

    const queued = (async () => {
      try { await ahead; } catch { /* their failure is not ours */ }
      pendingRef.current = null;
      const p = doRefresh();
      inflightRef.current = p;
      try {
        await p;
      } finally {
        if (inflightRef.current === p) inflightRef.current = null;
      }
    })();
    pendingRef.current = queued;
    return queued;
  }, [doRefresh]);

  // Kick off page 1 on mount + whenever `deps` change. We intentionally
  // depend on `reload` (stable) and the caller-supplied deps so e.g.
  // an adminPassword change re-fetches from scratch.
  useEffect(() => {
    reload();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reload, ...deps]);

  // IntersectionObserver: when the sentinel comes within 300px of the
  // viewport, request the next page. 300px is enough headroom that
  // most users see no perceptible "load more" delay on a normal scroll.
  useEffect(() => {
    const node = sentinelRef.current;
    if (!node) return;
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0]?.isIntersecting) loadMore();
      },
      { rootMargin: "300px 0px" },
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, [loadMore]);

  return {
    items,
    hasMore,
    loadingMore,
    error,
    sentinelRef,
    reload,
    refresh,
    setItems,
  };
}
