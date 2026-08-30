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
 *   - `loadMore()`      : fetch the next page explicitly. Callers
 *                         should put a button on this: auto-load
 *                         depends on the browser noticing the sentinel
 *                         moved, and a button does not
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
  // A next-page request that arrived while something else held the
  // loading flag. See releaseLoading() for why this exists.
  const wantMoreRef = useRef(false);
  // True while `releaseLoading` is handing the loading flag straight to
  // a queued page rather than freeing it. The queued call owns the flag
  // already and must not test for it. See releaseLoading().
  const drainingRef = useRef(false);
  const loadMoreRef = useRef(null);
  // The refresh currently in flight, so a caller that needs post-change
  // data can queue behind it instead of being turned away.
  const inflightRef = useRef(null);
  const pendingRef = useRef(null);
  const sentinelRef = useRef(null);
  const fetcherRef = useRef(fetcher);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => { fetcherRef.current = fetcher; }, [fetcher]);

  // DROPPING A PAGE REQUEST IS PERMANENT, WHICH IS WHY PAGING FELT
  // BROKEN. The observer only fires on a CHANGE in intersection. If the
  // sentinel came into view while the poll's refresh held the loading
  // flag -- and that refresh probes file metadata for every loaded row,
  // so on production it is in flight a good fraction of the time -- the
  // callback ran, found the flag set, and returned. The sentinel was
  // then still on screen, so no further event ever came: the page sat
  // there saying "scroll for more" until you scrolled away and back.
  //
  // So a blocked request is remembered rather than discarded, and
  // whatever was holding the flag runs it on the way out.
  const releaseLoading = useCallback(() => {
    if (wantMoreRef.current) {
      wantMoreRef.current = false;
      // HAND THE FLAG OVER, DO NOT DROP IT.
      //
      // This cleared the flag first and scheduled the drain on a
      // macrotask, which leaves a window with the flag free. The
      // Production page polls every three seconds while anything is
      // producing and that refresh probes file metadata for every
      // loaded row, so it is slow and frequent: it takes the flag
      // inside the window, the drained loadMore finds it held, queues
      // ITSELF again, and the page never arrives. `setLoadingMore(true)`
      // is after that check, so the button did not even change to
      // "Loading more uploads…" -- it looked dead, which is what it was.
      //
      // The flag stays held across the handover and the queued page
      // releases it when it is done. Nothing can get in between.
      drainingRef.current = true;
      // Out of this call stack: the state updates from the load that
      // just finished have to land first.
      setTimeout(() => loadMoreRef.current?.(), 0);
      return;
    }
    loadingRef.current = false;
  }, []);

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
      releaseLoading();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pageSize, releaseLoading]);

  const loadMore = useCallback(async () => {
    // Whether this call already owns the loading flag, handed over by
    // releaseLoading rather than taken here.
    const owned = drainingRef.current;
    drainingRef.current = false;
    if (!hasMore) {
      if (owned) loadingRef.current = false;
      setLoadingMore(false);
      return;
    }
    if (!owned && loadingRef.current) {
      wantMoreRef.current = true;
      // SAY THAT THE CLICK LANDED. `setLoadingMore(true)` was below
      // this check, so a queued page left the button reading "Load 25
      // more uploads" exactly as before it was pressed -- indis-
      // tinguishable from a dead button, which is how it was reported.
      setLoadingMore(true);
      return;
    }
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
      releaseLoading();
    }
  }, [hasMore, pageSize, releaseLoading]);

  // The drain in releaseLoading needs to reach the CURRENT loadMore,
  // not the one captured when the flag was taken. Assigned during
  // render rather than in an effect: a queued page must not depend on
  // an effect having run, which is exactly the kind of ordering
  // assumption that made paging unreliable in the first place.
  loadMoreRef.current = loadMore;

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
      releaseLoading();
    }
    return undefined;
  }, [pageSize, reload, releaseLoading]);

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
    loadMore,
    reload,
    refresh,
    setItems,
  };
}
