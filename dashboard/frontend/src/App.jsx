import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api.js";

/* ------------------------------------------------------------------ */
/* World clocks — Chicago, NYC, Phoenix, London, Sydney (in order)     */
/* ------------------------------------------------------------------ */
const CLOCK_CITIES = [
  { label: "Chicago", tz: "America/Chicago" },
  { label: "New York", tz: "America/New_York" },
  { label: "Phoenix", tz: "America/Phoenix" },
  { label: "London", tz: "Europe/London" },
  { label: "Sydney", tz: "Australia/Sydney" },
];

function cityTime(tz, now) {
  const time = new Intl.DateTimeFormat("en-US", {
    timeZone: tz,
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
    hour12: true,
  }).format(now);
  const day = new Intl.DateTimeFormat("en-US", { timeZone: tz, weekday: "short" }).format(now);
  return { time, day };
}

function ClockBar() {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  }, []);
  return (
    <div className="dash-clocks" role="group" aria-label="World clocks">
      {CLOCK_CITIES.map(({ label, tz }) => {
        const { time, day } = cityTime(tz, now);
        return (
          <div className="dash-clock" key={tz}>
            <div className="dash-clock-city">{label}</div>
            <div className="dash-clock-time">{time}</div>
            <div className="dash-clock-day">{day}</div>
          </div>
        );
      })}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Week line chart (5 trading days of intraday closes)                 */
/* ------------------------------------------------------------------ */
const CHART_W = 320;
const CHART_H = 110;
const PAD = { top: 8, right: 8, bottom: 18, left: 44 };

function WeekChart({ series, tz, formatPrice }) {
  const [hover, setHover] = useState(null); // index into series
  const svgRef = useRef(null);

  const geo = useMemo(() => {
    if (!series || series.length < 2) return null;
    const w = CHART_W - PAD.left - PAD.right;
    const h = CHART_H - PAD.top - PAD.bottom;
    const prices = series.map((d) => d.p);
    let min = Math.min(...prices);
    let max = Math.max(...prices);
    if (min === max) { min -= 1; max += 1; }
    const span = max - min;
    min -= span * 0.06;
    max += span * 0.06;
    // Points spaced by index, not wall-clock time, so overnight/weekend
    // gaps don't render as long flat runs.
    const x = (i) => PAD.left + (i / (series.length - 1)) * w;
    const y = (p) => PAD.top + (1 - (p - min) / (max - min)) * h;
    const pts = series.map((d, i) => [x(i), y(d.p)]);
    const line = pts.map(([px, py], i) => `${i ? "L" : "M"}${px.toFixed(1)},${py.toFixed(1)}`).join("");
    const area = `${line}L${pts[pts.length - 1][0].toFixed(1)},${(PAD.top + h).toFixed(1)}L${pts[0][0].toFixed(1)},${(PAD.top + h).toFixed(1)}Z`;

    // Day tick labels centered under each trading day's run of points.
    const dayFmt = new Intl.DateTimeFormat("en-US", { timeZone: tz, weekday: "short" });
    const groups = [];
    series.forEach((d, i) => {
      const label = dayFmt.format(new Date(d.t * 1000));
      const last = groups[groups.length - 1];
      if (last && last.label === label) last.end = i;
      else groups.push({ label, start: i, end: i });
    });
    const gridYs = [PAD.top, PAD.top + h / 2, PAD.top + h];
    const gridVals = [max, (max + min) / 2, min];
    return { pts, line, area, groups, x, gridYs, gridVals, h };
  }, [series, tz]);

  const onMove = useCallback(
    (e) => {
      if (!geo || !svgRef.current) return;
      const rect = svgRef.current.getBoundingClientRect();
      const px = ((e.clientX - rect.left) / rect.width) * CHART_W;
      const frac = (px - PAD.left) / (CHART_W - PAD.left - PAD.right);
      const i = Math.round(frac * (series.length - 1));
      setHover(Math.max(0, Math.min(series.length - 1, i)));
    },
    [geo, series]
  );

  if (!geo) return <div className="dash-chart-empty">No chart data</div>;

  const last = geo.pts[geo.pts.length - 1];
  const hoverPt = hover != null ? geo.pts[hover] : null;
  const hoverDatum = hover != null ? series[hover] : null;
  const timeFmt = new Intl.DateTimeFormat("en-US", {
    timeZone: tz, weekday: "short", hour: "numeric", minute: "2-digit",
  });

  return (
    <div className="dash-chart">
      <svg
        ref={svgRef}
        viewBox={`0 0 ${CHART_W} ${CHART_H}`}
        className="dash-chart-svg"
        onPointerMove={onMove}
        onPointerLeave={() => setHover(null)}
        role="img"
        aria-label="One-week price chart"
      >
        {geo.gridYs.map((gy, i) => (
          <g key={i}>
            <line x1={PAD.left} x2={CHART_W - PAD.right} y1={gy} y2={gy} className="dash-grid" />
            <text x={PAD.left - 6} y={gy + 3} className="dash-axis-text" textAnchor="end">
              {formatPrice(geo.gridVals[i], { compact: true })}
            </text>
          </g>
        ))}
        {geo.groups.map((g) => (
          <text
            key={g.start}
            x={(geo.x(g.start) + geo.x(g.end)) / 2}
            y={CHART_H - 4}
            className="dash-axis-text"
            textAnchor="middle"
          >
            {g.label}
          </text>
        ))}
        <path d={geo.area} className="dash-area" />
        <path d={geo.line} className="dash-line" />
        {hoverPt && (
          <line
            x1={hoverPt[0]} x2={hoverPt[0]}
            y1={PAD.top} y2={PAD.top + geo.h}
            className="dash-crosshair"
          />
        )}
        <circle
          cx={(hoverPt || last)[0]} cy={(hoverPt || last)[1]}
          r="4" className="dash-dot"
        />
      </svg>
      {hoverDatum && (
        <div
          className="dash-tooltip"
          style={{
            left: `${((hoverPt[0] / CHART_W) * 100).toFixed(1)}%`,
            transform: hoverPt[0] > CHART_W * 0.62 ? "translate(-100%, 0)" : "none",
          }}
        >
          <span className="dash-tooltip-value">{formatPrice(hoverDatum.p)}</span>
          <span className="dash-tooltip-time">{timeFmt.format(new Date(hoverDatum.t * 1000))}</span>
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Market quote card                                                   */
/* ------------------------------------------------------------------ */
const EXCHANGE_TZ = { "QBE.AX": "Australia/Sydney" };

function money(value, currency, { compact = false } = {}) {
  if (value == null || Number.isNaN(value)) return "—";
  const digits = compact && Math.abs(value) >= 1000 ? 0 : 2;
  const num = value.toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
  if (currency === "index") return num;
  if (currency === "AUD") return `A$${num}`;
  return `$${num}`;
}

function QuoteCard({ q }) {
  if (q.error) {
    return (
      <div className="card">
        <div className="dash-quote-head">
          <div>
            <div className="dash-quote-name">{q.name}</div>
            <div className="dash-quote-symbol">{q.symbol}</div>
          </div>
        </div>
        <div className="dash-error">{q.error}</div>
      </div>
    );
  }
  const currency = q.symbol === "^GSPC" ? "index" : q.currency;
  const fmt = (v, opts) => money(v, currency, opts);
  const up = (q.day_change ?? 0) >= 0;
  const tz = EXCHANGE_TZ[q.symbol] || "America/New_York";

  return (
    <div className="card">
      <div className="dash-quote-head">
        <div>
          <div className="dash-quote-name">{q.name}</div>
          <div className="dash-quote-symbol">{q.symbol}</div>
        </div>
        <div className="dash-quote-price-wrap">
          <div className="dash-quote-price">{fmt(q.price)}</div>
          {q.usd_price != null && (
            <div className="dash-quote-usd">≈ US${q.usd_price.toFixed(2)}</div>
          )}
        </div>
      </div>
      <div className="dash-quote-deltas">
        <span className={up ? "dash-delta-up" : "dash-delta-down"}>
          {q.day_change != null
            ? `${up ? "▲" : "▼"} ${fmt(Math.abs(q.day_change))} (${Math.abs(q.day_change_pct).toFixed(2)}%)`
            : "—"}
          <span className="dash-delta-label"> today</span>
        </span>
        {q.week_change_pct != null && (
          <span className="dash-delta-week">
            {q.week_change_pct >= 0 ? "+" : ""}
            {q.week_change_pct.toFixed(2)}% this week
          </span>
        )}
      </div>
      <WeekChart series={q.series} tz={tz} formatPrice={fmt} />
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* To-do list                                                          */
/* ------------------------------------------------------------------ */
function TodoCard() {
  const [todos, setTodos] = useState(null);
  const [text, setText] = useState("");
  const [err, setErr] = useState("");

  const refresh = useCallback(async () => {
    try {
      setTodos(await api.todos());
      setErr("");
    } catch (e) {
      setErr(String(e.message || e));
    }
  }, []);
  useEffect(() => { refresh(); }, [refresh]);

  async function add(e) {
    e.preventDefault();
    const t = text.trim();
    if (!t) return;
    setText("");
    try {
      await api.addTodo(t);
      await refresh();
    } catch (e2) {
      setErr(String(e2.message || e2));
    }
  }
  async function toggle(todo) {
    try {
      await api.updateTodo(todo.id, { done: !todo.done });
      await refresh();
    } catch (e2) {
      setErr(String(e2.message || e2));
    }
  }
  async function remove(todo) {
    try {
      await api.deleteTodo(todo.id);
      await refresh();
    } catch (e2) {
      setErr(String(e2.message || e2));
    }
  }

  const open = (todos || []).filter((t) => !t.done).length;
  return (
    <div className="card dash-todo">
      <div className="dash-section-title">
        To-do
        {todos && <span className="dash-count">{open} open</span>}
      </div>
      <form onSubmit={add} className="dash-todo-form">
        <input
          type="text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Add a task…"
          maxLength={500}
          aria-label="New to-do"
        />
        <button type="submit" disabled={!text.trim()}>Add</button>
      </form>
      {err && <div className="dash-error">{err}</div>}
      {todos == null && !err && <div className="dash-muted">Loading…</div>}
      {todos != null && todos.length === 0 && (
        <div className="dash-muted">Nothing yet — add your first task above.</div>
      )}
      <ul className="dash-todo-list">
        {(todos || []).map((t) => (
          <li key={t.id} className={t.done ? "done" : ""}>
            <label>
              <input type="checkbox" checked={t.done} onChange={() => toggle(t)} />
              <span className="dash-todo-text">{t.text}</span>
            </label>
            <button
              className="dash-todo-del"
              onClick={() => remove(t)}
              aria-label={`Delete "${t.text}"`}
              title="Delete"
            >
              ×
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* News columns                                                        */
/* ------------------------------------------------------------------ */
function NewsCard({ section }) {
  return (
    <div className="card">
      <div className="dash-section-title">
        {section.label}
        <span className="dash-count">AP News</span>
      </div>
      {section.error && <div className="dash-error">{section.error}</div>}
      <ul className="dash-news-list">
        {section.items.map((item) => (
          <li key={item.url}>
            <a href={item.url} target="_blank" rel="noreferrer noopener">{item.title}</a>
          </li>
        ))}
      </ul>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* App                                                                 */
/* ------------------------------------------------------------------ */
const QUOTES_POLL_MS = 60_000;
const NEWS_POLL_MS = 10 * 60_000;

export default function App() {
  const [quotes, setQuotes] = useState(null);
  const [quotesErr, setQuotesErr] = useState("");
  const [news, setNews] = useState(null);
  const [newsErr, setNewsErr] = useState("");

  useEffect(() => {
    let alive = true;
    async function loadQuotes() {
      try {
        const data = await api.quotes();
        if (alive) { setQuotes(data); setQuotesErr(""); }
      } catch (e) {
        if (alive) setQuotesErr(String(e.message || e));
      }
    }
    loadQuotes();
    const id = setInterval(loadQuotes, QUOTES_POLL_MS);
    return () => { alive = false; clearInterval(id); };
  }, []);

  useEffect(() => {
    let alive = true;
    async function loadNews() {
      try {
        const data = await api.news();
        if (alive) { setNews(data); setNewsErr(""); }
      } catch (e) {
        if (alive) setNewsErr(String(e.message || e));
      }
    }
    loadNews();
    const id = setInterval(loadNews, NEWS_POLL_MS);
    return () => { alive = false; clearInterval(id); };
  }, []);

  const updatedAt = quotes?.fetched_at
    ? new Date(quotes.fetched_at).toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" })
    : null;

  return (
    <div className="dash-wrap">
      <ClockBar />

      <div className="dash-header">
        <h1>Dashboard</h1>
        {updatedAt && <span className="dash-muted">Markets updated {updatedAt}</span>}
      </div>

      <section aria-label="Markets">
        {quotesErr && !quotes && <div className="card dash-error">Markets: {quotesErr}</div>}
        {!quotes && !quotesErr && <div className="card dash-muted">Loading market data…</div>}
        {quotes && (
          <div className="dash-quotes-grid">
            {quotes.quotes.map((q) => <QuoteCard key={q.symbol} q={q} />)}
          </div>
        )}
      </section>

      <div className="dash-bottom-grid">
        <TodoCard />
        {newsErr && !news && <div className="card dash-error">News: {newsErr}</div>}
        {!news && !newsErr && <div className="card dash-muted">Loading headlines…</div>}
        {news && news.sections.map((s) => <NewsCard key={s.key} section={s} />)}
      </div>
    </div>
  );
}
