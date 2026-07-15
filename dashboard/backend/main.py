"""Standalone personal dashboard.

One small FastAPI app that serves both the API and the built frontend:

  * /api/todos   — to-do list, persisted in a local SQLite file (dashboard.db)
  * /api/news    — Associated Press headlines for Business / Macroeconomics /
                   Politics, scraped from apnews.com hub pages (AP has no free
                   public API). Falls back to a Google News RSS query scoped
                   to site:apnews.com — still AP stories — if the scrape fails.
  * /api/quotes  — Yahoo Finance v8 chart endpoint: current price, previous
                   close and a week of intraday closes for NVDA, QQQM, the
                   S&P 500, QBE.AX (AUD, with a USD conversion) and RTX.
  * /            — the built SPA from ../frontend/dist

News is cached 10 minutes and quotes 90 seconds in-process, so a dashboard
left open and polling doesn't hammer either upstream.

Run:  uvicorn main:app --port 8100        (or ./start.sh from dashboard/)
"""
from __future__ import annotations

import asyncio
import html as html_lib
import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

app = FastAPI(title="Personal Dashboard", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

# ---------------------------------------------------------------------------
# Tiny in-process TTL cache
# ---------------------------------------------------------------------------
_cache: dict[str, tuple[float, object]] = {}


def _cache_get(key: str):
    hit = _cache.get(key)
    if hit and hit[0] > time.monotonic():
        return hit[1]
    return None


def _cache_put(key: str, value: object, ttl: float) -> None:
    _cache[key] = (time.monotonic() + ttl, value)


# ---------------------------------------------------------------------------
# To-dos (SQLite)
# ---------------------------------------------------------------------------
DB_PATH = Path(__file__).resolve().parent / "dashboard.db"


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE IF NOT EXISTS todos (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               text TEXT NOT NULL,
               done INTEGER NOT NULL DEFAULT 0,
               created_at TEXT NOT NULL,
               completed_at TEXT
           )"""
    )
    return conn


class TodoIn(BaseModel):
    text: str = Field(min_length=1, max_length=500)


class TodoPatch(BaseModel):
    text: Optional[str] = Field(default=None, min_length=1, max_length=500)
    done: Optional[bool] = None


def _todo_out(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "text": row["text"],
        "done": bool(row["done"]),
        "created_at": row["created_at"],
        "completed_at": row["completed_at"],
    }


@app.get("/api/todos")
def list_todos() -> list[dict]:
    with _db() as conn:
        rows = conn.execute(
            "SELECT * FROM todos ORDER BY done ASC, created_at DESC, id DESC"
        ).fetchall()
    return [_todo_out(r) for r in rows]


@app.post("/api/todos", status_code=201)
def create_todo(body: TodoIn) -> dict:
    now = datetime.utcnow().isoformat()
    with _db() as conn:
        cur = conn.execute(
            "INSERT INTO todos (text, done, created_at) VALUES (?, 0, ?)",
            (body.text.strip(), now),
        )
        row = conn.execute("SELECT * FROM todos WHERE id = ?", (cur.lastrowid,)).fetchone()
    return _todo_out(row)


@app.patch("/api/todos/{todo_id}")
def update_todo(todo_id: int, body: TodoPatch) -> dict:
    with _db() as conn:
        row = conn.execute("SELECT * FROM todos WHERE id = ?", (todo_id,)).fetchone()
        if not row:
            raise HTTPException(404, "todo not found")
        if body.text is not None:
            conn.execute("UPDATE todos SET text = ? WHERE id = ?", (body.text.strip(), todo_id))
        if body.done is not None:
            completed = datetime.utcnow().isoformat() if body.done else None
            conn.execute(
                "UPDATE todos SET done = ?, completed_at = ? WHERE id = ?",
                (1 if body.done else 0, completed, todo_id),
            )
        row = conn.execute("SELECT * FROM todos WHERE id = ?", (todo_id,)).fetchone()
    return _todo_out(row)


@app.delete("/api/todos/{todo_id}", status_code=204)
def delete_todo(todo_id: int) -> None:
    with _db() as conn:
        cur = conn.execute("DELETE FROM todos WHERE id = ?", (todo_id,))
        if cur.rowcount == 0:
            raise HTTPException(404, "todo not found")


# ---------------------------------------------------------------------------
# AP News
# ---------------------------------------------------------------------------
NEWS_SECTIONS = [
    # (key, label, AP hub slug, Google News fallback query)
    ("business", "Business", "business", "business"),
    ("macro", "Macroeconomics", "economy", 'economy OR inflation OR "federal reserve"'),
    ("politics", "Politics", "politics", "politics"),
]
NEWS_TTL = 600  # seconds
NEWS_LIMIT = 8

_AP_LINK_RE = re.compile(
    r'<a[^>]+href="(https://apnews\.com/article/[^"?#]+)"[^>]*>(.*?)</a>',
    re.DOTALL | re.IGNORECASE,
)
_TAG_RE = re.compile(r"<[^>]+>")
_RSS_ITEM_RE = re.compile(r"<item>.*?<title>(.*?)</title>.*?<link>(.*?)</link>", re.DOTALL)


def _clean_text(raw: str) -> str:
    text = _TAG_RE.sub(" ", raw)
    text = html_lib.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _parse_ap_hub(page_html: str) -> list[dict]:
    """Pull article headlines out of an apnews.com hub page.

    We match on the article-URL shape rather than CSS classes so minor AP
    redesigns don't break us. Anchors whose text is too short (icons,
    "Read more" chips) are skipped.
    """
    seen: set[str] = set()
    items: list[dict] = []
    for url, inner in _AP_LINK_RE.findall(page_html):
        title = _clean_text(inner)
        if len(title) < 25 or url in seen:
            continue
        seen.add(url)
        items.append({"title": title, "url": url})
        if len(items) >= NEWS_LIMIT:
            break
    return items


def _parse_google_news_rss(xml: str) -> list[dict]:
    items: list[dict] = []
    for title, link in _RSS_ITEM_RE.findall(xml):
        title = _clean_text(title)
        # Google News suffixes the source: "Headline - AP News"
        title = re.sub(r"\s+-\s+AP News$", "", title)
        link = _clean_text(link)
        if title and link:
            items.append({"title": title, "url": link})
        if len(items) >= NEWS_LIMIT:
            break
    return items


async def _fetch_section(
    client: httpx.AsyncClient, key: str, label: str, slug: str, fallback_q: str
) -> dict:
    # Primary: scrape the AP hub page directly.
    try:
        r = await client.get(f"https://apnews.com/hub/{slug}")
        r.raise_for_status()
        items = _parse_ap_hub(r.text)
        if items:
            return {"key": key, "label": label, "source": "apnews.com", "items": items}
    except Exception:
        pass

    # Fallback: Google News RSS restricted to apnews.com (still AP stories).
    try:
        q = httpx.QueryParams(
            {"q": f"site:apnews.com {fallback_q}", "hl": "en-US", "gl": "US", "ceid": "US:en"}
        )
        r = await client.get(f"https://news.google.com/rss/search?{q}")
        r.raise_for_status()
        items = _parse_google_news_rss(r.text)
        if items:
            return {"key": key, "label": label, "source": "AP via Google News", "items": items}
    except Exception:
        pass

    return {"key": key, "label": label, "source": None, "items": [],
            "error": "Couldn't reach AP News right now."}


@app.get("/api/news")
async def get_news() -> dict:
    cached = _cache_get("news")
    if cached is not None:
        return cached  # type: ignore[return-value]

    headers = {"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"}
    async with httpx.AsyncClient(headers=headers, timeout=15, follow_redirects=True) as client:
        sections = await asyncio.gather(
            *[_fetch_section(client, *cfg) for cfg in NEWS_SECTIONS]
        )
    payload = {"sections": list(sections), "fetched_at": datetime.utcnow().isoformat() + "Z"}
    # Don't cache a fully-failed fetch; let the next poll retry.
    if any(s["items"] for s in sections):
        _cache_put("news", payload, NEWS_TTL)
    return payload


# ---------------------------------------------------------------------------
# Yahoo Finance quotes + weekly charts
# ---------------------------------------------------------------------------
QUOTE_SYMBOLS = [
    # (yahoo symbol, display name)
    ("NVDA", "Nvidia"),
    ("QQQM", "QQQM (Nasdaq-100 ETF)"),
    ("^GSPC", "S&P 500"),
    ("QBE.AX", "QBE Insurance (ASX)"),
    ("RTX", "RTX"),
]
FX_SYMBOL = "AUDUSD=X"  # to show QBE.AX in USD alongside AUD
QUOTES_TTL = 90  # seconds
YAHOO_HOSTS = ["query1.finance.yahoo.com", "query2.finance.yahoo.com"]


async def _fetch_chart(client: httpx.AsyncClient, symbol: str) -> Optional[dict]:
    """One symbol's 5-day / 30-minute chart from Yahoo's v8 endpoint."""
    params = {"range": "5d", "interval": "30m", "includePrePost": "false"}
    for host in YAHOO_HOSTS:
        try:
            r = await client.get(f"https://{host}/v8/finance/chart/{symbol}", params=params)
            r.raise_for_status()
            result = (r.json().get("chart") or {}).get("result") or []
            if result:
                return result[0]
        except Exception:
            continue
    return None


def _series_from_chart(chart: dict) -> list[dict]:
    timestamps = chart.get("timestamp") or []
    quote = ((chart.get("indicators") or {}).get("quote") or [{}])[0]
    closes = quote.get("close") or []
    return [
        {"t": ts, "p": round(float(c), 4)}
        for ts, c in zip(timestamps, closes)
        if c is not None
    ]


def _previous_close(chart: dict, series: list[dict]) -> Optional[float]:
    """Prior trading day's close, for the daily change figure.

    Yahoo's meta.previousClose is the most direct answer but isn't always
    present; if missing, take the last close of the second-to-last trading
    day in the series (using the exchange's UTC offset to bucket by day);
    finally fall back to chartPreviousClose (close before the 5d window —
    that makes the figure a weekly change, still better than nothing).
    """
    meta = chart.get("meta") or {}
    prev = meta.get("previousClose")
    if prev:
        return float(prev)

    offset = meta.get("gmtoffset") or 0
    days: list[tuple[int, float]] = []  # (yyyymmdd, last close that day)
    for point in series:
        local = datetime.utcfromtimestamp(point["t"] + offset)
        day = local.year * 10000 + local.month * 100 + local.day
        if days and days[-1][0] == day:
            days[-1] = (day, point["p"])
        else:
            days.append((day, point["p"]))
    if len(days) >= 2:
        return days[-2][1]

    chart_prev = meta.get("chartPreviousClose")
    return float(chart_prev) if chart_prev else None


def _quote_payload(symbol: str, name: str, chart: Optional[dict], aud_usd: Optional[float]) -> dict:
    if not chart:
        return {"symbol": symbol, "name": name, "error": "Couldn't reach Yahoo Finance."}

    meta = chart.get("meta") or {}
    series = _series_from_chart(chart)
    price = meta.get("regularMarketPrice")
    if price is None and series:
        price = series[-1]["p"]
    if price is None:
        return {"symbol": symbol, "name": name, "error": "No price data returned."}
    price = float(price)

    prev_close = _previous_close(chart, series)
    day_change = day_change_pct = None
    if prev_close:
        day_change = price - prev_close
        day_change_pct = day_change / prev_close * 100

    week_change_pct = None
    if series:
        first = series[0]["p"]
        if first:
            week_change_pct = (price - first) / first * 100

    out = {
        "symbol": symbol,
        "name": name,
        "currency": meta.get("currency") or "USD",
        "price": price,
        "prev_close": prev_close,
        "day_change": day_change,
        "day_change_pct": day_change_pct,
        "week_change_pct": week_change_pct,
        "market_time": meta.get("regularMarketTime"),
        "exchange": meta.get("exchangeName"),
        "series": series,
    }
    # QBE trades in AUD on the ASX; surface a USD equivalent alongside.
    if symbol == "QBE.AX" and aud_usd:
        out["usd_price"] = price * aud_usd
        out["aud_usd_rate"] = aud_usd
    return out


@app.get("/api/quotes")
async def get_quotes() -> dict:
    cached = _cache_get("quotes")
    if cached is not None:
        return cached  # type: ignore[return-value]

    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    async with httpx.AsyncClient(headers=headers, timeout=15, follow_redirects=True) as client:
        charts = await asyncio.gather(
            *[_fetch_chart(client, sym) for sym, _ in QUOTE_SYMBOLS],
            _fetch_chart(client, FX_SYMBOL),
        )

    fx_chart = charts[-1]
    aud_usd = None
    if fx_chart:
        fx_meta = fx_chart.get("meta") or {}
        rate = fx_meta.get("regularMarketPrice")
        if rate is None:
            fx_series = _series_from_chart(fx_chart)
            rate = fx_series[-1]["p"] if fx_series else None
        aud_usd = float(rate) if rate else None

    quotes = [
        _quote_payload(sym, name, chart, aud_usd)
        for (sym, name), chart in zip(QUOTE_SYMBOLS, charts[:-1])
    ]
    payload = {"quotes": quotes, "fetched_at": datetime.utcnow().isoformat() + "Z"}
    if any("error" not in q for q in quotes):
        _cache_put("quotes", payload, QUOTES_TTL)
    return payload


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "app": "dashboard"}


# ---------------------------------------------------------------------------
# Static frontend (built SPA) — mounted last so /api/* wins.
# ---------------------------------------------------------------------------
FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="spa")
