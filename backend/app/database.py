from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from .config import settings


def _normalize_db_url(url: str) -> str:
    # Replit / Heroku-style URLs use bare "postgres://" or "postgresql://",
    # which SQLAlchemy routes to psycopg2 (v2). We ship psycopg v3, so
    # rewrite the scheme to force the v3 driver.
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://") :]
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://") :]
    return url


db_url = _normalize_db_url(settings.database_url)
connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}

# pool_pre_ping: Neon (serverless Postgres) closes idle connections after a
# short window, so a pooled connection can be dead by the next request —
# surfacing as "SSL connection has been closed unexpectedly" / "terminating
# connection due to administrator command" and a 500. pre_ping tests each
# connection before handing it out and transparently reconnects a dead one.
# pool_recycle caps a connection's age so it never outlives Neon's timeout.
engine_kwargs: dict = {
    "connect_args": connect_args,
    "future": True,
    "pool_pre_ping": True,
}
if not db_url.startswith("sqlite"):
    engine_kwargs["pool_recycle"] = 300  # seconds

engine = create_engine(db_url, **engine_kwargs)
# expire_on_commit=False: after a commit, ORM objects keep their loaded
# values instead of re-SELECTing on next attribute access. The re-SELECT
# silently OPENED A NEW TRANSACTION inside long-running workers (produce
# spends minutes on tracer work between DB touches), and Postgres then
# killed the connection with 'idle-in-transaction timeout'.
SessionLocal = sessionmaker(
    bind=engine, autoflush=False, autocommit=False, future=True,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
