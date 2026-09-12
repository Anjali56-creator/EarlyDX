"""SQLAlchemy models and session handling.

Persistence is best-effort: if the database cannot be reached the API still
serves predictions (it just does not record them) and ``/health`` reports the
database as unavailable. Production uses PostgreSQL (see docker-compose.yml);
without a ``DATABASE_URL`` the app falls back to a local SQLite file.
"""
from __future__ import annotations

import datetime as dt
import json
from typing import Any, Iterator

from sqlalchemy import (
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    text,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

from backend.app.config import get_settings


class Base(DeclarativeBase):
    pass


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class AssessmentSession(Base):
    __tablename__ = "assessment_session"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[dt.datetime] = mapped_column(default=_utcnow)
    client_context: Mapped[str] = mapped_column(String(32), default="app")
    mode: Mapped[str] = mapped_column(String(16), default="single")  # single | all

    results: Mapped[list["PredictionResult"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class PredictionResult(Base):
    __tablename__ = "prediction_result"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("assessment_session.id"))
    disease: Mapped[str] = mapped_column(String(64))
    model_id: Mapped[str] = mapped_column(String(64))
    risk_score: Mapped[float] = mapped_column(Float)
    risk_level: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[dt.datetime] = mapped_column(default=_utcnow)

    session: Mapped[AssessmentSession] = relationship(back_populates="results")


class ModelVersion(Base):
    __tablename__ = "model_version"

    model_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    disease: Mapped[str] = mapped_column(String(64))
    version: Mapped[str] = mapped_column(String(16))
    algorithm: Mapped[str] = mapped_column(String(64))
    metrics_json: Mapped[str] = mapped_column(Text, default="{}")
    status: Mapped[str] = mapped_column(String(16), default="active")
    artifact_path: Mapped[str] = mapped_column(String(256))
    synced_at: Mapped[dt.datetime] = mapped_column(default=_utcnow)


class DatasetMetadata(Base):
    __tablename__ = "dataset_metadata"

    disease: Mapped[str] = mapped_column(String(64), primary_key=True)
    dataset_name: Mapped[str] = mapped_column(String(128), default="unknown")
    source: Mapped[str] = mapped_column(String(256), default="unknown")
    license: Mapped[str] = mapped_column(String(256), default="unknown")
    records: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(32), default="unknown")
    synced_at: Mapped[dt.datetime] = mapped_column(default=_utcnow)


class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ts: Mapped[dt.datetime] = mapped_column(default=_utcnow)
    action: Mapped[str] = mapped_column(String(64))
    entity: Mapped[str] = mapped_column(String(64))
    detail_json: Mapped[str] = mapped_column(Text, default="{}")


# ── engine / session ────────────────────────────────────────────────────────
_settings = get_settings()
_url = _settings.resolved_database_url
_connect_args = {"check_same_thread": False} if _url.startswith("sqlite") else {}
engine = create_engine(_url, connect_args=_connect_args, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

DB_AVAILABLE = False
DB_ERROR: str | None = None


def init_db() -> bool:
    """Create tables if the database is reachable. Never raises."""
    global DB_AVAILABLE, DB_ERROR
    try:
        Base.metadata.create_all(engine)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        DB_AVAILABLE = True
        DB_ERROR = None
    except Exception as exc:  # noqa: BLE001
        DB_AVAILABLE = False
        DB_ERROR = f"{type(exc).__name__}: {exc}"
    return DB_AVAILABLE


def get_session() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def record_assessment(
    db: Session, *, mode: str, client_context: str, results: list[dict[str, Any]]
) -> int | None:
    """Persist one assessment and its results. Returns the session id or None."""
    if not DB_AVAILABLE:
        return None
    try:
        sess = AssessmentSession(mode=mode, client_context=client_context)
        db.add(sess)
        db.flush()
        for r in results:
            db.add(
                PredictionResult(
                    session_id=sess.id,
                    disease=r.get("disease_key", r.get("disease", "unknown")),
                    model_id=r["model_version"],
                    risk_score=float(r["risk_score"]),
                    risk_level=r["risk_level"],
                )
            )
        db.add(
            AuditLog(
                action="predict",
                entity=mode,
                detail_json=json.dumps(
                    {"n_results": len(results), "client_context": client_context}
                ),
            )
        )
        db.commit()
        return sess.id
    except Exception:  # noqa: BLE001
        db.rollback()
        return None


def recent_assessments(db: Session, *, limit: int = 10) -> dict[str, Any]:
    """The most recent stored prediction results, newest first, plus the total
    number of assessment sessions. Best-effort: returns an empty, flagged
    payload if the database is unavailable rather than failing the request."""
    if not DB_AVAILABLE:
        return {"available": False, "total_sessions": 0, "results": []}
    try:
        total = db.query(AssessmentSession).count()
        rows = (
            db.query(PredictionResult)
            .order_by(PredictionResult.created_at.desc(), PredictionResult.id.desc())
            .limit(limit)
            .all()
        )
        return {
            "available": True,
            "total_sessions": total,
            "results": [
                {
                    "session_id": r.session_id,
                    "disease_key": r.disease,
                    "model_id": r.model_id,
                    "risk_score": r.risk_score,
                    "risk_level": r.risk_level,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in rows
            ],
        }
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        return {"available": False, "total_sessions": 0, "results": [], "error": str(exc)}


def sync_registries(db: Session, models: dict[str, Any], datasets: dict[str, Any]) -> None:
    if not DB_AVAILABLE:
        return
    try:
        for entry in models.get("models", {}).values():
            db.merge(
                ModelVersion(
                    model_id=entry["model_id"],
                    disease=entry["disease"],
                    version=entry.get("version", "v1"),
                    algorithm=entry.get("algorithm", "unknown"),
                    metrics_json=json.dumps(entry.get("metrics", {})),
                    status=entry.get("status", "active"),
                    artifact_path=entry.get("artifact_path", ""),
                )
            )
        for key, entry in datasets.items():
            if key.startswith("_"):
                continue
            db.merge(
                DatasetMetadata(
                    disease=key,
                    dataset_name=str(entry.get("dataset_name", "unknown"))[:128],
                    source=str(entry.get("source", "unknown"))[:256],
                    license=str(entry.get("license", "unknown"))[:256],
                    records=int(entry.get("records", 0) or 0),
                    status=str(entry.get("status", "unknown")),
                )
            )
        db.commit()
    except Exception:  # noqa: BLE001
        db.rollback()
