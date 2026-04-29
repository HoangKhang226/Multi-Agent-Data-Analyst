"""
database.py — SQLAlchemy database manager (OOP refactor)

Provides a `DatabaseManager` class that owns the engine, session factory,
and the declarative `Base`. Module-level singletons are exposed for
backward-compatibility with existing importers (models.py, FastAPI deps).
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, Engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from src.core.config import settings
from src.utils.logger import logger


class DatabaseManager:
    """Encapsulates SQLAlchemy engine, session factory, and schema helpers.

    Usage
    -----
    db_manager = DatabaseManager(url="sqlite:///storage/app.db")
    db_manager.init_db()          # create all tables
    with db_manager.session() as db:
        db.execute(...)
    """

    def __init__(self, url: str | None = None) -> None:
        self._url: str = url or settings.database.url
        self._engine: Engine | None = None
        self._session_factory: sessionmaker | None = None
        self._base = declarative_base()

        self._ensure_storage_dir()
        self._create_engine()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_storage_dir(self) -> None:
        """Create the directory for SQLite DB files if it does not exist."""
        if not self._url.startswith("sqlite"):
            return
        db_path = self._url.replace("sqlite:///", "")
        if db_path.startswith(":memory:"):
            return
        abs_path = os.path.abspath(db_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        logger.debug(f"Ensured storage directory: {os.path.dirname(abs_path)}")

    def _create_engine(self) -> None:
        """Instantiate the SQLAlchemy engine with appropriate driver kwargs."""
        connect_args = {"check_same_thread": False} if "sqlite" in self._url else {}
        self._engine = create_engine(self._url, connect_args=connect_args)
        self._session_factory = sessionmaker(
            autocommit=False, autoflush=False, bind=self._engine
        )
        logger.debug(f"Database engine created for: {self._url}")

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def engine(self) -> Engine:
        """Return the underlying SQLAlchemy engine."""
        if self._engine is None:
            raise RuntimeError("Engine not initialised. Call _create_engine() first.")
        return self._engine

    @property
    def base(self):
        """Return the declarative Base used by all ORM models."""
        return self._base

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    def init_db(self) -> None:
        """Create all tables declared against ``Base.metadata``.

        **Important**: all ORM model modules must be imported *before* calling
        this method so their table definitions are registered with the shared
        ``Base.metadata``.  The ``src.db`` package ``__init__.py`` handles
        this automatically when you do ``from src.db import init_db``.
        """
        try:
            self._base.metadata.create_all(bind=self._engine)
            logger.info("Database tables created / verified successfully.")
        except Exception as exc:
            logger.error(f"Error initialising database: {exc}")
            raise

    def drop_all(self) -> None:
        """Drop ALL tables. Intended for tests only — use with caution."""
        self._base.metadata.drop_all(bind=self._engine)
        logger.warning("All database tables have been dropped.")

    def health_check(self) -> bool:
        """Return True if the database is reachable, False otherwise."""
        try:
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as exc:
            logger.error(f"Database health check failed: {exc}")
            return False

    # ------------------------------------------------------------------
    # Session helpers
    # ------------------------------------------------------------------

    def get_session(self) -> Session:
        """Return a new raw Session. Caller is responsible for closing it."""
        return self._session_factory()

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """Context-manager that yields a session and handles commit/rollback.

        Example
        -------
        with db_manager.session() as db:
            db.add(SomeModel(...))
        """
        db: Session = self._session_factory()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def get_db(self) -> Generator[Session, None, None]:
        """FastAPI-compatible generator dependency.

        Yields a session and ensures it is closed after the request.

        Example
        -------
        @app.get("/")
        def route(db: Session = Depends(db_manager.get_db)):
            ...
        """
        db: Session = self._session_factory()
        try:
            yield db
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Module-level singleton — created once at import time
# ---------------------------------------------------------------------------

db_manager = DatabaseManager()

# ---------------------------------------------------------------------------
# Module-level aliases for backward compatibility
# (models.py, crud.py, and FastAPI routers import these directly)
# ---------------------------------------------------------------------------

#: The shared SQLAlchemy Engine instance.
engine = db_manager.engine

#: Session factory bound to the shared engine.
SessionLocal = db_manager._session_factory

#: Declarative Base — all ORM models must inherit from this.
Base = db_manager.base


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a database session per request."""
    yield from db_manager.get_db()


def init_db() -> None:
    """Create all registered ORM tables. Call once at application startup."""
    db_manager.init_db()
