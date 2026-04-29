# Import models first so their table definitions register with Base.metadata
# before init_db() calls Base.metadata.create_all().
from src.db import models as models  # noqa: F401 — side-effect import

from src.db.database import (
    DatabaseManager,
    db_manager,
    get_db,
    init_db,
    SessionLocal,
    engine,
    Base,
)

__all__ = [
    "DatabaseManager",
    "db_manager",
    "get_db",
    "init_db",
    "SessionLocal",
    "engine",
    "Base",
    "models",
]
