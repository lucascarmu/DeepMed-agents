"""
Database service — PostgreSQL connection pool and LangGraph checkpointer.

Uses psycopg_pool.ConnectionPool for production-grade connection management
and langgraph-checkpoint-postgres PostgresSaver for state persistence.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager

from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres import PostgresSaver

from app.config.settings import get_settings

logger = logging.getLogger(__name__)

# Module-level pool — initialised lazily via init_pool()
_pool: ConnectionPool | None = None


def init_pool() -> ConnectionPool:
    """
    Initialise the PostgreSQL connection pool.

    Must be called once at application startup (e.g. in FastAPI lifespan).
    """
    global _pool
    if _pool is not None:
        return _pool

    settings = get_settings()
    logger.info("Initialising PostgreSQL connection pool")

    _pool = ConnectionPool(
        conninfo=settings.database_url,
        min_size=2,
        max_size=10,
        kwargs={"autocommit": True, "row_factory": dict_row},
    )

    logger.info("PostgreSQL connection pool created successfully")
    return _pool


def get_pool() -> ConnectionPool:
    """Return the existing connection pool. Raises if not initialised."""
    if _pool is None:
        raise RuntimeError(
            "Database pool not initialised. Call init_pool() at startup."
        )
    return _pool


def close_pool() -> None:
    """Close the connection pool. Call at application shutdown."""
    global _pool
    if _pool is not None:
        logger.info("Closing PostgreSQL connection pool")
        _pool.close()
        _pool = None


def get_checkpointer() -> PostgresSaver:
    """
    Create and setup a PostgresSaver checkpointer using the connection pool.

    Calls setup() to ensure the required LangGraph tables exist.
    """
    pool = get_pool()
    checkpointer = PostgresSaver(pool)
    checkpointer.setup()
    logger.info("PostgresSaver checkpointer ready")
    return checkpointer
