"""ACP routes package — assembles sub-module routers into a single router.

Usage (unchanged from the original single-file module)::

    from multihead.api.routes_acp import router as acp_router
"""

from __future__ import annotations

from fastapi import APIRouter

from ._connection import router as connection_router
from ._tasks import router as tasks_router
from ._marketplace import router as marketplace_router
from ._executions import router as executions_router

router = APIRouter()
router.include_router(connection_router)
router.include_router(tasks_router)
router.include_router(marketplace_router)
router.include_router(executions_router)

__all__ = ["router"]
