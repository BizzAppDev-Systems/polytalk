# SPDX-FileCopyrightText: 2026 BizzAppDev Systems Pvt. Ltd.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
PolyTalk - Speech-to-Speech Translation Web App

Main FastAPI application entry point.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import get_config
from .routers import api_router, web_router
from .routers.api import close_pipeline_service, close_visual_context_service
from .utils.logger import get_logger
from .version import __version__

logger = get_logger(__name__)


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance
    """
    app = FastAPI(
        title="PolyTalk",
        description="Speech-to-speech translation web application",
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    allowed_origins = os.environ.get(
        "ALLOWED_ORIGINS", "http://localhost:9000,http://127.0.0.1:9000"
    ).split(",")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(web_router)
    app.include_router(api_router)

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    media_dir = get_config().media_output_dir
    media_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/media/output", StaticFiles(directory=str(media_dir)), name="media")

    logger.info("PolyTalk application initialized")
    logger.info(f"Mock mode enabled: {get_config().whisper.get('mock_mode', True)}")

    return app


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    logger.info("PolyTalk starting up...")
    logger.info(
        f"Server will be available at http://{get_config().host}:{get_config().port}"
    )
    try:
        yield
    finally:
        await close_pipeline_service()
        await close_visual_context_service()
        logger.info("PolyTalk shutting down...")


app = create_app()


if __name__ == "__main__":
    import uvicorn

    config = get_config()
    uvicorn.run(
        "app.main:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
    )
