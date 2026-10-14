# SPDX-FileCopyrightText: 2026 BizzAppDev Systems Pvt. Ltd.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Web router for PolyTalk application.

Serves the frontend pages and static assets.
"""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

from ..config import get_config
from ..i18n import preferred_locale, public_catalog, t, ui_locale_options
from ..utils.config import get_custom_instruction_max_chars
from ..utils.translation_buffer import (
    TRANSLATION_BUFFER_SECONDS_MAX,
    TRANSLATION_BUFFER_SECONDS_MIN,
    TRANSLATION_BUFFER_SECONDS_STEP,
    clamp_translation_buffer_seconds,
)

router = APIRouter(tags=["web"])

templates = Jinja2Templates(directory=Path(__file__).parent.parent / "templates")


@router.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    """
    Serve the main application page.

    Args:
        request: FastAPI request object

    Returns:
        HTML response with the main page
    """
    config = get_config()
    app_config = config.app
    translation_config = config.translation
    auto_play = app_config.get("auto_play_audio", True)
    ui_locale = preferred_locale(request, config)
    custom_instruction_max_chars = get_custom_instruction_max_chars(
        translation_config.get("custom_instruction_max_chars")
    )
    translation_buffer_seconds = (
        clamp_translation_buffer_seconds(
            app_config.get("translation_flush_seconds", 2.0)
        )
        or TRANSLATION_BUFFER_SECONDS_MIN
    )

    template = templates.get_template("index.html")
    html = template.render(
        auto_play=auto_play,
        custom_instruction_max_chars=custom_instruction_max_chars,
        translation_buffer_seconds=translation_buffer_seconds,
        translation_buffer_seconds_min=TRANSLATION_BUFFER_SECONDS_MIN,
        translation_buffer_seconds_max=TRANSLATION_BUFFER_SECONDS_MAX,
        translation_buffer_seconds_step=TRANSLATION_BUFFER_SECONDS_STEP,
        request=request,
        ui_locale=ui_locale,
        ui_locale_options=ui_locale_options(ui_locale, config),
        i18n_messages=public_catalog(ui_locale, config),
        t=lambda key, **params: t(key, ui_locale, config, **params),
    )

    return HTMLResponse(content=html)
