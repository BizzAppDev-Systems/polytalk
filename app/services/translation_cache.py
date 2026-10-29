# SPDX-FileCopyrightText: 2026 BizzAppDev Systems Pvt. Ltd.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Redis-backed translation cache for PolyTalk.

Caches full-sentence translations keyed by (source_language, target_language,
normalized_transcript) so that repeated sentences skip the LLM call and
return the cached result instantly.

Gracefully degrades: if Redis is unavailable the cache is silently bypassed
and all translation requests flow through to the LLM as normal.
"""

import json
import logging
import re
from contextlib import suppress
from functools import lru_cache
from hashlib import sha256
from typing import Any

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

CACHE_PREFIX = "pt:cache"
DEFAULT_TTL_SECONDS = 86400  # 24 hours
HASH_LENGTH = 32
EMPTY_HASH = sha256(b"").hexdigest()[:HASH_LENGTH]


def normalize_text(text: str) -> str:
    """Normalize text for consistent cache key generation.

    Lowercases, strips whitespace, collapses internal whitespace, and removes
    non-alphanumeric characters (except spaces and common diacritics).
    """
    text = text.lower().strip()
    text = re.sub(r"[^\w\s'-]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


@lru_cache(maxsize=1024)
def _hash_instruction(instruction: str | None) -> str:
    """Hash a custom instruction string for cache key inclusion.

    Uses lru_cache with maxsize=1024 to prevent unbounded growth.
    """
    if instruction is None:
        return "none"
    stripped = instruction.strip().lower()
    if not stripped:
        return EMPTY_HASH
    return sha256(stripped.encode("utf-8")).hexdigest()[:HASH_LENGTH]


@lru_cache(maxsize=512)
def _hash_glossary_tuple(sorted_tuple: tuple[str, ...]) -> str:
    """Hash a sorted tuple of glossary terms for cache key inclusion.

    Uses lru_cache with maxsize=512 to prevent unbounded growth.
    """
    if not sorted_tuple:
        return "none"
    joined = "\x1f".join(sorted_tuple)
    return sha256(joined.encode("utf-8")).hexdigest()[:HASH_LENGTH]


def _hash_glossary_terms(glossary_terms: list[str] | None) -> str:
    """Hash a sorted list of glossary terms for cache key inclusion.

    Converts the list to a tuple so it can be used with lru_cache.
    """
    if glossary_terms is None or len(glossary_terms) == 0:
        return "none"
    sorted_tuple = tuple(sorted(glossary_terms))
    return _hash_glossary_tuple(sorted_tuple)


def make_cache_key(
    source_language: str,
    target_language: str,
    text: str,
    custom_instruction: str | None = None,
    glossary_terms: list[str] | None = None,
) -> str:
    """Build a deterministic Redis key for a translation pair.

    Format: pt:cache:{source}:{target}:{text_hash}:{instruction_hash}:{glossary_hash}
    """
    normalized = normalize_text(text)
    if not normalized:
        return ""
    text_hash = sha256(normalized.encode("utf-8")).hexdigest()[:HASH_LENGTH]
    instruction_hash = _hash_instruction(custom_instruction)
    glossary_hash = _hash_glossary_terms(glossary_terms)
    return f"{CACHE_PREFIX}:{source_language}:{target_language}:{text_hash}:{instruction_hash}:{glossary_hash}"


class TranslationCache:
    """Async Redis-backed cache for translation results.

    All public methods are safe to call even when Redis is not configured or
    not reachable -- they simply return None / do nothing.
    """

    def __init__(
        self,
        redis_url: str,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        max_entries: int = 100_000,
    ) -> None:
        self._redis_url = redis_url
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries
        self._client: aioredis.Redis | None = None
        self._connected = False
        self._stats = {"hits": 0, "misses": 0, "sets": 0, "errors": 0, "bypasses": 0}

    async def connect(self) -> None:
        """Connect to Redis. Sets self._connected on success."""
        try:
            self._client = aioredis.from_url(
                self._redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
            )
            await self._client.ping()
            self._connected = True
            logger.info("Translation cache connected to Redis at %s", self._redis_url)
        except Exception:
            self._connected = False
            self._client = None
            logger.warning(
                "Translation cache failed to connect to Redis; cache will be bypassed",
                exc_info=True,
            )

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._client is not None:
            with suppress(Exception):
                await self._client.close()
            self._client = None
            self._connected = False
            logger.info("Translation cache disconnected from Redis")

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    async def get(
        self,
        source_language: str,
        target_language: str,
        text: str,
        custom_instruction: str | None = None,
        glossary_terms: list[str] | None = None,
    ) -> dict[str, str] | None:
        """Look up a cached translation.

        Returns a dict with keys ``translated_text``, ``source_language``,
        and ``target_language`` on hit, or ``None`` on miss/error.
        """
        if not self._connected:
            self._stats["bypasses"] += 1
            return None

        cache_key = make_cache_key(
            source_language, target_language, text, custom_instruction, glossary_terms
        )
        if not cache_key:
            self._stats["misses"] += 1
            return None

        try:
            raw = await self._client.get(cache_key)  # type: ignore[union-attr]
            if raw is None:
                self._stats["misses"] += 1
                return None
            self._stats["hits"] += 1
            data = json.loads(raw)
            if isinstance(data, dict) and data.get("translated_text"):
                return data
            self._stats["misses"] += 1
            return None
        except Exception:
            self._stats["errors"] += 1
            logger.warning(
                "Translation cache get error",
                exc_info=True,
            )
            return None

    async def set(
        self,
        source_language: str,
        target_language: str,
        text: str,
        translated_text: str,
        custom_instruction: str | None = None,
        glossary_terms: list[str] | None = None,
        translation_quality: float | None = None,
    ) -> None:
        """Store a translation result in the cache."""
        if not self._connected:
            return

        cache_key = make_cache_key(
            source_language, target_language, text, custom_instruction, glossary_terms
        )
        if not cache_key:
            return

        value: dict[str, Any] = {
            "translated_text": translated_text,
            "source_language": source_language,
            "target_language": target_language,
        }
        if translation_quality is not None:
            value["translation_quality"] = translation_quality

        try:
            await self._client.set(  # type: ignore[union-attr]
                cache_key,
                json.dumps(value),
                ex=self._ttl_seconds,
            )
            self._stats["sets"] += 1
        except Exception:
            self._stats["errors"] += 1
            logger.warning(
                "Translation cache set error",
                exc_info=True,
            )

    async def get_stats(self) -> dict[str, Any]:
        """Return cache statistics and Redis connection info."""
        info: dict[str, Any] = {"stats": dict(self._stats)}
        if self._connected and self._client is not None:
            try:
                db_info = await self._client.info("memory")  # type: ignore[union-attr]
                info["memory"] = {
                    "used_memory_human": db_info.get("used_memory_human"),
                    "maxmemory_human": (db_info.get("maxmemory_human") or "unlimited"),
                }
            except Exception:
                info["memory"] = {"error": "could not fetch memory info"}
        else:
            info["memory"] = {"status": "not connected"}
        return info
