from __future__ import annotations

import logging
from functools import lru_cache

from app.core.config import get_settings
from app.services.providers.base import RealtimeContextProvider, SportsDataProvider
from app.services.providers.cache_store import JSONFileCacheStore
from app.services.providers.cricapi_realtime_provider import CricAPIRealtimeProvider
from app.services.providers.local_demo_provider import LocalDemoProvider
from app.services.providers.mock_realtime_provider import MockRealtimeProvider

logger = logging.getLogger(__name__)


def _build_cricapi_provider() -> CricAPIRealtimeProvider:
    settings = get_settings()
    cache_store = JSONFileCacheStore(path=settings.live_cache_path)
    return CricAPIRealtimeProvider(
        api_key=settings.live_cricket_api_key,
        base_url=settings.live_cricket_api_base_url,
        timeout_seconds=settings.live_provider_timeout_seconds,
        max_retries=settings.live_provider_max_retries,
        backoff_seconds=settings.live_provider_backoff_seconds,
        cache_ttl_seconds=settings.live_cache_ttl_seconds,
        stale_cache_ttl_seconds=settings.live_cache_stale_ttl_seconds,
        cache_store=cache_store,
    )


@lru_cache(maxsize=1)
def get_catalog_provider() -> SportsDataProvider:
    settings = get_settings()
    provider_key = settings.data_provider.strip().lower()

    if provider_key == "local-demo":
        logger.info("Catalog provider: local-demo")
        return LocalDemoProvider(processed_dir=settings.data_processed_dir)

    logger.warning("Unknown data provider '%s'. Falling back to local-demo.", provider_key)
    return LocalDemoProvider(processed_dir=settings.data_processed_dir)


@lru_cache(maxsize=1)
def get_live_provider() -> RealtimeContextProvider:
    settings = get_settings()
    provider_key = settings.realtime_provider.strip().lower()

    if provider_key in {"mock-realtime", "mock", "fallback"}:
        logger.info("Realtime provider: mock-realtime")
        return MockRealtimeProvider()

    if provider_key in {"cricapi", "cricketdata", "live-cricket"}:
        logger.info("Realtime provider: cricapi")
        return _build_cricapi_provider()

    logger.warning(
        "Unknown realtime provider '%s'. Falling back to cricapi provider without synthetic mock feed.",
        provider_key,
    )
    return _build_cricapi_provider()


@lru_cache(maxsize=1)
def get_realtime_provider() -> RealtimeContextProvider:
    return get_live_provider()
