# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""LiteLLM cache configuration utilities."""

import os
from typing import Any, Literal


def configure_cache(
    backend: Literal["disk", "r2", "redis", "s3"] = "r2",
    **kwargs: Any,
) -> None:
    """Configure LiteLLM cache.

    Args:
        backend: Cache backend type
            - "r2": Cloudflare R2 (default, reads from R2_* env vars)
            - "disk": Local disk cache
            - "redis": Redis cache
            - "s3": AWS S3 or compatible
        **kwargs: Additional arguments passed to litellm.Cache()

    Examples:
        # Cloudflare R2 (default, set R2_BUCKET_NAME, R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY)
        configure_cache("r2")

        # Disk cache
        configure_cache("disk")

        # Custom S3
        configure_cache("s3", s3_bucket_name="my-bucket", s3_region_name="us-west-2")
    """
    import litellm
    from litellm import Cache

    if backend == "disk":
        cache_dir = kwargs.pop("disk_cache_dir", os.path.expanduser("~/.cache/litellm"))
        litellm.cache = Cache(type="disk", disk_cache_dir=cache_dir, **kwargs)

    elif backend == "r2":
        litellm.cache = Cache(
            type="s3",
            s3_bucket_name=kwargs.pop("s3_bucket_name", os.environ["R2_BUCKET_NAME"]),
            s3_endpoint_url=kwargs.pop("s3_endpoint_url", os.environ["R2_ENDPOINT_URL"]),
            s3_aws_access_key_id=kwargs.pop("s3_aws_access_key_id", os.environ["R2_ACCESS_KEY_ID"]),
            s3_aws_secret_access_key=kwargs.pop("s3_aws_secret_access_key", os.environ["R2_SECRET_ACCESS_KEY"]),
            s3_region_name=kwargs.pop("s3_region_name", "auto"),
            **kwargs,
        )

    elif backend == "redis":
        litellm.cache = Cache(type="redis", **kwargs)

    elif backend == "s3":
        litellm.cache = Cache(type="s3", **kwargs)

    else:
        raise ValueError(f"Unknown cache backend: {backend}")


def disable_cache() -> None:
    """Disable LiteLLM caching."""
    import litellm

    litellm.cache = None
