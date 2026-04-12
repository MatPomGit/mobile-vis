"""Shared helpers for detector modules.

This module centralises common validation and retry logic reused by
multiple detection backends.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import ParamSpec, TypeVar

import numpy as np

from .utils import validate_image

P = ParamSpec("P")
T = TypeVar("T")


def validate_bgr_uint8_image(image: object) -> None:
    """Validate that *image* is a 3-channel BGR ``uint8`` NumPy array.

    Args:
        image: Value to validate.

    Raises:
        TypeError: If *image* is not a ``np.ndarray``.
        ValueError: If the array is not shaped like ``(H, W, 3)`` or is not ``uint8``.
    """
    validate_image(image)
    if (
        not isinstance(image, np.ndarray)
        or image.ndim != 3
        or image.shape[2] != 3
        or image.dtype != np.uint8
    ):
        raise ValueError(
            "Expected uint8 3-channel BGR image with shape (H, W, 3), "
            f"got shape {getattr(image, 'shape', 'N/A')} and "
            f"dtype {getattr(image, 'dtype', 'N/A')}"
        )


def load_with_retry(
    factory: Callable[P, T],
    *factory_args: P.args,
    max_retries: int,
    retry_delay: float,
    logger: logging.Logger,
    retry_message: str,
    failure_message: str,
    **factory_kwargs: P.kwargs,
) -> T:
    """Call *factory* with retry and exponential backoff.

    Message templates support ``str.format`` placeholders:
    ``target``, ``attempt``, ``max_retries``, ``error``, ``delay``.

    Args:
        factory: Callable creating the required object.
        *factory_args: Positional arguments forwarded to *factory*.
        max_retries: Total number of attempts, including the first one.
        retry_delay: Initial delay before retry; doubles after each failure.
        logger: Logger used for retry/final failure messages.
        retry_message: Warning message template for intermediate failures.
        failure_message: Error message template for terminal failure.
        **factory_kwargs: Keyword arguments forwarded to *factory*.

    Returns:
        Object returned by *factory*.

    Raises:
        RuntimeError: If all attempts fail.
    """
    if max_retries < 1:
        raise ValueError(f"max_retries must be >= 1, got {max_retries}")
    if retry_delay < 0.0:
        raise ValueError(f"retry_delay must be >= 0.0, got {retry_delay}")

    last_error: BaseException | None = None
    current_delay = retry_delay
    target = str(factory_args[0]) if factory_args else factory.__name__

    for attempt in range(1, max_retries + 1):
        try:
            return factory(*factory_args, **factory_kwargs)
        except Exception as exc:
            last_error = exc
            if attempt == max_retries:
                break

            logger.warning(
                retry_message.format(
                    target=target,
                    attempt=attempt,
                    max_retries=max_retries,
                    error=exc,
                    delay=current_delay,
                )
            )
            time.sleep(current_delay)
            current_delay *= 2

    final_message = failure_message.format(target=target, max_retries=max_retries)
    logger.error(final_message)
    raise RuntimeError(final_message) from last_error


# Rejestr publicznych symboli modułu używany przez image_analysis.__init__.
PUBLIC_EXPORTS: dict[str, str] = {
    "load_with_retry": "load_with_retry",
    "validate_bgr_uint8_image": "validate_bgr_uint8_image",
}
