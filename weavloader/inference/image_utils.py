"""Caption LLM image prep (shared by llm model providers in weavloader)."""

import logging
import os
from io import BytesIO

from PIL import Image

_DEFAULT_LLM_MAX_IMAGE_BYTES = 12 * 1024 * 1024
_DEFAULT_LLM_MAX_IMAGE_SIDE = 6144


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in ("1", "true", "yes")


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        value = int(str(raw).strip())
    except ValueError:
        logging.warning("Invalid %s=%r; using default %d", name, raw, default)
        return default
    if value <= 0:
        logging.warning("Invalid %s=%d (must be > 0); using default %d", name, value, default)
        return default
    return value


def llm_image_byte_limiting_enabled() -> bool:
    return _env_bool("LLM_IMAGE_BYTE_LIMITING", False)


def llm_max_image_bytes() -> int:
    return _positive_int_env("LLM_MAX_IMAGE_BYTES", _DEFAULT_LLM_MAX_IMAGE_BYTES)


def llm_max_image_side() -> int:
    return _positive_int_env("LLM_MAX_IMAGE_SIDE", _DEFAULT_LLM_MAX_IMAGE_SIDE)


def ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    return image.convert("RGB")


def _cap_longest_side(image: Image.Image, max_side: int) -> Image.Image:
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image
    scale = max_side / longest
    return image.resize(
        (max(1, int(width * scale)), max(1, int(height * scale))),
        Image.LANCZOS,
    )


def _downscale_until_rgb_under_bytes(image: Image.Image, max_bytes: int) -> Image.Image:
    width, height = image.size
    while width * height * 3 > max_bytes:
        width = max(1, int(width * 0.85))
        height = max(1, int(height * 0.85))
        image = image.resize((width, height), Image.LANCZOS)
    return image


def prepare_llm_image(image: Image.Image) -> Image.Image:
    rgb = ensure_rgb(image)
    if not llm_image_byte_limiting_enabled():
        return rgb

    rgb = _cap_longest_side(rgb, llm_max_image_side())
    return _downscale_until_rgb_under_bytes(rgb, llm_max_image_bytes())


def prepare_llm_image_bytes(image: Image.Image, jpeg_quality: int = 85) -> tuple[bytes, str]:
    rgb = prepare_llm_image(image)
    max_bytes = llm_max_image_bytes()

    if not llm_image_byte_limiting_enabled():
        buffer = BytesIO()
        rgb.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
        return buffer.getvalue(), "jpeg"

    quality = jpeg_quality
    while quality >= 50:
        buffer = BytesIO()
        rgb.save(buffer, format="JPEG", quality=quality, optimize=True)
        data = buffer.getvalue()
        if len(data) <= max_bytes:
            return data, "jpeg"
        quality -= 10

    buffer = BytesIO()
    rgb.save(buffer, format="JPEG", quality=50, optimize=True)
    return buffer.getvalue(), "jpeg"
