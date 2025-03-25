import base64
from io import BytesIO
from PIL import Image

from transformers.utils.import_utils import _is_package_available


_fastapi_available = _is_package_available("fastapi")
_pydantic_available = _is_package_available("pydantic")
_uvicorn_available = _is_package_available("uvicorn")
_vllm_available = _is_package_available("vllm")
_requests_available = _is_package_available("requests")

def is_fastapi_available() -> bool:
    return _fastapi_available


def is_pydantic_available() -> bool:
    return _pydantic_available

def is_uvicorn_available() -> bool:
    return _uvicorn_available


def is_vllm_available() -> bool:
    return _vllm_available

def is_requests_available() -> bool:
    return _requests_available


def is_pil_image(image) -> bool:
    return isinstance(image, Image.Image)


def encode_image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """
    Encode a PIL Image to a base64 string.

    Args:
        image (PIL.Image): The image to encode.
        format (str): Image format to use (e.g., "PNG", "JPEG"). Default is "PNG".

    Returns:
        str: Base64-encoded string of the image.
    """
    buffer = BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    encoded_string = base64.b64encode(buffer.read()).decode("utf-8")
    return encoded_string

def decode_base64_to_image(base64_str: str) -> Image.Image:
    """
    Decode a base64 string back to a PIL Image.

    Args:
        base64_str (str): Base64-encoded string of the image.

    Returns:
        PIL.Image: Decoded image.
    """
    image_data = base64.b64decode(base64_str)
    buffer = BytesIO(image_data)
    image = Image.open(buffer)
    return image
