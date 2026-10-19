"""Decoded-size caps for request media.

A compressed image or video can be tiny on the wire and enormous once decoded
(a flat PNG compresses about 1000:1, a still video far more), so the loaders
check what a file will decode to, not how big it is. A refusal is a
``ValueError``, which the data worker answers with a 400.

The pixel caps hold for a whole request: while the data worker loads a
request's files inside :class:`RequestMediaBudget`, every check also counts
against what the request's earlier files used.
"""
import os
from contextlib import contextmanager
from contextvars import ContextVar

import torch

# PIL's DecompressionBombWarning threshold (about 9459 x 9459 RGB)
DEFAULT_MAX_IMAGE_PIXELS = 89_478_485
# about 39 s of 720p30, or 2.6 min of 640x360 at 30 fps, decoded at once: a
# full decode of every short (up to 2 min) Video-MME clip fits
DEFAULT_MAX_VIDEO_PIXELS = 1 << 30
# the least a decoded frame counts against the video cap
MIN_FRAME_PIXELS = 64 * 64
DEFAULT_MAX_MEDIA_FILES = 16
# torchvision decodes these (GIF is read with Pillow); anything else is a 400
IMAGE_FORMATS = ("PNG", "JPEG", "WEBP", "GIF")
# a loader that decodes with Pillow also reads the other image types the API accepts
PILLOW_IMAGE_FORMATS = IMAGE_FORMATS + ("BMP", "TIFF")


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {raw!r}")
    return value


def max_image_pixels() -> int:
    """``MSTAR_MAX_IMAGE_PIXELS``: pixels a request's images may decode to,
    summed over them."""
    return _positive_int_env("MSTAR_MAX_IMAGE_PIXELS", DEFAULT_MAX_IMAGE_PIXELS)


def max_video_pixels() -> int:
    """``MSTAR_MAX_VIDEO_PIXELS``: pixels a request's videos may decode to,
    summed over the frames the loader decodes from each."""
    return _positive_int_env("MSTAR_MAX_VIDEO_PIXELS", DEFAULT_MAX_VIDEO_PIXELS)


def max_media_files() -> int:
    """``MSTAR_MAX_MEDIA_FILES``: images, audio clips and videos one request
    may attach, together."""
    return _positive_int_env("MSTAR_MAX_MEDIA_FILES", DEFAULT_MAX_MEDIA_FILES)


def check_media_limits() -> None:
    """Read the limits once at startup: a bad value then stops the server,
    instead of turning every media request into a 400 that blames the client.

    Also lifts Pillow's own bomb check to the image limit, which it would
    otherwise apply first (at about 179M pixels) whatever the limit is.
    """
    from PIL import Image

    limit = max_image_pixels()
    max_video_pixels()
    max_media_files()
    if Image.MAX_IMAGE_PIXELS is not None and limit > Image.MAX_IMAGE_PIXELS:
        Image.MAX_IMAGE_PIXELS = limit


_request_budget: ContextVar["RequestMediaBudget | None"] = ContextVar("mstar_request_media_budget", default=None)


class RequestMediaBudget:
    """Holds one request's media to the caps together, not file by file, so
    many files in one request cannot multiply them; refuses more than
    ``max_media_files()`` files before any is decoded.

    The data worker loads each file inside :meth:`file`; the image and video
    checks charge the request through it."""

    _ENV = {"image": "MSTAR_MAX_IMAGE_PIXELS", "video": "MSTAR_MAX_VIDEO_PIXELS"}

    def __init__(self, num_files: int) -> None:
        limit = max_media_files()
        if num_files > limit:
            raise ValueError(
                f"the request attaches {num_files} media files; the limit is {limit} (MSTAR_MAX_MEDIA_FILES)"
            )
        self.limits = {"image": max_image_pixels(), "video": max_video_pixels()}
        self.used = dict.fromkeys(self.limits, 0)
        self._file = dict.fromkeys(self.limits, 0)

    @contextmanager
    def file(self):
        """Load one file. It is charged the most any single check counted for
        it, so a header estimate and the decode after it are not added up."""
        token = _request_budget.set(self)
        try:
            yield
        finally:
            _request_budget.reset(token)
            for kind, pixels in self._file.items():
                self.used[kind] += pixels
                self._file[kind] = 0

    def charge(self, kind: str, pixels: int) -> None:
        self._file[kind] = max(self._file[kind], pixels)
        if self.used[kind] + self._file[kind] > self.limits[kind]:
            raise ValueError(
                f"the request's {kind}s decode to more than {self.limits[kind]} pixels together; "
                f"send fewer or smaller files (the limit is {self._ENV[kind]})"
            )


def _charge_request(kind: str, pixels: int) -> None:
    budget = _request_budget.get()
    if budget is not None:
        budget.charge(kind, pixels)


def check_image_size(filepath: str, formats: tuple[str, ...] = IMAGE_FORMATS) -> tuple[str, bool]:
    """Refuse an image whose header promises more than ``max_image_pixels()``
    or that is not in ``formats`` (the ones the caller's decoder reads);
    returns Pillow's name for the format and whether the file holds more than
    one frame.

    Reads the header only. The decoders allocate what the header declares, so
    it cannot understate the decode.
    """
    import warnings

    from PIL import Image, UnidentifiedImageError

    limit = max_image_pixels()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(filepath, formats=formats) as image:
                width, height = image.size
                fmt = image.format or ""
                animated = bool(getattr(image, "is_animated", False))
    except Image.DecompressionBombError:
        raise ValueError(f"image is too large to decode; the limit is {limit} pixels") from None
    except (UnidentifiedImageError, OSError, SyntaxError):
        # the decoder's message can name the server path; the client needs only this
        raise ValueError("the image is not in a format this server can read") from None
    if width * height > limit:
        raise ValueError(f"image is {width}x{height} ({width * height} pixels); the limit is {limit} pixels")
    _charge_request("image", width * height)
    return fmt, animated


def read_image(filepath: str, *, rgb: bool = False) -> torch.Tensor:
    """A request image as uint8 ``[C, H, W]``, refused first if it is too large.

    ``rgb`` forces three channels; otherwise the file's own channels are kept,
    as ``torchvision.io.decode_image`` does. A GIF or an animated WebP is read
    with Pillow, as its first frame in RGB: torchvision's GIF decoder is not
    safe on untrusted files before torchvision 0.28, and its WebP decoder
    refuses animation.
    """
    import torchvision

    fmt, animated = check_image_size(filepath)
    if fmt == "GIF" or (fmt == "WEBP" and animated):
        import numpy as np
        from PIL import Image

        with Image.open(filepath) as image:
            array = np.array(image.convert("RGB"))
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()
    mode = torchvision.io.ImageReadMode.RGB if rgb else torchvision.io.ImageReadMode.UNCHANGED
    image = torchvision.io.decode_image(filepath, mode=mode)
    if image.dtype == torch.uint16:
        # a 16-bit PNG: its high byte, as Pillow reads a 16-bit RGB PNG
        image = (image.to(torch.int32) >> 8).to(torch.uint8)
    return image


def probe_video(filepath: str) -> tuple[int, float, int, int] | None:
    """``(num_frames, fps, height, width)`` from the container header, or None
    when neither torchcodec nor PyAV can read it."""
    try:
        from torchcodec.decoders import VideoDecoder

        # "exact" would scan every packet to count frames; the header is enough here
        meta = VideoDecoder(filepath, seek_mode="approximate").metadata
        if meta.num_frames and meta.average_fps and meta.height and meta.width:
            return int(meta.num_frames), float(meta.average_fps), int(meta.height), int(meta.width)
    except Exception:  # noqa: BLE001, S110 -- missing or unable to read it: try PyAV
        pass
    try:
        import av

        with av.open(filepath) as container:
            stream = container.streams.video[0]
            rate = stream.average_rate or stream.guessed_rate or stream.base_rate
            fps = float(rate) if rate else 0.0
            frames = stream.frames or (
                int(float(stream.duration * stream.time_base) * fps) if stream.duration and fps else 0
            )
            ctx = stream.codec_context
            if frames and fps and ctx.height and ctx.width:
                return int(frames), fps, int(ctx.height), int(ctx.width)
    except Exception:  # noqa: BLE001, S110 -- unreadable: the loader's own decoder reports it
        pass
    return None


class VideoBudget:
    """Counts decoded frames against ``max_video_pixels()``, so a loader stops
    before it holds more than the cap whatever the container claims.

    A frame counts as at least ``MIN_FRAME_PIXELS``: each decoded frame costs
    about 2 KB of tensor and frame objects whatever its size, so a clip of
    millions of 2x2 frames would otherwise pass the cap at hundreds of GB."""

    def __init__(self) -> None:
        self.limit = max_video_pixels()
        self.pixels = 0

    def add(self, height: int, width: int, frames: int = 1) -> None:
        self.pixels += max(int(height) * int(width), MIN_FRAME_PIXELS) * int(frames)
        if self.pixels > self.limit:
            raise ValueError(
                f"video decodes to more than {self.limit} pixels; send a shorter "
                "or smaller clip (the limit is MSTAR_MAX_VIDEO_PIXELS)"
            )
        _charge_request("video", self.pixels)
