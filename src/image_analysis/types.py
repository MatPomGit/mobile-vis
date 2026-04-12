"""Type aliases used across :mod:`image_analysis`.

The aliases intentionally focus on image shape/dtype contracts and bounding-box formats
used by detector/classifier/preprocessing modules.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

# Komentarz: Skróty typów obrazów dla najczęściej używanych kontraktów.
ImageU8: TypeAlias = NDArray[np.uint8]
ImageF32: TypeAlias = NDArray[np.float32]

# Komentarz: Jawne aliasy kanałowe - grayscale i BGR.
GrayImageU8: TypeAlias = NDArray[np.uint8]
GrayImageF32: TypeAlias = NDArray[np.float32]
BgrImageU8: TypeAlias = NDArray[np.uint8]
BgrImageF32: TypeAlias = NDArray[np.float32]

# Komentarz: Uniwersalny obraz wejściowy obsługiwany przez walidatory.
Image: TypeAlias = ImageU8 | ImageF32

# Komentarz: Bounding boxes w formatach XYXY oraz XYWH.
BboxXYXY: TypeAlias = tuple[int, int, int, int]
BboxXYWH: TypeAlias = tuple[int, int, int, int]
