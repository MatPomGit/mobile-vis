"""Type aliases used across :mod:`image_analysis`.

The aliases intentionally focus on image shape/dtype contracts and bounding-box formats
used by detector/classifier/preprocessing modules.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

# Komentarz: Bazowe aliasy tablic NumPy dla danych obrazowych o wspieranych dtype.
ImageU8: TypeAlias = NDArray[np.uint8]
ImageF32: TypeAlias = NDArray[np.float32]
Image: TypeAlias = ImageU8 | ImageF32

# Komentarz: Aliasy semantyczne - ten sam typ ndarray, ale inny kontrakt kształtu.
GrayImageU8: TypeAlias = NDArray[np.uint8]
GrayImageF32: TypeAlias = NDArray[np.float32]
GrayImage: TypeAlias = GrayImageU8 | GrayImageF32

BgrImageU8: TypeAlias = NDArray[np.uint8]
BgrImageF32: TypeAlias = NDArray[np.float32]
BgrImage: TypeAlias = BgrImageU8 | BgrImageF32

# Komentarz: Ujednolicone aliasy bboxów dla dwóch najczęstszych konwencji.
BboxXYXY: TypeAlias = tuple[int, int, int, int]
BboxXYWH: TypeAlias = tuple[int, int, int, int]
