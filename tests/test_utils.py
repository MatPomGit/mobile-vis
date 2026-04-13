"""Tests for image_analysis.utils module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from image_analysis.utils import (
    get_project_root,
    list_images,
    safe_makedirs,
    validate_bbox_xywh,
    validate_bbox_xyxy,
    validate_bgr_image,
    validate_gray_image,
    validate_image,
)

# ---------------------------------------------------------------------------
# validate_image
# ---------------------------------------------------------------------------


class TestValidateImage:
    @pytest.mark.parametrize(
        "image",
        [
            np.zeros((100, 100), dtype=np.uint8),  # grayscale
            np.zeros((100, 100, 1), dtype=np.uint8),  # single-channel
            np.zeros((100, 100, 3), dtype=np.uint8),  # BGR
            np.zeros((100, 100, 4), dtype=np.uint8),  # BGRA
            np.zeros((100, 100, 3), dtype=np.float32),  # float BGR
        ],
    )
    def test_valid_images_do_not_raise(self, image: np.ndarray) -> None:
        validate_image(image)  # should not raise

    def test_raises_for_non_ndarray(self) -> None:
        with pytest.raises(TypeError):
            validate_image([[1, 2, 3]])  # type: ignore[arg-type]

    def test_raises_for_invalid_channels(self) -> None:
        image = np.zeros((100, 100, 2), dtype=np.uint8)
        with pytest.raises(ValueError):
            validate_image(image)

    def test_raises_for_1d_array(self) -> None:
        image = np.zeros((100,), dtype=np.uint8)
        with pytest.raises(ValueError):
            validate_image(image)

    def test_raises_for_invalid_dtype(self) -> None:
        image = np.zeros((100, 100, 3), dtype=np.int32)
        with pytest.raises(ValueError):
            validate_image(image)


class TestValidateBgrImage:
    def test_accepts_1x1_uint8_bgr(self) -> None:
        image = np.zeros((1, 1, 3), dtype=np.uint8)
        validate_bgr_image(image, allowed_dtypes=(np.uint8,))

    def test_rejects_grayscale_shape(self) -> None:
        image = np.zeros((8, 8), dtype=np.uint8)
        with pytest.raises(ValueError):
            validate_bgr_image(image)

    def test_rejects_invalid_dtype(self) -> None:
        image = np.zeros((8, 8, 3), dtype=np.int16)
        with pytest.raises(ValueError):
            validate_bgr_image(image)

    def test_rejects_float32_out_of_range(self) -> None:
        image = np.full((4, 4, 3), 1.5, dtype=np.float32)
        with pytest.raises(ValueError):
            validate_bgr_image(image)


class TestValidateGrayImage:
    @pytest.mark.parametrize("shape", [(1, 1), (8, 8, 1)])
    def test_accepts_gray_shapes(self, shape: tuple[int, ...]) -> None:
        image = np.zeros(shape, dtype=np.uint8)
        validate_gray_image(image, allowed_dtypes=(np.uint8,))

    def test_rejects_bgr_image(self) -> None:
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            validate_gray_image(image)

    def test_rejects_float32_out_of_range(self) -> None:
        image = np.full((8, 8), -0.2, dtype=np.float32)
        with pytest.raises(ValueError):
            validate_gray_image(image)


class TestValidateBboxXywh:
    def test_accepts_valid_bbox(self) -> None:
        assert validate_bbox_xywh((1, 2, 3, 4)) == (1, 2, 3, 4)

    @pytest.mark.parametrize("bbox", [(0, 1, 0, 5), (2, 3, 1, -1), (1, 2, 3)])
    def test_rejects_invalid_bbox_geometry_or_length(self, bbox: object) -> None:
        with pytest.raises((ValueError, TypeError)):
            validate_bbox_xywh(bbox)  # type: ignore[arg-type]


class TestValidateBboxXyxy:
    def test_accepts_valid_bbox(self) -> None:
        assert validate_bbox_xyxy((0, 1, 4, 5)) == (0, 1, 4, 5)

    @pytest.mark.parametrize("bbox", [(0, 1, 0, 5), (2, 3, 1, 9), (1, 2, 3)])
    def test_rejects_invalid_bbox_geometry_or_length(self, bbox: object) -> None:
        with pytest.raises((ValueError, TypeError)):
            validate_bbox_xyxy(bbox)  # type: ignore[arg-type]

    def test_rejects_non_finite_bbox(self) -> None:
        with pytest.raises(ValueError):
            validate_bbox_xyxy((0, 0, np.inf, 10))


# ---------------------------------------------------------------------------
# get_project_root
# ---------------------------------------------------------------------------


class TestGetProjectRoot:
    def test_returns_path_instance(self) -> None:
        root = get_project_root()
        assert isinstance(root, Path)

    def test_root_contains_pyproject_toml(self) -> None:
        root = get_project_root()
        assert (root / "pyproject.toml").exists()


# ---------------------------------------------------------------------------
# safe_makedirs
# ---------------------------------------------------------------------------


class TestSafeMakedirs:
    def test_creates_directory(self, tmp_path: Path) -> None:
        new_dir = tmp_path / "a" / "b" / "c"
        result = safe_makedirs(new_dir)
        assert result.is_dir()

    def test_does_not_raise_if_already_exists(self, tmp_path: Path) -> None:
        safe_makedirs(tmp_path)  # tmp_path already exists
        safe_makedirs(tmp_path)  # calling again should not raise

    def test_raises_when_path_is_file(self, tmp_path: Path) -> None:
        file_path = tmp_path / "artifact.txt"
        file_path.write_text("content", encoding="utf-8")
        with pytest.raises(NotADirectoryError):
            safe_makedirs(file_path)


# ---------------------------------------------------------------------------
# list_images
# ---------------------------------------------------------------------------


class TestListImages:
    def test_returns_only_image_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.jpg").touch()
        (tmp_path / "b.PNG").touch()
        (tmp_path / "c.txt").touch()
        (tmp_path / "d.jpeg").touch()

        result = list_images(tmp_path)
        names = {p.name for p in result}
        assert "a.jpg" in names
        assert "d.jpeg" in names
        assert "c.txt" not in names

    def test_returns_sorted_list(self, tmp_path: Path) -> None:
        for name in ["c.jpg", "a.jpg", "b.jpg"]:
            (tmp_path / name).touch()
        result = list_images(tmp_path)
        assert [p.name for p in result] == ["a.jpg", "b.jpg", "c.jpg"]

    def test_raises_for_non_directory(self, tmp_path: Path) -> None:
        with pytest.raises(NotADirectoryError):
            list_images(tmp_path / "nonexistent_dir")
