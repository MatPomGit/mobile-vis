"""Tests for image_analysis.utils module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from image_analysis.utils import (
    get_project_root,
    list_images,
    safe_makedirs,
    validate_image,
)

# ---------------------------------------------------------------------------
# validate_image
# ---------------------------------------------------------------------------


class TestValidateImage:
    @pytest.mark.parametrize(
        "image",
        [
            np.zeros((100, 100), dtype=np.uint8),           # grayscale
            np.zeros((100, 100, 1), dtype=np.uint8),         # single-channel
            np.zeros((100, 100, 3), dtype=np.uint8),         # BGR
            np.zeros((100, 100, 4), dtype=np.uint8),         # BGRA
            np.zeros((100, 100, 3), dtype=np.float32),       # float BGR
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
