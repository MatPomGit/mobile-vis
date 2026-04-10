"""Command-line interface for project utilities."""

from __future__ import annotations

import argparse
from pathlib import Path

from image_analysis.versioning import get_android_version, get_python_package_version


def build_parser() -> argparse.ArgumentParser:
    """Create CLI parser for image_analysis utilities."""
    parser = argparse.ArgumentParser(
        prog="mobilecv-version",
        description="Show current MobileCV application version.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="version",
        choices=("version",),
        help="CLI command to execute.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Optional repository root used to resolve Android version.",
    )
    return parser


def main() -> int:
    """Run CLI entrypoint and print version details."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "version":
        version_name, version_code = get_android_version(args.project_root)
        package_version = get_python_package_version()
        print(f"MobileCV Android version: {version_name} (code {version_code})")
        print(f"image-analysis package version: {package_version}")
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
