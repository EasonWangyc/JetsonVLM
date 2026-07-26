"""Read-only environment snapshot for study provenance."""

from __future__ import annotations

import platform
import socket
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


def capture_environment() -> dict[str, Any]:
    """Capture portable host facts and Jetson L4T metadata when available."""
    snapshot: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": sys.version.split()[0],
        "python_packages": _package_versions(
            ("parksight-vlm", "torch", "transformers", "Pillow")
        ),
    }
    l4t_release_path = Path("/etc/nv_tegra_release")
    if l4t_release_path.is_file():
        snapshot["nv_tegra_release"] = l4t_release_path.read_text(
            encoding="utf-8"
        ).strip()
    return snapshot


def _package_versions(distributions: tuple[str, ...]) -> dict[str, str]:
    installed: dict[str, str] = {}
    for distribution in distributions:
        try:
            installed[distribution] = version(distribution)
        except PackageNotFoundError:
            continue
    return installed
