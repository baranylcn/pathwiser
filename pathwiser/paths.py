"""Locations of the repository directories."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
CONFIG_DIR = REPO_ROOT / "configs"
REPORTS_DIR = REPO_ROOT / "reports"
