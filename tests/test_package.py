"""Tests for the bootstrap package metadata."""

from importlib.metadata import version

import tuesday


def test_package_exposes_expected_version() -> None:
    """The package imports and exposes its initial development version."""
    assert tuesday.__version__ == "0.1.0.dev0"


def test_distribution_version_matches_package_version() -> None:
    """Installed distribution metadata uses the package version source."""
    assert version("tuesday-ai") == tuesday.__version__
