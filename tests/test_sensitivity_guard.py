import pytest

from src.simulation import guard_sensitivity


def test_sensitivity_guard_trips_on_identical_totals():
    base = {"Zostera": 10.0, "Halophila": 5.0}
    alt = {"Zostera": 10.0, "Halophila": 5.0}
    with pytest.raises(AssertionError):
        guard_sensitivity(base, alt)


def test_sensitivity_guard_passes_on_change():
    base = {"Zostera": 10.0, "Halophila": 5.0}
    alt = {"Zostera": 10.1, "Halophila": 5.0}
    guard_sensitivity(base, alt)

