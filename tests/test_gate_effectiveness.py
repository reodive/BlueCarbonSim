import numpy as np
import pytest

from src.simulation import assert_gate_effective


def test_gate_effectiveness_trips_on_small_median():
    # median(|dy|) ~ 0.2 px -> should raise
    samples = [0.1, 0.2, 0.3, 0.0, 0.2]
    with pytest.raises(AssertionError):
        assert_gate_effective(samples, min_median=0.8)


def test_gate_effectiveness_passes_on_wide_median():
    samples = [1.0, 0.9, 0.8, 1.2, 0.7, 1.5]
    # Should not raise
    assert_gate_effective(samples, min_median=0.8)

