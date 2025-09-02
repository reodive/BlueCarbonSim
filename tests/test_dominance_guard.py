import pytest

from src.simulation import guard_dominance


def test_dominance_guard_triggers_after_streak():
    shares = {"Zostera": 0.7, "Other": 0.3}
    # Two times should not raise yet
    guard_dominance(shares, max_share=0.65, streak=3)
    guard_dominance(shares, max_share=0.65, streak=3)
    # Third time should raise
    with pytest.raises(AssertionError):
        guard_dominance(shares, max_share=0.65, streak=3)

