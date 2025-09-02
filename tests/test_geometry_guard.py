import pytest

from src.simulation import validate_geometry
from src.models.plant import Plant


def make_plant(x=10, y=10, r=5, name="Plant"):
    return Plant(
        name=name,
        fixation_ratio=0.7,
        release_ratio=0.05,
        structure_density=1.0,
        opt_temp=20.0,
        light_tolerance=0.3,
        light_half_saturation=0.5,
        salinity_range=(20, 35),
        absorption_efficiency=1.0,
        growth_rate=0.01,
        x=x,
        y=y,
        radius=r,
    )


def test_geometry_guard_raises_when_overlapping():
    plants = [make_plant(x=30, y=28, r=11, name="Zostera marina")]
    sources = [(30, 28)]  # directly on top
    with pytest.raises(ValueError):
        validate_geometry(sources, plants, width=100, height=100, margin_px=3, auto_shift=False)


def test_geometry_guard_autoshift_success():
    plants = [make_plant(x=30, y=28, r=11, name="Zostera marina")]
    sources = [(30, 28)]
    adjusted = validate_geometry(sources, plants, width=100, height=100, margin_px=3, auto_shift=True)
    # y should have shifted off the exact overlap
    assert adjusted[0][1] != 28

