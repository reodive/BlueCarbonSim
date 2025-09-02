import os

from src.simulation import save_pref1_snapshot
from src.models.plant import Plant
from src.models.particle import Particle


def test_pref1_outputs_pngs(tmp_path):
    outdir = tmp_path / "outputs"
    plants = [
        Plant(
            name="Zostera",
            fixation_ratio=0.7,
            release_ratio=0.05,
            structure_density=1.0,
            opt_temp=20.0,
            light_tolerance=0.3,
            light_half_saturation=0.5,
            salinity_range=(20, 35),
            absorption_efficiency=1.0,
            growth_rate=0.01,
            x=30,
            y=28,
            radius=10,
        )
    ]
    sources = [(35, 28)]
    particles = [Particle(x=35.0, y=28.0, origin="test", x0=35.0, y0=28.0)]
    save_pref1_snapshot((100, 100), plants, sources, particles, outdir=str(outdir))
    assert (outdir / "debug_pref1.png").exists()
    assert (outdir / "dy_hist.png").exists()
    assert (outdir / "dist_hist.png").exists()
