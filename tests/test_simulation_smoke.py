import os
import json

from src.simulation import run_simulation


def test_simulation_smoke(tmp_path):
    # Run a shortened simulation to keep CI fast
    os.environ["MPLCONFIGDIR"] = str(tmp_path / ".mpl")
    run_simulation(total_steps=40)

    # Check required artifacts
    assert os.path.exists("results/summary_totals.csv"), "summary_totals.csv missing"
    assert os.path.exists("outputs/metrics.json"), "metrics.json missing"
    with open("outputs/metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)
    assert "species_share" in metrics and metrics["species_share"], "species_share missing"

