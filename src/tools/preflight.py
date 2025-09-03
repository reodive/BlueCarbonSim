import sys
from typing import Optional


def main(argv: Optional[list] = None) -> int:
    """Run a one-step preflight visualization via simulation pref1 mode.

    Produces quick artifacts under `outputs/` such as `debug_pref1.png`.
    """
    try:
        # Local import to ensure package resolution when run as a module
        from ..simulation import run_simulation
    except Exception as e:
        print(f"[error] failed to import simulation: {e}")
        return 2

    try:
        run_simulation(pref1=True)
        print("[ok] preflight snapshot written under `outputs/`.")
        return 0
    except Exception as e:
        print(f"[error] preflight failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

