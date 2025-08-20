import numpy as np


class Particle:
    """Simple representation of a CO2 particle in the water column."""

    def __init__(self, x: float, y: float, **kwargs):
        self.x = x
        self.y = y
        self.mass = kwargs.get("mass", 1.0)
        self.form = kwargs.get("form", "CO2")
        self.reactivity = kwargs.get("reactivity", 1.0)


def initialize_particles(num_particles: int, terrain: np.ndarray) -> np.ndarray:
    """Create particles near the water surface on valid terrain cells."""
    height, width = terrain.shape
    particles = []
    while len(particles) < num_particles:
        x = np.random.uniform(0, width)
        y = np.random.uniform(0, 3)
        if terrain[int(y), int(x)] == 1:
            particles.append(Particle(x=x, y=y, mass=1.0, form="CO2", reactivity=1.0))
    return np.array(particles)