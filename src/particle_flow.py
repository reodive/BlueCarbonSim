import numpy as np

from models.particle import Particle


def diffuse_particles(particles, terrain, flow_field):
    height, width = terrain.shape
    new_particles = []
    for particle in particles:
        x, y = particle.x, particle.y
        ix, iy = int(y), int(x)
        if 0 <= iy < flow_field.shape[0] and 0 <= ix < flow_field.shape[1]:
            flow_x, flow_y = flow_field[iy, ix]
        else:
            flow_x, flow_y = 0, 0
        dx = np.random.normal(flow_x, 0.5)
        dy = np.random.normal(flow_y, 0.5)
        new_x = np.clip(x + dx, 0, width - 1)
        new_y = np.clip(y + dy, 0, height - 1)
        if terrain[int(new_y), int(new_x)] == 1:
            particle.x = new_x
            particle.y = new_y
        new_particles.append(particle)
    return np.array(new_particles)


def generate_dynamic_flow_field(width, height, step):
    flow_field = np.zeros((height, width, 2))
    for y in range(height):
        for x in range(width):
            angle = np.sin((x + step * 0.1) * 0.1) + np.cos((y + step * 0.1) * 0.1)
            flow_x = 0.1 * np.cos(angle)
            flow_y = 0.1 * np.sin(angle)
            flow_field[y, x] = [flow_x, flow_y]
    return flow_field


def inject_particles(particles, terrain, num_new_particles=20, sources=None):
    height, width = terrain.shape
    new_particles = []
    if sources is None:
        sources = [(0, 50), (99, 20)]
        n_sources = len(sources)
    base, remainder = divmod(num_new_particles, n_sources)
    for i, (sx, sy) in enumerate(sources):
        count = base + (1 if i < remainder else 0)
        for _ in range(count):
            x = sx + np.random.normal(scale=1.0)
            y = sy + np.random.normal(scale=1.0)
            if 0 <= int(y) < height and 0 <= int(x) < width:
                if terrain[int(y), int(x)] == 1:
                    new_particles.append(Particle(x=x, y=y, mass=1.0, form="CO2", reactivity=1.0))
<<<<<<< HEAD
        if isinstance(particles, np.ndarray):
        return np.concatenate([particles, np.array(new_particles, dtype=object)])
    else:
        particles.extend(new_particles)
=======
        if isinstance(particles, list):
        particles.extend(new_particles)
        return particles
    elif len(new_particles) > 0:
        return np.concatenate((particles, new_particles))
    else:
>>>>>>> 9e0959a (ver mac)
        return particles