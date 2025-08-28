import numpy as np


def generate_depth_map(mode="random", width=50, height=50):
    if mode == "flat":
        return np.full((width, height), 1.0)
    elif mode == "slope":
        return np.tile(np.linspace(0.2, 1.0, width), (height, 1)).T
    elif mode == "bay":
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        X, Y = np.meshgrid(x, y)
        radius = np.sqrt(X ** 2 + Y ** 2)
        return 1.0 - np.clip(radius, 0, 1)
    else:
        return np.random.uniform(0.2, 1.0, size=(width, height))


def create_terrain(width=100, height=100):
    terrain = np.ones((height, width))
    terrain[40:60, 45:55] = 0
    depth_map = np.tile(np.arange(height).reshape((height, 1)), (1, width)) / (height - 1)
    return terrain, depth_map


def mark_absorption_area(absorption_map, x, y, radius):
    height, width = absorption_map.shape
    for dx in range(-int(radius), int(radius) + 1):
        for dy in range(-int(radius), int(radius) + 1):
            px = int(x + dx)
            py = int(y + dy)
            if 0 <= px < width and 0 <= py < height:
                if dx ** 2 + dy ** 2 <= radius ** 2:
                    absorption_map[py, px] += 1
