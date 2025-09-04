import numpy as np
from .models.particle import Particle


def diffuse_particles(particles, terrain, flow_field, *, cfl: float = 0.5, reflect_boundaries: bool = False):
    """
    粒子の拡散・移流（簡易）と開境界の処理。
    - 左端: 河川側（主に入流想定）
    - 右端: 外洋側（外向き速度で到達時に流出として削除）
    - 上下端: 外向き到達で削除（海面・海底の弱開境界）
    数値安定化: フィールドの最大流速に基づきΔtを自動縮小（CFL近似）。
    戻り値: (更新後粒子配列, 流出した総質量)
    """
    height, width = terrain.shape

    # Δtの自動調整（セル幅=1とみなす）
    u = flow_field[:, :, 0]
    v = flow_field[:, :, 1]
    max_speed = float(np.max(np.sqrt(u * u + v * v))) if flow_field.size > 0 else 0.0
    eps = 1e-6
    if max_speed < eps:
        dt = 1.0
    else:
        dt = min(1.0, cfl / max_speed)

    new_particles = []
    outflow_mass = 0.0
    for particle in particles:
        x, y = particle.x, particle.y
        # remember previous position for swept-contact checks
        try:
            particle.x_prev = x
            particle.y_prev = y
        except Exception:
            pass
        ix, iy = int(x), int(y)
        if 0 <= iy < flow_field.shape[0] and 0 <= ix < flow_field.shape[1]:
            flow_x, flow_y = flow_field[iy, ix]
        else:
            flow_x, flow_y = 0.0, 0.0

        # 速度×dt と乱流拡散（分散はdtに比例）
        dx = np.random.normal(flow_x * dt, 0.5 * np.sqrt(dt))
        dy = np.random.normal(flow_y * dt, 0.5 * np.sqrt(dt))

        trial_x = x + dx
        trial_y = y + dy
        new_x = float(np.clip(trial_x, 0, width - 1))
        new_y = float(np.clip(trial_y, 0, height - 1))

        # 開境界: 外向きに境界を跨いだ場合は流出扱い
        crossed_left = (new_x == 0 and trial_x < 0)
        crossed_right = (new_x == width - 1 and trial_x > (width - 1))
        crossed_top = (new_y == 0 and trial_y < 0)
        crossed_bottom = (new_y == height - 1 and trial_y > (height - 1))
        if crossed_left or crossed_right or crossed_top or crossed_bottom:
            if reflect_boundaries:
                # Simple specular reflection at domain borders
                if crossed_left or crossed_right:
                    trial_x = x - dx  # invert
                if crossed_top or crossed_bottom:
                    trial_y = y - dy
                new_x = float(np.clip(trial_x, 0, width - 1))
                new_y = float(np.clip(trial_y, 0, height - 1))
            else:
                outflow_mass += particle.mass
                continue

        if terrain[int(new_y), int(new_x)] == 1:
            particle.x = new_x
            particle.y = new_y
        new_particles.append(particle)
    return np.array(new_particles, dtype=object), outflow_mass


def generate_dynamic_flow_field(width, height, step, *, scale: float = 1.0):
    """
    動的流速場に右向きの基底流（外向き）と表層シアを追加。
    目的:
      - 左浅場に滞留しがちな粒子を中央/右へ搬送（Chlorella の独占回避）
      - 右側（外洋）へも供給し、Posidonia/Kelp の遭遇率を上げる
    """
    flow_field = np.zeros((height, width, 2))
    # 基底の東向き（右向き）流。表層ほど強く、ステップで弱い季節性（緩い振動）。
    base_east = 0.06 * (1.0 + 0.2 * np.sin(2 * np.pi * step / 60.0)) * float(scale)
    for y in range(height):
        depth_frac = y / max(height - 1, 1)
        surface_shear = 1.0 - depth_frac  # 表層1.0 → 底0.0
        for x in range(width):
            # 既存の渦成分（弱め）
            angle = np.sin((x + step * 0.1) * 0.1) + np.cos((y + step * 0.1) * 0.1)
            flow_x = 0.06 * np.cos(angle) * float(scale)
            flow_y = 0.06 * np.sin(angle) * float(scale)
            # 基底流を加算（表層シア付き）
            flow_x += base_east * (0.6 + 0.4 * surface_shear)
            # ごく弱い上向き成分（再浮上を助ける）
            flow_y += 0.01 * (surface_shear - 0.5)
            flow_field[y, x] = [flow_x, flow_y]
    return flow_field


def inject_particles(particles, terrain, num_new_particles=20, sources=None):
    height, width = terrain.shape
    new_particles = []
    if sources is None:
        # 左端: 河川流入源 / 右端: 外洋側（主に流出）
        sources = [(0, 50), (99, 20)]
    n_sources = len(sources)
    base, remainder = divmod(num_new_particles, n_sources)
    actually_added = 0
    for i, (sx, sy) in enumerate(sources):
        count = base + (1 if i < remainder else 0)
        for _ in range(count):
            x = sx + np.random.normal(scale=1.0)
            y = sy + np.random.normal(scale=1.0)
            # 厳密に境界内かを x, y の実数値で判定
            if 0.0 <= x < (width) and 0.0 <= y < (height):
                if terrain[int(y), int(x)] == 1:
                    new_particles.append(Particle(x=x, y=y, mass=1.0, form="CO2", reactivity=1.0))
                    actually_added += 1
    if isinstance(particles, list):
        particles.extend(new_particles)
        return particles, actually_added
    elif len(new_particles) > 0:
        return np.concatenate((particles, new_particles)), actually_added
    else:
        return particles, 0
