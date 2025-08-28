class Plant:
    """Model of a plant with carbon absorption behaviour."""

    def __init__(
        self,
        name,
        absorb_efficiency,
        growth_speed,
        fixation_ratio,
        release_ratio,
        structure_density,
        opt_temp=20,
        light_tolerance=1.0,
        salinity_range=(20, 35),
        x=50,
        y=95,
        radius=3,
    ):
        self.name = name
        self.absorb_efficiency = absorb_efficiency
        self.growth_speed = growth_speed
        self.fixation_ratio = fixation_ratio
        self.release_ratio = release_ratio
        self.structure_density = structure_density
        self.opt_temp = opt_temp
        self.light_tolerance = light_tolerance
        self.salinity_range = salinity_range
        self.x = x
        self.y = y
        self.radius = radius
        self.total_absorbed = 0
        self.total_fixed = 0
        self.total_growth = 0

    def absorb(self, base_absorption, efficiency_score=1.0):
        absorbed = self.absorb_efficiency * base_absorption * efficiency_score
        self.total_absorbed += absorbed
        fixed = absorbed * self.fixation_ratio
        self.total_fixed += fixed
        growth = absorbed * self.growth_speed
        self.total_growth += growth
        return absorbed, fixed, growth
