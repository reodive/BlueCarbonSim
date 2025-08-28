class Plant:
    """Model of a plant with carbon absorption behaviour."""

    def __init__(
        self,
        name,
        fixation_ratio,
        release_ratio,
        structure_density,
        opt_temp=20,
        light_tolerance=1.0,
        salinity_range=(20, 35),
        absorption_efficiency=1.0,
        growth_rate=1.0,
        x=50,
        y=95,
        radius=3,
    ):
        self.name = name
        self.fixation_ratio = fixation_ratio
        self.release_ratio = release_ratio
        self.structure_density = structure_density
        self.opt_temp = opt_temp
        self.light_tolerance = light_tolerance
        self.salinity_range = salinity_range
        self.absorption_efficiency = absorption_efficiency
        self.growth_rate = growth_rate
        self.x = x
        self.y = y
        self.radius = radius
        self.total_absorbed = 0
        self.total_fixed = 0
        self.total_growth = 0

    def absorb(self, base_absorption, efficiency_score=1.0):
        absorbed = self. absorption_efficiency * base_absorption * efficiency_score
        self.total_absorbed += absorbed
        fixed = absorbed * self.fixation_ratio
        self.total_fixed += fixed
        growth = absorbed * self.growth_late
        self.total_growth += growth
        return absorbed, fixed, growth