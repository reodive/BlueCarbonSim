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
        light_half_saturation=0.5,
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
        self.light_half_saturation = light_half_saturation
        self.salinity_range = salinity_range
        self.absorption_efficiency = absorption_efficiency
        self.growth_rate = growth_rate
        self.x = x
        self.y = y
        self.radius = radius
        self.total_absorbed = 0
        self.total_fixed = 0
        self.total_growth = 0

    def absorb(self, absorbed_mass):
        """
        引数は既に「水から取り去られた吸収量（mgC単位の質量）」と解釈する。
        ここでは二重に効率を掛けない（質量保存のため）。
        """
        absorbed = float(absorbed_mass)
        self.total_absorbed += absorbed
        fixed = absorbed * self.fixation_ratio
        self.total_fixed += fixed
        growth = absorbed * self.growth_rate
        self.total_growth += growth
        return absorbed, fixed, growth
