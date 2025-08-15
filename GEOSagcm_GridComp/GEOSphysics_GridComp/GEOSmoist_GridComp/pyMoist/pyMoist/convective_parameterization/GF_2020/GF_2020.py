from pyMoist.convective_parameterization.GF_2020.setup import Setup
from pyMoist.convective_parameterization.GF_2020.config import GF2020Config
from ndsl import StencilFactory, QuantityFactory
from pyMoist.convective_parameterization.GF_2020.temporaries import Temporaries
from pyMoist.convective_parameterization.GF_2020.state import MixingRatios
from pyMoist.saturation_tables.tables.main import SaturationVaporPressureTable


class GF2020:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        GF2020_config: GF2020Config,
    ):
        if self.stencil_factory.grid_indexing.n_halo != 0:
            raise ValueError("halo needs to be zero for GFDL Single Moment microphysics")

        # Initalize saturation tables
        self.saturation_tables = SaturationVaporPressureTable(self.stencil_factory.backend)

        # Initalize extra quantities
        self.temporaries = Temporaries.make(self.quantity_factory)

        # Initalize submodules and build stencils
        self.setup = Setup(stencil_factory)

    def __call__(
        self,
        p_interface,
        geopotential_height_interface,
        t,
        area,
        convection_fraction,
        temporaries: Temporaries,
        mixing_ratios: MixingRatios,
        saturation_tables: SaturationVaporPressureTable,
    ):
        self.setup(
            p_interface,
            geopotential_height_interface,
            t,
            mixing_ratios,
            area,
            convection_fraction,
            self.temporaries,
            self.saturation_tables,
        )
