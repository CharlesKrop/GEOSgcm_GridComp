from ndsl import NDSLRuntime, QuantityFactory, StencilFactory

from pyMoist.microphysics.gfdl_1m.config import GFDL1MConfig
from pyMoist.microphysics.gfdl_1m.diagnostics import GFDL1MDiagnostics
from pyMoist.microphysics.gfdl_1m.locals import GFDL1MLocals
from pyMoist.microphysics.gfdl_1m.macrophysics import GFDL1MMacrophysics
from pyMoist.microphysics.gfdl_1m.microphysics import GFDL1MMicrophysics
from pyMoist.microphysics.gfdl_1m.optimization import get_optimization_config
from pyMoist.microphysics.gfdl_1m.setup import GFDL1MSetup
from pyMoist.microphysics.gfdl_1m.state import GFDL1MState
from pyMoist.saturation_tables import get_saturation_vapor_pressure_table


class GFDL1M(NDSLRuntime):
    """GFDL single-moment microphysics composed from four runtime components."""

    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        config: GFDL1MConfig,
    ):
        super().__init__(stencil_factory, get_optimization_config(stencil_factory))

        saturation_tables = get_saturation_vapor_pressure_table(stencil_factory)
        self._locals = GFDL1MLocals.empty(quantity_factory)

        self._setup = GFDL1MSetup(
            stencil_factory=stencil_factory,
            quantity_factory=quantity_factory,
            config=config,
            saturation_tables=saturation_tables,
        )
        self._macrophysics = GFDL1MMacrophysics(
            stencil_factory=stencil_factory,
            quantity_factory=quantity_factory,
            config=config,
            saturation_tables=saturation_tables,
            lid_level=self._setup.lid_level,
        )
        self._microphysics = GFDL1MMicrophysics(
            stencil_factory=stencil_factory,
            quantity_factory=quantity_factory,
            config=config,
            locals=self._locals,
            lid_level=self._setup.lid_level,
        )
        self._diagnostics = GFDL1MDiagnostics(
            stencil_factory=stencil_factory,
            locals=self._locals,
        )

    def __call__(self, state: GFDL1MState):
        self._setup(state=state, locals=self._locals)
        self._macrophysics(state=state, locals=self._locals)
        self._microphysics(state=state)
        self._diagnostics(state=state)
