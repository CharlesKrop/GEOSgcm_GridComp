import dataclasses

from ndsl import Local, LocalState, NDSLRuntime, QuantityFactory, StencilFactory
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM
from ndsl.dsl.gt4py import PARALLEL, computation, interval
from ndsl.dsl.typing import Float, FloatField, Int
from ndsl.stencils.basic_operations import set_value, copy, add

from pyMoist.constants import MAPL_GRAV
from pyMoist.microphysics.GFDL_1M.config import GFDL1MConfig
from pyMoist.saturation_tables import GlobalTable_saturation_tables, SaturationVaporPressureTable, saturation_specific_humidity
from pyMoist.microphysics.GFDL_1M.locals import GFDL1MLocals
from pyMoist.microphysics.GFDL_1M.state import GFDL1MState


def calculate_derived_states(
    p_interface: FloatField,
    p_interface_mb: FloatField,
    p_mb: FloatField,
    geopotential_height_interface: FloatField,
    edge_height_above_surface: FloatField,
    layer_height_above_surface: FloatField,
    layer_thickness: FloatField,
    dp: FloatField,
    mass: FloatField,
    mass_inverse: FloatField,
    t: FloatField,
    esx: GlobalTable_saturation_tables,
    sat: FloatField,
    dsat: FloatField,
    u: FloatField,
    u_unmodified: FloatField,
    v: FloatField,
    v_unmodified: FloatField,
):
    """
    Computes derived state fields required for the rest of the GFDL single moment
    microphysics module.

    Stencil MUST be built using K_INTERFACE_DIM to function properly

    Args:
        p_interface (FloatField)
        p_interface_mb (FloatField)
        p_mb (FloatField)
        geopotential_height_interface (FloatField)
        edge_height_above_surface (FloatField)
        layer_height_above_surface (FloatField)
        layer_thickness (FloatField)
        dp (FloatField)
        mass (FloatField)
        mass_inverse (FloatField)
        t (FloatField)
        esx (GlobalTable_saturation_tables)
        sat (FloatField)
        dsat (FloatField)
        u (FloatField)
        u_unmodified (FloatField)
        v (FloatField)
        v_unmodified (FloatField)
    """
    from __externals__ import k_end

    with computation(PARALLEL), interval(...):
        p_interface_mb = p_interface * 0.01
        edge_height_above_surface = geopotential_height_interface - geopotential_height_interface.at(K=k_end)

    with computation(PARALLEL), interval(0, -1):
        p_mb = 0.5 * (p_interface_mb + p_interface_mb[0, 0, 1])
        layer_height_above_surface = 0.5 * (edge_height_above_surface + edge_height_above_surface[0, 0, 1])
        layer_thickness = edge_height_above_surface - edge_height_above_surface[0, 0, 1]
        dp = p_interface[0, 0, 1] - p_interface
        mass = dp / MAPL_GRAV
        mass_inverse = 1 / mass
        sat, dsat = saturation_specific_humidity(t=t, p=p_mb * 100.0, esx=esx)
        sat, dsat = saturation_specific_humidity(t=t, p=p_mb * 100.0, esx=esx)
        u_unmodified = u
        v_unmodified = v


def update_precipitation(
    mixing_ratio: FloatField,
    shallow_convection_values: FloatField,
):
    """Update precipitate mixing ratio

    Args:
        mixing_ratio (FloatField)
        shallow_convection_values (FloatField)
    """
    from __externals__ import DT_MOIST

    with computation(PARALLEL), interval(...):
        mixing_ratio = mixing_ratio + shallow_convection_values * DT_MOIST


@dataclasses.dataclass
class GFDL1MSetupLocals(LocalState):
    temporary_3d: Local = dataclasses.field(
        metadata={
            "name": "temporary_3d",
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )


class GFDL1MSetup(NDSLRuntime):
    """
    Conglomeration of small stencils required to setup the main macro/micro physics schemes within the GFDL1M module. Contains the following stencils:

    prepare_tendencies: preloads macrophysics tendencies for post-phase_change calculations
    calculate_derived_states: computes fields required for the module but not provided by the model
    update_precipitation (conditional): updates precipitation (rain and snow) using shallow convection values
    """

    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        config: GFDL1MConfig,
        saturation_tables: SaturationVaporPressureTable,
    ):
        """Initialize the GFDL1M microphysics setup class

        Args:
            stencil_factory (StencilFactory)
            quantity_factory (QuantityFactory)
            config (GFDL1MConfig)
            saturation_tables (SaturationVaporPressureTable)
        """
        # init NDSLRuntime
        super().__init__(stencil_factory)

        # make configuration and saturation tables visible at runtime
        self.config = config
        self.saturation_tables = saturation_tables

        # initialize locals
        self._locals = GFDL1MSetupLocals.make_locals(quantity_factory)

        # construct stencils
        self._set_value = stencil_factory.from_dims_halo(
            func=set_value,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        self._calculate_derived_states = stencil_factory.from_dims_halo(
            func=calculate_derived_states,
            compute_dims=[I_DIM, J_DIM, K_INTERFACE_DIM],
        )

        self._find_lcl_level = stencil_factory.from_dims_halo(
            func=find_lcl_level,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        self._add = stencil_factory.from_dims_halo(
            func=add,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        self._copy = stencil_factory.from_dims_halo(
            func=copy,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        self._update_precipitation = stencil_factory.from_dims_halo(
            func=update_precipitation,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={
                "DT_MOIST": config.DT_MOIST,
            },
        )

        # Dev NOTE: this is an orchestration workaround. Direct call to
        #           `self.saturation_tables.X` fails closure capture for
        #           argument reconstruction at call time
        self._esx = self.saturation_tables.esx

    def __call__(
        self,
        state: GFDL1MState,
        locals: GFDL1MLocals,
    ):
        """Setup the GFDL1M microphysics module

        Args:
            state (GFDL1MState): Variables associated with the larger model outside of the GFDL1M microphysics module.
            locals (GFDL1MLocals): The local fields for the GFDL1M microphysics module.
        """
        # Initialize reflectivity
        self._set_value(
            field=locals.reflectivity,
            value=Float(-30.0),
        )

        self._calculate_derived_states(
            p_interface=state.p_interface,
            p_interface_mb=locals.p_interface_mb,
            p_mb=locals.p_mb,
            geopotential_height_interface=locals.z_interface,
            edge_height_above_surface=locals.edge_height_above_surface,
            layer_height_above_surface=locals.layer_height_above_surface,
            layer_thickness=locals.layer_thickness,
            dp=locals.dp,
            mass=locals.mass,
            mass_inverse=locals.mass_inverse,
            t=state.t,
            esx=self._esx,
            sat=locals.saturation_specific_humidity,
            dsat=locals.dsaturation_specific_humidity,
            u=state.u,
            u_unmodified=locals.u_unmodified,
            v=state.v,
            v_unmodified=locals.v_unmodified,
        )

        self._find_lcl_level(
            t=state.t,
            p_mb=locals.p_mb,
            vapor=state.mixing_ratio.vapor,
            esx=self._esx,
            lcl_level=locals.lcl_level,
        )

        if self.config.GFDL_MP_KLID > 0.0:
            locals.lid_level = Int(self.config.GFDL_MP_KLID)
        else:
            locals.lid_level = Int(1)

        # set unused exports to zero
        self._set_value(field=state.precipitation_at_surface.shallow_convective_precipitation, value=Float(0.0))
        self._set_value(field=state.precipitation_at_surface.deep_convective_precipitation, value=Float(0.0))
        self._set_value(field=state.precipitation_at_surface.anvil_precipitation, value=Float(0.0))
        self._set_value(field=state.precipitation_at_surface.shallow_convective_snow, value=Float(0.0))
        self._set_value(field=state.precipitation_at_surface.deep_convective_snow, value=Float(0.0))
        self._set_value(field=state.precipitation_at_surface.anvil_snow, value=Float(0.0))

        # pre-fill macrophysics exports
        self._copy(input=state.u, output=state.tendencies.dudt_macro)
        self._copy(input=state.v, output=state.tendencies.dvdt_macro)
        self._copy(input=state.t, output=state.tendencies.dtdt_macro)
        self._copy(input=state.mixing_ratio.vapor, output=state.tendencies.dvapordt_macro)
        self._add(summand_1=state.mixing_ratio.convective_liquid, summand_2=state.mixing_ratio.large_scale_liquid, output=locals.placeholder.temporary_3d)
        self._copy(input=self._locals.temporary_3d, output=state.tendencies.dliquiddt_macro)
        self._add(summand_1=state.mixing_ratio.convective_ice, summand_2=state.mixing_ratio.large_scale_ice, output=self._locals.temporary_3d)
        self._copy(input=self._locals.temporary_3d, output=state.tendencies.dicedt_macro)
        self._add(summand_1=state.cloud_fraction.convective, summand_2=state.cloud_fraction.large_scale, output=self._locals.temporary_3d)
        self._copy(input=self._locals.temporary_3d, output=state.tendencies.dcloud_fractiondt_macro)
        self._copy(input=state.mixing_ratio.graupel, output=state.tendencies.dgraupeldt_macro)
        self._copy(input=state.mixing_ratio.rain, output=state.tendencies.draindt_macro)
        self._copy(input=state.mixing_ratio.snow, output=state.tendencies.dsnowdt_macro)
        self._set_value(field=state.cloud_liquid_evaporation, value=Float(0.0))
        self._set_value(field=state.cloud_ice_sublimation, value=Float(0.0))
        self._set_value(field=state.hydrostatic_pdf_iterations, value=Float(0.0))
        self._set_value(field=state.relative_humidity_after_pdf, value=Float(0.0))

        # include shallow precip condensated if present
        if state.shallow_convection_rain is not None:
            self._update_precipitation(state.mixing_ratio.rain, state.shallow_convection_rain)

        if state.shallow_convection_snow is not None:
            self._update_precipitation(state.mixing_ratio.snow, state.shallow_convection_snow)
