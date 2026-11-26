import dataclasses

from ndsl import Local, LocalState, NDSLRuntime, QuantityFactory, StencilFactory
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM
from ndsl.dsl.gt4py import PARALLEL, computation, interval, FORWARD
from ndsl.dsl.typing import Float, FloatField, Int, IntFieldIJ
from ndsl.stencils.basic_operations import add, copy, set_value
from ndsl.stencils.basic_operations_2d import set_value_2d

from pyMoist.constants import MAPL_GRAV
from pyMoist.microphysics.gfdl_1m.config import GFDL1MConfig
from pyMoist.microphysics.gfdl_1m.locals import GFDL1MLocals
from pyMoist.microphysics.gfdl_1m.state import GFDL1MState
from pyMoist.saturation_tables import (
    GlobalTable_saturation_tables,
    SaturationVaporPressureTable,
    saturation_specific_humidity,
    saturation_specific_humidity_frozen_surface,
    saturation_specific_humidity_liquid_surface,
)
from pyMoist.shared.cloud_processes import find_lcl_level


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
    esw: GlobalTable_saturation_tables,
    lqu: Float,
    ese: GlobalTable_saturation_tables,
    frz: Float,
    sat_liquid: FloatField,
    sat_ice: FloatField,
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
        sat_liquid, _ = saturation_specific_humidity_liquid_surface(esw, lqu, t, p_mb * 100.0)
        sat_ice, _ = saturation_specific_humidity_frozen_surface(ese, frz, t, p_mb * 100.0)
        sat, dsat = saturation_specific_humidity(t=t, p=p_mb * 100.0, esx=esx)
        u_unmodified = u
        v_unmodified = v


def set_value_2d_int(field: IntFieldIJ, value: Int) -> None:
    """
    Sets every element of a field to a single value.

    Args:
        field: output field
        value: value of Int type
    """
    with computation(FORWARD), interval(0, 1):
        field = value


def update_precipitation(
    mixing_ratio: FloatField,
    shallow_convection_values: FloatField,
):
    """Update precipitate mixing ratio

    Args:
        mixing_ratio (FloatField)
        shallow_convection_values (FloatField)
    """
    from __externals__ import DT

    with computation(PARALLEL), interval(...):
        mixing_ratio = mixing_ratio + shallow_convection_values * DT


@dataclasses.dataclass
class GFDL1MSetupLocals(LocalState):
    temporary_3d: Local = dataclasses.field(
        metadata={
            "name": "temporary_3d",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
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
        state: GFDL1MState,
        locals: GFDL1MLocals,
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

        # make saturation tables visible at runtime
        self._saturation_tables = saturation_tables
        # NOTE: This is an orchestration workaround. Direct call to `self._saturation_tables.X`
        # fails closure capture for argument reconstruction at call time.
        self._esx = self._saturation_tables.esx
        self._esw = self._saturation_tables.esw
        self._lqu = self._saturation_tables.lqu
        self._ese = self._saturation_tables.ese
        self._frz = self._saturation_tables.frz

        # make config and fields visible at runtime
        self.config = config
        self._state = state
        self._gfdl_1m_locals = locals

        # # if GFDL_MP_KLID is non-negative (default value is -999), macrophysics will be restricted to start at that vertical level
        self.lid_level = max(0, int(config.GFDL_MP_KLID))

        # initialize class specific locals
        self._locals = GFDL1MSetupLocals.make_locals(quantity_factory)

        # construct stencils
        self._add = stencil_factory.from_dims_halo(
            func=add,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        self._calculate_derived_states = stencil_factory.from_dims_halo(
            func=calculate_derived_states,
            compute_dims=[I_DIM, J_DIM, K_INTERFACE_DIM],
        )

        self._copy = stencil_factory.from_dims_halo(
            func=copy,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        self._find_lcl_level = stencil_factory.from_dims_halo(
            func=find_lcl_level,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        self._set_value = stencil_factory.from_dims_halo(
            func=set_value,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        self._set_value_2d = stencil_factory.from_dims_halo(
            func=set_value_2d,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        self._set_value_2d_int = stencil_factory.from_dims_halo(
            func=set_value_2d_int,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        self._update_precipitation = stencil_factory.from_dims_halo(
            func=update_precipitation,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={
                "DT": config.DT,
            },
        )

    def __call__(self):
        """Setup the GFDL1M microphysics module

        Args:
            state (GFDL1MState): Variables associated with the larger model outside of the GFDL1M microphysics module.
            locals (GFDL1MLocals): The local fields for the GFDL1M microphysics module.
        """
        # Initialize reflectivity
        self._set_value(
            field=self._gfdl_1m_locals.reflectivity,
            value=Float(-30.0),
        )

        self._calculate_derived_states(
            p_interface=self._state.p_interface,
            p_interface_mb=self._gfdl_1m_locals.p_interface_mb,
            p_mb=self._gfdl_1m_locals.p_mb,
            geopotential_height_interface=self._state.z_interface,
            edge_height_above_surface=self._gfdl_1m_locals.edge_height_above_surface,
            layer_height_above_surface=self._gfdl_1m_locals.layer_height_above_surface,
            layer_thickness=self._gfdl_1m_locals.layer_thickness,
            dp=self._gfdl_1m_locals.dp,
            mass=self._gfdl_1m_locals.mass,
            mass_inverse=self._gfdl_1m_locals.mass_inverse,
            t=self._state.t,
            esx=self._esx,
            sat=self._gfdl_1m_locals.saturation_specific_humidity,
            dsat=self._gfdl_1m_locals.dsaturation_specific_humidity,
            esw=self._esw,
            lqu=self._lqu,
            ese=self._ese,
            frz=self._frz,
            sat_liquid=self._gfdl_1m_locals.saturation_specific_humidity_liquid,
            sat_ice=self._gfdl_1m_locals.saturation_specific_humidity_ice,
            u=self._state.u,
            u_unmodified=self._gfdl_1m_locals.u_unmodified,
            v=self._state.v,
            v_unmodified=self._gfdl_1m_locals.v_unmodified,
        )

        self._find_lcl_level(
            t=self._state.t,
            p_mb=self._gfdl_1m_locals.p_mb,
            vapor=self._state.mixing_ratio.vapor,
            esx=self._esx,
            lcl_level=self._gfdl_1m_locals.lcl_level,
        )

        # set unused exports to zero
        self._set_value_2d(field=self._state.precipitation_at_surface.shallow_convective_precipitation, value=Float(0.0))
        self._set_value_2d(field=self._state.precipitation_at_surface.deep_convective_precipitation, value=Float(0.0))
        self._set_value_2d(field=self._state.precipitation_at_surface.anvil_precipitation, value=Float(0.0))
        self._set_value_2d(field=self._state.precipitation_at_surface.shallow_convective_snow, value=Float(0.0))
        self._set_value_2d(field=self._state.precipitation_at_surface.deep_convective_snow, value=Float(0.0))
        self._set_value_2d(field=self._state.precipitation_at_surface.anvil_snow, value=Float(0.0))

        # pre-fill macrophysics exports
        self._copy(input=self._state.u, output=self._state.tendencies.du_dt_macro)
        self._copy(input=self._state.v, output=self._state.tendencies.dv_dt_macro)
        self._copy(input=self._state.t, output=self._state.tendencies.dt_dt_macro)
        self._copy(input=self._state.mixing_ratio.vapor, output=self._state.tendencies.dvapor_dt_macro)
        self._add(summand_1=self._state.mixing_ratio.convective_liquid, summand_2=self._state.mixing_ratio.large_scale_liquid, sum=self._locals.temporary_3d)
        self._copy(input=self._locals.temporary_3d, output=self._state.tendencies.dliquid_dt_macro)
        self._add(summand_1=self._state.mixing_ratio.convective_ice, summand_2=self._state.mixing_ratio.large_scale_ice, sum=self._locals.temporary_3d)
        self._copy(input=self._locals.temporary_3d, output=self._state.tendencies.dice_dt_macro)
        self._add(summand_1=self._state.cloud_fraction.convective, summand_2=self._state.cloud_fraction.large_scale, sum=self._locals.temporary_3d)
        self._copy(input=self._locals.temporary_3d, output=self._state.tendencies.dcloud_fraction_dt_macro)
        self._copy(input=self._state.mixing_ratio.graupel, output=self._state.tendencies.dgraupel_dt_macro)
        self._copy(input=self._state.mixing_ratio.rain, output=self._state.tendencies.drain_dt_macro)
        self._copy(input=self._state.mixing_ratio.snow, output=self._state.tendencies.dsnow_dt_macro)
        self._set_value(field=self._state.cloud_liquid_evaporation, value=Float(0.0))
        self._set_value(field=self._state.cloud_ice_sublimation, value=Float(0.0))
        self._set_value(field=self._state.hydrostatic_pdf_iterations, value=Float(0.0))
        self._set_value(field=self._state.relative_humidity_after_pdf, value=Float(0.0))

        # include shallow precip condensated if present
        if self._state.shallow_convection_rain is not None:
            self._update_precipitation(self._state.mixing_ratio.rain, self._state.shallow_convection_rain)

        if self._state.shallow_convection_snow is not None:
            self._update_precipitation(self._state.mixing_ratio.snow, self._state.shallow_convection_snow)
