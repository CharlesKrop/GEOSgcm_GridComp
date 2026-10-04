from ndsl import NDSLRuntime, QuantityFactory, StencilFactory, ndsl_log
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM
from ndsl.dsl.gt4py import PARALLEL, computation, interval, FORWARD
from ndsl.dsl.typing import Float, FloatField
from ndsl.stencils.basic_operations import add, copy, set_value

from pyMoist.microphysics.GFDL_1M.config import GFDL1MConfig
from pyMoist.microphysics.GFDL_1M.locals import GFDL1MLocals
from pyMoist.microphysics.GFDL_1M.macrophysics import GFDL1MMacrophysics
from pyMoist.microphysics.GFDL_1M.microphysics.gfdl_mp_v3 import GFDLMPV3
from pyMoist.microphysics.GFDL_1M.optimization import get_optimization_config
from pyMoist.microphysics.GFDL_1M.setup import GFDL1MSetup
from pyMoist.microphysics.GFDL_1M.state import GFDL1MState
from pyMoist.saturation_tables import get_saturation_vapor_pressure_table


def convert_mm_per_day_to_kg_per_m2_per_s(field: FloatField):
    """Convert a field from mm day^-1 to kg m^-2 s^-1

    Args:
        field (FloatField)
    """
    with computation(PARALLEL), interval(...):
        field = field * 1.0e-3 / 86400.0


def flip_sign(input: FloatField, output: FloatField):
    with computation(PARALLEL), interval(...):
        output = -1.0 * input


def min_with_one(field: FloatField):
    with computation(PARALLEL), interval(...):
        field = min(field, 1.0)


def negative_adjustment(
    graupel: FloatField,
    ice: FloatField,
    liquid: FloatField,
    rain: FloatField,
    snow: FloatField,
    vapor: FloatField,
):
    with computation(FORWARD), interval(0, 1):
        # initialize 2d internals
        n_solid_fix = 0
        n_liquid_fix = 0
        n_vapor_col = 0

    # 1. Local phase borrowing (GFDL neg_adj style)
    with computation(PARALLEL), interval(...):
        # Solid phase adjustments
        if ice < 0.0 and snow > 0.0:
            dq = min(-ice, snow)
            ice = ice + dq
            snow = snow - dq
            n_solid_fix = n_solid_fix + 1

        if snow < 0.0 and graupel > 0.0:
            dq = min(-snow, graupel)
            snow = snow + dq
            graupel = graupel - dq
            n_solid_fix = n_solid_fix + 1

        if graupel < 0.0 and rain > 0.0:
            dq = min(-graupel, rain)
            graupel = graupel + dq
            rain = rain - dq
            n_solid_fix = n_solid_fix + 1

        # Liquid phase adjustments
        if rain < 0.0 and liquid > 0.0:
            dq = min(-rain, liquid)
            rain = rain + dq
            liquid = liquid - dq
            n_liquid_fix = n_liquid_fix + 1

        if liquid < 0.0 and vapor > 0.0:
            dq = min(-liquid, vapor)
            liquid = liquid + dq
            vapor = vapor - dq
            n_liquid_fix = n_liquid_fix + 1

    # 2. Column water vapor borrowing
    with computation(FORWARD), interval(0, -1):
        if vapor < 0.0:
            vapor[0, 0, 1] = vapor[0, 0, 1] + vapor
            vapor = 0.0
            n_vapor_col = n_vapor_col + 1

    with computation(FORWARD), interval(-1, None):
        if vapor < 0.0 and vapor[0, 0, -1] > 0.0:
            dq = min(-vapor, vapor[0, 0, -1])
            vapor[0, 0, -1] = vapor[0, 0, -1] - dq
            vapor = vapor + dq
            n_vapor_col = n_vapor_col + 1

    # Final safety clamp
    with computation(PARALLEL), interval(...):
        graupel = max(graupel, 0.0)
        ice = max(ice, 0.0)
        liquid = max(liquid, 0.0)
        rain = max(rain, 0.0)
        snow = max(snow, 0.0)
        vapor = max(vapor, 1.0e-15)


class GFDL1M(NDSLRuntime):
    """
    GFDL Single Moment microphysics

    The primary purpose of this code is to compute macro/microphysical tendencies to be applied to state
    variables (p, t, wind, etc.). This code requires all fields to be preloaded with Fortran memory or
    otherwise supplied between the __init__ and __call__ steps.

    Performs the following functions to achieve this goal:
    __init__
        - initialize saturation vapor pressure tables, intialize temporary/output fields, construct stencils
        Arguments: StencilFactory, QuantityFactory, GFDL1MConfig

    __call__
        - setup: compute additional required fields, create pristine copies of input variables
        - phase_change: create new condensates, perform phase change operations
        - driver: precipitate condensates
        - finalize: compute tendencies, prepare fields to be returned to the larger model
        Arguments: none (data needs to be pre-loaded)
    """

    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        config: GFDL1MConfig,
    ):
        super().__init__(stencil_factory, get_optimization_config(stencil_factory))

        # WARNING - to be removed when 11.10.1 update is complete
        ndsl_log.warning(
            "pyMoist.GFDL_1M: This NDSL version of GFDL_1M was ported from GEOS v11.8.1, and has not yet been updated to v11.10. "
            "Stable execution is not guarenteed, as v11.10 made siginificant changes to the source Fortran."
        )

        # initialize saturation tables
        saturation_tables = get_saturation_vapor_pressure_table(stencil_factory)

        # initialize locals
        self._locals = GFDL1MLocals.make_locals(quantity_factory)

        # make config visible at runtime
        self._config = config

        # build subcomponents and stencils
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
        )

        self._add = stencil_factory.from_dims_halo(
            func=add,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        self._convert_mm_per_day_to_kg_per_m2_per_s = stencil_factory.from_dims_halo(
            func=convert_mm_per_day_to_kg_per_m2_per_s,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        self._convert_mm_per_day_to_kg_per_m2_per_s_interface = stencil_factory.from_dims_halo(
            func=convert_mm_per_day_to_kg_per_m2_per_s,
            compute_dims=[I_DIM, J_DIM, K_INTERFACE_DIM],
        )

        self._copy = stencil_factory.from_dims_halo(
            func=copy,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        self._flip_sign = stencil_factory.from_dims_halo(
            func=flip_sign,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        self._min_with_one = stencil_factory.from_dims_halo(
            func=min_with_one,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        self._negative_adjustment = stencil_factory.from_dims_halo(
            func=negative_adjustment,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        self._set_value = stencil_factory.from_dims_halo(
            func=set_value,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        self._set_value_k_interface = stencil_factory.from_dims_halo(
            func=set_value,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        self._gfdl_microphysics_v3 = GFDLMPV3()

    def __call__(
        self,
        state: GFDL1MState,
    ):
        # miscellaneous setup required for macro and/or microphysics schemes
        self._setup(
            state=state,
            locals=self._locals,
        )

        # --------------------------------------------------
        # MACROPHYSICS
        # --------------------------------------------------
        # compute macrophysical tendencies, use the hydrostatic pdf to distribute particles,
        # then melt, freeze, and evaporate, all according to options defined in namelist
        self._macrophysics(
            state=state,
            locals=self._locals,
        )

        # print a debug warning, if any non-physical values are identified
        if self._config.DEBUG_TQ_ERRORS:
            option_not_implemented = True

        # --------------------------------------------------
        # MICROPHYSICS
        # --------------------------------------------------
        self._copy(input=state.mixing_ratio.vapor, output=state.tendencies.dvapordt_micro)
        self._add(summand_1=state.mixing_ratio.convective_ice, summand_2=state.mixing_ratio.large_scale_ice, sum=self._locals.temporary_3d)
        self._copy(input=self._locals.temporary_3d, output=state.tendencies.dicedt_micro)
        self._add(summand_1=state.mixing_ratio.convective_liquid, summand_2=state.mixing_ratio.large_scale_liquid, sum=self._locals.temporary_3d)
        self._copy(input=self._locals.temporary_3d, output=state.tendencies.dliquiddt_micro)
        self._add(summand_1=state.cloud_fraction.convective, summand_2=state.cloud_fraction.large_scale, sum=self._locals.temporary_3d)
        self._copy(input=self._locals.temporary_3d, output=state.tendencies.dcloud_fractiondt_micro)
        self._copy(input=state.mixing_ratio.graupel, output=state.tendencies.dgraupeldt_micro)
        self._copy(input=state.mixing_ratio.rain, output=state.tendencies.draindt_micro)
        self._copy(input=state.mixing_ratio.snow, output=state.tendencies.dsnowdt_micro)
        self._copy(input=state.t, output=state.tendencies.dtdt_micro)
        self._copy(input=state.u, output=state.tendencies.dudt_micro)
        self._copy(input=state.v, output=state.tendencies.dvdt_micro)

        # delta-z layer thickness (gfdl mp v3 expects this to be negative)
        self._flip_sign(input=self._locals.layer_thickness, output=self._locals.layer_thickness_negative)

        # zero out GFDLMPV3 outputs
        self._set_value(field=self._locals.dcloud_fractiondt, value=Float(0.0))
        self._set_value_k_interface(fleid=state.non_anvil_large_scale.ice_precip_flux, value=Float(0.0))
        self._set_value_k_interface(fleid=state.non_anvil_large_scale.liquid_precip_flux, value=Float(0.0))

        # cloud fractions and condensates for radiation
        self._add(summand_1=state.cloud_fraction.convective, summand_2=state.cloud_fraction.large_scale, sum=self._locals.temporary_3d)
        self._min_with_one(field=self._locals.temporary_3d)
        self._copy(input=self._locals.temporary_3d, output=state.radiation_field.cloud_fraction)
        self._add(summand_1=state.mixing_ratio.convective_liquid, summand_2=state.mixing_ratio.large_scale_liquid, sum=self._locals.temporary_3d)
        self._copy(input=self._locals.temporary_3d, output=state.radiation_field.liquid)
        self._add(summand_1=state.mixing_ratio.convective_ice, summand_2=state.mixing_ratio.large_scale_ice, sum=self._locals.temporary_3d)
        self._copy(input=self._locals.temporary_3d, output=state.radiation_field.ice)
        self._copy(input=state.mixing_ratio.vapor, output=state.radiation_field.vapor)
        self._copy(input=state.mixing_ratio.graupel, output=state.radiation_field.graupel)
        self._copy(input=state.mixing_ratio.rain, output=state.radiation_field.rain)
        self._copy(input=state.mixing_ratio.snow, output=state.radiation_field.snow)

        if self._config.GFDL_MP3:
            # run the driver
            self._gfdl_microphysics_v3(state=state, locals=self._locals)

            # convert evap/subl/cloud/precipitation flux exports from (mm/day) to (kg m-2 s-1)
            self._convert_mm_per_day_to_kg_per_m2_per_s(state.non_anvil_large_scale.evaporation)
            self._convert_mm_per_day_to_kg_per_m2_per_s(state.non_anvil_large_scale.sublimation)
            self._convert_mm_per_day_to_kg_per_m2_per_s_interface(state.non_anvil_large_scale.ice_precip_flux)
            self._convert_mm_per_day_to_kg_per_m2_per_s_interface(state.non_anvil_large_scale.liquid_precip_flux)
            self._convert_mm_per_day_to_kg_per_m2_per_s_interface(state.non_anvil_large_scale.rain_precip_flux)
            self._convert_mm_per_day_to_kg_per_m2_per_s_interface(state.non_anvil_large_scale.snow_precip_flux)
            self._convert_mm_per_day_to_kg_per_m2_per_s_interface(state.non_anvil_large_scale.graupel_precip_flux)

            if self._config.DO_REF:
                option_not_impemented = True

        if self._config.REPORT_GFDL1M_NEGATIVES:
            ndsl_log.warning(
                "[GFDL_1M] Negative adjustment requested with debug output, but NDSL does not support in-stencil prints. "
                "The math has still been applied as expected, please use NDSL debug tools to see requested values"
            )
            self._negative_adjustment(
                graupel=state.radiation_field.graupel,
                ice=state.radiation_field.ice,
                liquid=state.radiation_field.liquid,
                rain=state.radiation_field.rain,
                snow=state.radiation_field.snow,
                vapor=state.radiation_field.vapor,
            )
        else:
            self._negative_adjustment(
                graupel=state.radiation_field.graupel,
                ice=state.radiation_field.ice,
                liquid=state.radiation_field.liquid,
                rain=state.radiation_field.rain,
                snow=state.radiation_field.snow,
                vapor=state.radiation_field.vapor,
            )

        # update cloud fraction, redistribute clouds, and fill vapor/precip states
        