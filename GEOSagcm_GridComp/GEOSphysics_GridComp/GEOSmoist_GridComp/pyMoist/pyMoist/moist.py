from pyMoist.state import MoistState
from pyMoist.convection_tracers import ConvectionTracers
from ndsl.stencils import set_value, copy, add, divide
from pyMoist.saturation_tables import get_saturation_vapor_pressure_table, compute_saturation_specific_humidity, GlobalTable_saturation_tables
from ndsl import StencilFactory, QuantityFactory, NDSLRuntime
from ndsl.dsl.gt4py import computation, PARALLEL, interval, FORWARD, K, BACKWARD
from ndsl.dsl.typing import FloatField, FloatFieldIJ, FloatFieldK, Float, IntFieldIJ
import pyMoist.constants as constants
from ndsl.constants import I_DIM, J_DIM, K_DIM
from pyMoist.shared.incloud_processes import fix_mixing_ratio, buoyancy_1, Buoyancy2
from pyMoist.locals import MoistLocals
from pyMoist.config import MoistConfig


def get_alarm():
    pass


def turn_alarm_off():
    pass


def set_surface_type(
    land_fraction: FloatFieldIJ,
    land_ice_fraction: FloatFieldIJ,
    ice_covered_fraction_of_tile: FloatFieldIJ,
    snow_mass: FloatFieldIJ,
    surface_type: FloatField,
):
    with computation(FORWARD), interval(0, 1):
        if land_ice_fraction > 0.5 or ice_covered_fraction_of_tile > 0.5:
            surface_type = 3.0  # ice
        elif snow_mass > 0.1 and snow_mass != constants.MAPL_UNDEF:
            surface_type = 2.0  # snow
        elif land_fraction > 0.1:
            surface_type = 1.0  # land
        else:
            surface_type = 0.0  # ocean


def compute_derived_state(
    mass: FloatField,
    p_interface: FloatField,
    p_interface_mb: FloatField,
    p_mb: FloatField,
    p_kappa_interface: FloatField,
    p_kappa: FloatField,
    z_interface: FloatField,
    edge_height_above_surface: FloatField,
    layer_height_above_surface: FloatField,
    layer_thickness: FloatField,
    t: FloatField,
    ese: GlobalTable_saturation_tables,
    esx: GlobalTable_saturation_tables,
    saturation_specific_humidity: FloatField,
    dsaturation_specific_humidity: FloatField,
):
    from __externals__ import k_end

    mass = (p_interface[0, 0, 1] - p_interface) / constants.MAPL_GRAV

    p_interface_mb = p_interface * 0.01
    p_kappa_interface = (p_interface / constants.MAPL_P00) ** constants.MAPL_KAPPA
    p_mb = 0.5 * (p_interface_mb + p_interface_mb[0, 0, 1])
    p_kappa = (100.0 * p_mb / constants.MAPL_P00) ** constants.MAPL_KAPPA

    edge_height_above_surface = z_interface - z_interface.at(K=k_end)
    layer_height_above_surface = 0.5 * (edge_height_above_surface + edge_height_above_surface[0, 0, 1])
    layer_thickness = edge_height_above_surface - edge_height_above_surface[0, 0, 1]

    saturation_specific_humidity, dsaturation_specific_humidity = compute_saturation_specific_humidity(t=t, p=p_mb, ese=ese, esx=esx)


def find_highest_level_interface(
    scalar_diffusivity: FloatField,
    value: Float,
    highest_level: IntFieldIJ,
):
    """Find the highest level where a field is greater than a specified value.

    This stencil must be build with K_INTERFACE_DIM to function properly.

    Args:
        field (FloatField): field to be analyzed
        value (Float): threshold for comparison
        highest_level (IntFieldIJ): highest level where field is greater than value
    """
    with computation(FORWARD), interval(0, 1):
        highest_level = -1

    with computation(FORWARD), interval(...):
        if highest_level == -1 and scalar_diffusivity > value:
            highest_level = K


def find_lowest_level_interface(
    scalar_diffusivity: FloatField,
    value: Float,
    lowest_level: IntFieldIJ,
):
    """Find the lowest level where a field is greater than a specified value.

    This stencil must be build with K_INTERFACE_DIM to function properly.

    Args:
        field (FloatField): field to be analyzed
        value (Float): threshold for comparison
        lowest_level (IntFieldIJ): lowest level where field is greater than value
    """
    with computation(FORWARD), interval(0, 1):
        lowest_level = -1

    with computation(BACKWARD), interval(...):
        if lowest_level == -1 and scalar_diffusivity > value:
            lowest_level = K


def export_cbl_level(
    cbl_level_before_moist: IntFieldIJ,
    pbl_level: IntFieldIJ,
    reference_pressure: FloatFieldK,
    p_min_cbl: Float,
):
    from __externals__ import k_end

    with computation(FORWARD), interval(0, 1):
        min_cbl_level: IntFieldIJ = 0

    with computation(FORWARD), interval(...):
        if reference_pressure < p_min_cbl:
            min_cbl_level += 1

    with computation(FORWARD), interval(0, 1):
        if pbl_level != 0:
            cbl_level_before_moist = max(min(pbl_level + 1, k_end - 1), 1)
        else:
            cbl_level_before_moist = k_end - 1
        cbl_level_before_moist = max(cbl_level_before_moist, min_cbl_level)


def compute_convection_fraction(convection_fraction: FloatFieldIJ, cape: FloatFieldIJ):
    from __externals__ import CONVECTION_FRACTION_MAX, CONVECTION_FRACTION_MIN, CONVECTION_FRACTION_EXP

    with computation(FORWARD), interval(0, 1):
        convection_fraction = 0.0
        if CONVECTION_FRACTION_MAX > CONVECTION_FRACTION_MIN:
            if cape != constants.MAPL_UNDEF:
                convection_fraction = max(1.0e-6, min(1.0, (cape - convection_fraction) / (CONVECTION_FRACTION_MAX - CONVECTION_FRACTION_MIN)))

        if CONVECTION_FRACTION_EXP != 1.0:
            convection_fraction = convection_fraction**CONVECTION_FRACTION_EXP


def initialize_convection_tracers():
    pass


def export_concentrations():
    pass


def update_cloud_fraction():
    pass


def get_saturation_specific_humidity():
    pass


def export_relative_humidity_wrt_ice():
    pass


def export_output_saturation_ratio():
    pass


def export_relative_humidity_wrt_liquid():
    pass


def rain_out_excessive_rh():
    pass


def cloud_cleanup():
    pass


class Moist(NDSLRuntime):
    def __init__(self, stencil_factory: StencilFactory, quantity_factory: QuantityFactory, config: MoistConfig):
        super().__init__(stencil_factory)

        self._config = config

        # initialize saturation vapor pressure tables
        self._saturation_tables = get_saturation_vapor_pressure_table(stencil_factory.backend)

        # initialize locals
        self._locals = MoistLocals.make_locals(quantity_factory)

        # construct stencils
        self._set_value = stencil_factory.from_dims_halo(func=set_value, compute_dims=[I_DIM, J_DIM, K_DIM])

        self._set_surface_type = stencil_factory.from_dims_halo(func=set_surface_type, compute_dims=[I_DIM, J_DIM, K_DIM])

        self._compute_derived_state = stencil_factory.from_dims_halo(func=compute_derived_state, compute_dims=[I_DIM, J_DIM, K_DIM])

        self._fix_mixing_ratio = stencil_factory.from_dims_halo(func=fix_mixing_ratio, compute_dims=[I_DIM, J_DIM, K_DIM])

        self._copy = stencil_factory.from_dims_halo(func=copy, compute_dims=[I_DIM, J_DIM, K_DIM])

        self._add = stencil_factory.from_dims_halo(func=add, compute_dims=[I_DIM, J_DIM, K_DIM])

        self._divide = stencil_factory.from_dims_halo(func=divide, compute_dims=[I_DIM, J_DIM, K_DIM])

        self._find_highest_level_interface = stencil_factory.from_dims_halo(func=find_highest_level_interface, compute_dims=[I_DIM, J_DIM, K_DIM])

        self._find_lowest_level_interface = stencil_factory.from_dims_halo(func=find_lowest_level_interface, compute_dims=[I_DIM, J_DIM, K_DIM])

        self._export_cbl_level = stencil_factory.from_dims_halo(func=export_cbl_level, compute_dims=[I_DIM, J_DIM, K_DIM])

        self._buoyancy_2 = Buoyancy2(stencil_factory=stencil_factory, quantity_factory=quantity_factory)

        self._buoyancy_1 = stencil_factory.from_dims_halo(func=buoyancy_1, compute_dims=[I_DIM, J_DIM, K_DIM])

        self._compute_convection_fraction = stencil_factory.from_dims_halo(
            func=compute_convection_fraction,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={
                "CONVECTION_FRACTION_MAX": config.CONVECTION_FRACTION_MAX,
                "CONVECTION_FRACTION_MIN": config.CONVECTION_FRACTION_MIN,
                "CONVECTION_FRACTION_EXP": config.CONVECTION_FRACTION_EXP,
            },
        )

    def __call__(self, state: MoistState, convection_tracers: ConvectionTracers):
        # Get alarm - is on (ringing) when pulled
        ALARM_IS_RINGING = get_alarm()

        if ALARM_IS_RINGING:
            ALARM_IS_RINGING = turn_alarm_off()

            # reset mass fluxes
            self._set_value(state.convective_diagnostics.total_cumulative_mass_flux, 0.0)
            self._set_value(state.convective_diagnostics.total_detraining_mass_flux, 0.0)

            # update surface_type for ice_fraction
            self._set_surface_type(
                land_fraction=state.surface_conditions.land_fraction,
                land_ice_fraction=state.surface_conditions.land_ice_fraction,
                ice_covered_fraction_of_tile=state.surface_conditions.ice_covered_fraction_of_tile,
                snow_mass=state.surface_conditions.snow_mass,
                surface_type=state.surface_conditions.surface_type,
            )

            # compute derived states
            self._compute_derived_state(
                mass=self._locals.mass,
                p_interface=state.atmospheric_state.p_interface,
                p_interface_mb=self._locals.p_interface_mb,
                p_mb=self._locals.p_mb,
                p_kappa_interface=self._locals.p_kappa_interface,
                p_kappa=self._locals.p_kappa,
                z_interface=state.atmospheric_state.z_interface,
                edge_height_above_surface=self._locals.edge_height_above_surface,
                layer_height_above_surface=self._locals.layer_height_above_surface,
                layer_thickness=self._locals.layer_thickness,
                t=state.atmospheric_state.t,
                ese=self._saturation_tables.ese,
                esx=self._saturation_tables.esx,
                saturation_specific_humidity=self._locals.saturation_specific_humidity,
                dsaturation_specific_humidity=self._locals.dsaturation_specific_humidity,
            )

            # fill negative specific humidity with zero
            self._fix_mixing_ratio(
                mixing_ratio=state.cloud_condensates.specific_humidity,
                mass=self._locals.mass,
                adjustment=self._locals.temporary_2d,
            )
            if state.diagnostics.negative_vapor_adjustment_start is not None:
                self._copy(self._locals.temporary_2d, state.diagnostics.negative_vapor_adjustment_start)

            # save a copy of the state at the start of the computation
            # all fields are unmodified within Moist at this point
            # one save requires the derived state field p_kappa,
            # forcing this entire block slightly lower than otherwise necessary
            self._copy(state.atmospheric_state.u, state.state_at_input.u)
            self._copy(state.atmospheric_state.v, state.state_at_input.v)
            if state.state_at_input.specific_humidity is not None:
                self._copy(state.cloud_condensates.specific_humidity, state.state_at_input.specific_humidity)
            if state.state_at_input.large_scale_cloud_fraction is not None:
                self._copy(state.cloud_condensates.large_scale_cloud_fraction, state.state_at_input.large_scale_cloud_fraction)
            if state.state_at_input.large_scale_liquid is not None:
                self._copy(state.cloud_condensates.large_scale_liquid, state.state_at_input.large_scale_liquid)
            if state.state_at_input.large_scale_ice is not None:
                self._copy(state.cloud_condensates.large_scale_ice, state.state_at_input.large_scale_ice)
            if state.state_at_input.convective_cloud_fraction is not None:
                self._copy(state.cloud_condensates.convective_cloud_fraction, state.state_at_input.convective_cloud_fraction)
            if state.state_at_input.convective_ice is not None:
                self._copy(state.cloud_condensates.convective_ice, state.state_at_input.convective_ice)
            if state.state_at_input.convective_condensates is not None:
                self._add(state.cloud_condensates.convective_ice, state.cloud_condensates.convective_liquid, state.state_at_input.convective_liquid)
            if state.state_at_input.large_scale_condensates is not None:
                self._add(state.cloud_condensates.large_scale_ice, state.cloud_condensates.large_scale_liquid, state.state_at_input.large_scale_condensates)
            if state.state_at_input.pt is not None:
                self._divide(state.atmospheric_state.t, self._locals.p_kappa, state.state_at_input.pt)
            if state.state_at_input.specific_humidity is not None:
                self._copy(state.cloud_condensates.specific_humidity, state.state_at_input.specific_humidity)
            if state.state_at_input.t_surface is not None:
                self._copy(state.surface_conditions.t_surface, state.state_at_input.t_surface)

            # save scalar diffusivity levels
            if state.diagnostics.highest_level_of_scalar_diffusivity_gt_2 is not None:
                self._find_highest_level_interface(
                    scalar_diffusivity=state.atmospheric_state.scalar_diffusivity_interface,
                    value=Float(2.0),
                    highest_level=state.diagnostics.highest_level_of_scalar_diffusivity_gt_2,
                )

            if state.diagnostics.lowest_level_of_scalar_diffusivity_gt_2 is not None:
                self._find_lowest_level_interface(
                    scalar_diffusivity=state.atmospheric_state.scalar_diffusivity_interface,
                    value=Float(2.0),
                    lowest_level=state.diagnostics.lowest_level_of_scalar_diffusivity_gt_2,
                )

            if state.levels.cbl_level_before_moist is not None:
                self._export_cbl_level(
                    cbl_level_before_moist=state.levels.cbl_level_before_moist,
                    pbl_level=state.levels.pbl_level,
                    reference_pressure=state.atmospheric_state.reference_pressure,
                    p_min_cbl=P_MIN_CBL,
                )

            # compute buoyancy and related parameters
            self._buoyancy_2(
                t=state.atmospheric_state.t,
                specific_humidity=state.cloud_condensates.specific_humidity,
                p_interface_mb=self._locals.p_interface_mb,
                p_mb=self._locals.p_mb,
                layer_height_above_surface=self._locals.layer_height_above_surface,
                layer_thickness=self._locals.layer_thickness,
                saturation_specific_humidity=self._locals.saturation_specific_humidity,
                dsaturation_specific_humidity=self._locals.dsaturation_specific_humidity,
                buoyancy_surface_parcel=state.convective_diagnostics.buoyancy_surface_parcel,
                sbcape=state.convective_diagnostics.sbcape,
                mlcape=state.convective_diagnostics.mlcape,
                mucape=state.convective_diagnostics.mucape,
                sbcin=state.convective_diagnostics.sbcin,
                mlcin=state.convective_diagnostics.mlcin,
                mucin=state.convective_diagnostics.mucin,
                lfc=state.convective_diagnostics.lfc,
                lnb=state.convective_diagnostics.lnb,
            )

            self._buoyancy_1(
                t=state.atmospheric_state.t,
                layer_height_above_surface=self._locals.layer_height_above_surface,
                layer_thickness=self._locals.layer_thickness,
                specific_humidity=state.cloud_condensates.specific_humidity,
                saturation_specific_humidity=self._locals.saturation_specific_humidity,
                dsaturation_specific_humidity=self._locals.dsaturation_specific_humidity,
                buoyancy=state.convective_diagnostics.buoyancy_surface_parcel,
                cape=state.convective_diagnostics.cape_surface_parcel,
                cin=state.convective_diagnostics.cin_surface_parcel,
            )

            # initialize diagnosed convective fraction
            self._compute_convection_fraction(convection_fraction=state.convective_diagnostics.convection_fraction, cape=state.convective_diagnostics.cape_surface_parcel)

            # extract convective tracers from the TR bundle
            initialize_convection_tracers()

            # get aerosol activation properties
            if self._config.USE_AEROSOL_NN:
                do_aerosol_activateion = True
            else:
                do_aerosol_activateion = False

            # export concentrations
            export_concentrations()

            # run convection and microphysics
            if SH_MD_DEEP:
                if SHALLOW_OPTION == "UW":
                    run_UW = True
                if CONVPAR_OPTION == "RAS":
                    run_RAS = True
                if CONVPAR_OPTION == "GF":
                    run_GF = True
            else:
                if CONVPAR_OPTION == "RAS":
                    run_RAS = True
                if CONVPAR_OPTION == "GF":
                    run_GF = True
                if SHALLOW_OPTION == "UW":
                    run_UW = True

            if CLDMICR_OPTION == "BACM_1M":
                raise ValueError(f"{CLDMICR_OPTION} microphysics not implemented. Please choose a different option.")
            if CLDMICR_OPTION == "GFDL_1M":
                run_GFDL1M = True
            if CLDMICR_OPTION == "THOM_1M":
                raise ValueError(f"{CLDMICR_OPTION} microphysics not implemented. Please choose a different option.")
            if CLDMICR_OPTION == "MGB2_2M":
                raise ValueError(f"{CLDMICR_OPTION} microphysics not implemented. Please choose a different option.")

            # export cloud fractions
            if state.cloud_condensates.large_scale_ice_cloud_fraction is not None:
                update_cloud_fraction()

            if state.cloud_condensates.large_scale_liquid_cloud_fraction is not None:
                update_cloud_fraction()

            # rain-out and relative humidity where RH > 110%
            copy(state.atmospheric_state.t, state.tendencies.dt_dt_from_rh_cleanup)
            copy(state.cloud_condensates.specific_humidity, state.tendencies.dspecific_humidity_dt_from_rh_cleanup)

            # compute saturation specific humidity values for current P and T
            get_saturation_specific_humidity()

            if state.cloud_condensates.relative_humidity_wrt_ice is not None:
                export_relative_humidity_wrt_ice()

            if state.state_at_output.saturation_ratio is not None:
                export_output_saturation_ratio()

            if CLDMICR_OPTION == "MGB2_2M":
                raise ValueError(f"{CLDMICR_OPTION} microphysics not implemented. Please choose a different option.")
            else:
                get_saturation_specific_humidity()

            if state.cloud_condensates.relative_humidity_wrt_liquid is not None:
                export_relative_humidity_wrt_liquid()

            # rain out excessive RH
            rain_out_excessive_rh()

            # cleanup any negative specific_humidity/QC/CF
            cloud_cleanup()
            if state.diagnostics.negative_vapor_adjustment_end is not None:
                export_negative_vapor_adjustment_end()
