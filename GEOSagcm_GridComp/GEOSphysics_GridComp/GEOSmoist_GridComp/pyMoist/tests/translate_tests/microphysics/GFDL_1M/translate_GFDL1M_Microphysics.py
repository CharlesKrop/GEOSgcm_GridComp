from f90nml import Namelist
from ndsl import StencilFactory
from ndsl.stencils.testing.grid import Grid
from ndsl.stencils.testing.savepoint import DataLoader
from ndsl.stencils.testing.translate import TranslateFortranData2Py
from ndsl.utils import safe_assign_array
from ndsl.dsl.typing import Int

from pyMoist import GFDL1M, GFDL1MConfig, GFDL1MState
from pyMoist.microphysics.GFDL_1M.locals import GFDL1MLocals


class TranslateGFDL1M_Microphysics(TranslateFortranData2Py):
    """Translate the complete GFDL1M-Microphysics In/Out section.

    Includes driver preparation, precipitation/radiation post-processing and
    tendencies, but neither macrophysics nor final full-driver diagnostics.
    GFDL MP3 initialization reads input.nml from config.CWD, as in TranslateGFDL1M.
    """

    def __init__(self, grid: Grid, namelist: Namelist, stencil_factory: StencilFactory):
        super().__init__(grid, stencil_factory)
        self.quantity_factory = grid.quantity_factory
        self.in_vars["data_vars"] = {
            name: {}
            for name in (
                "Q",
                "QRAIN",
                "QSNOW",
                "QGRAUPEL",
                "QLLS",
                "QLCN",
                "QILS",
                "QICN",
                "CLLS",
                "CLCN",
                "T",
                "U",
                "V",
                "W",
                "NACTL",
                "NACTI",
                "DZET",
                "DP",
                "PLmb",
                "ZLE0",
                "MASS",
                "RHCRIT",
                "PHIS",
                "CNV_FRC",
                "EIS",
                "AREA",
                "SRF_TYPE",
                "FRLAND",
                "PRCP_WATER",
            )
        }
        self.out_vars = {
            **self.in_vars["data_vars"],
            **{
                name: {}
                for name in (
                    "RAD_CF",
                    "RAD_QV",
                    "RAD_QL",
                    "RAD_QI",
                    "RAD_QR",
                    "RAD_QS",
                    "RAD_QG",
                    "CLDREFFL",
                    "CLDREFFI",
                    "PRCP_RAIN",
                    "PRCP_SNOW",
                    "PRCP_ICE",
                    "PRCP_GRAUPEL",
                    "LS_PRCP",
                    "LS_SNR",
                    "ICE",
                    "FRZR",
                    "REV_LS",
                    "RSU_LS",
                    "PFL_AN",
                    "PFL_LS",
                    "PFI_AN",
                    "PFI_LS",
                    "PFR_LS",
                    "PFS_LS",
                    "PFG_LS",
                    "VFALL_ICE",
                    "VFALL_SNOW",
                    "VFALL_GRAUPEL",
                    "VFALL_RAIN",
                    "DQVDT_micro",
                    "DQIDT_micro",
                    "DQLDT_micro",
                    "DQADT_micro",
                    "DQRDT_micro",
                    "DQSDT_micro",
                    "DQGDT_micro",
                    "DUDT_micro",
                    "DVDT_micro",
                    "DTDT_micro",
                )
            },
        }

    def extra_data_load(self, data_loader: DataLoader):
        self.constants = data_loader.load("GFDL1M-Config")

    def compute(self, inputs):
        # Serialized Fortran levels are one-based; Python configuration is zero-based.
        constants = self.constants.copy()
        # shift GFDL_MP_KLID to be zero-based for Python
        lid = constants["GFDL_MP_KLID"]
        constants["GFDL_MP_KLID"] = lid - Int(1) if lid > 0 else Int(0)
        config = GFDL1MConfig(
            CWD="/Users/ckropiew/GEOS/v11.10.1/run/experiments/GCM_C24_L72_NX1_NY1/scratch",
            **constants,
        )
        if not config.GFDL_MP3:
            raise NotImplementedError("The Python GFDL1M microphysics translator requires GFDL_MP3.")
        state = GFDL1MState.zeros(self.quantity_factory)
        # Initialize the full runtime, but execute only _run_microphysics
        # after loading the savepoint.
        code = GFDL1M(
            stencil_factory=self.stencil_factory,
            quantity_factory=self.quantity_factory,
            config=config,
        )
        # Serialized phase-boundary fields are test inputs/outputs, not
        # transient runtime scratch. NDSL's testing API makes them Quantities
        # so the translator can load and inspect them outside the runtime.
        locals = GFDL1MLocals.make_as_state(self.quantity_factory)
        code._locals = locals
        fields = {
            "Q": state.mixing_ratio.vapor,
            "QRAIN": state.mixing_ratio.rain,
            "QSNOW": state.mixing_ratio.snow,
            "QGRAUPEL": state.mixing_ratio.graupel,
            "QLLS": state.mixing_ratio.large_scale_liquid,
            "QLCN": state.mixing_ratio.convective_liquid,
            "QILS": state.mixing_ratio.large_scale_ice,
            "QICN": state.mixing_ratio.convective_ice,
            "CLLS": state.cloud_fraction.large_scale,
            "CLCN": state.cloud_fraction.convective,
            "T": state.t,
            "U": state.u,
            "V": state.v,
            "W": state.vertical_motion.velocity,
            "NACTL": state.concentration.liquid,
            "NACTI": state.concentration.ice,
            "DZET": locals.layer_thickness,
            "DP": locals.dp,
            "PLmb": locals.p_mb,
            "ZLE0": locals.edge_height_above_surface,
            "MASS": locals.mass,
            "RHCRIT": state.critical_relative_humidity_for_pdf,
            "PHIS": state.surface_geopotential_height,
            "CNV_FRC": state.convection_fraction,
            "EIS": state.estimated_inversion_strength,
            "AREA": state.area,
            "SRF_TYPE": state.surface_type,
            "FRLAND": state.land_fraction,
            "PRCP_WATER": state.precipitation_at_surface.water,
            "RAD_CF": state.radiation_field.cloud_fraction,
            "RAD_QV": state.radiation_field.vapor,
            "RAD_QL": state.radiation_field.liquid,
            "RAD_QI": state.radiation_field.ice,
            "RAD_QR": state.radiation_field.rain,
            "RAD_QS": state.radiation_field.snow,
            "RAD_QG": state.radiation_field.graupel,
            "CLDREFFL": state.cloud_particle_effective_radius.liquid,
            "CLDREFFI": state.cloud_particle_effective_radius.ice,
            "PRCP_RAIN": state.precipitation_at_surface.rain,
            "PRCP_SNOW": state.precipitation_at_surface.snow,
            "PRCP_ICE": state.precipitation_at_surface.ice,
            "PRCP_GRAUPEL": state.precipitation_at_surface.graupel,
            "LS_PRCP": state.non_anvil_large_scale.precip,
            "LS_SNR": state.non_anvil_large_scale.snow,
            "ICE": state.icefall,
            "FRZR": state.freezing_rainfall,
            "REV_LS": state.non_anvil_large_scale.evaporation,
            "RSU_LS": state.non_anvil_large_scale.sublimation,
            "PFL_AN": state.anvil.liquid_precip_flux,
            "PFL_LS": state.non_anvil_large_scale.liquid_precip_flux,
            "PFI_AN": state.anvil.ice_precip_flux,
            "PFI_LS": state.non_anvil_large_scale.ice_precip_flux,
            "PFR_LS": state.non_anvil_large_scale.rain_precip_flux,
            "PFS_LS": state.non_anvil_large_scale.snow_precip_flux,
            "PFG_LS": state.non_anvil_large_scale.graupel_precip_flux,
            "VFALL_ICE": state.fall_speed.ice,
            "VFALL_SNOW": state.fall_speed.snow,
            "VFALL_GRAUPEL": state.fall_speed.graupel,
            "VFALL_RAIN": state.fall_speed.rain,
            "DQVDT_micro": state.tendencies.dvapor_dt_micro,
            "DQIDT_micro": state.tendencies.dice_dt_micro,
            "DQLDT_micro": state.tendencies.dliquid_dt_micro,
            "DQADT_micro": state.tendencies.dcloud_fraction_dt_micro,
            "DQRDT_micro": state.tendencies.drain_dt_micro,
            "DQSDT_micro": state.tendencies.dsnow_dt_micro,
            "DQGDT_micro": state.tendencies.dgraupel_dt_micro,
            "DUDT_micro": state.tendencies.du_dt_micro,
            "DVDT_micro": state.tendencies.dv_dt_micro,
            "DTDT_micro": state.tendencies.dt_dt_micro,
        }
        for name in self.in_vars["data_vars"]:
            safe_assign_array(fields[name].field[:], inputs[name])
        locals.lid_level.field[:] = code._setup.lid_level

        # Do not call code(state): it would rerun setup and macrophysics and
        # overwrite the serialized phase-boundary inputs.
        code._run_microphysics(state=state)

        return {name: fields[name].field[:] for name in self.out_vars}
