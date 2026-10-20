from f90nml import Namelist
from ndsl import StencilFactory
from ndsl.stencils.testing.grid import Grid
from ndsl.stencils.testing.savepoint import DataLoader
from ndsl.stencils.testing.translate import TranslateFortranData2Py
from ndsl.utils import safe_assign_array
from ndsl.dsl.typing import Int

from pyMoist import GFDL1M, GFDL1MConfig, GFDL1MState
from pyMoist.microphysics.GFDL_1M.locals import GFDL1MLocals


class TranslateGFDL1M_Macrophysics(TranslateFortranData2Py):
    """Translate the combined GFDL1M-Macrophysics In/Out savepoints.

    Derived fields are loaded at the phase boundary, not recomputed by the
    full-driver setup. Output-only diagnostics and tendencies are not inputs.
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
                "NACTL",
                "NACTI",
                "AREA",
                "PLmb",
                "PLEmb",
                "ZL0",
                "MASS",
                "QST3",
                "EIS",
                "SRF_TYPE",
                "CNV_FRC",
                "KPBLSC",
                "PDF_A",
                "WSL",
                "WQT",
                "SL2",
                "QT2",
                "SLQT",
                "W3",
                "W2",
                "QT3",
                "SL3",
                "WTHV2",
                "WQL",
                "SHLW_PRC3",
                "SHLW_SNO3",
            )
        }
        self.out_vars = {
            **self.in_vars["data_vars"],
            **{
                name: {}
                for name in (
                    "EVAPC",
                    "SUBLC",
                    "PDFITERS",
                    "RHX",
                    "RHCRIT",
                    "DQVDT_macro",
                    "DQIDT_macro",
                    "DQLDT_macro",
                    "DQADT_macro",
                    "DQRDT_macro",
                    "DQSDT_macro",
                    "DQGDT_macro",
                    "DUDT_macro",
                    "DVDT_macro",
                    "DTDT_macro",
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
        state = GFDL1MState.zeros(self.quantity_factory)
        # Initialize the full runtime, but execute only the macrophysics phase.
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
            "NACTL": state.concentration.liquid,
            "NACTI": state.concentration.ice,
            "AREA": state.area,
            "PLmb": locals.p_mb,
            "PLEmb": locals.p_interface_mb,
            "ZL0": locals.layer_height_above_surface,
            "MASS": locals.mass,
            "QST3": locals.saturation_specific_humidity,
            "EIS": state.estimated_inversion_strength,
            "SRF_TYPE": state.surface_type,
            "CNV_FRC": state.convection_fraction,
            "KPBLSC": state.boundary_layer_level_for_uw_shallow_conv,
            "PDF_A": state.pdf_first_plume_fractional_area,
            "WSL": state.liquid_water_static_energy.flux,
            "WQT": state.total_water.flux,
            "SL2": state.liquid_water_static_energy.variance,
            "QT2": state.total_water.variance,
            "SLQT": state.covariance_liquid_water_static_energy_and_total_water_specific_humidity,
            "W3": state.vertical_motion.third_moment,
            "W2": state.vertical_motion.variance,
            "QT3": state.total_water.third_moment,
            "SL3": state.liquid_water_static_energy.third_moment,
            "WTHV2": state.buoyancy_flux,
            "WQL": state.liquid_water_flux,
            "SHLW_PRC3": state.shallow_convection_rain,
            "SHLW_SNO3": state.shallow_convection_snow,
            "EVAPC": state.cloud_liquid_evaporation,
            "SUBLC": state.cloud_ice_sublimation,
            "PDFITERS": state.hydrostatic_pdf_iterations,
            "RHX": state.relative_humidity_after_pdf,
            "RHCRIT": state.critical_relative_humidity_for_pdf,
            "DQVDT_macro": state.tendencies.dvapor_dt_macro,
            "DQIDT_macro": state.tendencies.dice_dt_macro,
            "DQLDT_macro": state.tendencies.dliquid_dt_macro,
            "DQADT_macro": state.tendencies.dcloud_fraction_dt_macro,
            "DQRDT_macro": state.tendencies.drain_dt_macro,
            "DQSDT_macro": state.tendencies.dsnow_dt_macro,
            "DQGDT_macro": state.tendencies.dgraupel_dt_macro,
            "DUDT_macro": state.tendencies.du_dt_macro,
            "DVDT_macro": state.tendencies.dv_dt_macro,
            "DTDT_macro": state.tendencies.dt_dt_macro,
        }
        for name in self.in_vars["data_vars"]:
            values = inputs[name] - 1 if name == "KPBLSC" else inputs[name]
            safe_assign_array(fields[name].field[:], values)

        # replicate lid_level initialization from GFDL1MSetup (needed for GFDL1MMacrophysics)
        locals.lid_level.field[:] = code._setup.lid_level

        # The savepoint precedes the tendency snapshots and shallow-precip
        # additions. Reproduce only that setup, preserving serialized locals.
        snapshots = {
            "DUDT_macro": state.u.field[:],
            "DVDT_macro": state.v.field[:],
            "DTDT_macro": state.t.field[:],
            "DQVDT_macro": state.mixing_ratio.vapor.field[:],
            "DQLDT_macro": state.mixing_ratio.convective_liquid.field[:] + state.mixing_ratio.large_scale_liquid.field[:],
            "DQIDT_macro": state.mixing_ratio.convective_ice.field[:] + state.mixing_ratio.large_scale_ice.field[:],
            "DQADT_macro": state.cloud_fraction.convective.field[:] + state.cloud_fraction.large_scale.field[:],
            "DQRDT_macro": state.mixing_ratio.rain.field[:],
            "DQSDT_macro": state.mixing_ratio.snow.field[:],
            "DQGDT_macro": state.mixing_ratio.graupel.field[:],
        }
        for name, values in snapshots.items():
            safe_assign_array(fields[name].field[:], values)
        state.mixing_ratio.rain.field[:] += state.shallow_convection_rain.field[:] * config.DT
        state.mixing_ratio.snow.field[:] += state.shallow_convection_snow.field[:] * config.DT

        # Skip full-driver setup, microphysics and diagnostics so the loaded
        # phase-boundary fields and tendency snapshots are preserved.
        code._run_macrophysics(state=state)

        outputs = {name: fields[name].field[:] for name in self.out_vars}
        outputs["KPBLSC"] = state.boundary_layer_level_for_uw_shallow_conv.field[:] + 1
        return outputs
