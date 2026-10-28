import numpy as np

from f90nml import Namelist
from ndsl import StencilFactory
from ndsl.dsl.typing import Int
from ndsl.stencils.testing.grid import Grid
from ndsl.stencils.testing.savepoint import DataLoader
from ndsl.stencils.testing.translate import TranslateFortranData2Py
from ndsl.utils import safe_assign_array

from pyMoist import GFDL1M, GFDL1MConfig, GFDL1MState
from pyMoist.microphysics.GFDL_1M.microphysics.locals import GFDLMPV3Locals
from pyMoist.microphysics.GFDL_1M.microphysics.mp_full.mp_full_locals import MPFullLocals


class TranslateGFDLMPV3_MPFull(TranslateFortranData2Py):
    """Check the complete MPFull loop against GFDLMPV3_MPFull-In/Out.

    These savepoints contain one column batch per rank/call, without the
    additional time-buffer axis used by the individual phase savepoints.
    MPFull itself executes all configured microphysics substeps.
    """

    def __init__(self, grid: Grid, namelist: Namelist, stencil_factory: StencilFactory):
        super().__init__(grid, stencil_factory)
        self.quantity_factory = grid.quantity_factory
        self.in_vars["data_vars"] = {
            f"{name}_data_buffered": {}
            for name in (
                "T",
                "QA",
                "QV",
                "QL",
                "QR",
                "QI",
                "QS",
                "QG",
                "DP",
                "DZ",
                "U",
                "V",
                "W",
                "DEN",
                "DENFAC",
                "CCN",
                "CIN",
                "H_VAR",
                "DTE",
                "WATER",
                "RAIN",
                "ICE",
                "SNOW",
                "GRAUPEL",
                "REVAP",
                "RSUBL",
                "PREFLUXW",
                "PREFLUXR",
                "PREFLUXI",
                "PREFLUXS",
                "PREFLUXG",
                "MPPCW",
                "MPPEW",
                "MPPE1",
                "MPPER",
                "MPPDI",
                "MPPD1",
                "MPPDS",
                "MPPDG",
                "MPPSI",
                "MPPS1",
                "MPPSS",
                "MPPSG",
                "MPPFW",
                "MPPFR",
                "MPPMI",
                "MPPMS",
                "MPPMG",
                "MPPM1",
                "MPPM2",
                "MPPM3",
                "MPPAR",
                "MPPAS",
                "MPPAG",
                "MPPRS",
                "MPPRG",
                "MPPXR",
                "MPPXS",
                "MPPXG",
                "CPAUT",
                "FAC_RC",
                "SRF_TYPE",
                "CNV_FRACTION",
                "ONEMSIG",
            )
        }
        self.out_vars = self.in_vars["data_vars"]

    def extra_data_load(self, data_loader: DataLoader):
        self.constants = data_loader.load("GFDL1M-Config")
        self.cwd = str(data_loader._data_path)

    def compute(self, inputs):
        constants = self.constants.copy()
        # Convert the Fortran lid once; retain the full physical grid.
        lid = constants["GFDL_MP_KLID"]
        constants["GFDL_MP_KLID"] = lid - Int(1) if lid > 0 else Int(0)
        config = GFDL1MConfig(CWD=self.cwd, **constants)
        if not config.GFDL_MP3:
            raise NotImplementedError("The Python GFDL1M microphysics translator requires GFDL_MP3.")

        state = GFDL1MState.zeros(self.quantity_factory)
        code = GFDL1M(
            stencil_factory=self.stencil_factory,
            quantity_factory=self.quantity_factory,
            config=config,
        )
        driver = code._gfdl_microphysics_v3._driver
        # Externally accessible test storage, rather than runtime-only locals.
        local = GFDLMPV3Locals.make_as_state(self.quantity_factory)
        driver._gfdl_mp_v3_locals = local
        driver._mp_full._mp_full_locals = MPFullLocals.make_as_state(self.quantity_factory)

        fields = {
            "T": local.t,
            "QA": local.cloud_fraction,
            "QV": local.mixing_ratio.vapor,
            "QL": local.mixing_ratio.liquid,
            "QR": local.mixing_ratio.rain,
            "QI": local.mixing_ratio.ice,
            "QS": local.mixing_ratio.snow,
            "QG": local.mixing_ratio.graupel,
            "DP": local.dry_dp,
            "DZ": local.dz,
            "U": local.u,
            "V": local.v,
            "W": local.w,
            "DEN": local.density,
            "DENFAC": local.density_factor,
            "CCN": local.ccn,
            "CIN": local.cin,
            "H_VAR": local.h_var,
            "DTE": local.total_energy.delta,
            "WATER": state.precipitation_at_surface.water,
            "RAIN": state.precipitation_at_surface.rain,
            "ICE": state.precipitation_at_surface.ice,
            "SNOW": state.precipitation_at_surface.snow,
            "GRAUPEL": state.precipitation_at_surface.graupel,
            "REVAP": state.non_anvil_large_scale.evaporation,
            "RSUBL": state.non_anvil_large_scale.sublimation,
            "PREFLUXW": state.non_anvil_large_scale.liquid_precip_flux,
            "PREFLUXR": state.non_anvil_large_scale.rain_precip_flux,
            "PREFLUXI": state.non_anvil_large_scale.ice_precip_flux,
            "PREFLUXS": state.non_anvil_large_scale.snow_precip_flux,
            "PREFLUXG": state.non_anvil_large_scale.graupel_precip_flux,
            "CPAUT": local.cpaut,
            "FAC_RC": local.factor_rc,
            "SRF_TYPE": local.surface_type,
            "CNV_FRACTION": local.convection_fraction,
            "ONEMSIG": local.one_minus_sigma,
        }
        for name in self.in_vars["data_vars"]:
            short_name = name.removesuffix("_data_buffered")
            if short_name.startswith("MPP"):
                fields[short_name] = getattr(local, short_name.lower())
        fields = {f"{name}_data_buffered": field for name, field in fields.items()}

        # Check every field before loading. Horizontal columns are Fortran-ordered.
        for name in self.in_vars["data_vars"]:
            field_shape = fields[name].field.shape
            shape = inputs[name].shape
            if (
                len(field_shape) not in (2, 3)
                or len(shape) != len(field_shape) - 1
                or shape[0] != field_shape[0] * field_shape[1]
                or (len(shape) == 2 and not 0 < shape[1] <= field_shape[2])
            ):
                raise ValueError(f"{name}: buffer shape {shape} is incompatible with field shape {field_shape}")

        for name in self.in_vars["data_vars"]:
            destination = fields[name].field[:]
            source = inputs[name]
            ni, nj = destination.shape[:2]
            destination[:] = 0
            if destination.ndim == 2:
                safe_assign_array(destination, source.reshape((ni, nj), order="F"))
            else:
                nlevels = source.shape[-1]
                # Cell fields end at the bottom cell; PREFLUX fields end at the
                # bottom interface. MPFull adds cell k's flux at interface k+1.
                safe_assign_array(
                    destination[:, :, -nlevels:],
                    source.reshape((ni, nj, nlevels), order="F"),
                )

        # Do not rerun driver preparation: all phase-boundary inputs are saved.
        driver._mp_full(state, local)

        outputs = {name: np.empty_like(inputs[name]) for name in self.out_vars}
        for name, output in outputs.items():
            field = fields[name].field[:]
            saved_levels = field if field.ndim == 2 else field[:, :, -output.shape[-1] :]
            safe_assign_array(output, saved_levels.reshape(output.shape, order="F"))
        return outputs
