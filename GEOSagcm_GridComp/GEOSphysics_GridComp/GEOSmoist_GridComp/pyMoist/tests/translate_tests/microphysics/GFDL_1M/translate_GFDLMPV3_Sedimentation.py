import numpy as np

from f90nml import Namelist
from ndsl import StencilFactory
from ndsl.stencils.testing.grid import Grid
from ndsl.stencils.testing.savepoint import DataLoader
from ndsl.stencils.testing.translate import TranslateFortranData2Py
from ndsl.utils import safe_assign_array
from ndsl.dsl.typing import Int

from pyMoist import GFDL1M, GFDL1MConfig, GFDL1MState
from pyMoist.microphysics.GFDL_1M.gfdl_mp_v3.mp_full.mp_full_locals import MPFullLocals
from pyMoist.microphysics.GFDL_1M.gfdl_mp_v3.locals import GFDLMPV3Locals
from pyMoist.microphysics.GFDL_1M.gfdl_mp_v3.constants import GRAV


class TranslateGFDLMPV3_Sedimentation(TranslateFortranData2Py):
    """Test the implementation of the GFDLMPV3 Sedimentation routine.

    Loads the serialized Fortran fields, maps them to the GFDL1M runtime state,
    runs the Sedimentation routine, and returns the resulting fields.
    """

    def __init__(self, grid: Grid, namelist: Namelist, stencil_factory: StencilFactory):
        super().__init__(grid, stencil_factory)
        self.quantity_factory = grid.quantity_factory
        self.in_vars["data_vars"] = {
            f"{name}_data_buffered_in_n_loop": {}
            for name in (
                "T",
                "QV",
                "QL",
                "QR",
                "QI",
                "QS",
                "QG",
                "DZ",
                "DP",
                "U",
                "V",
                "W",
                "DEN",
                "DENFAC",
                "DTE",
                "MPPM1",
                "MPPM2",
                "MPPM3",
                "CNV_FRACTION",
            )
        }
        self.out_vars = self.in_vars["data_vars"]

    def extra_data_load(self, data_loader: DataLoader):
        self.constants = data_loader.load("GFDL1M-Config")
        if "TURNRHCRIT" in self.constants:
            self.constants["TURNRHCRIT_SFC"] = self.constants.pop("TURNRHCRIT")
            self.constants["TURNRHCRIT_TOP"] = 3250.0
        self.constants.setdefault("MIN_RH_CRIT", 0.725)
        self.cwd = str(data_loader._data_path)
        lid = self.constants["GFDL_MP_KLID"]
        self.level_start = int(lid - Int(1)) if lid > 0 else 0

    def compute(self, inputs):
        # Serialized Fortran levels are one-based; Python configuration is zero-based.
        constants = self.constants.copy()
        # shift GFDL_MP_KLID to be zero-based for Python
        lid = constants["GFDL_MP_KLID"]
        constants["GFDL_MP_KLID"] = lid - Int(1) if lid > 0 else Int(0)
        # Retain the full namelist grid; the runtime restricts the stencil to the lid.
        config = GFDL1MConfig(
            CWD=self.cwd,
            **constants,
        )
        if not config.GFDL_MP3:
            raise NotImplementedError("The Python GFDL1M microphysics translator requires GFDL_MP3.")
        state = GFDL1MState.zeros(self.quantity_factory)
        # Initialize the runtime, but execute only sedimentation after loading
        # its phase-boundary fields.
        code = GFDL1M(
            stencil_factory=self.stencil_factory,
            quantity_factory=self.quantity_factory,
            config=config,
        )

        # replace gfdl_mp_v3_locals and mp_full_locals with a regular State version (instead of LocalState)
        # so that it can be read from outside and the data can be passed back to the translate test
        gfdl_mp_v3_locals = GFDLMPV3Locals.empty(self.quantity_factory)
        code._gfdl_microphysics_v3._driver._gfdl_mp_v3_locals = gfdl_mp_v3_locals

        mp_full_locals = MPFullLocals.empty(self.quantity_factory)
        code._gfdl_microphysics_v3._driver._mp_full._mp_full_locals = mp_full_locals

        # map fortran serialbox names to python state names
        fields = {
            "T_data_buffered_in_n_loop": code._gfdl_microphysics_v3._driver._gfdl_mp_v3_locals.t,
            "QV_data_buffered_in_n_loop": code._gfdl_microphysics_v3._driver._gfdl_mp_v3_locals.mixing_ratio.vapor,
            "QL_data_buffered_in_n_loop": code._gfdl_microphysics_v3._driver._gfdl_mp_v3_locals.mixing_ratio.liquid,
            "QR_data_buffered_in_n_loop": code._gfdl_microphysics_v3._driver._gfdl_mp_v3_locals.mixing_ratio.rain,
            "QI_data_buffered_in_n_loop": code._gfdl_microphysics_v3._driver._gfdl_mp_v3_locals.mixing_ratio.ice,
            "QS_data_buffered_in_n_loop": code._gfdl_microphysics_v3._driver._gfdl_mp_v3_locals.mixing_ratio.snow,
            "QG_data_buffered_in_n_loop": code._gfdl_microphysics_v3._driver._gfdl_mp_v3_locals.mixing_ratio.graupel,
            "DZ_data_buffered_in_n_loop": code._gfdl_microphysics_v3._driver._gfdl_mp_v3_locals.dz,
            "DP_data_buffered_in_n_loop": code._gfdl_microphysics_v3._driver._gfdl_mp_v3_locals.dry_dp,
            "U_data_buffered_in_n_loop": code._gfdl_microphysics_v3._driver._gfdl_mp_v3_locals.u,
            "V_data_buffered_in_n_loop": code._gfdl_microphysics_v3._driver._gfdl_mp_v3_locals.v,
            "W_data_buffered_in_n_loop": code._gfdl_microphysics_v3._driver._gfdl_mp_v3_locals.w,
            "DEN_data_buffered_in_n_loop": code._gfdl_microphysics_v3._driver._gfdl_mp_v3_locals.density,
            "DENFAC_data_buffered_in_n_loop": code._gfdl_microphysics_v3._driver._gfdl_mp_v3_locals.density_factor,
            "DTE_data_buffered_in_n_loop": code._gfdl_microphysics_v3._driver._gfdl_mp_v3_locals.total_energy.delta,
            "MPPM1_data_buffered_in_n_loop": code._gfdl_microphysics_v3._driver._gfdl_mp_v3_locals.mppm1,
            "MPPM2_data_buffered_in_n_loop": code._gfdl_microphysics_v3._driver._gfdl_mp_v3_locals.mppm2,
            "MPPM3_data_buffered_in_n_loop": code._gfdl_microphysics_v3._driver._gfdl_mp_v3_locals.mppm3,
            "CNV_FRACTION_data_buffered_in_n_loop": code._gfdl_microphysics_v3._driver._gfdl_mp_v3_locals.convection_fraction,
        }

        # Identify ntimes from the input data shape.
        ntimes = inputs[next(iter(self.in_vars["data_vars"]))].shape[1]

        # Create output arrays with the same shape as the input arrays.
        outputs = {name: np.empty_like(inputs[name]) for name in self.out_vars}

        # Accept the full column or the Fortran lid-to-bottom slice.
        for name in self.in_vars["data_vars"]:
            field_shape = fields[name].field.shape
            ni, nj = field_shape[:2]
            shape = inputs[name].shape
            if (
                len(field_shape) not in (2, 3)
                or len(shape) != len(field_shape)
                or shape[:2] != (ni * nj, ntimes)
                or (len(shape) == 3 and shape[2] not in (field_shape[2], field_shape[2] - self.level_start))
            ):
                raise ValueError(f"{name}: buffer shape {shape} is incompatible with " f"field shape {field_shape} and {ntimes} time samples")

        # Copy fill input data, run the test code, copy data to output.
        for n in range(ntimes):
            for name in self.in_vars["data_vars"]:
                destination = fields[name].field[:]
                source = inputs[name][:, n, ...]
                ni, nj = destination.shape[:2]
                # Preserve missing values outside the serialized domain.
                destination[:] = np.nan
                # Reshape and copy the serialized data into the state field.
                if destination.ndim == 2:
                    safe_assign_array(destination, source.reshape((ni, nj), order="F"))
                else:
                    nlevels = source.shape[-1]
                    unpacked = source.reshape((ni, nj, nlevels), order="F")
                    start = 0 if nlevels == destination.shape[-1] else self.level_start
                    safe_assign_array(destination[:, :, start : start + nlevels], unpacked)

            code._gfdl_microphysics_v3._driver._mp_full._sedimentation(
                code._gfdl_microphysics_v3._driver._gfdl_mp_v3_locals,
                code._gfdl_microphysics_v3._driver._mp_full._mp_full_locals,
            )

            # Fill output data for this iteration.
            # Restore the exact serialized layout for the translate test systems.
            for name in self.out_vars:
                output_sample = outputs[name][:, n, ...]
                field = fields[name].field[:]
                start = 0 if field.ndim == 2 or output_sample.shape[-1] == field.shape[-1] else self.level_start
                saved_levels = field if field.ndim == 2 else field[:, :, start : start + output_sample.shape[-1]]
                safe_assign_array(
                    output_sample,
                    saved_levels.reshape(output_sample.shape, order="F"),
                )

        return outputs
