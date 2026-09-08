import dataclasses

from pyMoist.microphysics.GFDL_1M.microphysics.config import GFDLMPV3CloudMPConfig, GFDLMPV3NamelistConfig
from ndsl import StencilFactory, QuantityFactory, LocalState, Local
from ndsl.stencils.basic_operations import set_value
from ndsl.stencils.basic_operations_2d import set_value_2d
from ndsl.dsl.typing import Float, Float64, FloatField
from ndsl.constants import I_DIM, J_DIM, K_DIM
from pyMoist.microphysics.GFDL_1M.state import GFDL1MState
from pyMoist.microphysics.GFDL_1M.locals import GFDL1MLocals
from pyMoist.microphysics.GFDL_1M.microphysics.locals import GFDLMPV3Locals
from pyMoist.microphysics.GFDL_1M.microphysics.mp_full.main import MPFullLocals
from pyMoist.microphysics.GFDL_1M.microphysics.shared import calc_mhc_lhc
from ndsl.dsl.gt4py import computation, PARALLEL, interval


def calc_mhc_lhc_wrapper(
    t: FloatField,
    vapor: FloatField,
    ice: FloatField,
    liquid: FloatField,
    graupel: FloatField,
    rain: FloatField,
    snow: FloatField,
    total_liquid: FloatField,
    total_solid: FloatField,
    cvm: FloatField,
    total_energy: FloatField,
    lcpk: FloatField,
    icpk: FloatField,
    tcpk: FloatField,
    tcp3: FloatField,
):
    from __externals__ import C1_VAP, C1_LIQ, C1_ICE, D1_ICE, D1_VAP, LI00, LI20, LV00, T_WFR

    with computation(PARALLEL), interval(...):
        total_liquid, total_solid, cvm, total_energy, lcpk, icpk, tcpk, tcp3 = calc_mhc_lhc(
            t=t,
            vapor=vapor,
            ice=ice,
            liquid=liquid,
            graupel=graupel,
            rain=rain,
            snow=snow,
            C1_VAP=C1_VAP,
            C1_LIQ=C1_LIQ,
            C1_ICE=C1_ICE,
            D1_ICE=D1_ICE,
            D1_VAP=D1_VAP,
            LI00=LI00,
            LI20=LI20,
            LV00=LV00,
            T_WFR=T_WFR,
        )


@dataclasses.dataclass
class SedimentationLocals(LocalState):
    cvm: Local = dataclasses.field(
        metadata={
            "name": "cvm",
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float64,
        }
    )
    icpk: Local = dataclasses.field(
        metadata={
            "name": "icpk",
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    lcpk: Local = dataclasses.field(
        metadata={
            "name": "lcpk",
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    tcpk: Local = dataclasses.field(
        metadata={
            "name": "tcpk",
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    tcp3: Local = dataclasses.field(
        metadata={
            "name": "tcp3",
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    total_energy: Local = dataclasses.field(
        metadata={
            "name": "total_energy",
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float64,
        }
    )
    total_liquid: Local = dataclasses.field(
        metadata={
            "name": "total_liquid",
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float64,
        }
    )
    total_solid: Local = dataclasses.field(
        metadata={
            "name": "total_solid",
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float64,
        }
    )


class Sedimentation:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        mp_config: GFDLMPV3CloudMPConfig,
        mp_namelist: GFDLMPV3NamelistConfig,
        CONV_FACTOR: Float,
    ):
        self._set_value_2d = stencil_factory.from_dims_halo(
            func=set_value_2d,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={"CONV_FACTOR": CONV_FACTOR},
        )
        self._set_value = stencil_factory.from_dims_halo(
            func=set_value,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={"CONV_FACTOR": CONV_FACTOR},
        )
        self._calc_mhc_lhc_wrapper = stencil_factory.from_dims_halo(
            func=calc_mhc_lhc_wrapper,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={"CONV_FACTOR": CONV_FACTOR},
        )

        self._sedimentation_locals = SedimentationLocals.make_locals(quantity_factory)


    def __call__(self, state: GFDL1MState, gfdl_1m_locals: GFDL1MLocals, gfdl_mp_v3_locals: GFDLMPV3Locals, mp_full_locals: MPFullLocals):

        # reset locals
        self._set_value_2d(mp_full_locals.surface_precip_ice, Float(0.0))
        self._set_value_2d(mp_full_locals.surface_precip_liquid, Float(0.0))
        self._set_value_2d(mp_full_locals.surface_precip_graupel, Float(0.0))
        self._set_value_2d(mp_full_locals.surface_precip_rain, Float(0.0))
        self._set_value_2d(mp_full_locals.surface_precip_snow, Float(0.0))

        self._set_value(mp_full_locals.precip_ice, Float(0.0))
        self._set_value(mp_full_locals.precip_liquid, Float(0.0))
        self._set_value(mp_full_locals.precip_graupel, Float(0.0))
        self._set_value(mp_full_locals.precip_rain, Float(0.0))
        self._set_value(mp_full_locals.precip_snow, Float(0.0))

        self._set_value_2d(mp_full_locals.surface_precip_ice, Float(0.0))
        self._set_value_2d(mp_full_locals.surface_precip_liquid, Float(0.0))
        self._set_value_2d(mp_full_locals.surface_precip_graupel, Float(0.0))
        self._set_value_2d(mp_full_locals.surface_precip_rain, Float(0.0))
        self._set_value_2d(mp_full_locals.surface_precip_snow, Float(0.0))

        # calculate heat capacities and latent heat coefficients
        self._calc_mhc_lhc_wrapper(
            t=gfdl_mp_v3_locals.t,
            vapor=gfdl_mp_v3_locals.vapor,
            ice=gfdl_mp_v3_locals.ice,
            liquid=gfdl_mp_v3_locals.liquid,
            graupel=gfdl_mp_v3_locals.graupel,
            rain=gfdl_mp_v3_locals.rain,
            snow=gfdl_mp_v3_locals.snow,
            total_liquid=self._sedimentation_locals.total_liquid,
            total_solid=self._sedimentation_locals.total_solid,
            cvm=self._sedimentation_locals.cvm,
            total_energy=self._sedimentation_locals.total_energy,
            lcpk=self._sedimentation_locals.lcpk,
            icpk=self._sedimentation_locals.icpk,
            tcpk=self._sedimentation_locals.tcpk,
            tcp3=self._sedimentation_locals.tcp3,
        )
