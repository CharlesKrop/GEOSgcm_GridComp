import dataclasses

from ndsl import Local, LocalState, QuantityFactory, StencilFactory
from ndsl.constants import I_DIM, J_DIM, K_DIM
from ndsl.dsl.gt4py import FORWARD, PARALLEL, computation, exp, interval
from ndsl.dsl.typing import Float, Float64, FloatField, FloatField64, FloatFieldIJ

from pyMoist.microphysics.GFDL_1M.config import GFDL1MConfig
from pyMoist.microphysics.GFDL_1M.microphysics.config import GFDLMPV3CloudMPConfig, GFDLMPV3NamelistConfig
from pyMoist.microphysics.GFDL_1M.microphysics.constants import CFMIN, QCMIN, TICE
from pyMoist.microphysics.GFDL_1M.microphysics.locals import GFDLMPV3Locals
from pyMoist.microphysics.GFDL_1M.microphysics.mp_full.mp_full import MPFullLocals
from pyMoist.microphysics.GFDL_1M.microphysics.shared import calc_mhc_lhc_wrapper, new_ice_condensate, new_liquid_condensate


def p_ice_melt_freeze(
        t: FloatField64,
        dry_dp: FloatField,
        density: FloatField,
        vapor: FloatField,
        ice: FloatField,
        liquid: FloatField,
        graupel: FloatField,
        rain: FloatField,
        snow: FloatField,
        cloud_fraction: FloatField,
        cvm: FloatField64,
        icpk: FloatField,
        lcpk: FloatField,
        tcpk: FloatField,
        tcp3: FloatField,
        total_energy: FloatField,
        one_minus_sigma: FloatFieldIJ,
        mppfw: FloatFieldIJ,
        mppmi: FloatFieldIJ,
        convection_fraction: FloatFieldIJ,
        surface_type: FloatFieldIJ,
):
    from __externals__ import CONV_FACTOR, D1_ICE, D1_VAP, DO_QA, DT, IN_CLOUD_ICE, LI00, LI20, LV00, PSAUT_QI_CRT, QL_MLT, T_WFR, TAU_FREZ, TAU_IMLT

    # psaut_qi_crt (ice to snow conversion) has strong resolution dependence
    # account for this using onemsig to convert more ice to snow at coarser resolutions
    with computation(FORWARD), interval(0, 1):
        critical_ice_factor: FloatFieldIJ = PSAUT_QI_CRT*(1.e-1*(1.0-one_minus_sigma) + one_minus_sigma)

        fac_imlt: FloatFieldIJ = 1. - exp (- DT / TAU_IMLT)
        fac_frez: FloatFieldIJ = 1. - exp (- DT / TAU_FREZ)

    with computation(PARALLEL), interval(...):

        if t  > TICE and ice  > QCMIN:

            # Use In-Cloud condensates with scale-aware blending
            if IN_CLOUD_ICE:
              # Enforce minimum bound to prevent vanishing values
              cloud_fraction_bounded = max(cloud_fraction, CFMIN)
            else:
              cloud_fraction_bounded = 1.0
            liquid_internal = liquid /cloud_fraction_bounded
            ice_internal = ice /cloud_fraction_bounded

            tmp = t
            newliq = new_liquid_condensate(t=tmp, liquid=liquid_internal, ice=ice_internal, convection_fraction=convection_fraction, surface_type=surface_type)
            sink = fac_imlt * min (ice_internal, newliq, (t  - TICE) / icpk  / cloud_fraction_bounded)
            tmp = min (sink, max(QL_MLT/cloud_fraction_bounded - liquid_internal, 0.0))

            tmp = tmp * cloud_fraction_bounded
            sink = sink * cloud_fraction_bounded
            mppmi = mppmi + sink * dry_dp  * CONV_FACTOR

            t, vapor, ice, liquid, graupel, rain, snow, cloud_fraction, cvm, total_energy, lcpk, icpk, tcpk, tcp3 = update_hydrometeors_and_temperature(
                cloud_fraction=cloud_fraction,
                vapor=vapor,
                ice=ice,
                liquid=liquid,
                graupel=graupel,
                rain=rain,
                snow=snow,
                dvapor=0.0,
                dice=-sink,
                dliquid=tmp,
                dgraupel=0.0,
                drain=sink-tmp,
                dsnow=0.0,
                DO_QA=DO_QA,
                D1_VAP=D1_VAP,
                D1_ICE=D1_ICE,
                LI00=LI00,
                LI20=LI20,
                LV00=LV00,
                T_WFR=T_WFR,
            )

        elif t  <= TICE and liquid  > QCMIN:

            # Use In-Cloud condensates with scale-aware blending
            if IN_CLOUD_ICE:
              # Enforce minimum bound to prevent vanishing values
              cloud_fraction_bounded = max(cloud_fraction, CFMIN)
            else:
              cloud_fraction_bounded = 1.0
            liquid_internal = liquid /cloud_fraction_bounded
            ice_internal = ice /cloud_fraction_bounded

            tmp = t
            newice = new_ice_condensate(t=tmp, liquid=liquid_internal, ice=ice_internal, convection_fraction=convection_fraction, surface_type=surface_type)
            sink = fac_frez * min(liquid_internal, newice, (TICE - t ) / icpk  / cloud_fraction_bounded)
            ice_modified = critical_ice_factor / density
            tmp = min (sink, max(ice_modified/cloud_fraction_bounded - ice_internal, 0.0))

            tmp = tmp*cloud_fraction_bounded
            sink = sink*cloud_fraction_bounded
            mppfw = mppfw + sink * dry_dp  * CONV_FACTOR

            t, vapor, ice, liquid, graupel, rain, snow, cloud_fraction, cvm, total_energy, lcpk, icpk, tcpk, tcp3 = update_hydrometeors_and_temperature(
                cloud_fraction=cloud_fraction,
                vapor=vapor,
                ice=ice,
                liquid=liquid,
                graupel=graupel,
                rain=rain,
                snow=snow,
                dvapor=0.0,
                dice=tmp,
                dliquid=-sink,
                dgraupel=0.0,
                drain=0.0,
                dsnow=sink-tmp,
                DO_QA=DO_QA,
                D1_VAP=D1_VAP,
                D1_ICE=D1_ICE,
                LI00=LI00,
                LI20=LI20,
                LV00=LV00,
                T_WFR=T_WFR,
            )


@dataclasses.dataclass
class IceCloudLocals(LocalState):
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


class IceCloud:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        gfdl_1m_config: GFDL1MConfig,
        mp_config: GFDLMPV3CloudMPConfig,
        mp_namelist: GFDLMPV3NamelistConfig,
        CONV_FACTOR: Float,
    ):
        # make config visible at runtime
        self._mp_namelist = mp_namelist

        # initialize locals for ice cloud
        self._ice_cloud_locals = IceCloudLocals.make_locals(quantity_factory)

        # construct stencils
        self._calc_mhc_lhc_wrapper = stencil_factory.from_dims_halo(
            func=calc_mhc_lhc_wrapper,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={
                "C1_ICE": mp_config.C1_ICE,
                "C1_LIQ": mp_config.C1_LIQ,
                "C1_VAP": mp_config.C1_VAP,
                "D1_ICE": mp_config.D1_ICE,
                "D1_VAP": mp_config.D1_VAP,
                "LI00": mp_config.LI00,
                "LI20": mp_config.LI20,
                "LV00": mp_config.LV00,
                "T_WFR": mp_config.T_WFR,
            },
        )
        self._p_ice_melt_freeze = stencil_factory.from_dims_halo(
            func=p_ice_melt_freeze,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={
               "CONV_FACTOR": CONV_FACTOR,
               "D1_ICE": mp_config.D1_ICE,
               "D1_VAP": mp_config.D1_VAP,
               "DO_QA": mp_namelist.DO_QA,
               "DT": gfdl_1m_config.DT_MOIST,
               "IN_CLOUD_ICE": mp_namelist.IN_CLOUD_ICE,
               "LI00": mp_config.LI00,
               "LI20": mp_config.LI20,
               "LV00": mp_config.LV00,
               "PSAUT_QI_CRT": mp_namelist.PSAUT_QI_CRT,
               "QL_MLT": mp_namelist.QL_MLT,
               "T_WFR": mp_config.T_WFR,
               "TAU_FREZ": mp_namelist.TAU_FREZ,
               "TAU_IMLT": mp_namelist.TAU_IMLT,
            },
        )

    def __call__(self, gfdl_mp_v3_locals: GFDLMPV3Locals, mp_full_locals: MPFullLocals):
        # --------------------------------------------------
        # calculate heat capacities and latent heat coefficients
        # --------------------------------------------------
        self._calc_mhc_lhc_wrapper(
            t=gfdl_mp_v3_locals.t,
            vapor=gfdl_mp_v3_locals.vapor,
            ice=gfdl_mp_v3_locals.ice,
            liquid=gfdl_mp_v3_locals.liquid,
            graupel=gfdl_mp_v3_locals.graupel,
            rain=gfdl_mp_v3_locals.rain,
            snow=gfdl_mp_v3_locals.snow,
            total_liquid=self._ice_cloud_locals.total_liquid,
            total_solid=self._ice_cloud_locals.total_solid,
            cvm=self._ice_cloud_locals.cvm,
            total_energy=self._ice_cloud_locals.total_energy,
            lcpk=self._ice_cloud_locals.lcpk,
            icpk=self._ice_cloud_locals.icpk,
            tcpk=self._ice_cloud_locals.tcpk,
            tcp3=self._ice_cloud_locals.tcp3,
        )

        if not self._mp_namelist.DO_WARM_RAIN_MP:
            # -----------------------------------------------------------------------
            # cloud ice/liq melt/freeze to form cloud water/ice and rain/snow
            # -----------------------------------------------------------------------
            self._p_ice_melt_freeze(
                t=gfdl_mp_v3_locals.t,
                dry_dp=gfdl_mp_v3_locals.dry_dp,
                density=gfdl_mp_v3_locals.density,
                vapor=gfdl_mp_v3_locals.mixing_ratio.vapor,
                ice=gfdl_mp_v3_locals.mixing_ratio.ice,
                liquid=gfdl_mp_v3_locals.mixing_ratio.liquid,
                graupel=gfdl_mp_v3_locals.mixing_ratio.graupel,
                rain=gfdl_mp_v3_locals.mixing_ratio.rain,
                snow=gfdl_mp_v3_locals.mixing_ratio.snow,
                cloud_fraction=gfdl_mp_v3_locals.cloud_fraction,
                cvm=self._ice_cloud_locals.cvm,
                icpk=self._ice_cloud_locals.icpk,
                lcpk=self._ice_cloud_locals.lcpk,
                tcpk=self._ice_cloud_locals.tcpk,
                tcp3=self._ice_cloud_locals.tcp3,
                total_energy=self._ice_cloud_locals.total_energy,
                one_minus_sigma=gfdl_mp_v3_locals.one_minus_sigma,
                mppfw=gfdl_mp_v3_locals.mppfw,
                mppmi=gfdl_mp_v3_locals.mppmi,
                convection_fraction=gfdl_mp_v3_locals.convection_fraction,
                surface_type=gfdl_mp_v3_locals.surface_type,
            )