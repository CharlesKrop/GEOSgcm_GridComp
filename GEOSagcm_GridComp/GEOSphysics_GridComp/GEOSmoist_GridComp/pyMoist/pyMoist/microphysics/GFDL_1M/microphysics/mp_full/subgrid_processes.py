import dataclasses

from ndsl import StencilFactory, QuantityFactory, Local, LocalState
from ndsl.constants import I_DIM, J_DIM, K_DIM
from ndsl.dsl.typing import Float, Float64, FloatField, FloatField64, FloatFieldIJ
from ndsl.dsl.gt4py import computation, interval, FORWARD, exp, PARALLEL

from pyMoist.microphysics.GFDL_1M.microphysics.config import GFDLMPV3CloudMPConfig, GFDLMPV3NamelistConfig, GFDLMPV3TableL20, GFDLMPV3TableL3xL10
from pyMoist.microphysics.GFDL_1M.config import GFDL1MConfig
from pyMoist.microphysics.GFDL_1M.state import GFDL1MState
from pyMoist.microphysics.GFDL_1M.microphysics.saturation_tables import GFDLMPV3Tables, GFDLMPV3SaturationTable
from pyMoist.microphysics.GFDL_1M.microphysics.shared import (
    calc_mhc_lhc_wrapper,
    moist_heat_capacity_3,
    update_hydrometeors_and_temperature,
    p_bigg,
    p_complete_freezing,
    p_graupel_deposition_and_sublimation,
    p_ice_deposition_and_sublimation,
    p_snow_deposition_and_sublimation,
    p_wbf,
)
from pyMoist.microphysics.GFDL_1M.microphysics.mp_full.mp_full import MPFullLocals
from pyMoist.microphysics.GFDL_1M.microphysics.locals import GFDLMPV3Locals
from pyMoist.microphysics.GFDL_1M.microphysics.constants import CFMIN, DT_FR, QCMIN, QVMIN
from pyMoist.microphysics.GFDL_1M.microphysics.saturation_table_functions import saturation_specific_humidity


def p_instant(
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
    h_var: FloatField,
    cvm: FloatField64,
    icpk: FloatField,
    lcpk: FloatField,
    tcpk: FloatField,
    tcp3: FloatField,
    total_energy: FloatField64,
    mppd1: FloatFieldIJ,
    mppe1: FloatFieldIJ,
    mpps1: FloatFieldIJ,
    one_minus_sigma: FloatFieldIJ,
    table_0: GFDLMPV3SaturationTable,
    table_2: GFDLMPV3SaturationTable,
    dtable_0: GFDLMPV3SaturationTable,
    dtable_2: GFDLMPV3SaturationTable,
):
    """instant processes (include deposition, evaporation, and sublimation)

    Args:
        t (FloatField64)
        dry_dp (FloatField)
        density (FloatField)
        vapor (FloatField)
        ice (FloatField)
        liquid (FloatField)
        graupel (FloatField)
        rain (FloatField)
        snow (FloatField)
        cloud_fraction (FloatField)
        h_var (FloatField)
        cvm (FloatField64)
        icpk (FloatField)
        lcpk (FloatField)
        tcpk (FloatField)
        tcp3 (FloatField)
        total_energy (FloatField64)
        mppd1 (FloatFieldIJ)
        mppe1 (FloatFieldIJ)
        mpps1 (FloatFieldIJ)
        one_minus_sigma (FloatFieldIJ)
        table_0 (GFDLMPV3SaturationTable)
        table_2 (GFDLMPV3SaturationTable)
        dtable_0 (GFDLMPV3SaturationTable)
        dtable_2 (GFDLMPV3SaturationTable)
    """
    from __externals__ import (
        CONV_FACTOR,
        D1_ICE,
        D1_VAP,
        DO_EVAP_TIMESCALE,
        DO_QA,
        DT,
        LI00,
        LI20,
        LV00,
        RH_FAC_EVAP,
        RH_INC,
        RHC_CEVAP,
        TAU_L2V,
        T_MIN,
        T_SUB,
        T_WFR,
        USE_RHC_CEVAP,
    )

    with computation(FORWARD), interval(0, 1):
        fac_l2v: FloatFieldIJ = 1.0 - exp(-DT / TAU_L2V)

    with computation(PARALLEL), interval(...):
        # -----------------------------------------------------------------------
        # instant deposit all water vapor to cloud ice when temperature is super low
        # -----------------------------------------------------------------------

        if t < T_MIN:

            subl = max(vapor - QCMIN, 0.0)
            mppd1 = mppd1 + subl * dry_dp * CONV_FACTOR

            t, vapor, ice, liquid, graupel, rain, snow, cloud_fraction, cvm, total_energy, lcpk, icpk, tcpk, tcp3 = update_hydrometeors_and_temperature(
                cloud_fraction=cloud_fraction,
                vapor=vapor,
                ice=ice,
                liquid=liquid,
                graupel=graupel,
                rain=rain,
                snow=snow,
                dvapor=-subl,
                dice=subl,
                dliquid=0.0,
                dgraupel=0.0,
                drain=0.0,
                dsnow=0.0,
                DO_QA=DO_QA,
                D1_VAP=D1_VAP,
                D1_ICE=D1_ICE,
                LI00=LI00,
                LI20=LI20,
                LV00=LV00,
                T_WFR=T_WFR,
            )

            # [WMP] avoid high cloud fractions for high troposhere cirrus clouds
            if not DO_QA:
                cloud_fraction = max(0.0, min(1.0, 1.0 - QCMIN / max(ice, QCMIN)))
                if cloud_fraction < CFMIN:
                    # remove clouds and qi if qa is too small
                    cloud_fraction = 0.0
                    subl = ice
                    mppd1 = mppd1 - subl * dry_dp * CONV_FACTOR
                    t, vapor, ice, liquid, graupel, rain, snow, cloud_fraction, cvm, total_energy, lcpk, icpk, tcpk, tcp3 = update_hydrometeors_and_temperature(
                        cloud_fraction=cloud_fraction,
                        vapor=vapor,
                        ice=ice,
                        liquid=liquid,
                        graupel=graupel,
                        rain=rain,
                        snow=snow,
                        dvapor=subl,
                        dice=-subl,
                        dliquid=0.0,
                        dgraupel=0.0,
                        drain=0.0,
                        dsnow=0.0,
                        DO_QA=DO_QA,
                        D1_VAP=D1_VAP,
                        D1_ICE=D1_ICE,
                        LI00=LI00,
                        LI20=LI20,
                        LV00=LV00,
                        T_WFR=T_WFR,
                    )

        # -----------------------------------------------------------------------
        # instant evaporation / sublimation of all clouds when rh < rh_adj
        # -----------------------------------------------------------------------

        combined_precip = vapor + liquid + ice
        t_in = (total_energy - LV00 * combined_precip + LI00 * (snow + graupel)) / moist_heat_capacity_3(combined_precip, rain, snow + graupel)

        if t_in > T_SUB + 6.0:

            # initialize to 0s
            evap = 0.0
            subl = 0.0

            rh_adj = 1.0 - h_var - RH_INC
            ice_saturation_humidity, _ = saturation_specific_humidity(t_in, density, table_2, dtable_2)
            rh = combined_precip / ice_saturation_humidity
            if rh < rh_adj:
                # instant evap of all liquid & ice
                evap = liquid
                subl = ice
            else:
                # partial evap of liquid
                t_in = t
                liquid_saturation_humidity, dliquid_saturation_humidity = saturation_specific_humidity(t_in, density, table_0, dtable_0)
                dliquid = liquid_saturation_humidity - vapor
                if dliquid > QVMIN:
                    if DO_EVAP_TIMESCALE:
                        factor = min(1.0, fac_l2v * (RH_FAC_EVAP * dliquid / liquid_saturation_humidity))
                    else:
                        factor = 1.0

                    evap = min(liquid, factor * liquid / (1.0 + tcp3 * dliquid_saturation_humidity))
                    # -----------------------------------------------------------------------
                    # use RH cap for cloud evaporation
                    # -----------------------------------------------------------------------
                    if USE_RHC_CEVAP:
                        rh_tem = combined_precip / liquid_saturation_humidity
                        if rh_tem >= RHC_CEVAP:
                            evap = 0.0
                    # endif
                # endif
                # nothing for ice
                subl = 0.0

            evap = evap * one_minus_sigma  # resolution dependent evap 0:1 coarse:fine
            subl = subl * one_minus_sigma  # resolution dependent subl 0:1 coarse:fine

            mppe1 = mppe1 + evap * dry_dp * CONV_FACTOR
            mpps1 = mpps1 + subl * dry_dp * CONV_FACTOR

            t, vapor, ice, liquid, graupel, rain, snow, cloud_fraction, cvm, total_energy, lcpk, icpk, tcpk, tcp3 = update_hydrometeors_and_temperature(
                cloud_fraction=cloud_fraction,
                vapor=vapor,
                ice=ice,
                liquid=liquid,
                graupel=graupel,
                rain=rain,
                snow=snow,
                dvapor=evap + subl,
                dice=-subl,
                dliquid=-evap,
                dgraupel=0.0,
                drain=0.0,
                dsnow=0.0,
                DO_QA=DO_QA,
                D1_VAP=D1_VAP,
                D1_ICE=D1_ICE,
                LI00=LI00,
                LI20=LI20,
                LV00=LV00,
                T_WFR=T_WFR,
            )


@dataclasses.dataclass
class SubgridProcessesLocals(LocalState):
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


class SubgridProcesses:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        saturation_tables: GFDLMPV3Tables,
        mp_config: GFDLMPV3CloudMPConfig,
        mp_namelist: GFDLMPV3NamelistConfig,
        CONV_FACTOR: Float,
        DRIVER_DT: Float,
    ):
        # make config visible at runtime
        self._mp_config = mp_config
        self._mp_namelist = mp_namelist
        self._saturation_tables = saturation_tables

        # initialize locals for ice cloud
        self._subgrid_processes_locals = SubgridProcessesLocals.make_locals(quantity_factory)

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
        self._p_bigg = stencil_factory.from_dims_halo(
            func=p_bigg,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={
                "CONV_FACTOR": CONV_FACTOR,
                "D1_ICE": mp_config.D1_ICE,
                "D1_VAP": mp_config.D1_VAP,
                "DO_BIGG": mp_namelist.DO_BIGG,
                "DO_PSD_WATER_NUM": mp_namelist.DO_PSD_WATER_NUM,
                "DO_QA": mp_namelist.DO_QA,
                "DT": DRIVER_DT,
                "LI00": mp_config.LI00,
                "LI20": mp_config.LI20,
                "LV00": mp_config.LV00,
                "MUW": mp_namelist.MUW,
                "PCAW": mp_config.PCAW,
                "PCBW": mp_config.PCBW,
                "T_WFR": mp_config.T_WFR,
            },
        )
        self._p_complete_freezing = stencil_factory.from_dims_halo(
            func=p_complete_freezing,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={
                "CONV_FACTOR": CONV_FACTOR,
                "D1_ICE": mp_config.D1_ICE,
                "D1_VAP": mp_config.D1_VAP,
                "DO_QA": mp_namelist.DO_QA,
                "LI00": mp_config.LI00,
                "LI20": mp_config.LI20,
                "LV00": mp_config.LV00,
                "T_WFR": mp_config.T_WFR,
            },
        )
        self._p_graupel_deposition_and_sublimation = stencil_factory.from_dims_halo(
            func=p_graupel_deposition_and_sublimation,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={
                "BLING": mp_namelist.BLING,
                "BLINH": mp_namelist.BLINH,
                "CONV_FACTOR": CONV_FACTOR,
                "D1_ICE": mp_config.D1_ICE,
                "D1_VAP": mp_config.D1_VAP,
                "DO_HAIL": mp_namelist.DO_HAIL,
                "DO_QA": mp_namelist.DO_QA,
                "DT": DRIVER_DT,
                "GS_FAC": mp_namelist.GS_FAC,
                "LI00": mp_config.LI00,
                "LI20": mp_config.LI20,
                "LV00": mp_config.LV00,
                "MUG": mp_namelist.MUG,
                "MUH": mp_namelist.MUH,
                "T_SUB": mp_namelist.T_SUB,
                "T_WFR": mp_config.T_WFR,
            },
        )
        self._p_ice_deposition_and_sublimation = stencil_factory.from_dims_halo(
            func=p_ice_deposition_and_sublimation,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={
                "CONV_FACTOR": CONV_FACTOR,
                "D1_ICE": mp_config.D1_ICE,
                "D1_VAP": mp_config.D1_VAP,
                "DO_PSD_ICE_NUM": mp_namelist.DO_PSD_ICE_NUM,
                "DO_QA": mp_namelist.DO_QA,
                "DT": DRIVER_DT,
                "IGFLAG": mp_namelist.IGFLAG,
                "INFLAG": mp_namelist.INFLAG,
                "IS_FAC": mp_namelist.IS_FAC,
                "LI00": mp_config.LI00,
                "LI20": mp_config.LI20,
                "LV00": mp_config.LV00,
                "MUI": mp_namelist.MUI,
                "PCAI": mp_config.PCAI,
                "PCBI": mp_config.PCBI,
                "PROG_CIN": mp_namelist.PROG_CIN,
                "QI_LIM": mp_namelist.QI_LIM,
                "T_SUB": mp_namelist.T_SUB,
                "T_WFR": mp_config.T_WFR,
            },
        )
        self._p_instant = stencil_factory.from_dims_halo(
            func=p_instant,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={
                "CONV_FACTOR": CONV_FACTOR,
                "D1_ICE": mp_config.D1_ICE,
                "D1_VAP": mp_config.D1_VAP,
                "DO_EVAP_TIMESCALE": mp_namelist.DO_EVAP_TIMESCALE,
                "DO_QA": mp_namelist.DO_QA,
                "DT": DRIVER_DT,
                "LI00": mp_config.LI00,
                "LI20": mp_config.LI20,
                "LV00": mp_config.LV00,
                "RH_FAC_EVAP": mp_namelist.RH_FAC_EVAP,
                "RH_INC": mp_namelist.RH_INC,
                "RHC_CEVAP": mp_namelist.RHC_CEVAP,
                "TAU_L2V": mp_namelist.TAU_L2V,
                "T_MIN": mp_namelist.T_MIN,
                "T_SUB": mp_namelist.T_SUB,
                "T_WFR": mp_config.T_WFR,
                "USE_RHC_CEVAP": mp_namelist.USE_RHC_CEVAP,
            },
        )
        self._p_snow_deposition_and_sublimation = stencil_factory.from_dims_halo(
            func=p_snow_deposition_and_sublimation,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={
                "BLINS": mp_namelist.BLINS,
                "CONV_FACTOR": CONV_FACTOR,
                "D1_ICE": mp_config.D1_ICE,
                "D1_VAP": mp_config.D1_VAP,
                "DO_QA": mp_namelist.DO_QA,
                "DT": DRIVER_DT,
                "LI00": mp_config.LI00,
                "LI20": mp_config.LI20,
                "LV00": mp_config.LV00,
                "MUS": mp_namelist.MUS,
                "SS_FAC": mp_namelist.SS_FAC,
                "T_SUB": mp_namelist.T_SUB,
                "T_WFR": mp_config.T_WFR,
            },
        )
        self._p_wbf = stencil_factory.from_dims_halo(
            func=p_wbf,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={
                "CONV_FACTOR": CONV_FACTOR,
                "D1_ICE": mp_config.D1_ICE,
                "D1_VAP": mp_config.D1_VAP,
                "DO_WBF": mp_namelist.DO_WBF,
                "DO_QA": mp_namelist.DO_QA,
                "DT": DRIVER_DT,
                "LI00": mp_config.LI00,
                "LI20": mp_config.LI20,
                "LV00": mp_config.LV00,
                "PWBF_QI_CRT": mp_namelist.PWBF_QI_CRT,
                "TAU_WBF": mp_namelist.TAU_WBF,
                "T_WFR": mp_config.T_WFR,
            },
        )

    def __call__(self, state: GFDL1MState, gfdl_mp_v3_locals: GFDLMPV3Locals):
        # -----------------------------------------------------------------------
        # calculate heat capacities and latent heat coefficients
        # -----------------------------------------------------------------------
        self._calc_mhc_lhc_wrapper(
            t=gfdl_mp_v3_locals.t,
            vapor=gfdl_mp_v3_locals.mixing_ratio.vapor,
            ice=gfdl_mp_v3_locals.mixing_ratio.ice,
            liquid=gfdl_mp_v3_locals.mixing_ratio.liquid,
            graupel=gfdl_mp_v3_locals.mixing_ratio.graupel,
            rain=gfdl_mp_v3_locals.mixing_ratio.rain,
            snow=gfdl_mp_v3_locals.mixing_ratio.snow,
            total_liquid=self._subgrid_processes_locals.total_liquid,
            total_solid=self._subgrid_processes_locals.total_solid,
            cvm=self._subgrid_processes_locals.cvm,
            total_energy=self._subgrid_processes_locals.total_energy,
            lcpk=self._subgrid_processes_locals.lcpk,
            icpk=self._subgrid_processes_locals.icpk,
            tcpk=self._subgrid_processes_locals.tcpk,
            tcp3=self._subgrid_processes_locals.tcp3,
        )

        # -----------------------------------------------------------------------
        # instant processes (include deposition, evaporation, and sublimation)
        # -----------------------------------------------------------------------
        self._p_instant(
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
            h_var=gfdl_mp_v3_locals.h_var,
            cvm=self._subgrid_processes_locals.cvm,
            icpk=self._subgrid_processes_locals.icpk,
            lcpk=self._subgrid_processes_locals.lcpk,
            tcpk=self._subgrid_processes_locals.tcpk,
            tcp3=self._subgrid_processes_locals.tcp3,
            total_energy=self._subgrid_processes_locals.total_energy,
            mppd1=gfdl_mp_v3_locals.mppd1,
            mppe1=gfdl_mp_v3_locals.mppe1,
            mpps1=gfdl_mp_v3_locals.mpps1,
            one_minus_sigma=gfdl_mp_v3_locals.one_minus_sigma,
            table_0=self._saturation_tables.table_0,
            table_2=self._saturation_tables.table_2,
            dtable_0=self._saturation_tables.dtable_0,
            dtable_2=self._saturation_tables.dtable_2,
        )
        if not self._mp_namelist.DO_WARM_RAIN_MP:
            if False:
                # WMP - something here causes large warm temperature spikes
                # WMP - partial evap is moved into pinst call above
                # WMP - ignoring condensation for now
                # -----------------------------------------------------------------------
                # cloud water condensation and evaporation
                # -----------------------------------------------------------------------
                if self._mp_namelist.DELAY_COND_EVAP:
                    cond_evap = self._mp_config.LAST_STEP
                else:
                    cond_evap = True

                if cond_evap:
                    for n in self._mp_namelist.NCONDS:
                        pcond_pevap_not_ported = True

        # -----------------------------------------------------------------------
        # enforce complete freezing below t_wfr
        # -----------------------------------------------------------------------
        self._p_complete_freezing(
            t=gfdl_mp_v3_locals.t,
            dry_dp=gfdl_mp_v3_locals.dry_dp,
            vapor=gfdl_mp_v3_locals.mixing_ratio.vapor,
            ice=gfdl_mp_v3_locals.mixing_ratio.ice,
            liquid=gfdl_mp_v3_locals.mixing_ratio.liquid,
            graupel=gfdl_mp_v3_locals.mixing_ratio.graupel,
            rain=gfdl_mp_v3_locals.mixing_ratio.rain,
            snow=gfdl_mp_v3_locals.mixing_ratio.snow,
            cloud_fraction=gfdl_mp_v3_locals.cloud_fraction,
            cvm=self._subgrid_processes_locals.cvm,
            icpk=self._subgrid_processes_locals.icpk,
            lcpk=self._subgrid_processes_locals.lcpk,
            tcpk=self._subgrid_processes_locals.tcpk,
            tcp3=self._subgrid_processes_locals.tcp3,
            total_energy=self._subgrid_processes_locals.total_energy,
            mppfw=gfdl_mp_v3_locals.mppfw,
        )

        # -----------------------------------------------------------------------
        # Wegener Bergeron Findeisen process
        # -----------------------------------------------------------------------
        self._p_wbf(
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
            cvm=self._subgrid_processes_locals.cvm,
            icpk=self._subgrid_processes_locals.icpk,
            lcpk=self._subgrid_processes_locals.lcpk,
            tcpk=self._subgrid_processes_locals.tcpk,
            tcp3=self._subgrid_processes_locals.tcp3,
            total_energy=self._subgrid_processes_locals.total_energy,
            one_minus_sigma=gfdl_mp_v3_locals.one_minus_sigma,
            mppfw=gfdl_mp_v3_locals.mppfw,
            table_0=self._saturation_tables.table_0,
            table_2=self._saturation_tables.table_2,
            dtable_0=self._saturation_tables.dtable_0,
            dtable_2=self._saturation_tables.dtable_2,
        )

        # -----------------------------------------------------------------------
        # Bigg freezing mechanism
        # -----------------------------------------------------------------------
        self._p_bigg(
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
            ccn=gfdl_mp_v3_locals.ccn,
            cvm=self._subgrid_processes_locals.cvm,
            icpk=self._subgrid_processes_locals.icpk,
            lcpk=self._subgrid_processes_locals.lcpk,
            tcpk=self._subgrid_processes_locals.tcpk,
            tcp3=self._subgrid_processes_locals.tcp3,
            total_energy=self._subgrid_processes_locals.total_energy,
            mppfw=gfdl_mp_v3_locals.mppfw,
        )

        # -----------------------------------------------------------------------
        # cloud ice deposition and sublimation
        # -----------------------------------------------------------------------
        self._p_ice_deposition_and_sublimation(
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
            cin=gfdl_mp_v3_locals.cin,
            rsubl=state.non_anvil_large_scale.sublimation,
            cvm=self._subgrid_processes_locals.cvm,
            icpk=self._subgrid_processes_locals.icpk,
            lcpk=self._subgrid_processes_locals.lcpk,
            tcpk=self._subgrid_processes_locals.tcpk,
            tcp3=self._subgrid_processes_locals.tcp3,
            total_energy=self._subgrid_processes_locals.total_energy,
            one_minus_sigma=gfdl_mp_v3_locals.one_minus_sigma,
            mppdi=gfdl_mp_v3_locals.mppdi,
            mppsi=gfdl_mp_v3_locals.mppsi,
            table_2=self._saturation_tables.table_2,
            dtable_2=self._saturation_tables.dtable_2,
        )

        # -----------------------------------------------------------------------
        # snow deposition and sublimation
        # -----------------------------------------------------------------------
        self._p_snow_deposition_and_sublimation(
            t=gfdl_mp_v3_locals.t,
            dry_dp=gfdl_mp_v3_locals.dry_dp,
            density=gfdl_mp_v3_locals.density,
            density_factor=gfdl_mp_v3_locals.density_factor,
            vapor=gfdl_mp_v3_locals.mixing_ratio.vapor,
            ice=gfdl_mp_v3_locals.mixing_ratio.ice,
            liquid=gfdl_mp_v3_locals.mixing_ratio.liquid,
            graupel=gfdl_mp_v3_locals.mixing_ratio.graupel,
            rain=gfdl_mp_v3_locals.mixing_ratio.rain,
            snow=gfdl_mp_v3_locals.mixing_ratio.snow,
            cloud_fraction=gfdl_mp_v3_locals.cloud_fraction,
            cvm=self._subgrid_processes_locals.cvm,
            icpk=self._subgrid_processes_locals.icpk,
            lcpk=self._subgrid_processes_locals.lcpk,
            tcpk=self._subgrid_processes_locals.tcpk,
            tcp3=self._subgrid_processes_locals.tcp3,
            total_energy=self._subgrid_processes_locals.total_energy,
            mppds=gfdl_mp_v3_locals.mppds,
            mppss=gfdl_mp_v3_locals.mppss,
            CSSUB=self._mp_config.CSSUB,
            table_2=self._saturation_tables.table_2,
            dtable_2=self._saturation_tables.dtable_2,
        )

        # -----------------------------------------------------------------------
        # graupel deposition and sublimation
        # -----------------------------------------------------------------------
        self._p_graupel_deposition_and_sublimation(
            t=gfdl_mp_v3_locals.t,
            dry_dp=gfdl_mp_v3_locals.dry_dp,
            density=gfdl_mp_v3_locals.density,
            density_factor=gfdl_mp_v3_locals.density_factor,
            vapor=gfdl_mp_v3_locals.mixing_ratio.vapor,
            ice=gfdl_mp_v3_locals.mixing_ratio.ice,
            liquid=gfdl_mp_v3_locals.mixing_ratio.liquid,
            graupel=gfdl_mp_v3_locals.mixing_ratio.graupel,
            rain=gfdl_mp_v3_locals.mixing_ratio.rain,
            snow=gfdl_mp_v3_locals.mixing_ratio.snow,
            cloud_fraction=gfdl_mp_v3_locals.cloud_fraction,
            cvm=self._subgrid_processes_locals.cvm,
            icpk=self._subgrid_processes_locals.icpk,
            lcpk=self._subgrid_processes_locals.lcpk,
            tcpk=self._subgrid_processes_locals.tcpk,
            tcp3=self._subgrid_processes_locals.tcp3,
            total_energy=self._subgrid_processes_locals.total_energy,
            mppdg=gfdl_mp_v3_locals.mppdg,
            mppsg=gfdl_mp_v3_locals.mppsg,
            CGSUB=self._mp_config.CGSUB,
            table_2=self._saturation_tables.table_2,
            dtable_2=self._saturation_tables.dtable_2,
        )
