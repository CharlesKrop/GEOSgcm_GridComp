import numpy as np
import gt4py.cartesian.gtscript as gtscript
from gt4py.cartesian.gtscript import (
    computation,
    interval,
    PARALLEL,
    FORWARD,
    BACKWARD,
    THIS_K,
    sqrt,
    round,
    max,
)

from ndsl import QuantityFactory, StencilFactory, orchestrate
from ndsl.constants import X_DIM, Y_DIM, Z_DIM, Z_INTERFACE_DIM
from ndsl.dsl.typing import Float, FloatField, FloatFieldIJ, Int
import pyMoist.constants as global_constants
import pyMoist.convective_parameterization.GF2020.constants as GF2020_constants
import pyMoist.convective_parameterization.shared_constants as convection_constants

# used only during construction
from pyMoist.convective_parameterization.GF2020.GF2020_flags import GF2020_flags


def set_excess(
    USE_EXCESS: Int,
    ocean_fraction: FloatFieldIJ,
    t_excess: FloatFieldIJ,
    q_excess: FloatFieldIJ,
):
    with computation(FORWARD), interval(0,1):
        # set minimum/max for excess of T and Q
        # NOTE clean up paths where  nothing happens, proabbly turn into a comment or something
        if USE_EXCESS == 0:  # no excess
            t_excess = 0.0
            q_excess = 0.0
        elif USE_EXCESS == 1:  # nothing happens
            t_excess = t_excess
            q_excess = q_excess
        elif USE_EXCESS == 2:  # recompute
            t_excess = min(0.5, max(0.2, t_excess))  # Kelvin
            q_excess = min(5.0e-4, max(1.0e-4, q_excess))  # kg kg^-1
        else:
            if ocean_fraction(i) > 0.98:  # ocean, so recomput
                t_excess = min(0.5, max(0.2, t_excess))  # Kelvin
                q_excess = min(5.0e-4, max(1.0e-4, q_excess))  # kg kg^-1
            else:  # land, so do nothing
                t_excess = t_excess
                q_excess = q_excess


def cumulus_parameterization_core(
    temp2m: FloatFieldIJ,
    temp_old: FloatField,
    temp_new: FloatField,
    temp_new_bl: FloatField,
    temp_new_adv: FloatField,
    t_excess: FloatField,
    qv_old: FloatField,
    qv_new: FloatField,
    qv_new_bl: FloatField,
    qv_new_adv: FloatField,
    q_excess: FloatField,
    ocean_fraction: FloatFieldIJ,
    dx2d: FloatFieldIJ,
    pbl_top_level: FloatFieldIJ,
    # forcings
    buoyancy_excess: FloatField,
    gsf_t: FloatField,
    gsf_q: FloatField,
    sgsf_t: FloatField,
    sgsf_q: FloatField,
    advf_t: FloatField,
    # end forcings
    ztexec: FloatFieldIJ,
    zqexec: FloatFieldIJ,
    zws: FloatFieldIJ,
    last_ierr: FloatFieldIJ,
    fixout_qv: FloatFieldIJ,
    conprr: FloatFieldIJ,
    out_chem_1_deep: FloatFieldIJ,
    out_chem_2_deep: FloatFieldIJ,
    out_chem_1_mid: FloatFieldIJ,
    out_chem_2_mid: FloatFieldIJ,
    out_chem_1_shal: FloatFieldIJ,
    out_chem_2_shal: FloatFieldIJ,
    topo_height_no_neg: FloatFieldIJ,
    lons_degrees: FloatFieldIJ,
    lats_degrees: FloatFieldIJ,
    revsu_gf: FloatField,
    prfil_gf_2d: FloatField,
    temp_tendqv: FloatField,
    outt_deep: FloatField,
    outt_mid: FloatField,
    outt_shal: FloatField,
    outu_deep: FloatField,
    outu_mid: FloatField,
    outu_shal: FloatField,
    outv_deep: FloatField,
    outv_mid: FloatField,
    outv_shal: FloatField,
    outq_deep: FloatField,
    outq_mid: FloatField,
    outq_shal: FloatField,
    outqc_deep: FloatField,
    outqc_mid: FloatField,
    outqc_shal: FloatField,
    outnice_deep: FloatField,
    outnice_mid: FloatField,
    outnice_shal: FloatField,
    outnliq_deep: FloatField,
    outnliq_mid: FloatField,
    outnliq_shal: FloatField,
    outbuoy_deep: FloatField,
    outbuoy_mid: FloatField,
    outbuoy_shal: FloatField,
    omeg_deep: FloatField,
    omeg_mid: FloatField,
    omeg_shal: FloatField,
    ccn: FloatField,
    sensible_heat_sfc_flux: FloatFieldIJ,
    latent_heat_sfc_flux: FloatFieldIJ,
    # outputs passed back to the rest of the model
    lightn_dens: FloatFieldIJ,
    ec3d: FloatField,
    # plume dependent constants
    HEI_DOWN_LAND: Float,
    HEI_DOWN_OCEAN: Float,
    HEI_UPDF_LAND: Float,
    HEI_UPDF_OCEAN: Float,
    MIN_EDT_LAND: Float,
    MIN_EDT_OCEAN: Float,
    MAX_EDT_LAND: Float,
    MAX_EDT_OCEAN: Float,
    FADJ_MASSFLX: Float,
    USE_EXCESS: Int,
    AVE_LAYER: Float,
):
