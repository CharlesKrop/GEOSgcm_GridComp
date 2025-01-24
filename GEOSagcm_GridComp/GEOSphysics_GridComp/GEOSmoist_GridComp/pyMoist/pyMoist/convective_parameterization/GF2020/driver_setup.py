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


def zero_outputs(
    wqt_dc: FloatField,
    cnv_mfc: FloatField,
    cnv_mf0: FloatField,
    cnv_prc3: FloatField,
    cnv_mfd: FloatField,
    cnv_dqcdt: FloatField,
    cnv_updf: FloatField,
    cnv_cvw: FloatField,
    cnv_qc: FloatField,
    entlam: FloatField,
    cnpcprate: FloatField,
    lightn_dens: FloatField,
    revsu: FloatField,
    prfil: FloatField,
    dqdt_gf: FloatField,
    dtdt_gf: FloatField,
    dudt_gf: FloatField,
    dvdt_gf: FloatField,
    sigma_deep: FloatField,
    sigma_mid: FloatField,
    mupdp: FloatField,
    mdndp: FloatField,
    mupsh: FloatField,
    mupmd: FloatField,
    mfdp: FloatField,
    mfsh: FloatField,
    mfmd: FloatField,
    errdp: FloatField,
    errsh: FloatField,
    errmd: FloatField,
    aa0: FloatField,
    aa1: FloatField,
    aa2: FloatField,
    aa3: FloatField,
    aa1_bl: FloatField,
    aa1_cin: FloatField,
    tau_bl: FloatField,
    tau_ec: FloatField,
):
    with computation(PARALLEL), interval(...):
        wqt_dc = 0.0
        cnv_mfc = 0.0
        cnv_mf0 = 0.0
        cnv_prc3 = 0.0
        cnv_mfd = 0.0
        cnv_dqcdt = 0.0
        cnv_updf = 0.0
        cnv_cvw = 0.0
        cnv_qc = 0.0
        entlam = 0.0
        cnpcprate = 0.0
        lightn_dens = 0.0
        revsu = 0.0
        prfil = 0.0

        dqdt_gf = 0.0
        dtdt_gf = 0.0
        dudt_gf = 0.0
        dvdt_gf = 0.0

        sigma_deep = 0.0
        sigma_mid = 0.0
        mupdp = 0.0
        mdndp = 0.0
        mupsh = 0.0
        mupmd = 0.0
        mfdp = 0.0
        mfsh = 0.0
        mfmd = 0.0
        errdp = 0.0
        errsh = 0.0
        errmd = 0.0
        aa0 = 0.0
        aa1 = 0.0
        aa2 = 0.0
        aa3 = 0.0
        aa1_bl = 0.0
        aa1_cin = 0.0
        tau_bl = 0.0
        tau_ec = 0.0


@gtscript.function
def setup_driver(
    # inputs
    dt: Float,
    t2m: FloatFieldIJ,
    evap: FloatFieldIJ,
    sh: FloatFieldIJ,
    ple: FloatField,  # +1 vertical level
    t: FloatField,
    q: FloatField,
    u: FloatField,
    v: FloatField,
    w: FloatField,
    phis: FloatFieldIJ,
    land_fraction: FloatFieldIJ,
    area: FloatFieldIJ,
    zle: FloatField,  # +1 vertical level
    plo: FloatField,
    zlo: FloatField,
    mass: FloatField,
    kh: FloatField,
    buoyancy: FloatField,
    kpblin: FloatFieldIJ,
    ple_dyn: FloatField,  # Z+1
    zle_dyn: FloatField,  # Z+1
    t_dyn: FloatField,
    qv_dyn: FloatField,
    u_dyn: FloatField,
    v_dyn: FloatField,
    dtdt_dyn: FloatField,
    dqvdt_dyn: FloatField,
    dtdt_bl: FloatField,
    dqdt_bl: FloatField,
    radsw: FloatField,
    radlw: FloatField,
    cnv_tr: FloatField,
    lons: FloatField,
    lats: FloatField,
    # outputs passed to driver but not handed back to the rest of the model
    temp2m: FloatFieldIJ,
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
    # outputs needed after the driver
    dz: FloatField,
    air_density: FloatField,
    temp: FloatField,
    pres: FloatField,
    rvap: FloatField,
    up: FloatField,
    vp: FloatField,
    wp: FloatField,
    zt: FloatField,
    zm: FloatField,
    dm: FloatField,
    khloc: FloatField,
    curr_rvap: FloatField,
    # outputs passed to the driver, then back to the rest of the model
    lightn_dens: FloatFieldIJ,
    ec3d: FloatField,
    # outputs passed back to the rest of the model
    entr3d: FloatField,
):
    from __externals__ import (
        kend,
        USE_SCALE_DEP,
        N_TRACERS,
        GF_ENV_SETTING,
        ENTRVERSION,
        CONVECTION_TRACER,
        USE_TRACER_TRANSP,
        AUTOCONV,
    )  # if pep8 states that constants should be capitals why is kend not capitalized?

    with computation(FORWARD), interval(0, 1):
        # as moist is called before surface, at the 1st time step all arrays
        # from surface are zero
        if max(t2m) < 1.0e-6:
            temp2m = t.at(k=kend)  # Kelvin
        else:
            temp2m = t2m  # or TA(:,:) # Kelvin

        # sensible heat flux (sh) comes in W m-2, below it is converted to K m s-1
        sh_converted = sh / (
            1004.0
            * ple.at(k=kend)
            / (287.04 * t.at(k=kend) * (1.0 + 0.608 * q.at(k=kend)))
        )  # K m s-1
        # topography height  (m)
        topo_height = phis / global_constants.MAPL_GRAV
        # land/ocean fraction: land if < 1 ,ocean if = 1
        ocean_fraction = 1.0 - land_fraction

        # grid length for the scale awareness
        dx2d = sqrt(area)  # meters

        # pbl heigth index
        if kpblin != 0.0:
            pbl_top_level = kpblin
        else:
            pbl_top_level = kend

    with computation(PARALLEL), interval(...):
        if GF_ENV_SETTING == "CURRENT":
            # 1st setting: enviromental state is the one already modified by dyn + physics
            dz = -(zle[0, 0, 1] - zle)
            air_density = plo / (287.04 * t * (1.0 + 0.608 * q))

            temp = t
            pres = plo  # Pa
            rvap = q
            up = u  # already @ A-grid (m/s)
            vp = v  # already @ A-grid (m/s)
            wp = w  # m/s
            zt = zlo  # mid -layer level
            zm = zle  # edge-layer level
            dm = mass
            khloc = kh
            curr_rvap = q  # current rvap

            if ENTRVERSION == 0:
                # eq 6 of https://doi.org/10.1029/2021JD034881
                ec3d = (
                    0.71 * max(0.5, w) ** -1.17 * max(0.1, buoyancy) ** -0.36
                )  # not flipped in fortran, may have to be flipped here
            else:
                ec3d = 1.0  # not flipped in fortran, may have to be flipped here
            entr3d = ec3d

    with computation(PARALLEL), interval(...):
        if GF_ENV_SETTING == "CURRENT":
            # Grid and sub-grid scale forcings for convection
            gsf_t = 0.0
            gsf_q = 0.0
            sgsf_t = 0.0
            sgsf_q = 0.0
            advf_t = 0.0

    with computation(PARALLEL), interval(...):
        if GF_ENV_SETTING == "DYNAMICS":
            # 2nd setting: environmental state is that one before any tendency
            # is applied (i.e, at begin of each time step).
            # Get back the model state, heights and others variables at time N
            # (or at the beggining of current time step)
            # In physics, the state vars (T,U,V,PLE) are untouched and represent the
            # model state after dynamics phase 1. But, "Q" is modified by physics, so
            # depending on what was called before this subroutine, "Q" may be already
            # changed from what it was just after dynamics phase 1. To solve this issue,
            # "Q" just after dynamics is saved in the var named "QV_DYN_IN" in "GEOS_AgcmGridComp.F90".
            mass_dyn = ple_dyn[0, 0, 1] - ple_dyn * (1.0 / global_constants.MAPL_GRAV)

            plo_dyn = 0.5 * (ple_dyn + ple_dyn[0, 0, 1])
            pke_dyn = (ple_dyn / global_constants.MAPL_P00) ** (
                global_constants.MAPL_RGAS / global_constants.MAPL_CP
            )
            pk_dyn = (plo / global_constants.MAPL_P00) ** (
                global_constants.MAPL_RGAS / global_constants.MAPL_CP
            )

    with computation(FORWARD), interval(-1, None):
        if GF_ENV_SETTING == "DYNAMICS":
            zle_dyn[0, 0, 1] = 0.0

    with computation(BACKWARD), interval(...):
        if GF_ENV_SETTING == "DYNAMICS":
            zle_dyn = (t_dyn / pk_dyn) * (1.0 + global_constants.MAPL_VIREPS * qv_dyn)
            zlo_dyn = (
                zle_dyn[0, 0, 1]
                + (global_constants.MAPL_CP / global_constants.MAPL_GRAV)
                * (pke_dyn[0, 0, 1] - pk_dyn)
                * zle_dyn
            )
            zle_dyn = (
                zlo_dyn
                + (global_constants.MAPL_CP / global_constants.MAPL_GRAV)
                * (pk_dyn - pke_dyn)
                * zle_dyn
            )

    with computation(PARALLEL), interval(...):
        if GF_ENV_SETTING == "DYNAMICS":
            dz = -(zle_dyn[0, 0, 1] - zle_dyn)
            air_density = plo / (287.04 * t_dyn * (1.0 + 0.608 * qv_dyn))

            temp = t_dyn  # (K)
            pres = plo_dyn  # (Pa) @ mid-layer level
            rvap = qv_dyn  # water vapor mix ratio
            up = u_dyn  # already @ A-grid (m/s)
            vp = v_dyn  # already @ A-grid (m/s)
            wp = w  # (m/s)
            zt = zlo_dyn  # mid -layer level (m)
            zm = zle_dyn  # edge-layer level (m)
            dm = mass_dyn
            khloc = kh
            curr_rvap = q  # current rvap (dyn+phys)

            if ENTRVERSION == 0:
                # eq 6 of https://doi.org/10.1029/2021JD034881
                ec3d = (
                    0.71 * max(0.5, w) ** -1.17 * max(0.1, buoyancy) ** -0.36
                )  # not flipped in fortran, may have to be flipped here
            else:
                ec3d = 1.0  # not flipped in fortran, may have to be flipped here
            entr3d = ec3d

            # Grid and sub-grid scale forcings for convection
            gsf_t = dtdt_dyn + radsw + radlw
            gsf_q = dqvdt_dyn
            sgsf_t = dtdt_bl
            sgsf_q = dqdt_bl
            advf_t = dtdt_dyn

        if CONVECTION_TRACER == 1:
            buoyancy_excess = cnv_tr
        else:
            buoyancy_excess = 0

    # reset a bunch of stuff to ensure there is nothing lingering from the previous call
    with computation(FORWARD), interval(-1, None):
        ztexec = 0.0
        zqexec = 0.0
        last_ierr = -999
        fixout_qv = 1.0

        conprr = 0.0
        lightn_dens = 0.0

        if USE_TRACER_TRANSP == 1:
            out_chem_1_deep = 0.0
            out_chem_2_deep = 0.0
            out_chem_1_mid = 0.0
            out_chem_2_mid = 0.0
            out_chem_1_shal = 0.0
            out_chem_2_shal = 0.0

        topo_height_no_neg = max(0.0, topo_height)
        lons_degrees = lons * 180.0 / 3.14159
        lats_degrees = lats * 180.0 / 3.14159

        if AUTOCONV == 2:
            ccn = max(100.0, (370.37 * 0.11) ** 1.555)
        else:
            ccn = 100.0

    with computation(PARALLEL), interval(...):
        revsu_gf = 0.0
        prfil_gf = 0.0
        temp_tendqv = 0.0
        # tendencies (w/ maxiens)
        outt_deep = 0.0
        outt_mid = 0.0
        outt_shal = 0.0
        outu_deep = 0.0
        outu_mid = 0.0
        outu_shal = 0.0
        outv_deep = 0.0
        outv_mid = 0.0
        outv_shal = 0.0
        outq_deep = 0.0
        outq_mid = 0.0
        outq_shal = 0.0
        outqc_deep = 0.0
        outqc_mid = 0.0
        outqc_shal = 0.0
        outnice_deep = 0.0
        outnice_mid = 0.0
        outnice_shal = 0.0
        outnliq_deep = 0.0
        outnliq_mid = 0.0
        outnliq_shal = 0.0
        outbuoy_deep = 0.0
        outbuoy_mid = 0.0
        outbuoy_shal = 0.0
        omeg_deep = 0.0
        omeg_mid = 0.0
        omeg_shal = 0.0

    with computation(PARALLEL), interval(1, None):
        # heigths, current pressure, temp and water vapor mix ratio
        height = zt + topo_height
        pres_mb = pres * 1.0e-2  # mbar
        temp_old = temp
        qv_old = rvap  # @ begin of the timestep
        qv_curr = (
            curr_rvap  # current (after dynamics + physical processes called before GF)
        )

        # air density, TKE and cloud liq water mixing ratio
        rhoi = 1.0e2 * pres_mb / (287.04 * temp_old * (1.0 + 0.608 * qv_old))
        tkeg = convection_constants.tkmin
        rcpg = 0.0

        # wind velocities
        omeg_deep = -global_constants.MAPL_GRAV * rhoi * w
        omeg_mid = -global_constants.MAPL_GRAV * rhoi * w
        omeg_shal = -global_constants.MAPL_GRAV * rhoi * w
        # temp/water vapor modified only by advection
        temp_new_ADV = temp_old + advf_t * dt
        qv_new_ADV = qv_old + gsf_q * dt

    with computation(FORWARD), interval(-1, None):
        pbl_height = height.at(K=pbl_top_level) - topo_height

        # get execess T and Q for source air parcels
        zrho = pres.at(K=kend + 1) / (
            287.04 * (temp_old.at(K=kend) * (1.0 + 0.608 * qv_old.at(k=kend)))
        )
        # sensible and latent sfc fluxes for the heat-engine closure
        sensible_heat_sfc_flux = zrho * convection_constants.CP * sh_converted  # W/m^2
        latent_heat_sfc_flux = zrho * convection_constants.XLV * evap  # W/m^2
        # local le and h fluxes for W*
        pahfs = -sh_converted * zrho * 1004.64  # W/m^2
        pqhfl = -evap  # kg/m^2/s
        # buoyancy flux (h+le)
        zkhvfl = (
            pahfs / 1004.64 + 0.608 * temp_old.at(K=kend) * pqhfl
        ) / zrho  # K m s-1
        # depth of 1st model layer
        # (zo(1)-top is ~ 1/2 of the depth of 1st model layer, => mult by 2)
        pgeoh = (
            2.0 * (height.at(K=kend) - topo_height) * global_constants.MAPL_GRAV
        )  # m+2 s-2
        # convective-scale velocity w*
        # in the future, change 0.001 by ustar^3
        zws = max(
            0.0, 0.001 - 1.5 * 0.41 * zkhvfl * pgeoh / temp_old.at(K=kend)
        )  # m+3 s-3

        if zws > 0:  # tiny(pgeoh): NOTE need better solution
            # convective-scale velocity w*
            zws = 1.2 * zws**0.3333
            # temperature excess
            ztexec = max(0.0, -1.5 * pahfs / (zrho * zws * 1004.64))  # K
            # moisture  excess
            zqexec = max(0.0, -1.5 * pqhfl / (zrho * zws))  # kg kg-1

        # zws for shallow convection closure (Grant 2001)
        # depth of the pbl
        pgeoh = pbl_top_level * global_constants.MAPL_GRAV
        # convective-scale velocity W* (m/s)
        zws = max(0.0, 0.001 - 1.5 * 0.41 * zkhvfl * pgeoh / temp_old.at(K=kend))
        zws = 1.2 * zws**0.3333

    # NOTE NEED GOOD SOLUTION FOR THIS
    # IF(USE_TRACER_TRANSP==1) THEN
    #   DO k=kts,kte
    #     DO i=its,itf
    #       kr=k !+1
    #       !- atmos composition
    #       DO ispc=1,mtp
    #          se_chem(ispc,i,k) = max(CNV_Tracers(ispc)%Q(i,j,flip(kr)),mintracer)
    #       ENDDO
    #     ENDDO
    #   ENDDO
    #  ENDIF
