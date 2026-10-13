from ndsl.dsl.gt4py import K, erfc, exp, float32, float64
from ndsl.dsl.gt4py import function as gtfunction
from ndsl.dsl.gt4py import log, sqrt
from ndsl.dsl.typing import Float, FloatField, Int

import pyMoist.constants as constants
from pyMoist.saturation_tables import GlobalTable_saturation_tables, saturation_specific_humidity
from pyMoist.shared.incloud_processes import ice_fraction


@gtfunction
def water_f(
    T: Float,
    iceramp: Float,
):
    Tmax=0.
    Tmin=Tmax-abs(iceramp)
    Tw=T-constants.MAPL_CELSIUS_TO_KELVIN

    if (Tw>Tmax):
        water_f=1.
    elif (Tw<Tmin):
        water_f=0.
    else:
        water_f=(Tw-Tmin)/(Tmax-Tmin)

    return water_f

@gtfunction
def get_alhl(
    T: Float,
    iceramp: Float,
):
    wf = water_f(T,iceramp)
    get_alhl=wf*constants.MAPL_ALHL+(1.-wf)*constants.MAPL_ALHS

    return get_alhl

@gtfunction
def condensation_edmfA(
    THV: Float,
    QT: Float,
    P: Float,
    ice_ramp: Float,
    esx: GlobalTable_saturation_tables,
):
   
    niter=20
    diff=2.e-5

    EXN=(P/constants.MAPL_P00)**constants.MAPL_KAPPA
    QC=0.

    i=1
    while i <= niter:
        T=EXN*THV/(1.+constants.MAPL_VIREPS*(QT-QC)-QC)
        ptmp = P / 100.0
        QS, _=saturation_specific_humidity(T,ptmp * 100.0,esx)
        QCOLD=QC
        QC=max(0.5*QC+0.5*(QT-QS),0.)
        if abs(QC-QCOLD)<diff:
            i = niter + 1
        i = i + 1
    
    alhl=get_alhl(T,ice_ramp)
    THL=(T-QC*alhl/constants.MAPL_CP)/EXN
    wf=water_f(T,ice_ramp)
    QL=QC*wf
    QI=QC*(1.-wf)

    return THL, QL, QI

@gtfunction
def condensation_edmf(
    QT: Float,
    THL: Float,
    P: Float,
    ice_ramp: Float,
    esx: GlobalTable_saturation_tables,
):
    niter=20
    diff=2.e-5

    EXN=(P/constants.MAPL_P00)**constants.MAPL_KAPPA
    QC=0.

    T=EXN*THL
    i = 1
    while i <= niter:
        alhl1 = get_alhl(T,ice_ramp)
        T=EXN*THL+alhl1/constants.MAPL_CP*QC
        QS, _=saturation_specific_humidity(T,P,esx)
        QCOLD=QC
        QC=max(0.5*QC+0.5*(QT-QS),0.)
        if abs(QC-QCOLD)<diff:
            i = niter + 1
        i = i + 1

    alhl2=get_alhl(T,ice_ramp)
    T=EXN*THL+alhl2/constants.MAPL_CP*QC
    QS, _=saturation_specific_humidity(T,P,esx)
    QC=max(QT-QS,0.)
    alhl3=get_alhl(T,ice_ramp)
    THV=(THL+alhl3/constants.MAPL_CP*QC/EXN)*(1.+constants.MAPL_VIREPS*(QT-QC)-QC)
    wf=water_f(T,ice_ramp)

    return THV,QC,wf