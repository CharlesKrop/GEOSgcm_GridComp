from dataclasses import dataclass

from ndsl import Local, NDSLRuntime, QuantityFactory
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM
from ndsl.dsl.typing import Int


@dataclass
class SHOCMFLocals:
    zi: Local
    zl: Local
    tkh: Local
    prsl: Local
    u: Local
    v: Local
    omega: Local
    tabs: Local
    qwv: Local
    qcl: Local
    qci: Local
    cld_sgs: Local
    tke: Local
    wthv_sec: Local
    wthv_mf: Local
    qv: Local
    thv: Local
    qpl: Local
    qpi: Local
    total_water: Local
    prespot: Local
    hl: Local
    adzi: Local
    adzl: Local
    def2: Local
    RI: Local
    prnum: Local
    tkesbdiss: Local
    tkesbshear: Local
    tkesbbuoy: Local
    brunt2: Local
    smixt: Local
    smixt1: Local
    smixt2: Local
    smixt3: Local
    brunt: Local
    tscale1: Local
    brunt_edge: Local
    isotropy: Local
    bet: Local
    dtqw: Local
    dtqi: Local
    gamaz: Local

    @classmethod
    def make(cls, runtime: NDSLRuntime, quantity_factory: QuantityFactory):
        # K_INTERFACE Fields
        zi = runtime.make_local(quantity_factory, [I_DIM, J_DIM, K_INTERFACE_DIM])
        adzi = runtime.make_local(quantity_factory, [I_DIM, J_DIM, K_INTERFACE_DIM])
        RI = runtime.make_local(quantity_factory, [I_DIM, J_DIM, K_INTERFACE_DIM])
        prnum = runtime.make_local(quantity_factory, [I_DIM, J_DIM, K_INTERFACE_DIM])
        brunt_edge = runtime.make_local(quantity_factory, [I_DIM, J_DIM, K_INTERFACE_DIM])

        # FloatFields
        zl = runtime.make_local(quantity_factory, [I_DIM, J_DIM, K_DIM])
        tkh = runtime.make_local(quantity_factory, [I_DIM, J_DIM, K_DIM])
        prsl = runtime.make_local(quantity_factory, [I_DIM, J_DIM, K_DIM])
        u = runtime.make_local(quantity_factory, [I_DIM, J_DIM, K_DIM])
        v = runtime.make_local(quantity_factory, [I_DIM, J_DIM, K_DIM])
        omega = runtime.make_local(quantity_factory, [I_DIM, J_DIM, K_DIM])
        tabs = runtime.make_local(quantity_factory, [I_DIM, J_DIM, K_DIM])
        qwv = runtime.make_local(quantity_factory, [I_DIM, J_DIM, K_DIM])
        qcl = runtime.make_local(quantity_factory, [I_DIM, J_DIM, K_DIM])
        qci = runtime.make_local(quantity_factory, [I_DIM, J_DIM, K_DIM])
        cld_sgs = runtime.make_local(quantity_factory, [I_DIM, J_DIM, K_DIM])
        tke = runtime.make_local(quantity_factory, [I_DIM, J_DIM, K_DIM])
        wthv_sec = runtime.make_local(quantity_factory, [I_DIM, J_DIM, K_DIM])
        wthv_mf = runtime.make_local(quantity_factory, [I_DIM, J_DIM, K_DIM])
        qv = runtime.make_local(quantity_factory, [I_DIM, J_DIM, K_DIM])
        thv = runtime.make_local(quantity_factory, [I_DIM, J_DIM, K_DIM])
        qpl = runtime.make_local(quantity_factory, [I_DIM, J_DIM, K_DIM])
        qpi = runtime.make_local(quantity_factory, [I_DIM, J_DIM, K_DIM])
        total_water = runtime.make_local(quantity_factory, [I_DIM, J_DIM, K_DIM])
        prespot = runtime.make_local(quantity_factory, [I_DIM, J_DIM, K_DIM])
        hl = runtime.make_local(quantity_factory, [I_DIM, J_DIM, K_DIM])
        adzl = runtime.make_local(quantity_factory, [I_DIM, J_DIM, K_DIM])
        def2 = runtime.make_local(quantity_factory, [I_DIM, J_DIM, K_DIM])
        tkesbdiss = runtime.make_local(quantity_factory, [I_DIM, J_DIM, K_DIM])
        tkesbshear = runtime.make_local(quantity_factory, [I_DIM, J_DIM, K_DIM])
        tkesbbuoy = runtime.make_local(quantity_factory, [I_DIM, J_DIM, K_DIM])
        brunt2 = runtime.make_local(quantity_factory, [I_DIM, J_DIM, K_DIM])
        smixt = runtime.make_local(quantity_factory, [I_DIM, J_DIM, K_DIM])
        smixt1 = runtime.make_local(quantity_factory, [I_DIM, J_DIM, K_DIM])
        smixt2 = runtime.make_local(quantity_factory, [I_DIM, J_DIM, K_DIM])
        smixt3 = runtime.make_local(quantity_factory, [I_DIM, J_DIM, K_DIM])
        brunt = runtime.make_local(quantity_factory, [I_DIM, J_DIM, K_DIM])
        tscale1 = runtime.make_local(quantity_factory, [I_DIM, J_DIM, K_DIM])
        bet = runtime.make_local(quantity_factory, [I_DIM, J_DIM, K_DIM])
        dtqw = runtime.make_local(quantity_factory, [I_DIM, J_DIM, K_DIM])
        dtqi = runtime.make_local(quantity_factory, [I_DIM, J_DIM, K_DIM])
        gamaz = runtime.make_local(quantity_factory, [I_DIM, J_DIM, K_DIM])
        isotropy = runtime.make_local(quantity_factory, [I_DIM, J_DIM, K_DIM])

        return cls(
            zi=zi,
            zl=zl,
            tkh=tkh,
            prsl=prsl,
            u=u,
            v=v,
            tabs=tabs,
            qwv=qwv,
            qcl=qcl,
            qci=qci,
            cld_sgs=cld_sgs,
            tke=tke,
            wthv_sec=wthv_sec,
            wthv_mf=wthv_mf,
            qv=qv,
            thv=thv,
            qpl=qpl,
            qpi=qpi,
            total_water=total_water,
            prespot=prespot,
            hl=hl,
            adzi=adzi,
            adzl=adzl,
            def2=def2,
            RI=RI,
            prnum=prnum,
            tkesbdiss=tkesbdiss,
            tkesbshear=tkesbshear,
            tkesbbuoy=tkesbbuoy,
            brunt2=brunt2,
            smixt=smixt,
            smixt1=smixt1,
            smixt2=smixt2,
            smixt3=smixt3,
            brunt=brunt,
            tscale1=tscale1,
            brunt_edge=brunt_edge,
            isotropy=isotropy,
            bet=bet,
            dtqw=dtqw,
            dtqi=dtqi,
            omega=omega,
            gamaz=gamaz,

        )
