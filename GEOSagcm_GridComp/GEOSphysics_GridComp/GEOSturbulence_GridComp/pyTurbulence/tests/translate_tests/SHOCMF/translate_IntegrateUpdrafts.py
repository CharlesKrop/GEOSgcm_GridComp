from f90nml import Namelist
from ndsl import QuantityFactory, StencilFactory
from ndsl.stencils.testing.savepoint import DataLoader
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM
from ndsl.stencils.testing.translate import TranslateFortranData2Py
from ndsl.utils import safe_assign_array
from ndsl.dsl.typing import Int

from pyTurbulence.SHOCMF.edmf import integrate_updrafts
from pyTurbulence.SHOCMF.config import SHOCMFConfiguration
from pyMoist.saturation_tables import GlobalTable_saturation_tables, get_saturation_vapor_pressure_table


class TranslateIntegrateUpdrafts(TranslateFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, stencil_factory)
        self.stencil_factory = stencil_factory
        self.quantity_factory = grid.quantity_factory

        # Inputs
        self.in_vars["data_vars"] = {
            "ENT": {},
            "UPA": {},
            "UPQT": {},
            "UPTHL": {},
            "UPTHV": {},
            "UPU": {},
            "UPW": {},
            "UPV": {},
            "exfh": {},
            "nup2": {},
            "qr": {},
            "qs": {},
            "qt": {},
            "thl": {},
            "thv": {},
            "u": {},
            "v": {},
            "wcfac": {},
            "zw": {},
            "p": {},
        }

        # Outputs
        self.out_vars = {
            "UPA": self.grid.compute_dict(),
            "UPQI": self.grid.compute_dict(),
            "UPQL": self.grid.compute_dict(),
            "UPQT": self.grid.compute_dict(),
            "UPTHL": self.grid.compute_dict(),
            "UPTHV": self.grid.compute_dict(),
            "UPU": self.grid.compute_dict(),
            "UPV": self.grid.compute_dict(),
            "UPW": self.grid.compute_dict(),
            "qr": self.grid.compute_dict(),
            "qs": self.grid.compute_dict(),
            "testvar": self.grid.compute_dict(),
        }

    def extra_data_load(self, data_loader: DataLoader):
        self.constants_edmf = data_loader.load("EDMF-constants")
        self.constants_shoc = data_loader.load("SHOCMF-constants")

    def compute(self, inputs):
        config = SHOCMFConfiguration(
            **self.constants_edmf,
            **self.constants_shoc
        )

        self.quantity_factory.add_data_dimensions(
            {
                "number_of_properties": int(config.NUP),
            }
        )

        _integrate_updrafts = self.stencil_factory.from_dims_halo(
            func=integrate_updrafts,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={"EntUFac":config.EntUFac, "ICERAMP":config.ICERAMP,"PRCPCRIT":config.PRCPCRIT,"WB":config.WB,"WC":config.WC,"WA":config.WA}
        )

        # Inputs
        ENT = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[
                I_DIM,
                J_DIM,
                K_DIM,
                "number_of_properties",
            ],
            units="n/a",
        )
        safe_assign_array(ENT.view[:], inputs["ENT"])

        UPA = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[
                I_DIM,
                J_DIM,
                K_INTERFACE_DIM,
                "number_of_properties",
            ],
            units="n/a",
        )
        safe_assign_array(UPA.view[:], inputs["UPA"])
        UPV = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[
                I_DIM,
                J_DIM,
                K_INTERFACE_DIM,
                "number_of_properties",
            ],
            units="n/a",
        )
        safe_assign_array(UPV.view[:], inputs["UPV"])

        UPQT = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[
                I_DIM,
                J_DIM,
                K_INTERFACE_DIM,
                "number_of_properties",
            ],
            units="n/a",
        )
        safe_assign_array(UPQT.view[:], inputs["UPQT"])

        UPTHL = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[
                I_DIM,
                J_DIM,
                K_INTERFACE_DIM,
                "number_of_properties",
            ],
            units="n/a",
        )
        safe_assign_array(UPTHL.view[:], inputs["UPTHL"])

        UPTHV = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[
                I_DIM,
                J_DIM,
                K_INTERFACE_DIM,
                "number_of_properties",
            ],
            units="n/a",
        )
        safe_assign_array(UPTHV.view[:], inputs["UPTHV"])

        UPU = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[
                I_DIM,
                J_DIM,
                K_INTERFACE_DIM,
                "number_of_properties",
            ],
            units="n/a",
        )
        safe_assign_array(UPU.view[:], inputs["UPU"])

        UPW = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[
                I_DIM,
                J_DIM,
                K_INTERFACE_DIM,
                "number_of_properties",
            ],
            units="n/a",
        )
        safe_assign_array(UPW.view[:], inputs["UPW"])

        exfh = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM, K_INTERFACE_DIM],
            units="n/a",
        )
        safe_assign_array(exfh.view[:, :], inputs["exfh"])

        nup2 = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM],
            units="n/a",
            dtype=Int,
        )
        safe_assign_array(nup2.view[:, :], inputs["nup2"])

        p = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM, K_INTERFACE_DIM],
            units="n/a",
        )
        safe_assign_array(p.view[:, :], inputs["p"])

        qr = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM,K_DIM],
            units="n/a",
        )
        safe_assign_array(qr.view[:, :], inputs["qr"])

        qs = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM,K_DIM],
            units="n/a",
        )
        safe_assign_array(qs.view[:, :], inputs["qs"])

        qt = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM,K_DIM],
            units="n/a",
        )
        safe_assign_array(qt.view[:, :], inputs["qt"])

        thl = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM, K_DIM],
            units="n/a",
        )
        safe_assign_array(thl.view[:], inputs["thl"])

        thv = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM, K_DIM],
            units="n/a",
        )
        safe_assign_array(thv.view[:], inputs["thv"])

        u = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM, K_DIM],
            units="n/a",
        )
        safe_assign_array(u.view[:], inputs["u"])

        v = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM, K_DIM],
            units="n/a",
        )
        safe_assign_array(v.view[:], inputs["v"])

        wcfac = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM, K_DIM],
            units="n/a",
        )
        safe_assign_array(wcfac.view[:], inputs["wcfac"])

        zw = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM, K_INTERFACE_DIM],
            units="n/a",
        )
        safe_assign_array(zw.view[:], inputs["zw"])


        saturation_vapor_pressure_table = get_saturation_vapor_pressure_table(self.stencil_factory)
        self.esx =  saturation_vapor_pressure_table.esx

 
        UPQI = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[
                I_DIM,
                J_DIM,
                K_INTERFACE_DIM,
                "number_of_properties",
            ],
            units="n/a",
        )
        UPQL = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[
                I_DIM,
                J_DIM,
                K_INTERFACE_DIM,
                "number_of_properties",
            ],
            units="n/a",
        )
        testvar = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[
                I_DIM,
                J_DIM,
            ],
            units="n/a",
        )

        _integrate_updrafts(
            ENT=ENT,
            UPA=UPA,
            UPQT=UPQT,
            UPTHL=UPTHL,
            UPTHV=UPTHV,
            UPU=UPU,
            UPW=UPW,
            UPV=UPV,
            UPQI=UPQI,
            UPQL=UPQL,
            exfh=exfh,
            nup2=nup2,
            qr=qr,
            qs=qs,
            qt=qt,
            thl=thl,
            thv=thv,
            u=u,
            v=v,
            wcfac=wcfac,
            zw=zw,
            esx=self.esx,
            p=p,
            testvar=testvar,
        )

        return {
            "UPA": UPA.view[:],
            "UPQI": UPQI.view[:],
            "UPQL": UPQL.view[:],
            "UPQT": UPQT.view[:],
            "UPTHL": UPTHL.view[:],
            "UPTHV": UPTHV.view[:],
            "UPU": UPU.view[:],
            "UPV": UPV.view[:],
            "UPW": UPW.view[:],
            "qr": qr.view[:],
            "qs": qs.view[:],
            "testvar": testvar.view[:],
        }