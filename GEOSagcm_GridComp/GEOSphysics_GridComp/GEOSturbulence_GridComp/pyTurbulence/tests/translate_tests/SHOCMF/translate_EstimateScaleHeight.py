from f90nml import Namelist
from ndsl import QuantityFactory, StencilFactory
from ndsl.stencils.testing.savepoint import DataLoader
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM
from ndsl.stencils.testing.translate import TranslateFortranData2Py
from ndsl.utils import safe_assign_array

from pyTurbulence.SHOCMF.edmf import estimate_scale_height
from pyTurbulence.SHOCMF.config import SHOCMFConfiguration
from pyMoist.saturation_tables import GlobalTable_saturation_tables, get_saturation_vapor_pressure_table

class TranslateEstimateScaleHeight(TranslateFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, stencil_factory)
        self.stencil_factory = stencil_factory
        self.quantity_factory = grid.quantity_factory

        # FloatField Inputs
        self.in_vars["data_vars"] = {
            "phis": {},
            "qv3": {},
            "t3": {},
            "tmp": {},
            "wqt": {},
            "wthv": {},
            "zlo3": {},
            "pw3": {},
            "zw3_ESH": {},
        }

        # FloatField Outputs
        self.out_vars = {
            "L0_ESH": self.grid.compute_dict(),
            "nup2": self.grid.compute_dict(),
            "pmid": self.grid.compute_dict(),
            "ztop": self.grid.compute_dict(),
        }

    def extra_data_load(self, data_loader: DataLoader):
        self.constants_edmf = data_loader.load("EDMF-constants")
        self.constants_shoc = data_loader.load("SHOCMF-constants")

    def compute(self, inputs):
        config = SHOCMFConfiguration(**self.constants_edmf,**self.constants_shoc)

        _estimate_scale_height = self.stencil_factory.from_dims_halo(
            func=estimate_scale_height,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={"NUP":config.NUP, "ET":config.ET,"L0_EDMF":config.L0_EDMF,"L0fac":config.L0fac},
        )

        # Inputs
        phis = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM], units="n/a")
        safe_assign_array(phis.view[:, :], inputs["phis"])
        qv3 = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(qv3.view[:, :, :], inputs["qv3"])
        t3 = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(t3.view[:, :, :], inputs["t3"])
        tmp = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM], units="n/a")
        safe_assign_array(tmp.view[:, :], inputs["tmp"])
        wqt = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM], units="n/a")
        safe_assign_array(wqt.view[:, :], inputs["wqt"])
        wthv = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM], units="n/a")
        safe_assign_array(wthv.view[:, :], inputs["wthv"])
        zlo3 = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(zlo3.view[:, :, :], inputs["zlo3"])
        pw3 = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_INTERFACE_DIM], units="n/a")
        safe_assign_array(pw3.view[:, :, :], inputs["pw3"])
        zw3 = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_INTERFACE_DIM], units="n/a")
        safe_assign_array(zw3.view[:, :, :], inputs["zw3_ESH"])



        # Outputs
        L0 = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM], units="n/a")
        nup2 = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM], units="n/a")
        pmid = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        ztop = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM], units="n/a")
        UPW= QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        UPTHL= QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        UPTHV= QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        UPQT= QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        UPA= QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        UPU= QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        UPV= QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        UPQI= QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        UPQL= QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        ENT= QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        QR= QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        QS= QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        z= QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")

        saturation_vapor_pressure_table = get_saturation_vapor_pressure_table(self.stencil_factory)
        self.esx =  saturation_vapor_pressure_table.esx
   
        _estimate_scale_height(
            wthv=wthv,
            tmp=tmp,
            phis=phis,
            UPW=UPW,
            UPTHL=UPTHL,
            UPTHV=UPTHV,
            UPQT=UPQT,
            UPA=UPA,
            UPU=UPU,
            UPV=UPV,
            UPQI=UPQI,
            UPQL=UPQL,
            ENT=ENT,
            QR=QR,
            QS=QS,
            pw3=pw3,
            t3=t3,
            wqt=wqt,
            qv3=qv3,
            zlo3=zlo3,
            nup2=nup2,
            L0=L0,
            pmid=pmid,
            esx=self.esx,
            ztop=ztop,
            zw3=zw3,
            z=z,
        )

        return {
            "L0_ESH": L0.view[:],
            "nup2": nup2.view[:],
            "pmid": pmid.view[:],
            "ztop": ztop.view[:],
            "z_test": z.view[:],
        }
