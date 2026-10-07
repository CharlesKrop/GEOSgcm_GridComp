from f90nml import Namelist
from ndsl import QuantityFactory, StencilFactory
from ndsl.stencils.testing.savepoint import DataLoader
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM
from ndsl.stencils.testing.translate import TranslateFortranData2Py
from ndsl.utils import safe_assign_array
from ndsl.dsl.typing import Float, Int
from ndsl.dsl.gt4py import int32

from pyTurbulence.SHOCMF.edmf import define_surface_properties
from pyTurbulence.SHOCMF.config import SHOCMFConfiguration


class TranslateDefineSurfaceProperties(TranslateFortranData2Py):
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
            "nup2": {},
            "qt": {},
            "thv": {},
            "u": {},
            "v": {},
            "sigmaQT": {},
            "sigmaTH": {},
            "sigmaW": {},
            "wmax": {},
            "wmin": {},
            "wthv": {},
        }

        # FloatField Outputs
        self.out_vars = {
            "UPA": self.grid.compute_dict(),
            "UPQT": self.grid.compute_dict(),
            "UPTHV": self.grid.compute_dict(),
            "UPU": self.grid.compute_dict(),
            "UPV": self.grid.compute_dict(),
            "UPW": self.grid.compute_dict(),
        }

    def extra_data_load(self, data_loader: DataLoader):
        self.constants_edmf = data_loader.load("EDMF-constants")
        self.constants_shoc = data_loader.load("SHOCMF-constants")

    def compute(self, inputs):
        config = SHOCMFConfiguration(**self.constants_edmf,**self.constants_shoc)


        self.quantity_factory.add_data_dimensions(
            {
                "number_of_properties": int(config.NUP),
            }
        )

        _define_surface_properties = self.stencil_factory.from_dims_halo(
            func=define_surface_properties,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={"UPABUOYDEP":config.UPABUOYDEP}
        )

        # Inputs
        qt = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(qt.view[:, :,:], inputs["qt"])
        thv = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(thv.view[:, :, :], inputs["thv"])
        u = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(u.view[:, :, :], inputs["u"])
        v = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(v.view[:, :], inputs["v"])
        sigmaQT = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM], units="n/a")
        safe_assign_array(sigmaQT.view[:, :], inputs["sigmaQT"])
        sigmaTH = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM], units="n/a")
        safe_assign_array(sigmaTH.view[:, :], inputs["sigmaTH"])
        sigmaW = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM], units="n/a")
        safe_assign_array(sigmaW.view[:, :], inputs["sigmaW"])
        wmax = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM], units="n/a")
        safe_assign_array(wmax.view[:, :], inputs["wmax"])
        wmin = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM], units="n/a")
        safe_assign_array(wmin.view[:, :], inputs["wmin"])
        wthv = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM], units="n/a")
        safe_assign_array(wthv.view[:, :], inputs["wthv"])
        nup2 = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM], units="n/a", dtype=Int)
        safe_assign_array(nup2.view[:, :], inputs["nup2"])


        # # Outputs
        UPA = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_INTERFACE_DIM, "number_of_properties"], units="n/a")
        UPQT = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM,K_INTERFACE_DIM, "number_of_properties"], units="n/a")
        UPTHV = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM,K_INTERFACE_DIM, "number_of_properties"], units="n/a")
        UPU = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM,K_INTERFACE_DIM, "number_of_properties"], units="n/a")
        UPV = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM,K_INTERFACE_DIM, "number_of_properties"], units="n/a")
        UPW = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM,K_INTERFACE_DIM, "number_of_properties"], units="n/a")
   
        _define_surface_properties(
            nup2=nup2,
            qt=qt,
            sigmaQT=sigmaQT,
            sigmaTH=sigmaTH,
            sigmaW=sigmaW,
            thv=thv,
            u=u,
            v=v,
            wmax=wmax,
            wmin=wmin,
            wthv=wthv,
            UPA=UPA,
            UPQT=UPQT,
            UPTHV=UPTHV,
            UPU=UPU,
            UPV=UPV,
            UPW=UPW,
        )

        return {
            "UPA": UPA.view[:],
            "UPQT": UPQT.view[:],
            "UPTHV": UPTHV.view[:],
            "UPU": UPU.view[:],
            "UPV": UPV.view[:],
            "UPW": UPW.view[:],
        }
