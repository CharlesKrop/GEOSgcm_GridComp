from f90nml import Namelist
from ndsl import QuantityFactory, StencilFactory
from ndsl.stencils.testing.savepoint import DataLoader
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM
from ndsl.stencils.testing.translate import TranslateFortranData2Py
from ndsl.utils import safe_assign_array

from pyTurbulence.SHOCMF.edmf import flip_variables
from pyTurbulence.SHOCMF.config import SHOCMFConfiguration


class TranslateFlipVariables(TranslateFortranData2Py):
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
            "qi3": {},
            "ql3": {},
            "qv3": {},
            "thl3": {},
            "thv3": {},
            "u3": {},
            "v3": {},
            "zlo3": {},
            "zw3_FV": {},
            "ztop": {},
            "wthv": {},
            "tmp": {},
            "phis": {},
            "rhoe3": {},
            "pw3_FV": {},
        }

        # FloatField Outputs
        self.out_vars = {
            "qi": self.grid.compute_dict(),
            "qii": self.grid.compute_dict(),
            "ql": self.grid.compute_dict(),
            "qli": self.grid.compute_dict(),
            "qv": self.grid.compute_dict(),
            "qvi": self.grid.compute_dict(),
            "thl": self.grid.compute_dict(),
            "thli": self.grid.compute_dict(),
            "thv": self.grid.compute_dict(),
            "u": self.grid.compute_dict(),
            "ui": self.grid.compute_dict(),
            "v": self.grid.compute_dict(),
            "vi": self.grid.compute_dict(),
            "zlo": self.grid.compute_dict(),
            "qt": self.grid.compute_dict(),
            "qti": self.grid.compute_dict(),
            "rhoe": self.grid.compute_dict(),
            "zw": self.grid.compute_dict(),
            "p": self.grid.compute_dict(),
            "dp": self.grid.compute_dict(),
        }

    def extra_data_load(self, data_loader: DataLoader):
        self.constants_edmf = data_loader.load("EDMF-constants")
        self.constants_shoc = data_loader.load("SHOCMF-constants")

    def compute(self, inputs):
        config = SHOCMFConfiguration(**self.constants_edmf,**self.constants_shoc)

        _flip_variables = self.stencil_factory.from_dims_halo(
            func=flip_variables,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={"DISCRETE":config.DISCRETE}
        )

        # Inputs
        ztop = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM],
            units="n/a",
        )
        safe_assign_array(ztop.view[:, :], inputs["ztop"])

        wthv = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM],
            units="n/a",
        )
        safe_assign_array(wthv.view[:, :], inputs["wthv"])

        phis = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM],
            units="n/a",
        )
        safe_assign_array(phis.view[:, :], inputs["phis"])

        tmp = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM],
            units="n/a",
        )
        safe_assign_array(tmp.view[:, :], inputs["tmp"])

        qi3 = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM, K_DIM],
            units="n/a",
        )
        safe_assign_array(qi3.view[:, :, :], inputs["qi3"])

        ql3 = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM, K_DIM],
            units="n/a",
        )
        safe_assign_array(ql3.view[:, :, :], inputs["ql3"])

        qv3 = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM, K_DIM],
            units="n/a",
        )
        safe_assign_array(qv3.view[:, :, :], inputs["qv3"])

        thl3 = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM, K_DIM],
            units="n/a",
        )
        safe_assign_array(thl3.view[:, :, :], inputs["thl3"])

        thv3 = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM, K_DIM],
            units="n/a",
        )
        safe_assign_array(thv3.view[:, :, :], inputs["thv3"])

        u3 = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM, K_DIM],
            units="n/a",
        )
        safe_assign_array(u3.view[:, :, :], inputs["u3"])

        v3 = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM, K_DIM],
            units="n/a",
        )
        safe_assign_array(v3.view[:, :, :], inputs["v3"])

        zlo3 = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM, K_DIM],
            units="n/a",
        )
        safe_assign_array(zlo3.view[:, :, :], inputs["zlo3"])

        zw3 = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM, K_INTERFACE_DIM],
            units="n/a",
        )
        safe_assign_array(zw3.view[:, :, :], inputs["zw3_FV"])

        rhoe3 = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM, K_INTERFACE_DIM],
            units="n/a",
        )
        safe_assign_array(rhoe3.view[:, :, :], inputs["rhoe3"])

        pw3 = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM, K_INTERFACE_DIM],
            units="n/a",
        )
        safe_assign_array(pw3.view[:, :, :], inputs["pw3_FV"])
        

        # Outputs
        qi = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM, K_DIM],
            units="n/a",
        )
        qii = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM, K_INTERFACE_DIM],
            units="n/a",
        )

        ql = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM, K_DIM],
            units="n/a",
        )
        qli = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM, K_INTERFACE_DIM],
            units="n/a",
        )

        qv = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM, K_DIM],
            units="n/a",
        )
        qvi = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM, K_INTERFACE_DIM],
            units="n/a",
        )

        thl = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM, K_DIM],
            units="n/a",
        )
        thli = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM, K_INTERFACE_DIM],
            units="n/a",
        )

        thv = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM, K_DIM],
            units="n/a",
        )

        u = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM, K_DIM],
            units="n/a",
        )
        ui = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM, K_INTERFACE_DIM],
            units="n/a",
        )

        v = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM, K_DIM],
            units="n/a",
        )
        vi = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM, K_INTERFACE_DIM],
            units="n/a",
        )

        zlo = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM, K_DIM],
            units="n/a",
        )

        qt = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM, K_DIM],
            units="n/a",
        )
        qti = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM, K_INTERFACE_DIM],
            units="n/a",
        )
        rhoe = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM, K_INTERFACE_DIM],
            units="n/a",
        )
        zw = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM, K_INTERFACE_DIM],
            units="n/a",
        )
        p = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM, K_INTERFACE_DIM],
            units="n/a",
        )
        dp = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM, K_DIM],
            units="n/a",
        )


        _flip_variables(
            wthv=wthv,
            tmp=tmp,
            phis=phis,
            ztop=ztop,
            zlo3=zlo3,
            zw3=zw3,
            zlo=zlo,
            u3=u3,
            u=u,
            v3=v3,
            v=v,
            thl3=thl3,
            thl=thl,
            thv3=thv3,
            thv=thv,
            qv3=qv3,
            qv=qv,
            ql3=ql3,
            ql=ql,
            qi3=qi3,
            qi=qi,
            ui=ui,
            vi=vi,
            thli=thli,
            qvi=qvi,
            qli=qli,
            qii=qii,
            qt=qt,
            qti=qti,
            rhoe=rhoe,
            zw=zw,
            p=p,
            dp=dp,
            rhoe3=rhoe3,
            pw3=pw3,
        )

        return {
            "qi": qi.view[:],
            "qii": qii.view[:],
            "ql": ql.view[:],
            "qli": qli.view[:],
            "qv": qv.view[:],
            "qvi": qvi.view[:],
            "thl": thl.view[:],
            "thli": thli.view[:],
            "thv": thv.view[:],
            "u": u.view[:],
            "ui": ui.view[:],
            "v": v.view[:],
            "vi": vi.view[:],
            "zlo": zlo.view[:],
            "qt": qt.view[:],
            "qti": qti.view[:],
            "rhoe": rhoe.view[:],
            "zw": zw.view[:],
            "p": p.view[:],
            "dp": dp.view[:],

        }