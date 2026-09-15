from f90nml import Namelist
from ndsl import QuantityFactory, StencilFactory
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM
from ndsl.stencils.testing.translate import TranslateFortranData2Py
from ndsl.utils import safe_assign_array

from pyTurbulence.SHOCMF.shoc import setup_derived_inputs


class TranslateSetupDerivedInputs(TranslateFortranData2Py):
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
            "omega": {},
            "prsl": {},
            "qci": {},
            "qcl": {},
            "qwv": {},
            "tabs": {},
            "zl": {},
        }

        # FloatField Outputs
        self.out_vars = {
            "bet": self.grid.compute_dict(),
            "gamaz": self.grid.compute_dict(),
            "hl": self.grid.compute_dict(),
            "prespot": self.grid.compute_dict(),
            "qpi": self.grid.compute_dict(),
            "qpl": self.grid.compute_dict(),
            "qv": self.grid.compute_dict(),
            "thv": self.grid.compute_dict(),
            "total_water": self.grid.compute_dict(),
            "w": self.grid.compute_dict(),
            "wrk": self.grid.compute_dict(),
        }

    def compute(self, inputs):

        _setup_derived_inputs = self.stencil_factory.from_dims_halo(
            func=setup_derived_inputs,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        # Inputs
        omega = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(omega.view[:, :, :], inputs["omega"])
        prsl = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(prsl.view[:, :, :], inputs["prsl"])
        qci = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(qci.view[:, :, :], inputs["qci"])
        qcl = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(qcl.view[:, :, :], inputs["qcl"])
        qwv= QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(qwv.view[:, :, :], inputs["qwv"])
        tabs = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(tabs.view[:, :, :], inputs["tabs"])
        zl = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(zl.view[:, :, :], inputs["zl"])
       

        # Outputs
        bet = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        gamaz = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        hl = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        prespot = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        qpi = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        qpl = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        qv = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        total_water = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        w = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        wrk = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        thv = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
    
        _setup_derived_inputs(
            wrk=wrk,
            prsl=prsl,
            qv=qv,
            qwv=qwv,
            thv=thv,
            tabs=tabs,
            qcl=qcl,
            qci=qci,
            w=w,
            omega=omega,
            qpl=qpl,
            qpi=qpi,
            total_water=total_water,
            prespot=prespot,
            gamaz=gamaz,
            zl=zl,
            hl=hl,
        )

        return {
            "bet": bet.view[:],
            "gamaz": gamaz.view[:],
            "hl": hl.view[:],
            "prespot": prespot.view[:],
            "qpi": qpi.view[:],
            "qpl": qpl.view[:],
            "qv": qv.view[:],
            "total_water": total_water.view[:],
            "w": w.view[:],
            "wrk": wrk.view[:],
            "thv": thv.view[:],
        }
