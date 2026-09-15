from f90nml import Namelist
from ndsl import QuantityFactory, StencilFactory
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM
from ndsl.stencils.testing.translate import TranslateFortranData2Py
from ndsl.utils import safe_assign_array

from pyTurbulence.SHOCMF.shoc import invert_inputs


class TranslateInvertInputs(TranslateFortranData2Py):
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
            "cld_sgs_inv": {},
            "omega_inv": {},
            "phii_inv": {},
            "phil_inv": {},
            "prsl_inv": {},
            "qc_inv": {},
            "qi_inv": {},
            "qwv_inv": {},
            "tabs_inv": {},
            "tke_inv": {},
            "tkh_inv": {},
            "u_inv": {},
            "v_inv": {},
            "wthv_mf_inv": {},
            "wthv_sec_inv": {},
        }

        # FloatField Outputs
        self.out_vars = {
            "cld_sgs": self.grid.compute_dict(),
            "omega": self.grid.compute_dict(),
            "prsl": self.grid.compute_dict(),
            "qci": self.grid.compute_dict(),
            "qcl": self.grid.compute_dict(),
            "qwv": self.grid.compute_dict(),
            "tabs": self.grid.compute_dict(),
            "tke": self.grid.compute_dict(),
            "tkh": self.grid.compute_dict(),
            "u": self.grid.compute_dict(),
            "v": self.grid.compute_dict(),
            "wthv_mf": self.grid.compute_dict(),
            "wthv_sec": self.grid.compute_dict(),
            "zl": self.grid.compute_dict(),
        }

    def compute(self, inputs):

        _invert_inputs = self.stencil_factory.from_dims_halo(
            func=invert_inputs,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        # Inputs
        cld_sgs_inv = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(cld_sgs_inv.view[:, :, :], inputs["cld_sgs_inv"])
        omega_inv = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(omega_inv.view[:, :, :], inputs["omega_inv"])
        phii_inv = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_INTERFACE_DIM], units="n/a")
        safe_assign_array(phii_inv.view[:, :, :], inputs["phii_inv"])
        phil_inv = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(phil_inv.view[:, :, :], inputs["phil_inv"])
        prsl_inv = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(prsl_inv.view[:, :, :], inputs["prsl_inv"])
        qc_inv = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(qc_inv.view[:, :, :], inputs["qc_inv"])
        qi_inv = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(qi_inv.view[:, :, :], inputs["qi_inv"])
        qwv_inv = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(qwv_inv.view[:, :, :], inputs["qwv_inv"])
        tabs_inv = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(tabs_inv.view[:, :, :], inputs["tabs_inv"])
        tke_inv = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(tke_inv.view[:, :, :], inputs["tke_inv"])
        tkh_inv = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(tkh_inv.view[:, :, :], inputs["tkh_inv"])
        u_inv = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(u_inv.view[:, :, :], inputs["u_inv"])
        v_inv = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(v_inv.view[:, :, :], inputs["v_inv"])
        wthv_mf_inv = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(wthv_mf_inv.view[:, :, :], inputs["wthv_mf_inv"])
        wthv_sec_inv = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(wthv_sec_inv.view[:, :, :], inputs["wthv_sec_inv"])

        # Outputs
        cld_sgs = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        omega = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        prsl = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        qci = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        qcl = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        qwv = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        tabs = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        tke = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        tkh = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        u = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        v = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        wthv_mf = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        wthv_sec = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        zl = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
   
        _invert_inputs(
            zl=zl,
            phil_inv=phil_inv,
            phii_inv=phii_inv,
            tkh=tkh,
            tkh_inv=tkh_inv,
            prsl=prsl,
            prsl_inv=prsl_inv,
            u=u,
            v=v,
            u_inv=u_inv,
            v_inv=v_inv,
            omega=omega,
            omega_inv=omega_inv,
            tabs=tabs,
            tabs_inv=tabs_inv,
            qwv=qwv,
            qwv_inv=qwv_inv,
            qcl=qcl,
            qc_inv=qc_inv,
            qci=qci,
            qi_inv=qi_inv,
            cld_sgs=cld_sgs,
            cld_sgs_inv=cld_sgs_inv,
            tke=tke,
            tke_inv=tke_inv,
            wthv_sec=wthv_sec,
            wthv_sec_inv=wthv_sec_inv,
            wthv_mf=wthv_mf,
            wthv_mf_inv=wthv_mf_inv,
        )

        return {
            "cld_sgs": cld_sgs.view[:],
            "omega": omega.view[:],
            "prsl": prsl.view[:],
            "qci": qci.view[:],
            "qcl": qcl.view[:],
            "qwv": qwv.view[:],
            "tabs": tabs.view[:],
            "tke": tke.view[:], 
            "tkh": tkh.view[:],
            "u": u.view[:],
            "v": v.view[:],
            "wthv_mf": wthv_mf.view[:],
            "wthv_sec": wthv_sec.view[:],
            "zl": zl.view[:],
        }
