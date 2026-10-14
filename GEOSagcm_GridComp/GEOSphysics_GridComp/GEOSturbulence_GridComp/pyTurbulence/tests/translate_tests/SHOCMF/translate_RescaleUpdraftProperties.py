from f90nml import Namelist
from ndsl import QuantityFactory, StencilFactory
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM
from ndsl.dsl.typing import Int
from ndsl.stencils.testing.savepoint import DataLoader
from ndsl.stencils.testing.translate import TranslateFortranData2Py
from ndsl.utils import safe_assign_array

from pyTurbulence.SHOCMF.config import SHOCMFConfiguration
from pyTurbulence.SHOCMF.edmf import rescale_updraft_properties


class TranslateRescaleUpdraftProperties(TranslateFortranData2Py):
    """Translate test for the RescaleUpdraftProperties savepoint."""

    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, stencil_factory)
        self.stencil_factory = stencil_factory
        self.quantity_factory = grid.quantity_factory

        # Inputs present in RescaleUpdraftProperties-In.nc
        self.in_vars["data_vars"] = {
            "UPA": {},
            "UPW": {},
            "nup2": {},
            "qr": {},
            "qs": {},
            "dqrdt": {},
            "dqsdt": {},
            "rhoe": {},
            "tke3": {},
            "zw": {},
        }

        # Outputs present in RescaleUpdraftProperties-Out.nc
        self.out_vars = {
            "UPA": self.grid.compute_dict(),
            "qr": self.grid.compute_dict(),
            "qs": self.grid.compute_dict(),
            "dqrdt": self.grid.compute_dict(),
            "dqsdt": self.grid.compute_dict(),
            "edmfmf": self.grid.compute_dict(),
        }

    def extra_data_load(self, data_loader: DataLoader):
        self.constants_edmf = data_loader.load("EDMF-constants")
        self.constants_shoc = data_loader.load("SHOCMF-constants")

    def compute(self, inputs):
        config = SHOCMFConfiguration(
            **self.constants_edmf,
            **self.constants_shoc,
        )

        self.quantity_factory.add_data_dimensions(
            {"number_of_properties": int(config.NUP)}
        )

        _rescale_updraft_properties = self.stencil_factory.from_dims_halo(
            func=rescale_updraft_properties,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={
                "dt":config.dt,
            },
        )

        def make_quantity(name, dims, dtype=None):
            kwargs = {"dims": dims, "units": "n/a"}
            if dtype is not None:
                kwargs["dtype"] = dtype
            quantity = QuantityFactory.zeros(self.quantity_factory, **kwargs)
            safe_assign_array(quantity.view[:], inputs[name])
            return quantity

        updraft_dims = [
            I_DIM,
            J_DIM,
            K_INTERFACE_DIM,
            "number_of_properties",
        ]
        interface_dims = [I_DIM, J_DIM, K_INTERFACE_DIM]
        layer_dims = [I_DIM, J_DIM, K_DIM]
        column_dims = [I_DIM, J_DIM]

        UPA = make_quantity("UPA", updraft_dims)
        UPW = make_quantity("UPW", updraft_dims)
        nup2 = make_quantity("nup2", column_dims, dtype=Int)
        qr = make_quantity("qr", layer_dims)
        qs = make_quantity("qs", layer_dims)
        dqrdt = make_quantity("dqrdt", layer_dims)
        dqsdt = make_quantity("dqsdt", layer_dims)
        rhoe = make_quantity("rhoe", interface_dims)
        tke3 = make_quantity("tke3", layer_dims)
        zw = make_quantity("zw", interface_dims)

        # factor is an output in the Fortran savepoint, so allocate it here.
        factor = QuantityFactory.zeros(
            self.quantity_factory,
            dims=column_dims,
            units="n/a",
        )
        edmfmf = QuantityFactory.zeros(
            self.quantity_factory,
            dims=interface_dims,
            units="n/a",
        )

        _rescale_updraft_properties(
            UPA=UPA,
            UPW=UPW,
            nup2=nup2,
            qr=qr,
            qs=qs,
            dqrdt=dqrdt,
            dqsdt=dqsdt,
            rhoe=rhoe,
            TKE3=tke3,
            zw=zw,
            edmfmf=edmfmf,
        )

        return {
            "UPA": UPA.view[:],
            "qr": qr.view[:],
            "qs": qs.view[:],
            "dqrdt": dqrdt.view[:],
            "dqsdt": dqsdt.view[:],
            "edmfmf": edmfmf.view[:],
        }

