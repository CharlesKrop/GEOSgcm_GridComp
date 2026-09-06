import dataclasses

from ndsl import NDSLRuntime, StencilFactory, QuantityFactory
from ndsl import Local, LocalState
from ndsl.constants import I_DIM, J_DIM, K_DIM
from ndsl.dsl.typing import Float, Int
from ndsl.stencils.basic_operations import set_value


@dataclasses.dataclass
class GFDLMPV3Locals(LocalState):
    mppcw: Local = dataclasses.field(
        metadata={
            "name": "mppcw",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )
    mppew: Local = dataclasses.field(
        metadata={
            "name": "mppew",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )
    mppe1: Local = dataclasses.field(
        metadata={
            "name": "mppe1",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )
    mpper: Local = dataclasses.field(
        metadata={
            "name": "mpper",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )
    mppdi: Local = dataclasses.field(
        metadata={
            "name": "mppdi",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )
    mppd1: Local = dataclasses.field(
        metadata={
            "name": "mppd1",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )
    mppds: Local = dataclasses.field(
        metadata={
            "name": "mppds",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )
    mppdg: Local = dataclasses.field(
        metadata={
            "name": "mppdg",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )
    mppsi: Local = dataclasses.field(
        metadata={
            "name": "mppsi",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )
    mpps1: Local = dataclasses.field(
        metadata={
            "name": "mpps1",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )
    mppss: Local = dataclasses.field(
        metadata={
            "name": "mppss",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )
    mppsg: Local = dataclasses.field(
        metadata={
            "name": "mppsg",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )
    mppfw: Local = dataclasses.field(
        metadata={
            "name": "mppfw",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )
    mppfr: Local = dataclasses.field(
        metadata={
            "name": "mppfr",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )
    mppar: Local = dataclasses.field(
        metadata={
            "name": "mppar",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )
    mppas: Local = dataclasses.field(
        metadata={
            "name": "mppas",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )
    mppag: Local = dataclasses.field(
        metadata={
            "name": "mppag",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )
    mpprs: Local = dataclasses.field(
        metadata={
            "name": "mpprs",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )
    mpprg: Local = dataclasses.field(
        metadata={
            "name": "mpprg",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )
    mppxr: Local = dataclasses.field(
        metadata={
            "name": "mppxr",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )
    mppxs: Local = dataclasses.field(
        metadata={
            "name": "mppxs",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )
    mppxg: Local = dataclasses.field(
        metadata={
            "name": "mppxg",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )
    mppmi: Local = dataclasses.field(
        metadata={
            "name": "mppmi",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )
    mppms: Local = dataclasses.field(
        metadata={
            "name": "mppms",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )
    mppmg: Local = dataclasses.field(
        metadata={
            "name": "mppmg",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )
    mppm1: Local = dataclasses.field(
        metadata={
            "name": "mppm1",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )
    mppm2: Local = dataclasses.field(
        metadata={
            "name": "mppm2",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )
    mppm3: Local = dataclasses.field(
        metadata={
            "name": "mppm3",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )


class GFDLMPV3Driver(NDSLRuntime):
    def __init__(self, stencil_factory: StencilFactory, quantity_factory: QuantityFactory):
        # initialize NDSLRuntime parent class
        super.__init__(stencil_factory)

        # initialize class specific locals
        self._mp_locals = GFDLMPV3Locals.make_locals(quantity_factory)

        # construct stencils
        self._set_value = stencil_factory.from_dims_halo(
            func=set_value,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

    def __call__(self, *args, **kwds):
        # reset mp locals to zero
        self._set_value(field=self._mp_locals.mppcw, value=Float(0, 0))
        self._set_value(field=self._mp_locals.mppew, value=Float(0, 0))
        self._set_value(field=self._mp_locals.mppe1, value=Float(0, 0))
        self._set_value(field=self._mp_locals.mpper, value=Float(0, 0))
        self._set_value(field=self._mp_locals.mppdi, value=Float(0, 0))
        self._set_value(field=self._mp_locals.mppd1, value=Float(0, 0))
        self._set_value(field=self._mp_locals.mppds, value=Float(0, 0))
        self._set_value(field=self._mp_locals.mppdg, value=Float(0, 0))
        self._set_value(field=self._mp_locals.mppsi, value=Float(0, 0))
        self._set_value(field=self._mp_locals.mpps1, value=Float(0, 0))
        self._set_value(field=self._mp_locals.mppss, value=Float(0, 0))
        self._set_value(field=self._mp_locals.mppsg, value=Float(0, 0))
        self._set_value(field=self._mp_locals.mppfw, value=Float(0, 0))
        self._set_value(field=self._mp_locals.mppfr, value=Float(0, 0))
        self._set_value(field=self._mp_locals.mppar, value=Float(0, 0))
        self._set_value(field=self._mp_locals.mppas, value=Float(0, 0))
        self._set_value(field=self._mp_locals.mppag, value=Float(0, 0))
        self._set_value(field=self._mp_locals.mpprs, value=Float(0, 0))
        self._set_value(field=self._mp_locals.mpprg, value=Float(0, 0))
        self._set_value(field=self._mp_locals.mppxr, value=Float(0, 0))
        self._set_value(field=self._mp_locals.mppxs, value=Float(0, 0))
        self._set_value(field=self._mp_locals.mppxg, value=Float(0, 0))
        self._set_value(field=self._mp_locals.mppmi, value=Float(0, 0))
        self._set_value(field=self._mp_locals.mppms, value=Float(0, 0))
        self._set_value(field=self._mp_locals.mppmg, value=Float(0, 0))
        self._set_value(field=self._mp_locals.mppm1, value=Float(0, 0))
        self._set_value(field=self._mp_locals.mppm2, value=Float(0, 0))
        self._set_value(field=self._mp_locals.mppm3, value=Float(0, 0))
