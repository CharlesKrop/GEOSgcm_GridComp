import dataclasses

from ndsl import Quantity, State
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM, Float


@dataclasses.dataclass
class SHOCMFState(State):
    @dataclasses.dataclass
    class Input:
        BUOYF: Quantity = dataclasses.field(
            metadata={
                "name": "BUOYF",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "",
                "intent": "?",
                "dtype": Float,
            }
        )
        DRYCBLH: Quantity = dataclasses.field(
            metadata={
                "name": "DRYCBLH",
                "dims": [I_DIM, J_DIM],
                "units": "m",
                "intent": "?",
                "dtype": Float,
            }
        )
        MFTKE: Quantity = dataclasses.field(
            metadata={
                "name": "MFTKE",
                "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
                "units": "m",
                "intent": "?",
                "dtype": Float,
            }
        )
        OMEGA: Quantity = dataclasses.field(
            metadata={
                "name": "OMEGA",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "m",
                "intent": "?",
                "dtype": Float,
            }
        )
        PLE: Quantity = dataclasses.field(
            metadata={
                "name": "PLE",
                "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
                "units": "m",
                "intent": "?",
                "dtype": Float,
            }
        )
        Q: Quantity = dataclasses.field(
            metadata={
                "name": "Q",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "m",
                "intent": "?",
                "dtype": Float,
            }
        )
        FCLD: Quantity = dataclasses.field(
            metadata={
                "name": "FCLD",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "m",
                "intent": "?",
                "dtype": Float,
            }
        )
        QITOT: Quantity = dataclasses.field(
            metadata={
                "name": "QITOT",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "m",
                "intent": "?",
                "dtype": Float,
            }
        )
        QLTOT: Quantity = dataclasses.field(
            metadata={
                "name": "QLTOT",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "m",
                "intent": "?",
                "dtype": Float,
            }
        )
        QSTOT: Quantity = dataclasses.field(
            metadata={
                "name": "QSTOT",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "m",
                "intent": "?",
                "dtype": Float,
            }
        )
        QGTOT: Quantity = dataclasses.field(
            metadata={
                "name": "QGTOT",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "m",
                "intent": "?",
                "dtype": Float,
            }
        )
        QRTOT: Quantity = dataclasses.field(
            metadata={
                "name": "QRTOT",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "m",
                "intent": "?",
                "dtype": Float,
            }
        )
   
        QPI: Quantity = dataclasses.field(
            metadata={
                "name": "QPI",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "m",
                "intent": "?",
                "dtype": Float,
            }
        )
        QPL: Quantity = dataclasses.field(
            metadata={
                "name": "QPL",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "m",
                "intent": "?",
                "dtype": Float,
            }
        )
        SH: Quantity = dataclasses.field(
            metadata={
                "name": "SH",
                "dims": [I_DIM, J_DIM],
                "units": "m",
                "intent": "?",
                "dtype": Float,
            }
        )
        T: Quantity = dataclasses.field(
            metadata={
                "name": "T",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "m",
                "intent": "?",
                "dtype": Float,
            }
        )
        U: Quantity = dataclasses.field(
            metadata={
                "name": "U",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "m",
                "intent": "?",
                "dtype": Float,
            }
        )
        V: Quantity = dataclasses.field(
            metadata={
                "name": "V",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "m",
                "intent": "?",
                "dtype": Float,
            }
        )
        WTHV2: Quantity = dataclasses.field(
            metadata={
                "name": "WTHV2",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "m",
                "intent": "?",
                "dtype": Float,
            }
        )
        Z: Quantity = dataclasses.field(
            metadata={
                "name": "Z",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "m",
                "intent": "?",
                "dtype": Float,
            }
        )
        ZLE: Quantity = dataclasses.field(
            metadata={
                "name": "ZLE",
                "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
                "units": "",
                "intent": "?",
                "dtype": Float,
            }
        )


    @dataclasses.dataclass
    class Input_Output:
        TKESHOC: Quantity = dataclasses.field(
            metadata={
                "name": "TKESHOC",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "",
                "intent": "?",
                "dtype": Float,
            }
        )
        TKH: Quantity = dataclasses.field(
            metadata={
                "name": "TKH",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "",
                "intent": "?",
                "dtype": Float,
            }
        )
     

    @dataclasses.dataclass
    class Output:

        ISOTROPY: Quantity = dataclasses.field(
            metadata={
                "name": "ISOTROPY",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "f",
                "intent": "?",
                "dtype": Float,
            }
        )

        KM: Quantity = dataclasses.field(
            metadata={
                "name": "KM",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "?",
                "intent": "?",
                "dtype": Float,
            }
        )
        TKEDISS: Quantity = dataclasses.field(
            metadata={
                "name": "TKEDISS",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "?",
                "intent": "?",
                "dtype": Float,
            }
        )
        TKESHOC: Quantity = dataclasses.field(
            metadata={
                "name": "TKESHOC",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "?",
                "intent": "?",
                "dtype": Float,
            }
        )
        TKH: Quantity = dataclasses.field(
            metadata={
                "name": "TKH",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "?",
                "intent": "?",
                "dtype": Float,
            }
        )
        BRUNTSHOC: Quantity | None = dataclasses.field(
            metadata={
                "name": "BRUNTSHOC",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "?",
                "intent": "?",
                "dtype": Float,
            }
        )

        TKEBUOY: Quantity | None = dataclasses.field(
            metadata={
                "name": "TKEBUOY",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "?",
                "intent": "?",
                "dtype": Float,
            }
        )

        TKESHEAR: Quantity | None = dataclasses.field(
            metadata={
                "name": "TKESHEAR",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "?",
                "intent": "?",
                "dtype": Float,
            }
        )

        LSHOC: Quantity | None = dataclasses.field(
            metadata={
                "name": "LSHOC",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "?",
                "intent": "?",
                "dtype": Float,
            }
        )

        LSHOC1: Quantity | None = dataclasses.field(
            metadata={
                "name": "LSHOC1",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "?",
                "intent": "?",
                "dtype": Float,
            }
        )

        LSHOC2: Quantity | None = dataclasses.field(
            metadata={
                "name": "LSHOC2",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "?",
                "intent": "?",
                "dtype": Float,
            }
        )

        LSHOC3: Quantity | None = dataclasses.field(
            metadata={
                "name": "LSHOC3",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "?",
                "intent": "?",
                "dtype": Float,
            }
        )

        RI: Quantity | None = dataclasses.field(
            metadata={
                "name": "RI",
                "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
                "units": "?",
                "intent": "?",
                "dtype": Float,
            }
        )
        SHOCPRNUM: Quantity | None = dataclasses.field(
            metadata={
                "name": "SHOCPRNUM",
                "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
                "units": "?",
                "intent": "?",
                "dtype": Float,
            }
        )
    

    input: Input
    input_output: Input_Output
    output: Output
