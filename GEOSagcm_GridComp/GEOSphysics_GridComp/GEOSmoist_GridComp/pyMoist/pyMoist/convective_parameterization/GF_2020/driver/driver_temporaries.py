from dataclasses import dataclass

from ndsl import Quantity, QuantityFactory
from ndsl.constants import X_DIM, Y_DIM, Z_DIM, Z_INTERFACE_DIM
from ndsl.dsl.typing import Int


@dataclass
class Temporaries:
    p: Quantity

    @classmethod
    def make(cls, quantity_factory: QuantityFactory):
        p = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")

        return cls(p)
    
    # needs to have all of these
    # NOTE MANY OF THESE ARE A PROBLEM. 4D/5D fields
    # do_this_column =0
    # ierr4d         =0
    # jmin4d         =0
    # klcl4d         =0
    # k224d          =0
    # kbcon4d        =0
    # ktop4d         =0
    # kstabi4d       =0
    # kstabm4d       =0
    # xmb4d          =0.
    # cprr4d         =0.
    # edt4d          =0.
    # pwav4d         =0.
    # sigma4d        =0.
    # pcup5d         =0.
    # entr5d         =0.                      
    # up_massentr5d  =0.
    # up_massdetr5d  =0.
    # dd_massentr5d  =0.
    # dd_massdetr5d  =0.
    # zup5d          =0.
    # zdn5d          =0.
    # prup5d         =0.
    # prdn5d         =0.
    # clwup5d        =0.
    # tup5d          =0.
    # conv_cld_fr5d  =0.
    # SRC_NI         =0.
    # SRC_NL         =0.
    # SRC_T          =0.
    # SRC_Q          =0.
    # SRC_CI         =0.
    # SRC_U          =0.
    # SRC_V          =0.
    # CNPCPRATE      =0.
    # SUB_MPQI       =0.
    # SUB_MPQL       =0.
    # SUB_MPCF       =0.
    # LIGHTN_DENS    =0.
    # SRC_BUOY       =0.
    # REVSU_GF       =0.
    # PRFIL_GF       =0.
