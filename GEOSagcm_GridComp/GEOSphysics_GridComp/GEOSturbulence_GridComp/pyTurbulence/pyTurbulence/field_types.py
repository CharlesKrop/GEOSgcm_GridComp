from ndsl.dsl.gt4py import Field, IJK
from ndsl.dsl.typing import Float, Int

from pyTurbulence.constants import NUP2


# NOTE must cast to int because numpy types are not acceptable for data dimensions
FloatField_UpdraftProperties = Field[IJK, (Float, int(NUP2))]
