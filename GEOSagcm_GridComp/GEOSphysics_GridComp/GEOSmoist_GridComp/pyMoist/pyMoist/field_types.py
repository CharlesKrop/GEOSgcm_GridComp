from ndsl.dsl.gt4py import Field, IJK, K
from ndsl.dsl.typing import Float
from pyMoist.constants import N_MODES
from pyMoist.saturation.constants import TABLESIZE


FloatField_NModes = Field[IJK, (Float, (N_MODES))]
FloatField_VaporSaturationTable = Field[K, (Float, (int(TABLESIZE)))]
