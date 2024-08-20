from gt4py.cartesian.gtscript import computation, interval, PARALLEL, stencil, max, min
import numpy as np
from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from ndsl.dsl.typing import FloatField, Int
from ndsl.quantity import Quantity
from ndsl import QuantityFactory, StencilFactory


def calc_slope(field: FloatField, p0: FloatField, slope: FloatField):
    with computation(PARALLEL), interval(0, 1):
        below_value = (field[0, 0, 1] - field[0, 0, 0]) / (p0[0, 0, 1] - p0[0, 0, 0])
    with computation(PARALLEL), interval(1, -1):
        below=field
        below_value = (field[0, 0, 1] - field[0, 0, 0]) / (p0[0, 0, 1] - p0[0, 0, 0])
        above_value = (field[0, 0, 0] - field[0, 0, -1]) / (p0[0, 0, 0] - p0[0, 0, -1])
        if above_value > 0.0:
            slope_value = max(0.0, min(above_value, below_value))
        else:
            slope_value = min(0.0, max(above_value, below_value))
        below[0, 0, 0] = above_value
        slope[0, 0, 0] = slope_value
    with computation(PARALLEL), interval(-1, None):
        slope_value = slope[0, 0, -1] 
        slope[0, 0, 0] = slope_value 


class Slope:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
    ) -> None:

        self._calc_slope = stencil_factory.from_dims_halo(
            func=calc_slope,
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
            
        )

    def __call__(
        self,
        thl0_test: FloatField,
        pmid0_test: FloatField,
        ssthl0_test: FloatField,
    ):
        """
        Calculate slope.

        Parameters:
        field (3D in): Variable of interest.
        p0 (3D in): Environmental pressure [ Pa ]
        slope (3D out): Slope [units?]
        """

        self._calc_slope(
            field=thl0_test,
            p0=pmid0_test,
            slope=ssthl0_test,
        )


'''
# Define dimensions
nx, ny, nz = 24, 24, 72  # Adjust dimensions as needed
shape = (nx*ny, nz)

arr = np.indices(shape, dtype=np.float64).sum(axis=0)  # Value of each entry is sum of the I and J index at each point
arr_zero = np.zeros_like(arr)

field = arr
p0 = arr


field = Quantity(data=arr,
                  dims=["IJ", "K"],
                  units="m",
                  gt4py_backend="numpy")
p0 = Quantity(data=arr,
                  dims=["IJ", "K"],
                  units="m",
                  gt4py_backend="numpy")
below = Quantity(data=arr_zero,
                  dims=["IJ", "K"],
                  units="m",
                  gt4py_backend="numpy")
slope = Quantity(data=arr_zero,
                  dims=["IJ", "K"],
                  units="m",
                  gt4py_backend="numpy")
                  


# Create 2D input data
input_data = [field,p0]

for arr in input_data:
    print("2D FIELD: ", arr.shape)

# Define the origin and domain to account for negative indexing
origin = (0, 0, 1)  # Start at k=1 to allow access to k-1
before = reshape_before(input_data)

# Execute the stencil with the correct origin and domain
#print("3D FIELDS: ", before)
for arr in before:
    print("3D FIELD: ", arr.shape)

field=Quantity(data=before[0],
                  dims=["I", "J", "K"],
                  units="m",
                  gt4py_backend="numpy")
p0 = Quantity(data=before[1],
                  dims=["I", "J", "K"],
                  units="m",
                  gt4py_backend="numpy")
below = Quantity(data=np.zeros_like(field),
                  dims=["I", "J", "K"],
                  units="m",
                  gt4py_backend="numpy")
slope = Quantity(data=np.zeros_like(field),
                  dims=["I", "J", "K"],
                  units="m",
                  gt4py_backend="numpy")


slope_3d = calc_slope(field,p0,below,slope,origin=origin)
print("SLOPE 3D: ", slope.data.shape)

after = reshape_after(outputs=slope.data)
print("2D FIELD: ", after.shape)

'''