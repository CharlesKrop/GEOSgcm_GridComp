import sys

import cffi
from mpi4py import MPI


ffi = cffi.FFI()

ffi.cdef(
    """
    typedef int MPI_Fint;
    void* MPI_Comm_f2c(MPI_Fint comm);
"""
)

# Dev NOTE: We would need to adjust the library name for MPICH
if sys.platform == "win32":
    mpi_lib_name = "msmpi.dll"
elif sys.platform == "darwin":
    mpi_lib_name = "libmpi.dylib"
else:
    mpi_lib_name = "libmpi.so"

# RTLD_GLOBAL is critical here so that MPI can resolve its internal global state symbols
libmpi = ffi.dlopen(mpi_lib_name, ffi.RTLD_GLOBAL | ffi.RTLD_LAZY)


def fortran_comm_to_mpi4py(f_comm_int):
    """
    Takes a raw Fortran communicator integer, calls MPI_Comm_f2c via CFFI,
    and returns a native mpi4py Comm object.
    """
    c_comm_ptr = libmpi.MPI_Comm_f2c(f_comm_int)
    c_comm_addr = int(ffi.cast("intptr_t", c_comm_ptr))
    return MPI.Comm.fromhandle(c_comm_addr)
