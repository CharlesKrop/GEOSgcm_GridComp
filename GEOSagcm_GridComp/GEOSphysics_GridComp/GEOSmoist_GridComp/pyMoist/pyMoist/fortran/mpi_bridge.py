from mpi4py import MPI


def fortran_comm_to_mpi4py(f_comm_int):
    """
    Rely on internal of `mpi4py` to deal with Fortran C-pointer casting
    """
    return MPI.Comm.fromhandle(f_comm_int)
