"""Temporary file with GF2020 global & namelist parameters"""

from gt4py.cartesian.gtscript import i32, f32


class GF2020_flags:
    def __init__(self, MAXIENS, N_TRACERS):
        self.MAXIENS = MAXIENS
        self.N_TRACERS = N_TRACERS
