from ndsl import StencilFactory, QuantityFactory, SubtileGridSizer, ndsl_log
from pyMoist.microphysics.GFDL_1M.microphysics.constants import SATURATION_TABLE_LENGTH, SATURATION_TABLE_TMIN, DELT, TICE, LV0, DC_VAP, RVGAS, E00, D2_ICE, LI2
from mpi4py import MPI
from ndsl.dsl.typing import FloatField, FloatField64, Int, Bool, FloatFieldIJ, Float32
from ndsl.dsl.gt4py import computation, FORWARD, interval, log, exp, K, log10
from pyMoist.shared.cloud_processes import ice_fraction


def compute_table_0(
    table_0: FloatField,
):
    with computation(FORWARD), interval(...):
        # ensure all temporaries are 64 bit
        tem: FloatField64 = 0.0
        fac0: FloatField64 = 0.0
        fac1: FloatField64 = 0.0
        fac2: FloatField64 = 0.0

    with computation(FORWARD), interval(...):
        tem = SATURATION_TABLE_TMIN + DELT * K
        fac0 = (tem - TICE) / (tem * TICE)
        fac1 = fac0 * LV0
        fac2 = (DC_VAP * log(tem / TICE) + fac1) / RVGAS
        table_0 = E00 * exp(fac2)


def compute_table_core(
    n_blend: Int,
    do_smith_table: Bool,
    table: FloatField,
):
    from __externals__ import k_end, N_MIN

    with computation(FORWARD), interval(0, 1):
        # initialize internal constants
        esbasw: FloatFieldIJ = 1013246.0
        tbasw: FloatFieldIJ = TICE + 100.0
        esbasi: FloatFieldIJ = 6107.1
        tmin: FloatFieldIJ = TICE - N_MIN * DELT

    with computation(FORWARD), interval(...):
        # enforce 64 bit precision on certain fields
        tmin = 0.0
        tem = 0.0
        esh = 0.0
        wice = 0.0
        wh2o = 0.0
        fac0 = 0.0
        fac1 = 0.0
        fac2 = 0.0
        esbasw = 0.0
        tbasw = 0.0
        esbasi = 0.0
        a = 0.0
        b = 0.0
        c = 0.0
        d = 0.0
        e = 0.0
        esupc = 0.0

    # compute es over ice between - (N_MIN * DELT) deg C and 0 deg C
    with computation(FORWARD), interval(...):
        if K <= N_MIN:
            if do_smith_table:
                tem = tmin + DELT * Float32(K)
                a = -9.09718 * (TICE / tem - 1.0)
                b = -3.56654 * log10(TICE / tem)
                c = 0.876793 * (1.0 - tem / TICE)
                e = log10(esbasi)
                table = 0.1 * exp((a + b + c + e) * log(10.0))
            else:
                tem = tmin + DELT * K
                fac0 = (tem - TICE) / (tem * TICE)
                fac1 = fac0 * LI2
                fac2 = (D2_ICE * log(tem / TICE) + fac1) / RVGAS
                table = E00 * exp(fac2)

    # compute es over water between - (n_blend * DELT) deg C and [ (n - n_min - 1) * DELT] deg C
    with computation(FORWARD), interval(...):
        if K <= k_end - N_MIN + n_blend:
            if do_smith_table:
                tem = TICE + DELT * Float32(K - n_blend)
                a = -7.90298 * (tbasw / tem - 1.0)
                b = 5.02808 * log10(tbasw / tem)
                c = -1.3816e-7 * (exp((1.0 - tem / tbasw) * 11.344 * log(10.0)) - 1.0)
                d = 8.1328e-3 * (exp((tbasw / tem - 1.0) * (-3.49149) * log(10.0)) - 1.0)
                e = log10(esbasw)
                esh = 0.1 * exp((a + b + c + d + e) * log(10.0))
                if K <= n_blend:
                    esupc = esh
                else:
                    table[K + N_MIN - n_blend] = esh
            else:
                tem = TICE + DELT * Float32(K - n_blend)
                fac0 = (tem - TICE) / (tem * TICE)
                fac1 = fac0 * LV0
                fac2 = (DC_VAP * log(tem / TICE) + fac1) / RVGAS
                esh = E00 * exp(fac2)
                if K <= n_blend:
                    esupc = esh
                else:
                    table[K + N_MIN - n_blend] = esh

    # derive blended es over ice and supercooled water between - (n_blend * delt) deg C and 0 deg C
    with computation(FORWARD), interval(...):
        if K <= n_blend:
            tem = TICE + DELT * Float32(K - n_blend)
            # WMP impose CALIPSO ice polynomial for mixed phase
            ifrac = ice_fraction(Float32(tem), 0.0, 0.0)
            wice = ifrac
            wh2o = 1.0 - wice
            table[K + N_MIN - n_blend] = wice * table[K + N_MIN - n_blend] + wh2o * esupc


def compute_dtable(table: FloatField, dtable: FloatField):
    from __externals__ import k_end

    with computation(FORWARD), interval(...):
        if K < k_end:
            dtable = max(0.0, table[0, 0, 1])

    with computation(FORWARD), interval(-1, None):
        dtable = dtable[0, 0, -1]


class GFDLMPV3Tables:
    """
    Initializes lookup tables for saturation water vapor pressure
    for the utility routines that are designed to return qs
    consistent with the assumptions in FV3.

    Reference Fortran: gfdl_cloud_microphys.F90: qsmith_init.py
    """

    def __init__(self, stencil_factory: StencilFactory) -> None:
        # enforce an absolute minimum size for tables
        N_MIN = 1600
        if SATURATION_TABLE_LENGTH < N_MIN:
            ndsl_log.error("FATAL ERROR in GFDLMPV3Tables:" "  n     = ", SATURATION_TABLE_LENGTH, "  n_min = ", N_MIN, "  Table length n must be >= n_min")
            raise ValueError("Table length n must be >= n_min")

        table_compute_domain = (1, 1, SATURATION_TABLE_LENGTH)

        sizer = SubtileGridSizer(
            nx=table_compute_domain[0],
            ny=table_compute_domain[1],
            nz=table_compute_domain[2],
            n_halo=0,
            data_dimensions={},
            backend=stencil_factory.backend,
        )
        quantity_factory = QuantityFactory(sizer, backend=stencil_factory.backend)

        # create table quantities
        self._table_0 = quantity_factory.zeros([1, 1, SATURATION_TABLE_LENGTH], "n/a")
        self._table_1 = quantity_factory.zeros([1, 1, SATURATION_TABLE_LENGTH], "n/a")
        self._table_2 = quantity_factory.zeros([1, 1, SATURATION_TABLE_LENGTH], "n/a")
        self._table_3 = quantity_factory.zeros([1, 1, SATURATION_TABLE_LENGTH], "n/a")
        self._table_4 = quantity_factory.zeros([1, 1, SATURATION_TABLE_LENGTH], "n/a")
        self._dtable_0 = quantity_factory.zeros([1, 1, SATURATION_TABLE_LENGTH], "n/a")
        self._dtable_1 = quantity_factory.zeros([1, 1, SATURATION_TABLE_LENGTH], "n/a")
        self._dtable_2 = quantity_factory.zeros([1, 1, SATURATION_TABLE_LENGTH], "n/a")
        self._dtable_3 = quantity_factory.zeros([1, 1, SATURATION_TABLE_LENGTH], "n/a")
        self._dtable_4 = quantity_factory.zeros([1, 1, SATURATION_TABLE_LENGTH], "n/a")

        # Cancel multi-node compile for tables
        # TODO: this should come for free with the rewrite of the gt:X stencils
        #       compilation mode
        if not stencil_factory.config.dace_config.do_compile:
            MPI.COMM_WORLD.Barrier()

        self._compute_table_0 = stencil_factory.from_origin_domain(
            func=compute_table_0,
            origin=(0, 0, 0),
            domain=table_compute_domain,
        )
        self._compute_table_core = stencil_factory.from_origin_domain(
            func=compute_table_core,
            origin=(0, 0, 0),
            domain=table_compute_domain,
            externals={"N_MIN": N_MIN},
        )
        self._compute_dtable = stencil_factory.from_origin_domain(
            func=compute_dtable,
            origin=(0, 0, 0),
            domain=table_compute_domain,
        )

        # set up n_blend values for tables 1-4
        n_blend = [200, 0, 200, 0]

        failing_tables = []
        for i, x in enumerate(n_blend):
            if x > N_MIN:
                failing_tables.append(i)

        if failing_tables:
            ndsl_log.error(
                "FATAL ERROR in GFDLMPV3Tables:\n"
                "  n_blend = {} (tables 1-4)\n"
                "  n_min   = {}\n"
                "  n_blend must be <= n_min to avoid negative array indices\n"
                "  in the blending loop: table(i + n_min - n_blend)".format(n_blend, N_MIN)
            )
            raise ValueError("n_blend must be <= n_min to avoid negative array indices for tables {}".format([t + 1 for t in failing_tables]))

        self._compute_table_0(self._table_0)
        self._compute_table_core(n_blend=n_blend[0], do_smith_table=False, table=self._table_1)
        self._compute_table_core(n_blend=n_blend[1], do_smith_table=False, table=self._table_2)
        self._compute_table_core(n_blend=n_blend[2], do_smith_table=True, table=self._table_3)
        self._compute_table_core(n_blend=n_blend[3], do_smith_table=True, table=self._table_4)

        self._compute_dtable(table=self._table_0, dtable=self._dtable_0)
        self._compute_dtable(table=self._table_1, dtable=self._dtable_1)
        self._compute_dtable(table=self._table_2, dtable=self._dtable_2)
        self._compute_dtable(table=self._table_3, dtable=self._dtable_3)
        self._compute_dtable(table=self._table_4, dtable=self._dtable_4)

        if stencil_factory.config.dace_config.do_compile:
            MPI.COMM_WORLD.Barrier()

        # NOTE do we still need to do this?
        self.table_0 = self._table_0[0, 0, :]
        self.table_1 = self._table_1[0, 0, :]
        self.table_2 = self._table_2[0, 0, :]
        self.table_3 = self._table_3[0, 0, :]
        self.table_4 = self._table_4[0, 0, :]
        self.dtable_0 = self._dtable_0[0, 0, :]
        self.dtable_1 = self._dtable_1[0, 0, :]
        self.dtable_2 = self._dtable_2[0, 0, :]
        self.dtable_3 = self._dtable_3[0, 0, :]
        self.dtable_4 = self._dtable_4[0, 0, :]


# Table needs to be calculated only once
_cached_table: GFDLMPV3Tables | None = None


def get_saturation_vapor_pressure_tables(
    stencil_factory: StencilFactory,
) -> GFDLMPV3Tables:
    if _cached_table is None:
        _cached_table = GFDLMPV3Tables(stencil_factory)
    return _cached_table
