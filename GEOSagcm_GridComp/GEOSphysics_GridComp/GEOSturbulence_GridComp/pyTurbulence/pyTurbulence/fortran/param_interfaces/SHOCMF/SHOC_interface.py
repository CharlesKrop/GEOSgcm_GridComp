from MAPL_PythonBridge import UserCode, get_MAPLPy
from MAPL_PythonBridge.types import CVoidPointer
from mpi4py import MPI
from ndsl.dsl.typing import Float, Int

from pyTurbulence.SHOCMF.shoc import RUN_SHOC, SHOCMFConfiguration, SHOCMFState
from pyMoist.fortran import get_NDSL_physics
from pyMoist.fortran.build_helper import StencilBackendCompilerOverride
from pyMoist.fortran.cuda_profiler import TimedCUDAProfiler
from pyMoist.fortran.managed_state import MAPLManagedState
from pyMoist.fortran.memory_factory import MAPLMemoryRepository


class SHOCGEOSInterface(UserCode):
    def __init__(self, name: str) -> None:
        # these must be defined during the first run call because they require the number of convection tracers
        self.config = None
        self._managed_state = None
        self._shoc = None
        self._first_run = True

    def init(self, mapl_state, import_state, export_state) -> None:
        # Make sure we have our NDSL stack setup
        if not self._first_run:
            return
        self._first_run = False

        MAPLPy = get_MAPLPy()
        ndsl_stack = get_NDSL_physics(mapl_state)
        self.config = SHOCMFConfiguration(
            dtn=MAPLPy.get_resource("DSL__SHOC_DT:", mapl_state, default=Float(0)),
            PRNUMBER=MAPLPy.get_resource("TURBULENCE_SHC_PRNUM:", mapl_state, default=Float(-0.9)),
            BUOYOPT=MAPLPy.get_resource("TURBULENCE_SHC_BUOY_OPTION:", mapl_state, default=Int(2)),
            LENOPT=MAPLPy.get_resource("TURBULENCE_SHC_LENOPT:", mapl_state, default=Int(3)),
            LENFAC1=MAPLPy.get_resource("TURBULENCE_SHC_LENFAC1:", mapl_state, default=Float(5.)),
            LENFAC2=MAPLPy.get_resource("TURBULENCE_SHC_LENFAC2:", mapl_state, default=Float(2.)),
            LENFAC3=MAPLPy.get_resource("TURBULENCE_SHC_LENFAC3:", mapl_state, default=Float(.5)),
            CeFAC=MAPLPy.get_resource("TURBULENCE_SHC_CEFAC:", mapl_state, default=Float(1.)),
            CesFAC=MAPLPy.get_resource("TURBULENCE_SHC_CESFAC:", mapl_state, default=Float(4.)),
            Ck=MAPLPy.get_resource("TURBULENCE_SHC_CK:", mapl_state, default=Float(.1)),
            shoc_lambda=MAPLPy.get_resource("TURBULENCE_SHC_LAMBDA:", mapl_state, default=Float(.25)),
        )

    def run(self, mapl_state, import_state, export_state) -> None:
        raise RuntimeError("SHOC requires pyTurbulence integration requires `run_with_internal`")

    def run_with_internal(
        self,
        mapl_state: CVoidPointer,
        import_state: CVoidPointer,
        export_state: CVoidPointer,
        internal_state: CVoidPointer,
    ) -> None:
        ndsl_stack = get_NDSL_physics(mapl_state)
        import_repository = MAPLMemoryRepository(import_state, ndsl_stack.quantity_factory)
        export_repository = MAPLMemoryRepository(export_state, ndsl_stack.quantity_factory)
        internal_repository = MAPLMemoryRepository(internal_state, ndsl_stack.quantity_factory)

        if self._managed_state is None:
            assert self.config
            # Initialize NDSL state
            self._managed_state = MAPLManagedState(
                SHOCMFState.empty(
                    ndsl_stack.quantity_factory,
                    data_dimensions=ndsl_stack.quantity_factory.sizer.data_dimensions,
                ),
                ndsl_stack.interface_type,
            )

        assert self._managed_state  # type: ignore[unreachable]
        self._managed_state.register_2D("input.SH", "SH", import_repository)
        self._managed_state.register_K_interface("input.PLE", "PLE", import_repository)
        self._managed_state.register_K_interface("input.ZLE", "ZLE", import_repository)
        self._managed_state.register("input.U", "U", import_repository)
        self._managed_state.register("input.V", "V", import_repository)
        self._managed_state.register("input.OMEGA", "OMEGA", import_repository)
        self._managed_state.register("input.T", "T", import_repository)
        self._managed_state.register("input.Q", "QV", import_repository)
        self._managed_state.register("input.QITOT", "QITOT", import_repository)
        self._managed_state.register("input.QSTOT", "QSTOT", import_repository)
        self._managed_state.register("input.QGTOT", "QGTOT", import_repository)
        self._managed_state.register("input.QLTOT", "QLTOT", import_repository)
        self._managed_state.register("input.QRTOT", "QRTOT", import_repository)
        self._managed_state.register("input.WTHV2", "WTHV2", import_repository)
        self._managed_state.register("input.FCLD", "FCLD", import_repository)
        self._managed_state.register("input.BUOYF", "DSL__BUOYF", internal_repository,alloc=True)
        self._managed_state.register_K_interface("input.MFTKE", "DSL__MFTKE", internal_repository,alloc=True)
        self._managed_state.register_2D("input.DRYCBLH", "DSL__DRYCBLH", internal_repository,alloc=True)

        self._managed_state.register("input_output.TKESHOC", "TKESHOC", internal_repository)
        self._managed_state.register_K_interface("input_output.TKH", "TKH", internal_repository)
        self._managed_state.register_K_interface("output.KH", "KH", export_repository,alloc=True)
        self._managed_state.register_K_interface("output.KM", "KM", export_repository, alloc=True)
        self._managed_state.register_K_interface("output.ISOTROPY", "ISOTROPY", export_repository, alloc=True)
        self._managed_state.register("output.TKEDISS", "TKEDISS", internal_repository)
        self._managed_state.register("output.TKEDISSx", "TKEDISS", export_repository)
        self._managed_state.register("output.TKEBUOY", "TKEBUOY", export_repository)    
        self._managed_state.register("output.TKESHEAR", "TKESHEAR", export_repository) 
        self._managed_state.register("output.LSHOC", "LSHOC", export_repository)     
        self._managed_state.register("output.LSHOC1", "LSHOC1", export_repository)
        self._managed_state.register("output.LSHOC2", "LSHOC2", export_repository)
        self._managed_state.register("output.LSHOC3", "LSHOC3", export_repository)
        self._managed_state.register("output.BRUNTSHOC", "BRUNTSHOC", export_repository, alloc=True) # alloc=True only if PDFSHAPE>=5
        self._managed_state.register_K_interface("output.RI", "RI", export_repository,alloc=True)  
        self._managed_state.register_K_interface("output.SHOCPRNUM", "SHOCPRNUM", export_repository)    

        if self._shoc is None:
            # Build SHOC
            with StencilBackendCompilerOverride(
                MPI.COMM_WORLD,
                ndsl_stack.stencil_factory.config.dace_config,
            ):
                self._shoc = RUN_SHOC(ndsl_stack.stencil_factory, ndsl_stack.quantity_factory, self.config)

        with TimedCUDAProfiler("SHOC", {}):
            with TimedCUDAProfiler("SHOC - State copy", {}):
                self._managed_state.fortran_to_ndsl()
                #self._managed_state.record("SHOC-In")

            with TimedCUDAProfiler("SHOC Numerics", {}):
                self._shoc(self._managed_state.ndsl_state)

            with TimedCUDAProfiler("SHOC - State copy-back", {}):
                self._managed_state.ndsl_to_fortran()
                #self._managed_state.record("SHOC-Out")

    def finalize(self, mapl_state, import_state, export_state) -> None:
        assert self._managed_state
        # rank = MPI.COMM_WORLD.Get_rank()
        # if rank == 0:
        #     self._managed_state.save_recorded()  # type: ignore[unreachable]


CODE = SHOCGEOSInterface("SHOC")