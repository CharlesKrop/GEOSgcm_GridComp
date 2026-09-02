from ndsl import NDSLRuntime, StencilFactory


class GFDLMPV3Driver(NDSLRuntime):
    def __init__(self, stencil_factory: StencilFactory):
        # initialize NDSLRuntime parent class
        super.__init__(stencil_factory)

    def __call__(self, *args, **kwds):
        pass
