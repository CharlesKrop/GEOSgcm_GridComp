from ndsl import OptimizationConfig, StencilFactory


def get_optimization_config(stencil_factory: StencilFactory) -> OptimizationConfig:
    return OptimizationConfig.get_default()
