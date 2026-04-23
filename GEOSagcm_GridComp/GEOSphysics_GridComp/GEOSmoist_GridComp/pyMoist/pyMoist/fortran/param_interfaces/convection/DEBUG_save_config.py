import netCDF4 as nc
import numpy as np
from dataclasses import fields
from typing import get_type_hints
import inspect


def dataclass_to_netcdf(instance, filepath: str, group_name: str = None):
    """
    Write a dataclass of scalars to a NetCDF4 file.
    
    Args:
        instance: An instance of a dataclass containing scalar fields
        filepath: Path to the output NetCDF4 file
        group_name: Optional group name within the NetCDF file
    """
    # Type mapping from Python/numpy types to NetCDF4 compatible types
    TYPE_MAP = {
        float: "f8",
        int: "i4",
        bool: "i1",  # NetCDF doesn't have bool; store as byte
        "Float": "f8",
        "Int": "i4",
    }

    with nc.Dataset(filepath, "w", format="NETCDF4") as ds:
        target = ds.createGroup(group_name) if group_name else ds

        # Add metadata
        target.description = f"Scalar configuration from {type(instance).__name__}"
        target.source_class = type(instance).__name__

        # Get type hints for the dataclass
        hints = get_type_hints(type(instance))

        for field in fields(instance):
            name = field.name
            value = getattr(instance, name)
            hint = hints.get(name, type(value))

            # Resolve the NetCDF type string
            hint_name = getattr(hint, "__name__", str(hint))
            nc_type = TYPE_MAP.get(hint, TYPE_MAP.get(hint_name, "f8"))

            # Create a scalar variable (no dimensions)
            var = target.createVariable(name, nc_type)

            # Convert bools to int for storage, annotate with attribute
            if isinstance(value, bool) or hint is bool:
                var.python_type = "bool"
                var[:] = int(value)
            else:
                var.python_type = hint_name
                var[:] = value


def netcdf_to_dataclass(filepath: str, dataclass_type, group_name: str = None):
    """
    Read a NetCDF4 file back into a dataclass instance.
    
    Args:
        filepath: Path to the NetCDF4 file
        dataclass_type: The dataclass class to instantiate
        group_name: Optional group name within the NetCDF file
    
    Returns:
        An instance of dataclass_type populated with values from the file
    """
    with nc.Dataset(filepath, "r") as ds:
        source = ds.groups[group_name] if group_name else ds

        kwargs = {}
        for var_name, var in source.variables.items():
            raw = var[:].item()  # Extract scalar from numpy array
            python_type = getattr(var, "python_type", None)

            if python_type == "bool":
                kwargs[var_name] = bool(raw)
            elif python_type == "Int":
                kwargs[var_name] = int(raw)
            elif python_type == "Float":
                kwargs[var_name] = float(raw)
            else:
                kwargs[var_name] = raw

        return dataclass_type(**kwargs)


# --- Example usage ---
if __name__ == "__main__":
    from dataclasses import dataclass

    # Simulate ndsl types as aliases (replace with real imports in your env)
    Float = float
    Int = int

    @dataclass
    class GF2020Config:
        DT_MOIST: Float
        LHYDROSTATIC: bool
        STOCHASTIC_CNV: bool
        STOCH_TOP: Float
        STOCH_BOT: Float
        GF_MIN_AREA: Float
        GF_ENV_SETTING: Int
        ENTRVERSION: Int
        CONVECTION_TRACER: Int
        C1: Float
        ADV_TRIGGER: Int
        AUTOCONV: Int
        USE_TRACER_TRANSPORT: Int
        SCLM_DEEP: Float
        FIX_CONVECTIVE_CLOUD: bool
        APPLY_SUBSIDENCE_MICROPHYSICS: Int
        NUMBER_OF_TRACERS: Int
        USE_MOMENTUM_TRANSPORT: Int

    config = GF2020Config(
        DT_MOIST=300.0,
        LHYDROSTATIC=True,
        STOCHASTIC_CNV=False,
        STOCH_TOP=150.0,
        STOCH_BOT=50.0,
        GF_MIN_AREA=1.0e8,
        GF_ENV_SETTING=1,
        ENTRVERSION=2,
        CONVECTION_TRACER=0,
        C1=0.02,
        ADV_TRIGGER=1,
        AUTOCONV=1,
        USE_TRACER_TRANSPORT=0,
        SCLM_DEEP=1.0,
        FIX_CONVECTIVE_CLOUD=True,
        APPLY_SUBSIDENCE_MICROPHYSICS=1,
        NUMBER_OF_TRACERS=5,
        USE_MOMENTUM_TRANSPORT=0,
    )

    filepath = "gf2020_config.nc"
    dataclass_to_netcdf(config, filepath, group_name="GF2020Config")
    print(f"Written to {filepath}")

    recovered = netcdf_to_dataclass(filepath, GF2020Config, group_name="GF2020Config")
    print(f"Recovered: {recovered}")
    assert config == recovered, "Round-trip mismatch!"
    print("Round-trip check passed ✓")