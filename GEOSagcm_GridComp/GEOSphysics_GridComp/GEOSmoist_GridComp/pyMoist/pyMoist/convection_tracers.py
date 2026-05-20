import dataclasses
import numpy as np
from ndsl import Quantity, State, QuantityFactory
from ndsl.constants import I_DIM, J_DIM, K_DIM
from ndsl.dsl.typing import Bool, Float


@dataclasses.dataclass
class ConvectionTracers(State):
    """
    Dataclass of Convection Tracers, contains both the numerical data of the tracers
    (stored in the "tracer" field) and metadata, each stored in its off-grid field

    Must be initialized with the following extra dimensions:
        "convection_tracers": number of convective tracers, must be defined prior to initalization
        "size_three_dimension": fixed dimension of size three for metadata
        "size_four_dimension": fixed dimension of size four for metadata
    """

    tracers: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM, "convection_tracers"],
            "dtype": Float,
        }
    )
    fscav: Quantity = dataclasses.field(
        metadata={
            "dims": ["convection_tracers"],
            "dtype": Float,
        }
    )
    vect_hcts: Quantity = dataclasses.field(
        metadata={
            "dims": ["convection_tracers", "size_four_dimension"],
            "dtype": Float,
        }
    )
    kc_scal: Quantity = dataclasses.field(
        metadata={
            "dims": ["convection_tracers", "size_three_dimension"],
            "dtype": Float,
        }
    )
    convfaci2g: Quantity = dataclasses.field(
        metadata={
            "dims": ["convection_tracers"],
            "dtype": Float,
        }
    )
    retfactor: Quantity = dataclasses.field(
        metadata={
            "dims": ["convection_tracers"],
            "dtype": Float,
        }
    )
    liq_and_gas: Quantity = dataclasses.field(
        metadata={
            "dims": ["convection_tracers"],
            "dtype": Float,
        }
    )
    online_cldliq: Quantity = dataclasses.field(
        metadata={
            "dims": ["convection_tracers"],
            "dtype": Float,
        }
    )
    online_vud: Quantity = dataclasses.field(
        metadata={
            "dims": ["convection_tracers"],
            "dtype": Float,
        }
    )
    ftemp_threshold: Quantity = dataclasses.field(
        metadata={
            "dims": ["convection_tracers"],
            "dtype": Float,
        }
    )
    use_gcc_washout: Quantity = dataclasses.field(
        metadata={
            "dims": ["convection_tracers"],
            "dtype": Bool,
        }
    )
    use_gocart: Quantity = dataclasses.field(
        metadata={
            "dims": ["convection_tracers"],
            "dtype": Bool,
        }
    )
    is_wetdep: Quantity = dataclasses.field(
        metadata={
            "name": "is_wetdep",
            "dims": ["convection_tracers"],
            "dtype": Bool,
        }
    )


# Table needs to be done
_initialized_convection_tracers = False


def initialize(
    cls,
    quantity_factory: QuantityFactory,
    tracer_bundle: ESMF_BUNDLE,
    n_total_tracers: int,
):
    n_friendly_tracers = 0
    for n in range(n_total_tracers):
        name: str = tracer_bundle.name  # string
        if tracer_bundle[name].FriendlyToMOIST.isPresent:
            if tracer_bundle[name].FriendlyToMOIST.isFriendly:
                n_friendly_tracers += 1

    if _initialized_convection_tracers:
        # check initialized size
        assert convecetion_tracers_ddim_size == n_friendly_tracers
    else:
        # initialize
        convection_tracers = ConvectionTracers.ones(
            quantity_factory,
            data_dimensions={
                "convection_tracers": n_friendly_tracers,
                "size_three_dimension": 3,
                "size_four_dimension": 4,
            },
        )

        # friendly index
        f = 0

        # TODO this loop requires runtime support for ESMF_AttributeGet and ESMFL_BundleGetPointerToData
        # or the ability to bring the entire tracer bundle across the fortran - python interface (which would require ESMF_FieldBundleGet)
        # until a decision has been made and the required support has been implemented, this code will not function
        # for now, place holders have been implemented in place of some calls (e.g. tracer_bundle[name].ATTRIBUTE requires ESMF_AttributeGet)
        for n in range(n_total_tracers):
            name: str = tracer_bundle.name  # string
            if tracer_bundle[name].FriendlyToMOIST.isPresent:
                if tracer_bundle[name].FriendlyToMOIST.isFriendly:
                    # get items scavenging fraction
                    convection_tracers.fscav[f, :] = -99.0
                    if tracer_bundle[name].ScavengingFractionPerKm.isPresent:
                        convection_tracers.fscav[f, :] = tracer_bundle[name].ScavengingFractionPerKm.field

                    # get items for the wet removal parameterization for gases based on the Henry's Law
                    convection_tracers.vect_hcts[f, :] = -99.0
                    if tracer_bundle[name].SetofHenryLawCts.isPresent:
                        convection_tracers.fscav[f, :] = tracer_bundle[name].SetofHenryLawCts.field

                    # additional items, needed for GEOS-Chem washout parameterization
                    convection_tracers.is_wetdep[f] = False
                    convection_tracers.use_gcc_washout[f] = False
                    convection_tracers.kc_scal[f, :] = 1.0
                    convection_tracers.retfactor[f] = 1.0
                    convection_tracers.liq_and_gas[f] = 0.0
                    convection_tracers.convfaci2g[f] = 0.0
                    convection_tracers.online_cldliq[f] = 0.0
                    convection_tracers.online_vud[f] = 1.0
                    convection_tracers.use_gocart[f] = False
                    convection_tracers.ftemp_threshold[f] = -999.0

                    # check if GEOS-Chem washout should be used
                    # assume this is hte case if kc scale factors are present
                    if tracer_bundle[name].SetofKcScalFactors.isPresent:
                        convection_tracers.use_gcc_washout[f] = tracer_bundle[name].SetofKcScalFactors.field

                    # if using GEOS-Chem parameterization, retrieve all necessary parameters
                    if convection_tracers.use_gcc_washout[f]:
                        # kc scale factors
                        if tracer_bundle[name].SetofKcScalFactors.isPresent:
                            convection_tracers.kc_scal.field[f, :] = tracer_bundle[name].SetofKcScalFactors.field
                        # is this a wetdep species
                        if tracer_bundle[name].IsWetDep.isPresent:
                            is_wetdep = tracer_bundle[name].IsWetDep.field
                            if is_wetdep == 1.0:
                                convection_tracers.is_wetdep.field[f] = True
                            else:
                                convection_tracers.is_wetdep.field[f] = False
                        # gas-phase washout parameter for GEOS-Chem
                        if tracer_bundle[name].RetentionFactor.isPresent:
                            convection_tracers.retfactor.field[f] = tracer_bundle[name].RetentionFactor.field
                        if tracer_bundle[name].LiqAndGas.isPresent:
                            convection_tracers.liq_and_gas.field[f] = tracer_bundle[name].LiqAndGas.field
                        if tracer_bundle[name].ConvFacI2G.isPresent:
                            convection_tracers.convfaci2g.field[f] = tracer_bundle[name].ConvFacI2G.field
                        if tracer_bundle[name].OnlineCLDLIQ.isPresent:
                            convection_tracers.online_cldliq.field[f] = tracer_bundle[name].OnlineCLDLIQ.field
                        if tracer_bundle[name].OnlineVUD.isPresent:
                            convection_tracers.online_vud.field[f] = tracer_bundle[name].OnlineVUD.field
                        if tracer_bundle[name].UseGOCART.isPresent:
                            use_gocart = tracer_bundle[name].UseGOCART.field
                            if use_gocart == 1.0:
                                convection_tracers.use_gocart.field[f] = True
                            else:
                                convection_tracers.use_gocart.field[f] = False
                        if tracer_bundle[name].GOCARTfTempThreshold.isPresent:
                            convection_tracers.ftemp_threshold.field[f] = tracer_bundle[name].GOCARTfTempThreshold.field

                    # get friendly tracer
                    convection_tracers.tracers[:, :, :, f] = tracer_bundle[name].ESMFL_BundleGetPointerToData.field

                    # iterate the friendly index
                    f += 1

        _initialized_convection_tracers = True

    return convection_tracers
