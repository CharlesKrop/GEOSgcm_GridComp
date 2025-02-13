"""GF2020 cumulus parameterization interface"""

import numpy as np
from gt4py.cartesian.gtscript import i32

from ndsl import QuantityFactory, StencilFactory, orchestrate
from ndsl.constants import X_DIM, Y_DIM, Z_DIM, Z_INTERFACE_DIM
from ndsl.dsl.typing import Float, FloatField, FloatFieldIJ, Int
from pyMoist.convective_parameterization.GF2020.GF2020_internal_variables import (
    outputs,
    temporaries,
    namelist_constants,
)

from pyMoist.convective_parameterization.GF2020.cumulus_parameterizaiton_setup import (
    which_plumes,
    get_constants,
    temporaries,
)
from pyMoist.convective_parameterization.GF2020.cumulus_parameterization_core import (
    set_excess,
    cumulus_parameterization_core,
)

# used only during construction
from pyMoist.convective_parameterization.GF2020.GF2020_flags import GF2020_flags


class cumulus_parameterization:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        GF2020_flags: GF2020_flags,
    ):
        self.stencil_factory = stencil_factory
        self.quantity_factory = quantity_factory
        self.GF2020_flags = GF2020_flags

        orchestrate(obj=self, config=stencil_factory.config.dace_config)
        self._cu_param = self.stencil_factory.from_dims_halo(
            func=cumulus_parameterization_core,
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
        )

        # Determine which plumes to paramterize and their order
        self.plume_order = which_plumes(GF2020_flags)

        # Create temporaries to be used on call
        self.temporaries = temporaries(self, quantity_factory)

    def __call__(
        temp2m: FloatFieldIJ,
        temp_old: FloatField,
        temp_new: FloatField,
        temp_new_bl: FloatField,
        temp_new_adv: FloatField,
        qv_old: FloatField,
        qv_new: FloatField,
        qv_new_bl: FloatField,
        qv_new_adv: FloatField,
        ocean_fraction: FloatFieldIJ,
        dx2d: FloatFieldIJ,
        pbl_top_level: FloatFieldIJ,
        # forcings
        buoyancy_excess,
        gsf_t,
        gsf_q,
        sgsf_t,
        sgsf_q,
        advf_t,
        # end forcings
        zws,
        last_ierr,
        fixout_qv,
        conprr,
        out_chem_1_deep,
        out_chem_2_deep,
        out_chem_1_mid,
        out_chem_2_mid,
        out_chem_1_shal,
        out_chem_2_shal,
        topo_height_no_neg,
        lons_degrees,
        lats_degrees,
        revsu_gf,
        prfil_gf,
        temp_tendqv,
        outt_deep,
        outt_mid,
        outt_shal,
        outu_deep,
        outu_mid,
        outu_shal,
        outv_deep,
        outv_mid,
        outv_shal,
        outq_deep,
        outq_mid,
        outq_shal,
        outqc_deep,
        outqc_mid,
        outqc_shal,
        outnice_deep,
        outnice_mid,
        outnice_shal,
        outnliq_deep,
        outnliq_mid,
        outnliq_shal,
        outbuoy_deep,
        outbuoy_mid,
        outbuoy_shal,
        omega,
        ccn,
        sensible_heat_sfc_flux,
        latent_heat_sfc_flux,
        # outputs passed back to the rest of the model
        lightn_dens,
        cnv_tr,
    ):
        # iterate over plume depths
        for i in range(self.plume_order):
            # pull the correct constants for the plume
            # NOTE probably want to come up with a better solution for this
            (
                HEI_DOWN_LAND,
                HEI_DOWN_OCEAN,
                HEI_UPDF_LAND,
                HEI_UPDF_OCEAN,
                MIN_EDT_LAND,
                MIN_EDT_OCEAN,
                MAX_EDT_LAND,
                MAX_EDT_OCEAN,
                FADJ_MASSFLX,
                USE_EXCESS,
                AVE_LAYER,
            ) = get_constants(self.plume_order[i], GF2020_flags)

            set_excess(
                USE_EXCESS,
                ocean_fraction,
                self.temporaries.t_excess,
                self.temporaries.q_excess,
                t_excess,
            )

            cumulus_parameterization_core(
                STUFF,
            )
