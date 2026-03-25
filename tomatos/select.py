from functools import partial

import jax
import jax.numpy as jnp
import relaxed
import numpy as np 

import tomatos.utils


# @partial(jax.jit, static_argnames=["config", "validate_only"])
# @jax.jit
def cuts(pars, data, config, validate_only):
    # remove approximation with large slope for validation --> sharp cuts
    slope = 1e20 if validate_only else config.slope
    # init with shape (samples, events)
    cut_weights = jnp.ones(data.shape[:2])
    # collect them over all cuts and apply to weights once
    for var, var_dict in config.opt_cuts.items():
        # get the sigmoid weights for var_cut
        cut_weights *= relaxed.cut(
            data=data[:, :, var_dict["idx"]],
            cut_val=pars["cut_" + var],
            slope=slope,
            keep=var_dict["keep"],
        )

    # NB
    # if you wonder why not doing cuts like this:
    # var = jnp.where(var > cut_param, var, 0)
    # discontinuous --> no gradient

    return cut_weights


# @partial(jax.jit, static_argnames=["config"])
# @jax.jit
def events(data, config, base_weights):
    # apply selections via weights
    # you can also put the computation of selections at the preselection stage
    btag_1 = data[:, :, config.vars.index("j1_tag")]
    btag_2 = data[:, :, config.vars.index("j2_tag")]

    btag_1_pass = btag_1 > 0.5
    btag_2_pass = btag_2 > 0.5

    Ntag_1 = btag_1_pass | btag_2_pass
    Ntag_2 = btag_1_pass & btag_2_pass

    h_mass_1 = data[:, :, config.vars.index("j1_mass")]
    h_mass_2 = data[:, :, config.vars.index("j2_mass")]

    h_m_idx = config.vars.index("m_hh")
    if config.objective == "cls_var":
        h_m = tomatos.utils.inverse_min_max_scale(
            config,
            data[:, :, h_m_idx],
            h_m_idx,
        )
    elif config.objective == "cls_nn":
        h_m = data[:, :, h_m_idx]

    SR = np.sqrt(((h_mass_1 - 124e3) / (1500e3/h_mass_1))**2 + ((h_mass_2 - 117e3) / (1900e3/h_mass_1))**2) < 1.6e3
    VR = np.sqrt(((h_mass_1 - 124e3) / (0.1 * np.log(h_mass_1)))**2 + ((h_mass_2 - 117e3) / (0.1 * np.log(h_mass_2)))**2) < 100e3
    CR = np.sqrt(((h_mass_1 - 124e3) / (0.1 *np.log(h_mass_1)))**2 + ((h_mass_2 - 117e3) / (0.1 *np.log(h_mass_2)))**2) < 170e3

    weights = {
        # "base_weights": base_weights,
        "SR_btag_1": base_weights * SR * Ntag_1,
        "SR_btag_2": base_weights * SR * Ntag_2,
        "VR_btag_1": base_weights * VR * Ntag_1,
        "VR_btag_2": base_weights * VR * Ntag_2,
        "CR_btag_1": base_weights * CR * Ntag_1,
        "CR_btag_2": base_weights * CR * Ntag_2,
        "SR_btag_2_my_sf_unc_up": base_weights
        * SR
        * Ntag_2
        * 1.2,
        "SR_btag_2_my_sf_unc_down": base_weights
        * SR
        * Ntag_2
        * 0.8
    }

    return weights
