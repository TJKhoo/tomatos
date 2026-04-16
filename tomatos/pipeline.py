import pprint
from functools import partial

import jax
import jax.numpy as jnp
import neos
import pyhf

import tomatos.constraints
import tomatos.histograms
import tomatos.select
import tomatos.train_utils
import tomatos.utils
import tomatos.workspace


def make_hists(
    pars, data, config, scale, validate_only=False, filter_return_hists=False
):
    # event manipulations are done via weights to the base weights
    base_weights = data[:, :, config.weight_idx]
    # base_weights = 1.
    cut_weights = tomatos.select.cuts(pars, data, config, validate_only)
    # apply cuts
    base_weights = jnp.multiply(base_weights, cut_weights)
    # get event selections
    sel_weights = tomatos.select.events(data, config, base_weights)

    # fill
    hists = tomatos.histograms.fill_hists(
        pars, data, config, sel_weights, scale, validate_only
    )
    # calculate additional hists based on existing hists
    hists = tomatos.workspace.hist_transforms(hists, validate_only)
    # flatten and filter if desired
    hists = tomatos.utils.filter_hists(config, hists) if filter_return_hists else hists
    return hists


def loss_fn(
    pars,  # OptaxSolver expects opt_pars as first arg
    data,
    config,
    scale,
    validate_only=False,
    filter_return_hists=True,
):
    
    # the main reason why not everything in here is jitted, is that the
    # config is not a jax compatible type (pytree), this will be a bit tedious
    # as in particular you have to get rid of all strings
    hists = make_hists(pars, data, config, scale, validate_only)
    
    model, hists = tomatos.workspace.pyhf_model(hists, config)

    if "bce" in config.objective:
        # adjist to data you want to use, lets see if anyone wants to use
        # this
        from tomatos.histograms import get_nn_output
        nn_output = get_nn_output(
            pars,
            data,
            config.nn_arch,
            config.nn_inputs_idx_end,
        )
        loss_value = tomatos.train_utils.bce(ones=nn_output[config.samples.index(config.signal_sample), :], zeros=nn_output[config.samples.index("bkg"), :])
    if "cls" in config.objective:
        loss_value = neos.loss_from_model(model, loss="cls")

        if not validate_only:
            loss_value = tomatos.constraints.penalize_loss(loss_value, hists)
    
    # Protect against NaN losses (can occur if histograms are empty or have zeros)
    loss_value = jnp.where(jnp.isnan(loss_value), 1e10, loss_value)

    # flatten and reduces to the configured filter
    hists = tomatos.utils.filter_hists(config, hists) if filter_return_hists else hists
    return loss_value, hists
