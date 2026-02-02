import itertools

import h5py
import jax.numpy as jnp
import numpy as np


def get_generator(config, split):
    with h5py.File(config.preprocess_files[split], "r") as f:
        ds = f["stack"]
        # this makes all non-repetitive combinations of batches, for r=2
        # [1,2,3] --> [[1,2], [1,3], [2,3]]
        # load the combinations together to avoid tiny rest batch, more data
        # variability
        # r slice combinations of the memory layout on disk
        batch_combi = list(
            itertools.combinations(list(ds.iter_chunks()), r=config.n_chunk_combine)
        )
        while True:
            for batch in batch_combi:
                # shape is (n_sample_sys, n_events, n_vars)
                batch = jnp.concatenate([ds[b] for b in batch], axis=1)
                # scale to total events
                batch_events = batch.shape[1]
                batch_sf = config.preprocess_md[split]["events"] / batch_events
                # this is an array of sample_sys size
                scale_factor = jnp.array(
                    np.array(config.preprocess_md[split]["scale_factor"]) * batch_sf
                )

                yield batch, scale_factor
            np.random.shuffle(batch_combi)

def get_generator_train_balanced(config, split):
    assert split == "train", "Balanced generator only for training!"

    f = h5py.File(config.preprocess_files[split], "r")
    ds = f["stack"]

    rng = np.random.default_rng()

    # --- identify samples ---
    signal_idx = config.sample_sys.index("ggZH125_vvbb_NOSYS")
    bkg_idx    = config.sample_sys.index("bkg_NOSYS")

    n_events = ds.shape[1]

    while True:
        # --- sample signal events (with replacement) ---
        idx_sig = np.sort(
            rng.choice(
            n_events,
            size=config.n_sig_events,
            replace=True,
    )
)
        sig_batch = ds[signal_idx, idx_sig, :]

        # --- sample background events (without replacement) ---
        idx_bkg = np.sort(
            rng.choice(
            n_events,
            size=config.n_bkg_events,
            replace=False,
    )
)
        bkg_batch = ds[bkg_idx, idx_bkg, :]

        # --- combine ---
        batch = jnp.stack([sig_batch, bkg_batch], axis=0)
        # shape: (2, n_sig_or_bkg, n_vars) - 2 samples (sig, bkg)

        # --- scale factors ---
        # For balanced batch: both sig and bkg have n_events, total is sig + bkg
        batch_events = config.n_sig_events + config.n_bkg_events
        batch_sf = config.preprocess_md[split]["events"] / batch_events

        scale_factor = jnp.array(
            np.array(config.preprocess_md[split]["scale_factor"]) * batch_sf
        )

        yield batch, scale_factor
