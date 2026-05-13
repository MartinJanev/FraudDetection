"""concurrentNumba/numba_kernels.py

Numba JIT-compiled kernels for hot numerical paths used in Phase 1 cleaning.

Kernels
-------
_fast_hash_mask         – parallel boolean mask for hash-based row sampling
_parse_amounts_kernel   – parallel validity check (non-negative, non-NaN amounts)
    _robust_z_kernel        – parallel robust Z-score (0.6745*(x-median)/MAD)
    _weighted_sum_scores    – parallel weighted sum for scoring

When Numba is not installed every kernel falls back to an equivalent pure-NumPy
implementation so the rest of the pipeline remains functional without GPU/JIT support.
"""

from __future__ import annotations

import logging

import numpy as np

# ---------------------------------------------------------------------------
# Numba import (optional)
# ---------------------------------------------------------------------------
try:
    import numba
    from numba import njit, prange
    _HAS_NUMBA = True
except ImportError:
    numba = None          # type: ignore[assignment]
    njit = None           # type: ignore[assignment]
    prange = None         # type: ignore[assignment]
    _HAS_NUMBA = False
    logging.warning("Numba not installed – falling back to pure-NumPy kernel paths.")


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

_WARMED_UP = False

if _HAS_NUMBA:
    @njit(parallel=True, cache=True)
    def _fast_hash_mask(hash_vals: np.ndarray, threshold: np.uint64) -> np.ndarray:
        """Return boolean array: ``hash_vals[i] % 10_000_000 < threshold``.

        Used for fast, deterministic fraction-based row sampling.

        Parameters
        ----------
        hash_vals:
            1-D uint64 array of pre-computed hash values (one per row).
        threshold:
            uint64 value equal to ``int(fraction * 10_000_000)``.

        Returns
        -------
        np.ndarray of dtype bool, same length as *hash_vals*.
        """
        n = hash_vals.shape[0]
        out = np.empty(n, dtype=numba.boolean)
        mod = np.uint64(10_000_000)
        for i in prange(n):
            out[i] = (hash_vals[i] % mod) < threshold
        return out

    @njit(parallel=True, cache=True)
    def _parse_amounts_kernel(amounts_f64: np.ndarray) -> np.ndarray:
        """Return boolean validity mask for a float64 amount array.

        A row is *valid* when its value is finite (not NaN) **and** non-negative.

        Parameters
        ----------
        amounts_f64:
            1-D float64 array of payment amounts.

        Returns
        -------
        np.ndarray of dtype bool.
        """
        n = amounts_f64.shape[0]
        valid = np.empty(n, dtype=numba.boolean)
        for i in prange(n):
            v = amounts_f64[i]
            valid[i] = (v == v) and (v >= 0.0)   # NaN check: NaN != NaN
        return valid

    @njit(parallel=True, cache=True)
    def _robust_z_kernel(x: np.ndarray, eps: float) -> np.ndarray:
        """Compute robust Z-score: ``0.6745 * (x - median) / (MAD + eps)``.

        Parameters
        ----------
        x:
            1-D float64 array.
        eps:
            Small regularisation constant added to MAD to avoid division by zero.

        Returns
        -------
        np.ndarray of dtype float64.
        """
        n = x.shape[0]
        # Median via sorted copy
        tmp = np.empty(n, dtype=np.float64)
        for i in range(n):
            tmp[i] = x[i]
        tmp.sort()
        mid = n // 2
        if n % 2 == 0:
            med = (tmp[mid - 1] + tmp[mid]) * 0.5
        else:
            med = tmp[mid]
        # MAD
        dev = np.empty(n, dtype=np.float64)
        for i in prange(n):
            v = x[i] - med
            dev[i] = v if v >= 0.0 else -v
        dev.sort()
        if n % 2 == 0:
            mad = (dev[mid - 1] + dev[mid]) * 0.5
        else:
            mad = dev[mid]
        denom = mad + eps
        out = np.empty(n, dtype=np.float64)
        for i in prange(n):
            out[i] = 0.6745 * (x[i] - med) / denom
        return out

    @njit(parallel=True, cache=True)
    def _weighted_sum_scores(
        z_in_w: np.ndarray,
        z_in_deg: np.ndarray,
        z_out_w: np.ndarray,
        z_out_deg: np.ndarray,
        w_in_w: float,
        w_in_deg: float,
        w_out_w: float,
        w_out_deg: float,
    ) -> np.ndarray:
        """Compute weighted risk score in parallel."""
        n = z_in_w.shape[0]
        out = np.empty(n, dtype=np.float64)
        for i in prange(n):
            out[i] = (
                w_in_w * z_in_w[i]
                + w_in_deg * z_in_deg[i]
                + w_out_w * z_out_w[i]
                + w_out_deg * z_out_deg[i]
            )
        return out

else:
    # ------------------------------------------------------------------
    # Pure-NumPy fallbacks (identical semantics, no parallelism)
    # ------------------------------------------------------------------

    def _fast_hash_mask(hash_vals: np.ndarray, threshold) -> np.ndarray:  # type: ignore[misc]
        mod = np.uint64(10_000_000)
        return (hash_vals % mod) < threshold

    def _parse_amounts_kernel(amounts_f64: np.ndarray) -> np.ndarray:  # type: ignore[misc]
        return np.isfinite(amounts_f64) & (amounts_f64 >= 0.0)

    def _robust_z_kernel(x: np.ndarray, eps: float) -> np.ndarray:  # type: ignore[misc]
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + eps
        return 0.6745 * (x - med) / mad

    def _weighted_sum_scores(  # type: ignore[misc]
        z_in_w: np.ndarray,
        z_in_deg: np.ndarray,
        z_out_w: np.ndarray,
        z_out_deg: np.ndarray,
        w_in_w: float,
        w_in_deg: float,
        w_out_w: float,
        w_out_deg: float,
    ) -> np.ndarray:
        return (
            w_in_w * z_in_w
            + w_in_deg * z_in_deg
            + w_out_w * z_out_w
            + w_out_deg * z_out_deg
        )


def warmup_all_kernels() -> None:
    """Pre-compile all Numba kernels so the first real call is not slowed by JIT.

    Safe to call even when Numba is unavailable (becomes a no-op).
    """
    global _WARMED_UP
    if _WARMED_UP or not _HAS_NUMBA:
        return
    _dummy_amt = np.array([1.0, -1.0, float("nan")], dtype=np.float64)
    _parse_amounts_kernel(_dummy_amt)
    _dummy_h = np.array([0, 1000, 9_999_999, 10_000_000], dtype=np.uint64)
    _fast_hash_mask(_dummy_h, np.uint64(5_000_000))
    _dummy = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    _robust_z_kernel(_dummy, 1e-9)
    _weighted_sum_scores(_dummy, _dummy, _dummy, _dummy, 0.55, 0.25, 0.10, 0.10)
    _WARMED_UP = True
    logging.info("[Numba] JIT warm-up complete")


def warmup_kernels() -> None:
    """Backward-compatible alias for full kernel warmup."""
    warmup_all_kernels()


# Public aliases for external imports
robust_z_kernel = _robust_z_kernel
weighted_sum_scores = _weighted_sum_scores

