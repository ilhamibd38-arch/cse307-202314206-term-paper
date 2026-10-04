"""Synthetic page-reference trace with a deliberate workload shift.

Phase 1 (first half): locality-heavy. A small working set that drifts slowly,
plus short sequential scans.
Phase 2 (second half): random / bursty. Uniform random pages with occasional
bursts on a few "hot" pages.
"""
import numpy as np

N_PAGES = 50


def _phase_local(rng, n, n_pages, ws=5):
    out = []
    base = int(rng.integers(n_pages))
    while len(out) < n:
        base = (base + 4) % n_pages          # working set drifts every block
        block = []
        while len(block) < 500:
            if rng.random() < 0.03:          # short sequential scan
                s = int(rng.integers(n_pages))
                length = int(rng.integers(3, 6))
                block.extend((s + i) % n_pages for i in range(length))
            else:                            # reference inside the working set
                block.append((base + int(rng.integers(ws))) % n_pages)
        out.extend(block)
    return out[:n]


def _phase_random(rng, n, n_pages):
    out = []
    while len(out) < n:
        if rng.random() < 0.04:              # occasional burst on 3 hot pages
            hot = rng.choice(n_pages, size=3, replace=False)
            length = int(rng.integers(8, 16))
            out.extend(int(x) for x in rng.choice(hot, size=length))
        else:                                # uniform random reference
            out.append(int(rng.integers(n_pages)))
    return out[:n]


def generate_trace(seed, length=5000, n_pages=N_PAGES):
    """Return (trace, shift_index). trace[:shift] = phase 1, trace[shift:] = phase 2."""
    rng = np.random.default_rng(seed)
    shift = length // 2
    first = _phase_local(rng, shift, n_pages)
    second = _phase_random(rng, length - shift, n_pages)
    return first + second, shift
