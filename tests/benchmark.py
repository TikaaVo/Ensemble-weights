#!/usr/bin/env python3
"""
Dynamic Ensemble Selection -- Multi-Seed Benchmark
===================================================
Runs the full showcase across N random seeds and reports mean +/- std for
every method on every dataset. This gives a much more reliable picture of
relative performance than a single seed, especially on smaller datasets like
Phoneme where individual runs have high variance.

Imports run_regression and run_classification from showcase.py, which must be
in the same directory (or on PYTHONPATH). All algorithm settings (K, temperature,
thresholds) are read directly from showcase.py so the two files stay in sync.

Usage
-----
  python benchmark.py              # 5 seeds, all 4 datasets
  python benchmark.py --seeds 10   # 10 seeds
  python benchmark.py --seeds 3 --verbose   # print each individual run too

Runtime (MacBook Air M3, default 5 seeds)
  ~25-40 min. Datasets are loaded once and reused across all seeds.
  Use --seeds 3 for a quick ~15 min run.
"""

import argparse
import contextlib
import io
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings('ignore')

try:
    from showcase import (
        run_regression, run_classification,
        load_california, load_bike,
        load_letter, load_phoneme,
        W,
    )
except ImportError:
    print("ERROR: Could not import from showcase.py.")
    print("Make sure showcase.py is in the same directory as benchmark.py.")
    sys.exit(1)


# ── Configuration ─────────────────────────────────────────────────────

DEFAULT_N_SEEDS = 30


def make_seeds(n):
    """Seeds 0..n-1: deterministic and easy to extend by raising n."""
    return list(range(n))


# Each entry: (loader, run_fn, extra_kwargs, label, metric_name, higher_is_better)
DATASETS = [
    (load_california, run_regression,     {},       'California Housing', 'MAE',      False),
    (load_bike,       run_regression,     {},       'Bike Sharing',       'MAE',      False),
    (load_letter,     run_classification, {'k': 20}, 'Letter Recognition', 'Accuracy', True),
    (load_phoneme,    run_classification, {'k': 10}, 'Phoneme',            'Accuracy', True),
]


# ── Data loading ──────────────────────────────────────────────────────

def load_silently(loader):
    """Call a loader and return its result without printing."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return loader()


def make_cached_loader(data):
    """Wrap pre-loaded data so it can be passed to run_* as a loader callable."""
    return lambda: data


# ── Multi-seed runner ─────────────────────────────────────────────────

def run_all_seeds(n_seeds, verbose_runs=False):
    """
    Run every dataset over all seeds.

    Returns
    -------
    dict[dataset_label -> dict[method_label -> list[float]]]
        Raw per-seed scores for aggregation.
    """
    seeds = make_seeds(n_seeds)

    # Load datasets once upfront — OpenML has network latency.
    print(f"\n  Pre-loading datasets (fetched once, reused across all {n_seeds} seeds)...")
    loaded = {}
    for loader, _, _, label, _, _ in DATASETS:
        print(f"    {label}...", end=' ', flush=True)
        t0 = time.time()
        loaded[label] = load_silently(loader)
        print(f"done  ({time.time()-t0:.1f}s)")

    results = {label: {} for _, _, _, label, _, _ in DATASETS}

    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'━' * W}")
        print(f"  Seed {seed}  ({seed_idx + 1}/{n_seeds})")
        print(f"{'━' * W}")

        for loader, run_fn, kwargs, label, _, _ in DATASETS:
            cached_loader = make_cached_loader(loaded[label])
            print(f"  [{label}]", end='  ', flush=True)
            t0 = time.time()
            scores = run_fn(cached_loader, seed=seed, verbose=verbose_runs, **kwargs)
            print(f"done  ({time.time()-t0:.0f}s)")

            for method, score in scores.items():
                results[label].setdefault(method, []).append(score)

    return results


# ── Summary display ───────────────────────────────────────────────────

def print_summary(results, n_seeds):
    seeds = make_seeds(n_seeds)

    print(f"\n\n{'━' * W}")
    print(f"  Multi-Seed Benchmark Summary  --  {n_seeds} seeds: {seeds}")
    print(f"{'━' * W}")

    for _, _, _, label, metric_name, higher in DATASETS:
        ds   = results[label]
        methods = list(ds.keys())
        vals    = {m: np.array(ds[m]) for m in methods}

        print(f"\n  {label}  ({metric_name}, {'higher' if higher else 'lower'} is better)")
        print(f"  {'-' * (W - 4)}")

        # "Best Single" is always the first row; use it as the vs-Best reference.
        ref_method = methods[0]
        ref_mean   = vals[ref_method].mean()

        best_mean = (max if higher else min)(v.mean() for v in vals.values())

        if higher:
            hdr = f"  {'Method':<44}  {'Mean':>9}  {'Std':>6}  {'Min':>9}  {'Max':>9}  {'vs Best':>9}"
        else:
            hdr = f"  {'Method':<44}  {'Mean':>9}  {'Std':>8}  {'Min':>9}  {'Max':>9}  {'vs Best':>9}"
        print(hdr)
        print(f"  {'-'*44}  {'-'*9}  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*9}")

        for method in methods:
            v     = vals[method]
            mean  = v.mean()
            std   = v.std()
            delta = (mean - ref_mean) / abs(ref_mean) * 100
            d_str = "    -    " if method == ref_method else \
                    f"{'+' if delta >= 0 else ''}{delta:.2f}%"
            marker = "  <" if abs(mean - best_mean) < 1e-12 else ""

            if higher:
                print(f"  {method:<44}  {mean*100:>8.2f}%  {std*100:>6.2f}%"
                      f"  {v.min()*100:>8.2f}%  {v.max()*100:>8.2f}%  {d_str:>9}{marker}")
            else:
                print(f"  {method:<44}  {mean:>9.4f}  {std:>8.4f}"
                      f"  {v.min():>9.4f}  {v.max():>9.4f}  {d_str:>9}{marker}")

        # Per-seed breakdown so you can spot outlier seeds at a glance.
        print(f"\n  Per-seed ({metric_name}):")
        seed_hdr = "  " + "".join(f"  seed {s:>2}" for s in seeds)
        print(seed_hdr)
        for method in methods:
            short = (method[:41] + "...") if len(method) > 44 else method
            if higher:
                row = "".join(f"  {v*100:>8.2f}%" for v in vals[method])
            else:
                row = "".join(f"  {v:>9.4f}" for v in vals[method])
            print(f"  {short:<44}{row}")


# ── Entry point ───────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-seed DES benchmark. Imports showcase.py and averages results over N seeds."
    )
    p.add_argument(
        '--seeds', type=int, default=DEFAULT_N_SEEDS,
        help=f"Number of seeds to run (default: {DEFAULT_N_SEEDS}). "
             "Seeds are always 0, 1, 2, ... so results are fully reproducible."
    )
    p.add_argument(
        '--verbose', action='store_true',
        help="Print the full per-run output for every seed (very long; off by default)."
    )
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    print(f"\n{'━' * W}")
    print(f"  Dynamic Ensemble Selection -- Multi-Seed Benchmark")
    print(f"{'━' * W}")
    print(f"  Seeds     : {args.seeds}  ({make_seeds(args.seeds)})")
    print(f"  Datasets  : California Housing, Bike Sharing, Letter Recognition, Phoneme")
    print(f"  Verbose   : {'yes' if args.verbose else 'no  (--verbose to enable per-run output)'}")
    print(f"{'━' * W}")

    t_total = time.time()
    results = run_all_seeds(n_seeds=args.seeds, verbose_runs=args.verbose)
    print_summary(results, n_seeds=args.seeds)

    elapsed = time.time() - t_total
    print(f"\n{'━' * W}")
    print(f"  Total runtime: {elapsed/60:.1f} min")
    print(f"{'━' * W}\n")