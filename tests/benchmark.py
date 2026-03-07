#!/usr/bin/env python3
"""
Dynamic Ensemble Selection -- Multi-Seed Benchmark
===================================================
Runs the full showcase across N random seeds and reports mean +/- std for
every method on every dataset. This gives a much more reliable picture of
relative performance than a single seed, especially on smaller datasets like
Abalone and Diabetes where individual runs have high variance.

Imports run_regression and run_classification from showcase.py, which must be
in the same directory (or on PYTHONPATH). All algorithm settings (K, temperature,
thresholds) are read directly from showcase.py so the two files stay in sync.

Usage
-----
  python benchmark.py              # 10 seeds, all 10 datasets
  python benchmark.py --seeds 30   # 30 seeds (more reliable)
  python benchmark.py --seeds 5 --verbose   # print each individual run too

Runtime (MacBook Air M3, default 10 seeds)
  ~60-90 min. Datasets are loaded once and reused across all seeds.
  Use --seeds 5 for a quick ~30 min run.
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
        load_california, load_bike, load_abalone,
        load_diabetes_data, load_concrete,
        load_har, load_yeast, load_segment, load_vowel, load_waveform,
        W,
    )
except ImportError:
    print("ERROR: Could not import from showcase.py.")
    print("Make sure showcase.py is in the same directory as benchmark.py.")
    sys.exit(1)


# ── Configuration ─────────────────────────────────────────────────────

DEFAULT_N_SEEDS = 100


def make_seeds(n):
    """Seeds 0..n-1: deterministic and easy to extend by raising n."""
    return list(range(n))


# Each entry: (loader, run_fn, extra_kwargs, label, metric_name, higher_is_better)
DATASETS = [
    (load_california,    run_regression,     {},        'California Housing', 'MAE',      False),
    (load_bike,          run_regression,     {},        'Bike Sharing',       'MAE',      False),
    (load_abalone,       run_regression,     {},        'Abalone',            'MAE',      False),
    (load_diabetes_data, run_regression,     {},        'Diabetes',           'MAE',      False),
    (load_concrete,      run_regression,     {},        'Concrete Strength',  'MAE',      False),
    (load_har,           run_classification, {'k': 20}, 'HAR',                'Accuracy', True),
    (load_yeast,         run_classification, {'k': 10}, 'Yeast',              'Accuracy', True),
    (load_segment,       run_classification, {'k': 10}, 'Image Segment',      'Accuracy', True),
    (load_vowel,         run_classification, {'k': 10}, 'Vowel',              'Accuracy', True),
    (load_waveform,      run_classification, {'k': 10}, 'Waveform',           'Accuracy', True),
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

    results     = {label: {} for _, _, _, label, _, _ in DATASETS}
    timing_fit  = {label: {} for _, _, _, label, _, _ in DATASETS}
    timing_pred = {label: {} for _, _, _, label, _, _ in DATASETS}

    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'━' * W}")
        print(f"  Seed {seed}  ({seed_idx + 1}/{n_seeds})")
        print(f"{'━' * W}")

        for loader, run_fn, kwargs, label, _, _ in DATASETS:
            cached_loader = make_cached_loader(loaded[label])
            print(f"  [{label}]", end='  ', flush=True)
            t0 = time.time()
            # Always call with verbose=True but suppress stdout — this ensures
            # DESlib (and any other block gated on verbose) always runs and
            # populates the returned scores dict. Per-run output is then
            # re-printed only when --verbose was requested.
            _buf = io.StringIO()
            with contextlib.redirect_stdout(_buf):
                result = run_fn(cached_loader, seed=seed, verbose=True, **kwargs)
            if verbose_runs:
                print(_buf.getvalue(), end='')
            print(f"done  ({time.time()-t0:.0f}s)")
            scores, fit_ms, pred_ms = result if isinstance(result, tuple) else (result, {}, {})

            for method, score in scores.items():
                # Normalize 'Best Single  (Model Name)' → 'Best Single'
                # so results from different seeds collapse into one row.
                key = 'Best Single' if method.startswith('Best Single') else method
                results[label].setdefault(key, []).append(score)
            for method, ms in fit_ms.items():
                timing_fit[label].setdefault(method, []).append(ms)
            for method, ms in pred_ms.items():
                timing_pred[label].setdefault(method, []).append(ms)

    return results, timing_fit, timing_pred


# ── Summary display ───────────────────────────────────────────────────

def print_summary(results, timing_fit, timing_pred, n_seeds):
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
        ref_clean  = vals[ref_method][vals[ref_method] != np.array(None)].astype(float)
        ref_mean   = ref_clean.mean()

        def _safe_mean(v):
            clean = v[v != np.array(None)].astype(float)
            return clean.mean() if len(clean) else (float('-inf') if higher else float('inf'))
        best_mean = (max if higher else min)(_safe_mean(v) for v in vals.values())

        if higher:
            hdr = f"  {'Method':<44}  {'Mean':>9}  {'Std':>6}  {'Min':>9}  {'Max':>9}  {'vs Best':>9}"
        else:
            hdr = f"  {'Method':<44}  {'Mean':>9}  {'Std':>8}  {'Min':>9}  {'Max':>9}  {'vs Best':>9}"
        print(hdr)
        print(f"  {'-'*44}  {'-'*9}  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*9}")

        for method in methods:
            v_raw  = vals[method]
            valid  = v_raw[v_raw != np.array(None)].astype(float)
            n_ok   = len(valid)
            if n_ok == 0:
                print(f"  {method:<44}  {'N/A':>9}  (all seeds failed)")
                continue
            mean  = valid.mean()
            std   = valid.std()
            delta = (mean - ref_mean) / abs(ref_mean) * 100
            d_str = "    -    " if method == ref_method else \
                    f"{'+' if delta >= 0 else ''}{delta:.2f}%"
            marker = "  <" if abs(mean - best_mean) < 1e-12 else ""
            miss   = f"  [{n_ok}/{len(v_raw)}]" if n_ok < len(v_raw) else ""

            if higher:
                print(f"  {method:<44}  {mean*100:>8.2f}%  {std*100:>6.2f}%"
                      f"  {valid.min()*100:>8.2f}%  {valid.max()*100:>8.2f}%  {d_str:>9}{marker}{miss}")
            else:
                print(f"  {method:<44}  {mean:>9.4f}  {std:>8.4f}"
                      f"  {valid.min():>9.4f}  {valid.max():>9.4f}  {d_str:>9}{marker}{miss}")

        # Per-seed breakdown so you can spot outlier seeds at a glance.
        print(f"\n  Per-seed ({metric_name}):")
        seed_hdr = "  " + "".join(f"  seed {s:>2}" for s in seeds)
        print(seed_hdr)
        for method in methods:
            short = (method[:41] + "...") if len(method) > 44 else method
            def _fmt(v, higher):
                if v is None: return f"{'N/A':>9}  " if higher else f"{'N/A':>10} "
                return f"  {v*100:>8.2f}%" if higher else f"  {v:>9.4f}"
            row = "".join(_fmt(v, higher) for v in vals[method])
            print(f"  {short:<44}{row}")

        # Timing summary — mean fit + predict ms across seeds
        t_fit  = timing_fit.get(label, {})
        t_pred = timing_pred.get(label, {})
        if t_fit:
            print(f"\n  Timing (mean ms, {n_seeds} seeds):")
            print(f"  {'Method':<44}  {'Fit':>8}  {'Predict':>9}  {'Total':>7}")
            print(f"  {'-'*44}  {'-'*8}  {'-'*9}  {'-'*7}")
            for m in t_fit:
                fa = [v for v in t_fit[m]          if v is not None and not np.isnan(v)]
                pa = [v for v in t_pred.get(m, []) if v is not None and not np.isnan(v)]
                if not fa: continue
                fm = np.mean(fa); pm = np.mean(pa) if pa else float('nan')
                lbl = (m[:41] + '...') if len(m) > 44 else m
                print(f"  {lbl:<44}  {fm:>8.1f}  {pm:>9.1f}  {fm+pm:>7.1f}")


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
    print(f"  Datasets  : California Housing, Bike Sharing, Abalone,")
    print(f"              Diabetes, Concrete Strength  (regression)")
    print(f"              HAR, Yeast, Image Segment, Vowel, Waveform  (classification)")
    print(f"  Pool      : KNN · Decision Tree · SVR/SVM-RBF · Ridge/NB · Bayesian Ridge/LR")
    print(f"  Verbose   : {'yes' if args.verbose else 'no  (--verbose to enable per-run output)'}")
    print(f"{'━' * W}")

    t_total = time.time()
    results, timing_fit, timing_pred = run_all_seeds(n_seeds=args.seeds, verbose_runs=args.verbose)
    print_summary(results, timing_fit, timing_pred, n_seeds=args.seeds)

    elapsed = time.time() - t_total
    print(f"\n{'━' * W}")
    print(f"  Total runtime: {elapsed/60:.1f} min")
    print(f"{'━' * W}\n")