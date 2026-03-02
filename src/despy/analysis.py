"""
DES suitability analysis.

Analyses a validation set and model pool to estimate whether Dynamic Ensemble
Selection is likely to help, and by how much.

The function prints a formatted report and returns a dict of raw metrics for
programmatic use.
"""
import numpy as np
from despy._config    import resolve_metric, prep_fit_inputs
from despy.base.knnbase import KNNBase
from despy.neighbors  import KNNNeighborFinder

# Internal helpers

class _AnalysisFitter(KNNBase):
    """Minimal KNNBase subclass used only to build the score matrix."""
    def predict(self, x, temperature=None, threshold=None):
        raise NotImplementedError("_AnalysisFitter is not used for prediction.")


def _entropy(probs):
    """Shannon entropy of a probability distribution, normalised to [0, 1]."""
    probs = np.asarray(probs, dtype=float)
    probs = probs[probs > 0]
    n     = len(probs) + (1 - len(probs)) % 1   # guard for single-element
    raw   = -np.sum(probs * np.log(probs))
    max_e = np.log(max(len(probs), 2))
    return raw / max_e if max_e > 0 else 0.0


def _bar(value, width=30, fill='█', empty='░'):
    """Render a simple ASCII bar for a value in [0, 1]."""
    n = round(value * width)
    return fill * n + empty * (width - n)


# Public API

def analyze(features, y, preds_dict, metric, mode, k=20, verbose=True):
    """
    Analyse a validation set and model pool for DES suitability.

    Computes four complementary signals and prints an interpreted report.

    Parameters
    ----------
    features : array-like, shape (n_val, n_features)
        Validation features. Should be the same set you would pass to fit().
    y : array-like, shape (n_val,)
        Validation ground-truth labels or values.
    preds_dict : dict[str, array-like]
        Validation predictions keyed by model name.
        Shape (n_val,) for scalar metrics; (n_val, n_classes) for probability
        metrics (log_loss, prob_correct).
    metric : str or callable
        Same metric you intend to use for the DES router.
    mode : str
        'min' if lower scores are better (mae, log_loss), 'max' if higher
        (accuracy, prob_correct).
    k : int
        Neighbourhood size. Should match the k you intend to use for routing.
        Default: 20.
    verbose : bool
        If True (default), print the formatted report. If False, return the
        metrics dict silently.

    Returns
    -------
    dict with keys:
        n_val            int    Number of validation samples.
        n_features       int    Number of features.
        n_models         int    Number of models in the pool.
        k                int    Neighbourhood size used.
        model_scores     dict   Mean score per model (higher-is-better scale).
        best_model       str    Name of the globally best model.
        oracle_gain      float  Fractional improvement of the local oracle over
                                the best single model. 0.0 = no gain possible;
                                0.15 = 15% headroom.
        estimated_gain   float  Conservative estimate of realised DES gain
                                (~5% of oracle_gain; empirical range 1-5% across benchmarks).
        regional_diversity  float  Entropy of model win shares across
                                   neighbourhoods, normalised to [0, 1].
                                   0 = one model wins everywhere;
                                   1 = wins perfectly distributed.
        model_win_shares    dict   Fraction of neighbourhoods each model wins.
        disagreement        float  Mean fraction of sample pairs where the
                                   locally best model differs. Regression only;
                                   None for probability inputs.
        val_quality         float  n_val / k, clamped to [0, 1] against a
                                   target of 100. Proxy for neighbourhood
                                   estimate reliability.
        recommendation      str    'USE DES', 'MAYBE', or 'SKIP DES'.
        reason              str    Plain-English explanation of recommendation.
    """
    metric_name, metric_fn = resolve_metric(metric)
    features, y, preds_dict = prep_fit_inputs(features, y, preds_dict, metric_name)

    n_val, n_features = features.shape
    n_models          = len(preds_dict)
    model_names       = list(preds_dict.keys())

    # Build score matrix
    finder = KNNNeighborFinder(k=k + 1)
    fitter = _AnalysisFitter(metric=metric_fn, mode=mode, neighbor_finder=finder)
    fitter.fit(features, y, preds_dict)

    matrix = fitter.matrix

    # Per-model global mean scores
    model_scores = {name: float(matrix[:, j].mean())
                    for j, name in enumerate(model_names)}
    best_model   = max(model_scores, key=model_scores.get)
    best_score   = model_scores[best_model]

    # Oracle gain
    oracle_scores  = matrix.max(axis=1)           # (n_val,)
    oracle_mean    = oracle_scores.mean()

    # Gain is relative to the best single model's mean score.
    if abs(best_score) > 1e-12:
        oracle_gain = float((oracle_mean - best_score) / abs(best_score))
    else:
        oracle_gain = 0.0

    # Empirically, DES captures roughly 1-5% of oracle headroom.
    estimated_gain = oracle_gain * 0.02

    # Regional diversity
    _, indices = fitter.model.kneighbors(features, k=k + 1)
    indices    = indices[:, 1:]                   # (n_val, k) — drop self

    neighborhood_scores = matrix[indices].mean(axis=1)   # (n_val, n_models)
    local_winners       = np.argmax(neighborhood_scores, axis=1)  # (n_val,)

    # Win share
    win_counts  = np.bincount(local_winners, minlength=n_models)
    win_shares  = win_counts / n_val
    model_wins  = {name: float(win_shares[j]) for j, name in enumerate(model_names)}

    # Regional diversity
    regional_diversity = float(_entropy(win_shares))

    # Local uplift
    global_best_idx  = model_names.index(best_model)
    nbhd_best        = neighborhood_scores.max(axis=1)          # (n_val,)
    nbhd_global_best = neighborhood_scores[:, global_best_idx]  # (n_val,)
    nbhd_improvement = nbhd_best - nbhd_global_best

    denom = abs(nbhd_global_best.mean())
    local_uplift = float(nbhd_improvement.mean() / denom) if denom > 1e-12 else 0.0

    # Model disagreement
    first_preds = next(iter(preds_dict.values()))
    if np.asarray(first_preds).ndim == 1:
        # Pairwise disagreement
        disagreement = float(1.0 - np.sum(win_shares ** 2))
    else:
        disagreement = None

    # Validation set quality
    # n_val / k >= 100 = stable neighbourhood estimates.
    val_quality = float(min(1.0, (n_val / k) / 100.0))

    # KNN learnability
    feature_score  = float(np.clip((n_features - 2) / 10.0, 0.0, 1.0))
    knn_learnability = val_quality * feature_score

    # Recommendation

    if val_quality < 0.2:
        recommendation = 'UNRELIABLE'
        reason = (
            f"Validation set too small relative to k "
            f"(n_val/k = {n_val/k:.0f}, recommended \u2265 100). "
            f"Reduce k or increase the validation set before drawing conclusions."
        )
    elif local_uplift < 0.01 and regional_diversity < 0.4:
        recommendation = 'SKIP DES'
        reason = (
            f"Local uplift is negligible ({local_uplift*100:.1f}%) and regional "
            f"diversity is low ({regional_diversity:.2f}). The routing signal is "
            f"too weak to improve on the best single model. Use it directly."
        )
    elif knn_learnability < 0.25 or local_uplift < 0.05 or regional_diversity < 0.45:
        weak_reasons = []
        if knn_learnability < 0.25:
            weak_reasons.append(
                f"KNN learnability is low ({knn_learnability:.2f}) — "
                f"{'the feature space is too small' if feature_score < 0.5 else 'the val set is too sparse'} "
                f"for stable competence regions"
            )
        if local_uplift < 0.05:
            weak_reasons.append(f"local uplift is modest ({local_uplift*100:.1f}%)")
        if regional_diversity < 0.45:
            weak_reasons.append(f"regional diversity is low ({regional_diversity:.2f})")
        recommendation = 'MAYBE \u2014 try Global Ensemble first'
        reason = (
            f"{'; '.join(weak_reasons).capitalize()}. "
            f"A fixed-weight global ensemble may capture most of the gain "
            f"with less variance. Run both and compare on held-out test data."
        )
    else:
        recommendation = 'USE DES'
        reason = (
            f"Local uplift ({local_uplift*100:.1f}%), regional diversity "
            f"({regional_diversity:.2f}), and KNN learnability ({knn_learnability:.2f}) "
            f"are all strong. DES is likely to improve on both the best single "
            f"model and the global ensemble."
        )

    # Assemble results
    global_best_win_share = float(model_wins[best_model])

    result = {
        'n_val':                 n_val,
        'n_features':            n_features,
        'n_models':              n_models,
        'k':                     k,
        'model_scores':          model_scores,
        'best_model':            best_model,
        'oracle_gain':           oracle_gain,
        'estimated_gain':        estimated_gain,
        'local_uplift':          local_uplift,
        'regional_diversity':    regional_diversity,
        'global_best_win_share': global_best_win_share,
        'model_win_shares':      model_wins,
        'disagreement':          disagreement,
        'val_quality':           val_quality,
        'feature_score':         feature_score,
        'knn_learnability':      knn_learnability,
        'recommendation':        recommendation,
        'reason':                reason,
    }

    if verbose:
        _print_report(result)

    return result

# Report formatting

def _print_report(r):
    W = 72

    def _section(title):
        print(f"\n  {title}")
        print(f"  {'─' * (W - 4)}")

    print(f"\n  {'━' * W}")
    print(f"  DES Suitability Analysis")
    print(f"  {'━' * W}")
    print(f"  {r['n_val']:,} val samples  ·  {r['n_features']} features  ·  "
          f"{r['n_models']} models  ·  k = {r['k']}")

    # Model scores
    _section("Model scores  (higher-is-better scale)")
    scores  = list(r['model_scores'].values())
    s_min   = min(scores)
    s_max   = max(scores)
    s_range = s_max - s_min
    for name, score in r['model_scores'].items():
        marker   = '  ← best' if name == r['best_model'] else ''
        # Bar shows each model's relative position in the pool score range.
        bar_val  = (score - s_min) / s_range if s_range > 0 else 1.0
        bar      = _bar(bar_val)
        print(f"    {name:<22}  {score:+.4f}  {bar}{marker}")

    # Oracle gain
    _section("Oracle / local uplift  (headroom and realisable gain)")
    og = r['oracle_gain']
    lu = r['local_uplift']
    eg = r['estimated_gain']
    bar_og = _bar(min(og / 0.20, 1.0))
    bar_lu = _bar(min(lu / 0.10, 1.0))
    print(f"    Oracle gain      {og*100:+6.2f}%  {bar_og}")
    print(f"    Local uplift     {lu*100:+6.2f}%  {bar_lu}")
    print(f"    Estimated DES    {eg*100:+6.2f}%  (≤2% of oracle; empirical range 1–5%)")
    print()
    if og < 0.02:
        note = "Very little headroom — models already agree on most samples."
    elif og < 0.05:
        note = "Moderate headroom — DES may help, depends on regional structure."
    elif og < 0.12:
        note = "Good headroom — DES has meaningful potential."
    else:
        note = "Large headroom — strong case for DES."
    print(f"    Oracle gain is the per-sample ceiling. Local uplift is the")
    print(f"    neighbourhood-level routing advantage that must generalise to")
    print(f"    test data. Low local uplift with high oracle gain usually means")
    print(f"    the routing signal is noisy (val set too small or too few features).")

    # Regional diversity
    _section("Regional diversity  (do different models win in different areas?)")
    rd  = r['regional_diversity']
    bar_rd = _bar(rd)
    print(f"    Diversity score  {rd:.3f}  {bar_rd}")
    print()
    print(f"    {'Model':<22}  {'Win share':>10}  {'Neighbourhoods'}")
    print(f"    {'─'*22}  {'─'*10}  {'─'*14}")
    for name, share in sorted(r['model_win_shares'].items(),
                               key=lambda x: -x[1]):
        bar = _bar(share, width=20)
        print(f"    {name:<22}  {share*100:>9.1f}%  {bar}")
    print()
    if rd < 0.35:
        note = "Low diversity — one model dominates most regions."
    elif rd < 0.65:
        note = "Moderate diversity — some regional structure present."
    else:
        note = "High diversity — models have distinct regional strengths."
    print(f"    {note}")
    # Warn when the local neighbourhood winner differs from the global best.
    gbws         = r['global_best_win_share']
    local_leader = max(r['model_win_shares'], key=r['model_win_shares'].get)
    if local_leader != r['best_model']:
        share_leader = r['model_win_shares'][local_leader] * 100
        print()
        print(f"    ⚠  Local winner '{local_leader}' ({share_leader:.0f}% of regions)")
        print(f"       differs from global best '{r['best_model']}' ({gbws*100:.0f}%).")
        print(f"       A weaker model dominating locally often signals noisy")
        print(f"       neighbourhood estimates. Cross-check local uplift.")

    # Disagreement
    if r['disagreement'] is not None:
        _section("Model disagreement  (how often does the locally-best model vary?)")
        d      = r['disagreement']
        bar_d  = _bar(d)
        print(f"    Disagreement     {d:.3f}  {bar_d}")
        print()
        if d < 0.3:
            note = "Low disagreement — models make similar errors."
        elif d < 0.6:
            note = "Moderate disagreement — routing has something to work with."
        else:
            note = "High disagreement — strong routing signal available."
        print(f"    {note}")

    # Validation set quality & KNN learnability
    _section("KNN learnability  (can the router learn stable competence regions?)")
    vq      = r['val_quality']
    fs      = r['feature_score']
    kl      = r['knn_learnability']
    ratio   = r['n_val'] / r['k']
    bar_vq  = _bar(vq)
    bar_fs  = _bar(fs)
    bar_kl  = _bar(kl)
    vq_status = '✓' if ratio >= 100 else ('~' if ratio >= 50 else '✗')
    fs_status = '✓' if r['n_features'] >= 12 else ('~' if r['n_features'] >= 6 else '✗')
    kl_status = '✓' if kl >= 0.5 else ('~' if kl >= 0.25 else '✗')
    print(f"    n_val / k  = {ratio:>5.0f}  {bar_vq}  {vq_status}  (sample density)")
    print(f"    n_features = {r['n_features']:>5d}  {bar_fs}  {fs_status}  (distance structure)")
    print(f"    learnability = {kl:.2f}  {bar_kl}  {kl_status}  (combined)")
    print()
    if kl < 0.25:
        note = ("Low — KNN is unlikely to find stable competence regions. "
                "Consider Global Ensemble instead.")
    elif kl < 0.5:
        note = "Moderate — competence regions may be noisy. Treat results with caution."
    else:
        note = "Good — KNN has enough data and feature structure to route reliably."
    print(f"    {note}")

    # Recommendation
    print(f"\n  {'━' * W}")
    rec = r['recommendation']
    symbols = {
        'USE DES':                       '✓',
        'MAYBE — try Global Ensemble first': '~',
        'SKIP DES':                      '✗',
        'UNRELIABLE':                    '?',
    }
    symbol = symbols.get(rec, ' ')
    print(f"  Recommendation:  [{symbol}]  {rec}")
    print(f"\n  {r['reason']}")
    print(f"  {'━' * W}\n")