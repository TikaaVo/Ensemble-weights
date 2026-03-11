import openml
import pandas as pd
import numpy as np
import time
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, Lasso, HuberRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
import warnings
from scipy.linalg import LinAlgWarning
warnings.filterwarnings('ignore', 'Ill-conditioned matrix', category=LinAlgWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning, module='deskit')

from deskit.des.dewsu  import DEWSU
from deskit.des.dewsi  import DEWSI
from deskit.des.dewst  import DEWST
from deskit.des.dewsv  import DEWSV
from deskit.des.dewsiv import DEWSIV
from deskit.des.lwseu  import LWSEU
from deskit.des.lwsei  import LWSEI

K_VALUES    = [10, 25, 50, 100]
POOL_NAMES  = {0: 'trees', 1: 'random_forest', 2: 'heterogeneous_large', 3: 'heterogeneous_small'}

des_methods = ['DEWS-U', 'DEWS-I', 'DEWS-T', 'DEWS-V', 'DEWS-IV', 'LWSE-U', 'LWSE-I']
des_classes = {
    'DEWS-U':  DEWSU,
    'DEWS-I':  DEWSI,
    'DEWS-T':  DEWST,
    'DEWS-V':  DEWSV,
    'DEWS-IV': DEWSIV,
    'LWSE-U':  LWSEU,
    'LWSE-I':  LWSEI,
}

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def get_datasets():
    suite = openml.study.get_suite(353)
    valid_tasks = []
    for task_id in suite.tasks:
        task    = openml.tasks.get_task(task_id)
        dataset = openml.datasets.get_dataset(task.dataset_id, download_data=False)
        q       = dataset.qualities
        if (q['NumberOfInstances'] >= 1000
                and q['NumberOfFeatures'] < 100
                and q['NumberOfMissingValues'] == 0):
            valid_tasks.append(task_id)
    print(f"Using {len(valid_tasks)} datasets.")
    return valid_tasks

def load_task(task_id):
    task    = openml.tasks.get_task(task_id)
    dataset = task.get_dataset()
    X, y, _, _ = dataset.get_data(target=task.target_name)
    X = pd.get_dummies(X)
    X = np.ascontiguousarray(X.values, dtype=np.float32)
    y = y.values.astype(np.float32)
    return X, y, dataset.name

# ---------------------------------------------------------------------------
# Pool constructors
# ---------------------------------------------------------------------------

def pool_trees(random_state=42):
    models = []
    for depth in [2, 3, 4, 5, 7, 10, 15, None]:
        for leaf in [1, 2, 4, 8, 16, 32, 64]:
            if depth is None or depth > 3 or leaf < 16:
                models.append(DecisionTreeRegressor(
                    max_depth=depth, min_samples_leaf=leaf, random_state=random_state))
    assert len(models) == 50, f"Expected 50, got {len(models)}"
    return models

def pool_large(random_state=42):
    models = []
    for n in [1, 2, 3, 5, 7, 10, 15, 20, 30, 50]:
        models.append(Pipeline([('scaler', StandardScaler()),
                                ('model',  KNeighborsRegressor(n_neighbors=n))]))
    for depth in [3, 5, 7, 10, None]:
        for leaf in [1, 5]:
            models.append(DecisionTreeRegressor(
                max_depth=depth, min_samples_leaf=leaf, random_state=random_state))
    for alpha in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]:
        models.append(Pipeline([('scaler', StandardScaler()),
                                ('model',  Ridge(alpha=alpha))]))
    for alpha in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0]:
        models.append(TransformedTargetRegressor(
            regressor=Pipeline([('scaler', StandardScaler()),
                                ('model',  Lasso(alpha=alpha, max_iter=10000))]),
            transformer=StandardScaler(), check_inverse=False))
    for epsilon in [1.1, 1.35, 1.5, 2.0, 3.0]:
        for alpha in [0.01, 0.1]:
            models.append(TransformedTargetRegressor(
                regressor=Pipeline([('scaler', StandardScaler()),
                                    ('model',  HuberRegressor(epsilon=epsilon, alpha=alpha, max_iter=2000))]),
                transformer=StandardScaler(), check_inverse=False))
    assert len(models) == 50, f"Expected 50, got {len(models)}"
    return models

def pool_small(random_state=42):
    models = []
    for n in [3, 10]:
        models.append(Pipeline([('scaler', StandardScaler()),
                                ('model',  KNeighborsRegressor(n_neighbors=n))]))
    for pair in [(5, 5), (None, 1)]:
        models.append(DecisionTreeRegressor(
            max_depth=pair[0], min_samples_leaf=pair[1], random_state=random_state))
    for alpha in [0.001, 1]:
        models.append(Pipeline([('scaler', StandardScaler()),
                                ('model',  Ridge(alpha=alpha))]))
    for alpha in [0.01, 1.0]:
        models.append(TransformedTargetRegressor(
            regressor=Pipeline([('scaler', StandardScaler()),
                                ('model',  Lasso(alpha=alpha, max_iter=10000))]),
            transformer=StandardScaler(), check_inverse=False))
    for pair in [(1.35, 0.1), (1.1, 0.01)]:
        models.append(TransformedTargetRegressor(
            regressor=Pipeline([('scaler', StandardScaler()),
                                ('model',  HuberRegressor(epsilon=pair[0], alpha=pair[1], max_iter=2000))]),
            transformer=StandardScaler(), check_inverse=False))
    assert len(models) == 10, f"Expected 10, got {len(models)}"
    return models

# ---------------------------------------------------------------------------
# Cross-validation split
# ---------------------------------------------------------------------------

def split_data(X, y, random_state=42):
    kf = KFold(n_splits=10, shuffle=True, random_state=random_state)
    for temp_idx, test_idx in kf.split(X):
        X_temp, X_test = X[temp_idx], X[test_idx]
        y_temp, y_test = y[temp_idx], y[test_idx]
        split   = int(len(X_temp) * 0.75)
        X_train = X_temp[:split];  X_comp = X_temp[split:]
        y_train = y_temp[:split];  y_comp = y_temp[split:]
        yield X_train, X_comp, X_test, y_train, y_comp, y_test

# ---------------------------------------------------------------------------
# DES algorithm factory
# ---------------------------------------------------------------------------

def des_algorithms(k, preset="balanced"):
    des = []
    for method in des_methods:
        if method not in ('LWSE-U', 'LWSE-I'):
            des.append((method, des_classes[method](
                task="regression", metric="mae", mode="min", preset=preset, k=k)))
        else:
            des.append((method, des_classes[method](
                task="regression", preset=preset, k=k)))
    return des

# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def baseline_simple_average(test_preds, pool_size):
    preds_matrix = np.array([test_preds[m] for m in range(pool_size)])
    return preds_matrix.mean(axis=0)

def baseline_val_weighted_average(val_preds, test_preds, y_comp, pool_size):
    """Weight each model by inverse comp-set MAE."""
    errors = np.array([mean_absolute_error(y_comp, val_preds[m]) for m in range(pool_size)])
    # avoid division by zero
    inv_errors = 1.0 / np.maximum(errors, 1e-12)
    weights = inv_errors / inv_errors.sum()
    preds_matrix = np.array([test_preds[m] for m in range(pool_size)])
    return np.dot(weights, preds_matrix)

def baseline_ridge_stacking(val_preds, test_preds, y_comp, pool_size):
    """Fit a Ridge meta-learner on comp-set predictions, predict on test."""
    X_meta_train = np.column_stack([val_preds[m]  for m in range(pool_size)])
    X_meta_test  = np.column_stack([test_preds[m] for m in range(pool_size)])
    meta = Ridge(alpha=1.0)
    meta.fit(X_meta_train, y_comp)
    return meta.predict(X_meta_test)

# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(mode, task_id, seeds=5):
    X, y, dataset_name = load_task(task_id)
    print(f"\n{dataset_name} | samples: {X.shape[0]}, features: {X.shape[1]}")
    print("-" * 80)

    baseline_names = ["Simple Average", "Val-Weighted Average", "Ridge Stacking"]

    # scores/times for baselines (k-independent)
    baseline_scores = {b: [] for b in baseline_names}

    # per-pool-model scores (k-independent)
    # will be populated once we know pool size
    model_scores = None

    # DES scores and timings, keyed by k
    des_scores      = {k: {m: [] for m in des_methods} for k in K_VALUES}
    des_fit_times   = {k: {m: [] for m in des_methods} for k in K_VALUES}
    des_pred_times  = {k: {m: [] for m in des_methods} for k in K_VALUES}

    first = True

    for seed in range(seeds):
        for X_train, X_comp, X_test, y_train, y_comp, y_test in split_data(X, y, seed):
            if first:
                t0 = time.perf_counter()

            preset = "balanced" if X_comp.shape[0] >= 2500 else "exact"

            # ---- build pool & get predictions ----
            val_preds, test_preds = {}, {}

            if mode == 1:
                rf = RandomForestRegressor(n_estimators=50, random_state=seed)
                rf.fit(X_train, y_train)
                pool = rf.estimators_
            else:
                pool_fn = {0: pool_trees, 2: pool_large, 3: pool_small}[mode]
                pool    = pool_fn(seed)
                for model in pool:
                    model.fit(X_train, y_train)

            for model_id, model in enumerate(pool):
                val_preds[model_id]  = model.predict(X_comp)
                test_preds[model_id] = model.predict(X_test)

            y_min, y_max = y_train.min(), y_train.max()
            for model_id in val_preds:
                val_preds[model_id]  = np.clip(val_preds[model_id],  y_min, y_max)
                test_preds[model_id] = np.clip(test_preds[model_id], y_min, y_max)

            pool_size = len(pool)
            if model_scores is None:
                model_scores = {f"pool_model_{i}": [] for i in range(pool_size)}

            # ---- per-model scores ----
            for i in range(pool_size):
                model_scores[f"pool_model_{i}"].append(
                    mean_absolute_error(y_test, test_preds[i]))

            # ---- baselines ----
            baseline_scores["Simple Average"].append(
                mean_absolute_error(y_test, baseline_simple_average(test_preds, pool_size)))
            baseline_scores["Val-Weighted Average"].append(
                mean_absolute_error(y_test, baseline_val_weighted_average(
                    val_preds, test_preds, y_comp, pool_size)))
            baseline_scores["Ridge Stacking"].append(
                mean_absolute_error(y_test, baseline_ridge_stacking(
                    val_preds, test_preds, y_comp, pool_size)))

            # ---- DES algorithms across all k ----
            for k in K_VALUES:
                for method_name, method in des_algorithms(k, preset):

                    t_fit = time.perf_counter()
                    method.fit(X_comp, y_comp, val_preds)
                    fit_time = time.perf_counter() - t_fit

                    t_pred = time.perf_counter()
                    results = method.predict(X_test)
                    pred_time = time.perf_counter() - t_pred

                    final_preds = np.array([
                        sum(weights[mid] * test_preds[mid][idx] for mid in weights)
                        for idx, weights in enumerate(results)
                    ])

                    des_scores[k][method_name].append(
                        mean_absolute_error(y_test, final_preds))
                    des_fit_times[k][method_name].append(fit_time)
                    des_pred_times[k][method_name].append(pred_time)

            if first:
                dur = time.perf_counter() - t0
                print(f"One fold took {dur:.2f}s, expected total: {dur * 10 * seeds:.1f}s")
                print("-" * 80)
                first = False

    # ---- average across folds/seeds ----
    avg_baseline = {b: np.mean(v) for b, v in baseline_scores.items()}
    avg_models   = {m: np.mean(v) for m, v in model_scores.items()}
    avg_des_scores     = {k: {m: np.mean(v) for m, v in ms.items()} for k, ms in des_scores.items()}
    avg_des_fit_times  = {k: {m: np.mean(v) for m, v in mt.items()} for k, mt in des_fit_times.items()}
    avg_des_pred_times = {k: {m: np.mean(v) for m, v in mt.items()} for k, mt in des_pred_times.items()}

    return dataset_name, avg_baseline, avg_models, avg_des_scores, avg_des_fit_times, avg_des_pred_times

# ---------------------------------------------------------------------------
# Run + collect records
# ---------------------------------------------------------------------------

def run(mode, seeds, records):
    pool_name = POOL_NAMES[mode]
    datasets  = get_datasets()

    for task_id in datasets:
        dataset_name, avg_baseline, avg_models, avg_des_scores, avg_des_fit_times, avg_des_pred_times = \
            train(mode, task_id, seeds)

        # baselines (no k, no timing)
        for method, score in avg_baseline.items():
            records.append({
                'pool': pool_name, 'dataset': dataset_name,
                'k': None, 'method': method,
                'score': score, 'fit_time': None, 'predict_time': None,
            })

        # individual pool models (no k, no timing)
        for method, score in avg_models.items():
            records.append({
                'pool': pool_name, 'dataset': dataset_name,
                'k': None, 'method': method,
                'score': score, 'fit_time': None, 'predict_time': None,
            })

        # DES algorithms (with k and timing)
        for k in K_VALUES:
            for method in des_methods:
                records.append({
                    'pool':         pool_name,
                    'dataset':      dataset_name,
                    'k':            k,
                    'method':       method,
                    'score':        avg_des_scores[k][method],
                    'fit_time':     avg_des_fit_times[k][method],
                    'predict_time': avg_des_pred_times[k][method],
                })

        # ---- per-dataset console summary ----
        print(f"\n{dataset_name}")
        for k in K_VALUES:
            print(f"  k={k}")
            ref = min(avg_models.values())   # best individual model as reference
            for method in des_methods:
                score = avg_des_scores[k][method]
                diff  = (ref - score) / ref * 100
                ft    = avg_des_fit_times[k][method]
                pt    = avg_des_pred_times[k][method]
                print(f"    {method}: {score:.4f}  ({diff:+.1f}% vs best model)"
                      f"  fit={ft*1000:.1f}ms  pred={pt*1000:.1f}ms")
        print(f"  Baselines:")
        ref = min(avg_models.values())
        for b, score in avg_baseline.items():
            diff = (ref - score) / ref * 100
            print(f"    {b}: {score:.4f}  ({diff:+.1f}% vs best model)")
        print(f"  Best individual model: {ref:.4f}")
        print("-" * 80)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

records = []

print("RUNNING DECISION TREE POOL")
run(0, 5, records)
print("\nRUNNING RANDOM FOREST ESTIMATOR POOL")
run(1, 5, records)
print("\nRUNNING BIG HETEROGENEOUS POOL")
run(2, 5, records)
print("\nRUNNING SMALL HETEROGENEOUS POOL")
run(3, 5, records)

df = pd.DataFrame(records)
df.to_csv('benchmark_results.csv', index=False)
print(f"\nSaved {len(df)} records to benchmark_results.csv")
print(df.head(20).to_string())