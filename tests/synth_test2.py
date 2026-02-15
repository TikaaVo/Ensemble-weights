from ensemble_weights import DynamicRouter
import pandas as pd
import numpy as np

# Parameters
np.random.seed(42)
n = 100000                     # number of validation samples
feature = np.random.uniform(0, 1, n)  # continuous feature in [0,1]

# True labels (binary, 50/50)
y_true = np.random.randint(0, 2, n)

# Define performance functions: accuracy of model A and B as functions of feature
# Model A: perfect at 0, random at 1   -> accuracy = 1 - feature
# Model B: random at 0, perfect at 1   -> accuracy = feature
# To generate predictions, we use these probabilities to decide whether the model
# predicts the true label or the opposite.

# Generate predictions for model A
prob_A_correct = 1 - feature                      # probability correct
rand_A = np.random.random(n)                      # random threshold
pred_A = np.where(rand_A < prob_A_correct, y_true, 1 - y_true)

# Generate predictions for model B
prob_B_correct = feature
rand_B = np.random.random(n)
pred_B = np.where(rand_B < prob_B_correct, y_true, 1 - y_true)

# Create DataFrame
df = pd.DataFrame({
    'feature': feature,
    'label': y_true,
    'pred_A': pred_A,
    'pred_B': pred_B
})

# Initialize router
router = DynamicRouter(
    task='classification',        # even though metric is accuracy, it's classification
    dtype='tabular',
    method='knn-dw',               # or 'ola'
    metric="accuracy",
    mode='max',
    finder='annoy',                 # or 'faiss', 'knn'
    k=10,
    # Optional: pass HNSW parameters for better stability at 100k
    # M=32, ef_construction=400, ef_search=200, backend='hnswlib'
)

router.fit(
    features=df[['feature']].values,
    y=df['label'].values,
    preds_dict={'A': df['pred_A'].values, 'B': df['pred_B'].values}
)

# Test points along the feature range
test_points = [0.0, 0.25, 0.5, 0.75, 1.0]
for x in test_points:
    weights = router.predict(np.array([[x]]))
    print(f"x = {x:.2f} -> {weights}")