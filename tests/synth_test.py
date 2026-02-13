from ensemble_weights import DynamicRouter
import pandas as pd
import numpy as np

# Synthetic data (as before)
np.random.seed(42)
n = 1000
real_feature = np.random.randint(0, 2, n)
y_true = np.random.randint(0, 2, n)
pred_A = np.where(real_feature == 0, y_true, 1 - y_true)
pred_B = np.where(real_feature == 1, y_true, 1 - y_true)

df = pd.DataFrame({
    'feature': real_feature,
    'label': y_true,
    'pred_A': pred_A,
    'pred_B': pred_B
})

router = DynamicRouter(
    task='regression',
    dtype='tabular',
    method='knn',
    metric="accuracy",
    mode = 'max',
    k=10
)

router.fit(
    features=df[['feature']].values,
    y=df['label'].values,
    preds_dict={'A': df['pred_A'].values, 'B': df['pred_B'].values}
)

print(router.predict(np.array([[0.0]])))   # {'A': ~1.0, 'B': ~0.0}
print(router.predict(np.array([[1.0]])))   # {'A': ~0.0, 'B': ~1.0}