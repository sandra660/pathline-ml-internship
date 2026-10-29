# Task 3 — Automated End-to-End ML Pipeline

A single scikit-learn `Pipeline` that turns raw Titanic passenger data into a survival prediction. It is leak-free, reproducible and ready to deploy.

**Notebook:** [`ml_pipeline.ipynb`](ml_pipeline.ipynb)

## Pipeline design
```
raw data ─► ColumnTransformer ─┬─ numerical:   SimpleImputer(median) ─► StandardScaler
                               └─ categorical: SimpleImputer(most_frequent) ─► OneHotEncoder(handle_unknown="ignore")
          ─► model (chosen by GridSearchCV) ─► survival probability
```
* Numerical and categorical columns are **detected automatically** from their data types.
* All preprocessing is learned **from the training data only**, and is re-fitted inside every cross-validation fold, so there is no data leakage.
* **One `GridSearchCV`** searches across both models (Random Forest and Gradient Boosting) and their hyper-parameters.
* The fitted pipeline is saved with `joblib` and can predict for **new raw passengers**, including ones with missing values.

## Results
| Model (5-fold CV on training set) | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | 0.784 | 0.854 |
| Random Forest | 0.788 | 0.864 |
| Gradient Boosting | 0.824 | 0.884 |
| **Tuned Gradient Boosting** (200 trees, depth 4, lr 0.05) | **0.827** | — |

**Held-out test set:** accuracy **0.816**, ROC-AUC 0.815.

Trade-off: survivor recall is 0.67, compared with 0.91 for non-survivors. If missing survivors mattered more, the next step would be to tune the decision threshold.
