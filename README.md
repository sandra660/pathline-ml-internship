# Machine Learning Internship Projects: Pathline Training Academy

Three machine learning tasks I completed during my **Machine Learning Traineeship at Pathline Training Academy**.
They cover deep learning, interpretable ensemble models, and production-style ML pipelines.

| # | Project | Techniques | Key result |
|---|---|---|---|
| 1 | [CNN from scratch: Cats vs. Dogs](01-cnn-cats-vs-dogs/) | TensorFlow/Keras CNN, data augmentation, batch norm, class imbalance (class weights) | 74% test accuracy; class weighting raised minority-class recall from 57% to 66% |
| 2 | [Random Forest & feature importance](02-random-forest-feature-importance/) | Feature engineering, Random Forest, cross-validation, MDI vs. permutation importance | 0.83 CV accuracy, 0.84 ROC-AUC, beats the gender-only baseline |
| 3 | [Automated ML pipeline](03-ml-pipeline/) | `Pipeline`, `ColumnTransformer`, model comparison, `GridSearchCV`, `joblib` | 0.82 test accuracy; one saved file predicts from raw data |

## Repository structure
```
├── 01-cnn-cats-vs-dogs/
│   ├── cnn_cats_vs_dogs.ipynb
│   ├── download_data.py        # fetches the 4,000-image subset (~100 MB)
│   └── images/                 # result plots
├── 02-random-forest-feature-importance/
│   ├── random_forest_feature_importance.ipynb
│   └── images/
├── 03-ml-pipeline/
│   ├── ml_pipeline.ipynb
│   └── titanic_pipeline.joblib # saved, ready-to-use pipeline
├── data/titanic.csv            # Titanic training set (891 passengers)
└── requirements.txt
```

## How to run
```bash
pip install -r requirements.txt
jupyter notebook
```
The CNN notebook downloads its images automatically on first run (requires `git`). The notebooks also run on Google Colab.

## Principles applied throughout
* **No data leakage:** data is split before any statistics are computed, and preprocessing is fitted on training data only.
* **Honest evaluation:** results are compared to simple baselines, checked with cross-validation, and reported with per-class metrics, not accuracy alone.
* **Reproducibility:** fixed random seeds and scripted data download.

## Tech stack
Python · TensorFlow/Keras · scikit-learn · pandas · NumPy · Matplotlib · Jupyter
