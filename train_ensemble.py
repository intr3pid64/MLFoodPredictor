#!/usr/bin/env python3

import pandas as pd
import numpy as np
import json
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer

def numpy_default(obj):
    """Convert NumPy ints/floats/arrays to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Cannot serialize type {type(obj)}")

def main():
    ############################
    # 1. LOAD & PREPARE DATA
    ############################
    data_path = "cleaned_data_advanced.csv"  # path to your cleaned CSV
    df = pd.read_csv(data_path)
    
    # separate label (y) and features (X)
    y = df["label"].values
    X = df.drop(columns=["id","label","movie_raw","drink_raw"]).copy()

    # one-hot encode relevant columns (already numeric except 'movie_std','drink_std')
    X = pd.get_dummies(X, columns=["movie_std","drink_std"], drop_first=False)

    # encode labels
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)

    # train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    # handle missing values
    numeric_cols = X_train.columns
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train[numeric_cols])
    X_val_imp   = imputer.transform(X_val[numeric_cols])

    ############################
    # 2. HYPERPARAMETER TUNING
    ############################
    
    # A) LOGISTIC REGRESSION
    logreg = LogisticRegression(max_iter=1000, random_state=42)
    param_grid_lr = {
      "C": [0.01, 0.1, 1, 10],
      "penalty": ["l2"]
    }
    grid_lr = GridSearchCV(logreg, param_grid_lr, cv=3, scoring="accuracy")
    grid_lr.fit(X_train_imp, y_train)
    best_lr = grid_lr.best_estimator_
    print("[LogisticRegression] best params:", grid_lr.best_params_)
    print("[LogisticRegression] train acc:", grid_lr.best_score_)

    # B) RANDOM FOREST
    rf = RandomForestClassifier(random_state=42)
    param_grid_rf = {
      "n_estimators": [50, 100],
      "max_depth": [None, 5, 10],
      "min_samples_split": [2, 5]
    }
    grid_rf = GridSearchCV(rf, param_grid_rf, cv=3, scoring="accuracy")
    grid_rf.fit(X_train_imp, y_train)
    best_rf = grid_rf.best_estimator_
    print("[RandomForest] best params:", grid_rf.best_params_)
    print("[RandomForest] train acc:", grid_rf.best_score_)

    # C) GRADIENT BOOSTING
    gb = GradientBoostingClassifier(random_state=42)
    param_grid_gb = {
      "n_estimators": [50, 100],
      "learning_rate": [0.01, 0.1],
      "max_depth": [3, 5]
    }
    grid_gb = GridSearchCV(gb, param_grid_gb, cv=3, scoring="accuracy")
    grid_gb.fit(X_train_imp, y_train)
    best_gb = grid_gb.best_estimator_
    print("[GradientBoosting] best params:", grid_gb.best_params_)
    print("[GradientBoosting] train acc:", grid_gb.best_score_)

    ############################
    # 3. RETRAIN BEST MODELS
    ############################
    best_lr.fit(X_train_imp, y_train)
    best_rf.fit(X_train_imp, y_train)
    best_gb.fit(X_train_imp, y_train)

    # Evaluate on validation
    val_acc_lr = best_lr.score(X_val_imp, y_val)
    val_acc_rf = best_rf.score(X_val_imp, y_val)
    val_acc_gb = best_gb.score(X_val_imp, y_val)
    print(f"Validation Accuracy (LR): {val_acc_lr:.4f}")
    print(f"Validation Accuracy (RF): {val_acc_rf:.4f}")
    print(f"Validation Accuracy (GB): {val_acc_gb:.4f}")

    ############################
    # 4. EXTRACT PARAMETERS
    ############################
    lr_params = {
       "coeff": best_lr.coef_.tolist(),
       "intercept": best_lr.intercept_.tolist(),
       "classes": list(best_lr.classes_)
    }
    rf_trees = []
    for estimator in best_rf.estimators_:
        tree = estimator.tree_
        rf_trees.append({
            "feature": tree.feature.tolist(),
            "threshold": tree.threshold.tolist(),
            "children_left": tree.children_left.tolist(),
            "children_right": tree.children_right.tolist(),
            "value": tree.value.tolist()
        })
    rf_params = {
      "n_classes": len(best_rf.classes_),
      "n_features": X_train_imp.shape[1],
      "trees": rf_trees
    }
    gb_trees = []
    for i in range(best_gb.n_estimators_):
        row = []
        for cls in range(len(best_gb.estimators_[i])):
            reg = best_gb.estimators_[i][cls]
            tree = reg.tree_
            row.append({
                "feature": tree.feature.tolist(),
                "threshold": tree.threshold.tolist(),
                "children_left": tree.children_left.tolist(),
                "children_right": tree.children_right.tolist(),
                "value": tree.value.tolist()
            })
        gb_trees.append(row)
    gb_params = {
      "n_estimators": best_gb.n_estimators_,
      "learning_rate": best_gb.learning_rate,
      "init_score": list(best_gb.init_.predict(X_train_imp)),
      "trees": gb_trees,
      "loss": "deviance"
    }

    ############################
    # 5. SAVE EVERYTHING
    ############################
    label_mapping = list(label_encoder.classes_)
    train_cols = list(X_train.columns)
    imputer_stats = imputer.statistics_.tolist()

    final_params = {
      "lr_params": lr_params,
      "rf_params": rf_params,
      "gb_params": gb_params,
      "label_mapping": label_mapping,
      "train_cols": train_cols,
      "imputer_strategy": "median",
      "imputer_stats": imputer_stats,
    }

    # JSON dump with a custom default to handle np.int64/float64
    with open("model_params.json", "w") as f:
        json.dump(final_params, f, default=numpy_default)

    print("[Done] Wrote model_params.json with logistic regression, RF, GB parameters.")
    print("You can now implement pred.py in pure Python using these parameters.")

if __name__ == "__main__":
    main()
