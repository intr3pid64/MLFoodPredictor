#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier

def main():
    # 1) Load data
    df = pd.read_csv("cleaned_data_advanced.csv")
    
    # 2) Separate label and features
    y = df["label"].values
    X = df.drop(columns=["id","label","movie_raw","drink_raw"]).copy()
    
    # 3) One-hot encode (if 'movie_std','drink_std' still have text)
    X = pd.get_dummies(X, columns=["movie_std","drink_std"], drop_first=False)
    
    # 4) Encode labels
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)
    
    # 5) Imputer for missing values
    numeric_cols = X.columns
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X[numeric_cols])
    
    # 6) Define final models with best hyperparameters
    #    (From your training script results)
    
    # Best Logistic Regression
    best_lr = LogisticRegression(
        max_iter=1000,
        random_state=42,
        C=10,          # from best params
        penalty="l2"
    )
    
    # Best Random Forest
    best_rf = RandomForestClassifier(
        random_state=42,
        n_estimators=100,   # from best params
        max_depth=10,
        min_samples_split=2
    )
    
    # Best Gradient Boosting
    best_gb = GradientBoostingClassifier(
        random_state=42,
        n_estimators=100,   # from best params
        learning_rate=0.1,
        max_depth=3
    )
    
    # 7) Combine them into a VotingClassifier for final ensemble
    ensemble = VotingClassifier(
        estimators=[
            ("lr", best_lr),
            ("rf", best_rf),
            ("gb", best_gb)
        ],
        voting="soft"
    )
    
    # 8) Cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(ensemble, X_imp, y_enc, cv=kf, scoring="accuracy")
    
    print("Cross-Validation Accuracy Scores:", scores)
    print("Mean Accuracy:", np.mean(scores))
    print("Std of Accuracy:", np.std(scores))

if __name__ == "__main__":
    main()
