# src/feature_selection.py

import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.tree import DecisionTreeClassifier


# =========================
# 1. CORRELATION ANALYSIS
# =========================
def correlation_analysis(X):
    return X.corr()


def remove_highly_correlated_features(X, threshold=0.9):
    corr_matrix = X.corr().abs()
    
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    to_drop = [
        column for column in upper_triangle.columns
        if any(upper_triangle[column] > threshold)
    ]
    
    X_reduced = X.drop(columns=to_drop)
    
    print(f"\n[Correlation] Fitur dihapus (> {threshold}):")
    print(to_drop if to_drop else "Tidak ada")
    
    return X_reduced, to_drop


# =========================
# 2. SELECT KBEST (STATISTICAL)
# =========================
def select_k_best_features(X, y, k=10):
    selector = SelectKBest(score_func=f_classif, k=k)
    
    X_new = selector.fit_transform(X, y)
    
    selected_mask = selector.get_support()
    selected_features = X.columns[selected_mask]
    
    print(f"\n[SelectKBest] Top {k} fitur terpilih:")
    print(list(selected_features))
    
    return pd.DataFrame(X_new, columns=selected_features), selected_features


# =========================
# 3. FEATURE IMPORTANCE (MODEL BASED)
# =========================
def select_features_by_importance(X, y, threshold=0.01):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)
    
    importances = model.feature_importances_
    
    feature_importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": importances
    }).sort_values(by="importance", ascending=False)
    
    selected_features = feature_importance_df[
        feature_importance_df["importance"] > threshold
    ]["feature"]
    
    print(f"\n[Feature Importance] Fitur terpilih (importance > {threshold}):")
    print(list(selected_features))
    
    return X[selected_features], selected_features, feature_importance_df


# =========================
# MAIN PIPELINE
# =========================
def feature_selection_pipeline(
    X, y,
    method="correlation",
    threshold=0.9,
    k=10
):
    """
    method:
    - 'correlation'
    - 'selectkbest'
    - 'importance'
    - 'none'
    """
    
    print("\n=== FEATURE SELECTION ADVANCED ===")
    
    if method == "correlation":
        X_selected, dropped = remove_highly_correlated_features(X, threshold)
        return X_selected, {"dropped": dropped}
    
    elif method == "selectkbest":
        X_selected, selected = select_k_best_features(X, y, k)
        return X_selected, {"selected": list(selected)}
    
    elif method == "importance":
        X_selected, selected, importance_df = select_features_by_importance(X, y)
        return X_selected, {
            "selected": list(selected),
            "importance": importance_df
        }
    
    elif method == "none":
        print("Tidak dilakukan feature selection")
        return X, {}
    
    else:
        raise ValueError("Method tidak valid")


# =========================
# DEMO
# =========================
if __name__ == "__main__":
    
    from preprocessing import load_data, clean_data, split_features_target
    
    path = "/data/wdbc.csv"
    
    df = load_data(path)
    df = clean_data(df)
    
    X, y = split_features_target(df)
    
    # Pilih metode di sini
    X_selected, info = feature_selection_pipeline(
        X, y,
        method="importance",   # ganti: correlation / selectkbest / importance
        threshold=0.9,
        k=10
    )
    
    print("\nJumlah fitur sebelum:", X.shape[1])
    print("Jumlah fitur sesudah:", X_selected.shape[1])