# src/train_decision_tree.py

import os
import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from preprocessing import preprocess_pipeline


# ──────────────────────────────────────────────────────────────
# CATATAN DESAIN:
# Decision Tree tidak memerlukan feature scaling karena algoritma
# ini bekerja berdasarkan threshold pada nilai fitur (split point),
# bukan berdasarkan jarak antar titik data seperti KNN.
# Oleh karena itu kita gunakan X_train dan X_test (data mentah),
# bukan versi yang sudah di-StandardScaler.
# ──────────────────────────────────────────────────────────────


def tune_with_cross_validation(X_train, y_train, param_grid, cv_folds=5, random_state=42):
    """
    Pilih hyperparameter terbaik menggunakan Stratified K-Fold
    Cross Validation pada training set — BUKAN pada test set.

    Ini mencegah data leakage dalam proses seleksi model:
    test set hanya disentuh satu kali di akhir untuk evaluasi final.

    Returns:
        best_params  : dict parameter terbaik
        best_cv_score: rata-rata CV accuracy tertinggi
        cv_results   : list semua hasil untuk ditampilkan
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    best_params    = None
    best_cv_score  = 0
    cv_results     = []

    print(f"  Menggunakan {cv_folds}-Fold Stratified Cross Validation")
    print(f"  (Tuning dilakukan pada training set — test set tidak disentuh)\n")
    print(f"  {'max_depth':>10}  {'min_split':>9}  {'min_leaf':>8}  {'CV Mean':>8}  {'CV Std':>7}")
    print(f"  {'─'*10}  {'─'*9}  {'─'*8}  {'─'*8}  {'─'*7}")

    for params in param_grid:
        model = DecisionTreeClassifier(
            max_depth         = params["max_depth"],
            min_samples_split = params["min_samples_split"],
            min_samples_leaf  = params["min_samples_leaf"],
            criterion         = params.get("criterion", "gini"),
            random_state      = random_state
        )

        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        mean_score = scores.mean()
        std_score  = scores.std()

        depth_label = str(params["max_depth"]) if params["max_depth"] else "None"
        print(f"  {depth_label:>10}  {params['min_samples_split']:>9}  "
              f"{params['min_samples_leaf']:>8}  {mean_score:.4f}  ±{std_score:.4f}")

        cv_results.append({
            "params"    : params,
            "cv_mean"   : mean_score,
            "cv_std"    : std_score,
        })

        if mean_score > best_cv_score:
            best_cv_score = mean_score
            best_params   = params

    return best_params, best_cv_score, cv_results


def train_decision_tree(data_path, random_state=42):
    """
    Train Decision Tree dengan proses:
      1. Hyperparameter tuning via Cross Validation (pada training set)
      2. Retrain model terbaik pada seluruh training set
      3. Evaluasi final pada test set (satu kali saja)
      4. Tampilkan struktur pohon & feature importance
    """
    X_train, X_test, y_train, y_test, _, _, _ = preprocess_pipeline(data_path)

    print("=" * 60)
    print("        TRAINING DECISION TREE — WDBC CLASSIFICATION")
    print("=" * 60)
    print()
    print("  Karakteristik Decision Tree:")
    print("  - Tidak memerlukan feature scaling")
    print("  - Bekerja berdasarkan threshold nilai fitur (bukan jarak)")
    print("  - Mudah diinterpretasikan: dapat divisualisasikan sebagai pohon")
    print("  - Rentan overfitting jika max_depth tidak dibatasi")
    print()

    # ── 1. Definisi kandidat hyperparameter ──────────────────────────
    param_grid = [
        {"max_depth": 3,    "min_samples_split": 10, "min_samples_leaf": 5,  "criterion": "gini"},
        {"max_depth": 4,    "min_samples_split": 10, "min_samples_leaf": 5,  "criterion": "gini"},
        {"max_depth": 5,    "min_samples_split": 10, "min_samples_leaf": 5,  "criterion": "gini"},
        {"max_depth": 5,    "min_samples_split": 5,  "min_samples_leaf": 3,  "criterion": "gini"},
        {"max_depth": 7,    "min_samples_split": 5,  "min_samples_leaf": 2,  "criterion": "gini"},
        {"max_depth": 3,    "min_samples_split": 10, "min_samples_leaf": 5,  "criterion": "entropy"},
        {"max_depth": 5,    "min_samples_split": 10, "min_samples_leaf": 5,  "criterion": "entropy"},
        {"max_depth": None, "min_samples_split": 2,  "min_samples_leaf": 1,  "criterion": "gini"},
    ]

    # ── 2. Tuning dengan Cross Validation ────────────────────────────
    print("  TAHAP 1: Hyperparameter Tuning via Cross Validation")
    print("  " + "─" * 56)
    best_params, best_cv_score, cv_results = tune_with_cross_validation(
        X_train, y_train, param_grid, cv_folds=5, random_state=random_state
    )

    print(f"\n  Parameter terbaik  : {best_params}")
    print(f"  CV Accuracy terbaik: {best_cv_score:.4f} ({best_cv_score*100:.2f}%)")

    # ── 3. Retrain pada seluruh training set ─────────────────────────
    print()
    print("  TAHAP 2: Retrain model terbaik pada seluruh training set")
    print("  " + "─" * 56)

    best_model = DecisionTreeClassifier(
        max_depth         = best_params["max_depth"],
        min_samples_split = best_params["min_samples_split"],
        min_samples_leaf  = best_params["min_samples_leaf"],
        criterion         = best_params.get("criterion", "gini"),
        random_state      = random_state
    )
    best_model.fit(X_train, y_train)

    depth_actual = best_model.get_depth()
    n_leaves     = best_model.get_n_leaves()
    print(f"  Model dilatih dengan {len(X_train)} sampel")
    print(f"  Kedalaman pohon aktual : {depth_actual}")
    print(f"  Jumlah leaf node       : {n_leaves}")

    # ── 4. Evaluasi final pada test set ──────────────────────────────
    print()
    print("  TAHAP 3: Evaluasi final pada test set")
    print("  " + "─" * 56)

    y_pred = best_model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    cm   = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"  Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print()
    print(f"  Confusion Matrix:")
    print(f"  {'':16s}  Pred Benign  Pred Malignant")
    print(f"  {'Actual Benign':16s}  TN={tn:<9}  FP={fp}")
    print(f"  {'Actual Malignant':16s}  FN={fn:<9}  TP={tp}")
    print()
    print("  Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=["Benign", "Malignant"]
    ))

    # ── 5. Feature importance ─────────────────────────────────────────
    print("  Top 10 Feature Importance:")
    print("  " + "─" * 40)
    feature_names  = X_train.columns.tolist()
    importances    = best_model.feature_importances_
    sorted_idx     = np.argsort(importances)[::-1]

    for rank, idx in enumerate(sorted_idx[:10], start=1):
        bar = "█" * int(importances[idx] * 40)
        print(f"  {rank:>2}. {feature_names[idx]:<20s}  {importances[idx]:.4f}  {bar}")

    # ── 6. Cuplikan struktur pohon (3 level pertama) ──────────────────
    print()
    print("  Struktur pohon (3 level pertama):")
    print("  " + "─" * 40)
    tree_rules = export_text(
        best_model,
        feature_names=feature_names,
        max_depth=3
    )
    for line in tree_rules.split("\n")[:25]:
        print("  " + line)

    return best_model, best_params, best_cv_score, X_test, y_test


def save_model(model, model_path="models/decision_tree_model.joblib"):
    """
    Simpan model Decision Tree terbaik.
    Disimpan langsung (bukan dict) karena DT tidak memerlukan scaler.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"\n  Model tersimpan di: {model_path}")


if __name__ == "__main__":
    data_path = "data/wdbc.csv"

    print("Memulai training Decision Tree...\n")

    model, best_params, best_cv_score, X_test, y_test = train_decision_tree(data_path)

    save_model(model)

    print()
    print("=" * 60)
    print("  SELESAI")
    print(f"  Parameter terbaik : {best_params}")
    print(f"  CV Accuracy       : {best_cv_score*100:.2f}%")
    print("=" * 60)