# src/train_naive_bayes.py

import os
import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from preprocessing import preprocess_pipeline


def train_naive_bayes(data_path):
    """
    Train Naive Bayes (GaussianNB) untuk data kontinu.
    GaussianNB adalah satu-satunya varian yang sesuai untuk dataset WDBC
    karena semua fitur bersifat kontinu (bukan count/frekuensi).
    """

    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = preprocess_pipeline(data_path)

    print("=" * 45)
    print("   TRAINING NAIVE BAYES (GaussianNB)")
    print("=" * 45)
    print()
    print("Catatan: MultinomialNB tidak digunakan karena")
    print("dataset WDBC berisi fitur kontinu (bukan count),")
    print("sehingga GaussianNB adalah pilihan yang tepat.")
    print()

    model = GaussianNB()

    # GaussianNB dilatih dengan data yang sudah di-scale
    # agar distribusi Gaussian lebih baik terpenuhi
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print()
    print("Confusion Matrix:")
    print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"  FN={cm[1][0]}  TP={cm[1][1]}")
    print()
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Benign", "Malignant"]))

    return model, scaler, X_test_scaled, y_test, acc


def save_model(model, model_path="models/naive_bayes_model.joblib"):
    """
    Simpan model GaussianNB.
    Disimpan langsung (bukan dict) agar konsisten
    dengan format Decision Tree.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model tersimpan di: {model_path}")


if __name__ == "__main__":
    data_path = "data/wdbc.csv"

    print("Memulai training Naive Bayes...\n")

    model, scaler, X_test_scaled, y_test, acc = train_naive_bayes(data_path)

    save_model(model)

    print()
    print("=" * 45)
    print(f"  SELESAI — Accuracy final: {acc*100:.2f}%")
    print("=" * 45)