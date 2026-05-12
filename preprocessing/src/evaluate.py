# src/evaluate.py

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from preprocessing import preprocess_pipeline


def load_models():
    """
    Load semua model dari folder models/.

    Format yang tersimpan:
      - KNN          : dict {"model": ..., "scaler": ...}
      - Decision Tree: model langsung
      - Naive Bayes  : model langsung (GaussianNB)
    """
    knn_data = joblib.load("models/knn_model.joblib")
    dt_model = joblib.load("models/decision_tree_model.joblib")
    nb_model = joblib.load("models/naive_bayes_model.joblib")

    return {
        "KNN": knn_data,
        "Decision Tree": dt_model,
        "Naive Bayes": nb_model,
    }


def predict(name, model_data, X_test, X_test_scaled):
    """
    Pilih data yang tepat berdasarkan jenis model:
      - KNN          : wajib scaled (berbasis jarak Euclidean)
      - Decision Tree: tidak perlu scaled (berbasis threshold)
      - Naive Bayes  : wajib scaled (dilatih dengan data scaled)
    """
    if name == "KNN":
        model = model_data["model"]
        return model, model.predict(X_test_scaled)

    elif name == "Decision Tree":
        return model_data, model_data.predict(X_test)

    elif name == "Naive Bayes":
        # GaussianNB dilatih dengan X_train_scaled,
        # maka evaluasi juga harus menggunakan X_test_scaled
        return model_data, model_data.predict(X_test_scaled)

    else:
        raise ValueError(f"Model tidak dikenal: {name}")


def print_confusion_matrix(cm):
    """Tampilkan confusion matrix dalam format yang mudah dibaca."""
    tn, fp, fn, tp = cm.ravel()
    print("  Confusion Matrix:")
    print(f"  {'':12s}  Pred Benign  Pred Malignant")
    print(f"  {'Actual Benign':12s}  TN={tn:<9}  FP={fp}")
    print(f"  {'Actual Malignant':16s}  FN={fn:<9}  TP={tp}")
    print()
    print(f"  Interpretasi (konteks medis):")
    print(f"  - True Negative  (TN={tn}): kanker jinak, diprediksi jinak     [BENAR]")
    print(f"  - True Positive  (TP={tp}): kanker ganas, diprediksi ganas     [BENAR]")
    print(f"  - False Positive (FP={fp}):  kanker jinak, diprediksi ganas   [Alarm palsu]")
    print(f"  - False Negative (FN={fn}):  kanker ganas, diprediksi jinak   [BERBAHAYA]")


def error_analysis(name, model, X_test, y_test, y_pred, feature_names):
    """
    Analisis sampel yang salah diklasifikasi.
    Menampilkan pola error dan rekomendasi tindak lanjut.
    """
    wrong_mask = y_pred != y_test.values
    wrong_indices = np.where(wrong_mask)[0]

    if len(wrong_indices) == 0:
        print("  Tidak ada sampel yang salah diklasifikasi!")
        return

    actual_wrong    = y_test.values[wrong_mask]
    predicted_wrong = y_pred[wrong_mask]

    fp_count = int(np.sum((actual_wrong == 0) & (predicted_wrong == 1)))
    fn_count = int(np.sum((actual_wrong == 1) & (predicted_wrong == 0)))

    print(f"  Total sampel salah  : {len(wrong_indices)} dari {len(y_test)}")
    print(f"  False Positive (FP) : {fp_count}  — jinak diprediksi ganas (alarm palsu)")
    print(f"  False Negative (FN) : {fn_count}  — ganas diprediksi jinak (BERBAHAYA)")
    print()

    if fn_count > 0:
        print("  [!] PERHATIAN: Model melewatkan beberapa kasus kanker ganas.")
        print("      Dalam konteks medis, False Negative jauh lebih berbahaya")
        print("      karena pasien tidak mendapat penanganan yang seharusnya.")
        print()

    if fp_count > 0:
        print("  [!] CATATAN: Beberapa kasus jinak diprediksi ganas.")
        print("      Ini menyebabkan pemeriksaan lanjutan yang tidak perlu,")
        print("      namun lebih aman dari pada melewatkan kanker ganas.")


def evaluate_models(data_path):
    """
    Evaluasi dan bandingkan semua model secara lengkap.
    Menghasilkan:
      - Confusion matrix per model
      - Classification report per model
      - Error analysis per model
      - Tabel perbandingan akhir
      - File CSV hasil evaluasi
    """
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, _ = preprocess_pipeline(data_path)
    feature_names = X_test.columns.tolist()

    models = load_models()
    results = []

    print("\n" + "=" * 60)
    print("           EVALUASI MODEL — WDBC CLASSIFICATION")
    print("=" * 60)
    print(f"  Ukuran test set : {len(y_test)} sampel")
    print(f"  Kelas           : 0=Benign, 1=Malignant")
    print(f"  Distribusi test : Benign={sum(y_test==0)}, Malignant={sum(y_test==1)}")

    for name, model_data in models.items():

        print(f"\n{'─'*60}")
        print(f"  MODEL: {name}")
        print(f"{'─'*60}")

        model, y_pred = predict(name, model_data, X_test, X_test_scaled)

        # ── Metrik utama ──────────────────────────────────────────
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        cm   = confusion_matrix(y_test, y_pred)

        report = classification_report(
            y_test, y_pred,
            target_names=["Benign", "Malignant"],
            output_dict=True
        )

        print(f"\n  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
        print(f"  Precision : {prec:.4f}  (dari semua prediksi ganas, berapa yang benar)")
        print(f"  Recall    : {rec:.4f}  (dari semua kasus ganas, berapa yang terdeteksi)")
        print(f"  F1-Score  : {f1:.4f}  (harmonic mean precision & recall)")
        print()

        # ── Confusion matrix ──────────────────────────────────────
        print_confusion_matrix(cm)

        # ── Classification report ─────────────────────────────────
        print("\n  Classification Report:")
        print(classification_report(
            y_test, y_pred,
            target_names=["Benign", "Malignant"]
        ))

        # ── Error analysis ────────────────────────────────────────
        print(f"  Error Analysis:")
        error_analysis(name, model, X_test, y_test, y_pred, feature_names)

        # ── Simpan hasil ──────────────────────────────────────────
        results.append({
            "Model"              : name,
            "Accuracy"           : round(acc,  4),
            "Precision"          : round(prec, 4),
            "Recall"             : round(rec,  4),
            "F1-Score"           : round(f1,   4),
            "False Positive"     : int(cm[0][1]),
            "False Negative"     : int(cm[1][0]),
            "Precision_Benign"   : round(report["Benign"]["precision"],    4),
            "Recall_Benign"      : round(report["Benign"]["recall"],       4),
            "F1_Benign"          : round(report["Benign"]["f1-score"],     4),
            "Precision_Malignant": round(report["Malignant"]["precision"], 4),
            "Recall_Malignant"   : round(report["Malignant"]["recall"],    4),
            "F1_Malignant"       : round(report["Malignant"]["f1-score"],  4),
        })

    # ── Tabel perbandingan akhir ──────────────────────────────────────
    df_results = pd.DataFrame(results)
    df_sorted  = df_results.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 60)
    print("           PERBANDINGAN AKHIR SEMUA MODEL")
    print("=" * 60)

    display_cols = ["Model", "Accuracy", "Precision", "Recall", "F1-Score",
                    "False Positive", "False Negative"]
    print(df_sorted[display_cols].to_string(index=False))

    print()
    best = df_sorted.iloc[0]
    print(f"  Model terbaik berdasarkan Accuracy : {best['Model']} ({best['Accuracy']*100:.2f}%)")

    # Cari model dengan FN terendah (paling aman secara medis)
    safest = df_results.loc[df_results["False Negative"].idxmin()]
    print(f"  Model teraman secara medis (FN min) : {safest['Model']} (FN={safest['False Negative']})")

    # ── Simpan ke CSV ─────────────────────────────────────────────────
    os.makedirs("reports", exist_ok=True)
    df_results.to_csv("reports/hasil_evaluasi.csv", index=False)
    print(f"\n  Hasil tersimpan di: reports/hasil_evaluasi.csv")

    return df_results


if __name__ == "__main__":
    data_path = "data/wdbc.csv"
    evaluate_models(data_path)