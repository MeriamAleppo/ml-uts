# src/evaluate.py

import joblib
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

from preprocessing import preprocess_pipeline


def load_models():
    """
    Load semua model dari folder models/
    """
    models = {}

    # KNN (pakai scaler)
    knn_data = joblib.load("models/knn_model.joblib")
    models["KNN"] = knn_data

    # Decision Tree
    dt_model = joblib.load("models/decision_tree_model.joblib")
    models["Decision Tree"] = dt_model

    # Naive Bayes
    nb_model = joblib.load("models/naive_bayes_model.joblib")
    models["Naive Bayes"] = nb_model

    return models


def evaluate_models(data_path):
    """
    Evaluasi semua model
    """
    
    # Preprocessing
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, _ = preprocess_pipeline(data_path)
    
    models = load_models()
    
    results = []
    
    print("\n=== HASIL EVALUASI MODEL ===\n")
    
    for name, model_data in models.items():
        
        print(f"\n--- {name} ---")
        
        # KNN (punya scaler)
        if name == "KNN":
            model = model_data["model"]
            scaler = model_data["scaler"]
            y_pred = model.predict(X_test_scaled)
        
        # Decision Tree & Naive Bayes
        else:
            model = model_data
            y_pred = model.predict(X_test)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        print("Confusion Matrix:")
        print(cm)
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Simpan hasil
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": report["1"]["precision"],
            "Recall": report["1"]["recall"],
            "F1-Score": report["1"]["f1-score"]
        })
    
    # Tabel perbandingan
    df_results = pd.DataFrame(results)    
    print("\n=== PERBANDINGAN MODEL ===\n")
    print(df_results.sort_values(by="Accuracy", ascending=False))
    
    # Simpan hasil
    df_results.to_csv("reports/hasil_evaluasi.csv", index=False)
    return df_results

if __name__ == "__main__":
    data_path = "data/wdbc.csv"
    
    evaluate_models(data_path)