# src/train_knn.py

import os
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from preprocessing import preprocess_pipeline


def train_knn(data_path):
    """
    Train KNN dengan tuning parameter k
    """
    
    # Ambil data dari preprocessing
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = preprocess_pipeline(data_path)
    
    # Kandidat nilai k
    k_values = [3, 5, 7, 9, 11]
    
    best_model = None
    best_score = 0
    best_k = None
    
    print("Mulai training KNN dengan berbagai nilai k...\n")
    
    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        
        # Training (pakai data yang sudah discale)
        model.fit(X_train_scaled, y_train)
        
        # Evaluasi
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"k = {k} -> Accuracy: {acc:.4f}")
        
        if acc > best_score:
            best_score = acc
            best_model = model
            best_k = k
    
    print("\nModel terbaik:")
    print(f"k terbaik: {best_k}")
    print(f"Accuracy: {best_score:.4f}")
    
    return best_model, scaler, best_k, X_test_scaled, y_test


def save_model(model, scaler, model_path="models/knn_model.joblib"):
    """
    Simpan model + scaler
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    joblib.dump({
        "model": model,
        "scaler": scaler
    }, model_path)


if __name__ == "__main__":
    data_path = "data/wdbc.csv"
    
    print("Training KNN (Tuning Mode)...")
    
    model, scaler, best_k, X_test, y_test = train_knn(data_path)
    
    save_model(model, scaler)
    
    print("\nModel KNN terbaik berhasil disimpan!")