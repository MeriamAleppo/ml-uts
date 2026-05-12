# src/train_naive_bayes.py

import os
import joblib
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score
from preprocessing import preprocess_pipeline


def train_naive_bayes(data_path):
    """
    Train Naive Bayes dengan beberapa varian model
    """
    
    # Ambil data dari preprocessing
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, _ = preprocess_pipeline(data_path)
    
    # Kandidat model NB
    models = {
        "GaussianNB": GaussianNB(),
        "MultinomialNB": MultinomialNB()
    }
    
    best_model = None
    best_score = 0
    best_name = None
    
    print("Mulai training Naive Bayes...\n")
    
    for name, model in models.items():
        try:
            # GaussianNB pakai data scaled
            if name == "GaussianNB":
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                # MultinomialNB pakai data asli (tanpa scaling)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            
            print(f"{name} -> Accuracy: {acc:.4f}")
            
            if acc > best_score:
                best_score = acc
                best_model = model
                best_name = name
        
        except Exception as e:
            print(f"{name} gagal dijalankan: {e}")
    
    print("\nModel terbaik:")
    print(f"Model: {best_name}")
    print(f"Accuracy: {best_score:.4f}")
    
    return best_model, best_name, X_test, y_test


def save_model(model, model_path="models/naive_bayes_model.joblib"):
    """
    Simpan model Naive Bayes
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)


if __name__ == "__main__":
    data_path = "data/wdbc.csv"
    
    print("Training Naive Bayes...")
    
    model, model_name, X_test, y_test = train_naive_bayes(data_path)
    
    save_model(model)
    
    print("\nModel Naive Bayes terbaik berhasil disimpan!")