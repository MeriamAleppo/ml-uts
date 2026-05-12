# src/train_decision_tree.py

import os
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from preprocessing import preprocess_pipeline


def train_decision_tree(data_path, random_state=42):
    """
    Train Decision Tree dengan beberapa konfigurasi parameter
    dan pilih model terbaik berdasarkan akurasi
    """
    
    # Ambil data
    X_train, X_test, y_train, y_test, _, _, _ = preprocess_pipeline(data_path)
    
    # Kandidat parameter (tuning sederhana)
    param_grid = [
        {"max_depth": 3, "min_samples_split": 10, "min_samples_leaf": 5},
        {"max_depth": 5, "min_samples_split": 10, "min_samples_leaf": 5},
        {"max_depth": 7, "min_samples_split": 5, "min_samples_leaf": 2},
        {"max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1}
    ]
    
    best_model = None
    best_score = 0
    best_params = None
    
    print("Mulai training Decision Tree dengan beberapa parameter...\n")
    
    for params in param_grid:
        model = DecisionTreeClassifier(
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            random_state=random_state
        )
        
        model.fit(X_train, y_train)
        
        # Evaluasi sederhana (accuracy)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"Params: {params} -> Accuracy: {acc:.4f}")
        
        if acc > best_score:
            best_score = acc
            best_model = model
            best_params = params
    
    print("\nModel terbaik:")
    print(f"Params: {best_params}")
    print(f"Accuracy: {best_score:.4f}")
    
    return best_model, best_params, X_test, y_test


def save_model(model, model_path="models/decision_tree_model.joblib"):
    """
    Simpan model terbaik
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)


if __name__ == "__main__":
    data_path = "data/wdbc.csv"
    
    print("Training Decision Tree (Tuning Mode)...")
    
    model, best_params, X_test, y_test = train_decision_tree(data_path)
    
    save_model(model)
    
    print("\nModel Decision Tree terbaik berhasil disimpan!")