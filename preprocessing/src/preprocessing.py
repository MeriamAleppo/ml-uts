# src/preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_data(path):
    """
    Load dataset TANPA header
    """
    df = pd.read_csv(path, header=None)
    
    # Tambahkan nama kolom
    columns = ["id", "diagnosis"] + [f"feature_{i}" for i in range(1, 31)]
    df.columns = columns
    
    return df


def clean_data(df):
    """
    Cleaning data:
    - Drop ID
    - Encode target
    """
    
    # Drop ID
    df = df.drop(columns=["id"])
    
    # Encode target: M=1, B=0
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
    
    return df


def split_features_target(df):
    X = df.drop("diagnosis", axis=1)
    y = df["diagnosis"]
    
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def scale_data(X_train, X_test):
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler


def preprocess_pipeline(path):
    df = load_data(path)
    df = clean_data(df)
    
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
    
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler


if __name__ == "__main__":
    path = "data/wdbc.csv"
    
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = preprocess_pipeline(path)
    
    print("Preprocessing berhasil!")
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    print("Distribusi target train:")
    print(y_train.value_counts())