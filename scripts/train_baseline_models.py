import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# CONFIGURATION
FEATURES_CSV = os.path.join("..", "data", "processed", "features.csv")
MODEL_OUTPUT = os.path.join("..", "models", "baseline_model.joblib")

def train():
    if not os.path.exists(FEATURES_CSV):
        print("Erreur : features.csv introuvable.")
        return
        
    df = pd.read_csv(FEATURES_CSV)
    X = df.drop('label', axis=1)
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "SVM": SVC(probability=True),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(),
        "XGBoost": XGBClassifier()
    }
    
    best_acc = 0
    best_model = None
    best_name = ""
    
    print("Entraînement des modèles classiques...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        print(f"--- {name} ---")
        print(f"Accuracy: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name
            
    print(f"\nMEILLEUR MODÈLE : {best_name} avec {best_acc:.4f} d'accuracy.")
    
    # Rapport détaillé pour le meilleur modèle
    y_pred = best_model.predict(X_test)
    print("\nRapport de classification :")
    print(classification_report(y_test, y_pred))
    
    # Sauvegarde
    os.makedirs(os.path.dirname(MODEL_OUTPUT), exist_ok=True)
    joblib.dump(best_model, MODEL_OUTPUT)
    print(f"Modèle sauvegardé dans : {MODEL_OUTPUT}")

if __name__ == "__main__":
    train()
