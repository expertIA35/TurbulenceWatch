import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# CONFIGURATION
DATA_DIR = os.path.join("..", "data", "hybrid")
METADATA_PATH = os.path.join("..", "data", "metadata", "metadata_hybrid.csv")
MODEL_SAVE_PATH = os.path.join("..", "models", "cnn_turbulence_model.h5")

# Paramètres audio
SR = 16000
DURATION = 2.0
N_MELS = 64  # Résolution fréquentielle
HOP_LENGTH = 512

def extract_spectrogram(file_path):
    """Calcule le Mel-Spectrogram d'un fichier audio et le normalise."""
    y, sr = librosa.load(file_path, sr=SR, duration=DURATION)
    # On s'assure que la durée est exacte
    if len(y) < SR * DURATION:
        y = np.pad(y, (0, int(SR * DURATION) - len(y)))
    
    ps = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
    ps_db = librosa.power_to_db(ps, ref=np.max)
    
    # Normalisation entre 0 et 1 pour le réseau de neurones
    ps_db = (ps_db - ps_db.min()) / (ps_db.max() - ps_db.min() + 1e-6)
    return ps_db

def load_dataset():
    if not os.path.exists(METADATA_PATH):
        print("Erreur : metadata_hybrid.csv introuvable.")
        return None, None
        
    df = pd.read_csv(METADATA_PATH)
    X = []
    y = []
    
    print(f"Chargement et prétraitement de {len(df)} échantillons...")
    for idx, row in df.iterrows():
        file_path = os.path.join(DATA_DIR, row['filename'])
        try:
            spectro = extract_spectrogram(file_path)
            X.append(spectro)
            y.append(row['label'])
        except Exception as e:
            print(f"Erreur sur {file_path}: {e}")
            
    X = np.array(X)
    X = X[..., np.newaxis]  # Ajouter une dimension pour le canal (1 pour Noir & Blanc)
    y = np.array(y)
    
    return X, y

def build_model(input_shape):
    """Architecture CNN légère optimisée pour le TinyML (Ultra-compact)."""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # Bloc 1 (Réduit)
        layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Bloc 2 (Réduit)
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Bloc 3 (Sortie compacte)
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.Flatten(),
        
        # Couche Dense (Réduite)
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    X, y = load_dataset()
    if X is None: return
    
    # Split Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Construction du modèle
    input_shape = (X.shape[1], X.shape[2], 1)
    model = build_model(input_shape)
    model.summary()
    
    print("\nDémarrage de l'entraînement IA (Deep Learning)...")
    history = model.fit(X_train, y_train, 
                        epochs=25, 
                        batch_size=16, 
                        validation_split=0.1,
                        verbose=1)
    
    # Évaluation
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nSCORE FINAL : Accuracy = {acc:.4f}")
    
    # Sauvegarde
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"Modèle sauvegardé dans : {MODEL_SAVE_PATH}")
    
    # Plot des courbes (Visuel pour le rapport)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title('Modèle : Précision')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title('Modèle : Perte (Loss)')
    plt.legend()
    
    plt.savefig(os.path.join("..", "docs", "reports", "cnn_training_curves.png"))
    print("Graphiques d'entraînement sauvegardés.")

if __name__ == "__main__":
    main()
