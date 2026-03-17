import tensorflow as tf
import numpy as np
import os
import pandas as pd
import librosa

# CONFIGURATION
MODEL_PATH = os.path.join("..", "models", "cnn_turbulence_model.h5")
TFLITE_PATH = os.path.join("..", "models", "turbulence_model_quantized.tflite")
METADATA_PATH = os.path.join("..", "data", "metadata", "metadata_hybrid.csv")
DATA_DIR = os.path.join("..", "data", "hybrid")

# Paramètres audio (Doivent correspondre à l'entraînement)
SR = 16000
DURATION = 2.0
N_MELS = 64
HOP_LENGTH = 512

def representative_dataset_gen():
    """Générateur d'échantillons représentatifs pour la quantification INT8."""
    df = pd.read_csv(METADATA_PATH).sample(50) # On prend 50 fichiers au hasard
    for idx, row in df.iterrows():
        file_path = os.path.join(DATA_DIR, row['filename'])
        y, sr = librosa.load(file_path, sr=SR, duration=DURATION)
        if len(y) < SR * DURATION:
            y = np.pad(y, (0, int(SR * DURATION) - len(y)))
        
        ps = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
        ps_db = librosa.power_to_db(ps, ref=np.max)
        ps_db = (ps_db - ps_db.min()) / (ps_db.max() - ps_db.min() + 1e-6)
        
        # Le modèle attend (batch, height, width, 1)
        data = ps_db[np.newaxis, ..., np.newaxis].astype(np.float32)
        yield [data]

def quantize():
    if not os.path.exists(MODEL_PATH):
        print(f"Erreur : Modèle {MODEL_PATH} introuvable.")
        return

    print("Chargement du modèle Keras...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("Démarrage de la conversion TFLite avec Quantification INT8...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Stratégie de quantification
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    
    # Forcer INT8 pour le TinyML
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_model = converter.convert()

    # Sauvegarde
    with open(TFLITE_PATH, 'wb') as f:
        f.write(tflite_model)
    
    size_kb = os.path.getsize(TFLITE_PATH) / 1024
    print(f"Quantification terminée !")
    print(f"Fichier : {TFLITE_PATH}")
    print(f"Taille finale : {size_kb:.2f} KB")
    
    if size_kb < 256:
        print("✅ SUCCÈS : Le modèle tient dans la RAM cible (<256 KB).")
    else:
        print("⚠️ ALERTE : Le modèle dépasse la taille cible.")

if __name__ == "__main__":
    quantize()
