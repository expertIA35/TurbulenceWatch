import os
import numpy as np
import pandas as pd
import tensorflow as tf
import librosa

# CONFIGURATION
MODEL_PATH = os.path.join("..", "models", "cnn_turbulence_model.h5")
HYBRID_DATA_DIR = os.path.join("..", "data", "hybrid")
METADATA_PATH = os.path.join("..", "data", "metadata", "metadata_hybrid.csv")
OUTPUT_CSV = os.path.join("..", "data", "processed", "turbulence_points_3d.csv")

# Paramètres de vol simulé
N_POINTS = 100
START_LAT, START_LON = 43.6045, 1.4442  # Toulouse (Siège Airbus/CNES)
END_LAT, END_LON = 48.8566, 2.3522      # Paris
ALTITUDE_BASE = 10000                   # 10,000 mètres

def load_data_samples():
    df = pd.read_csv(METADATA_PATH)
    return df

def extract_spectrogram(file_path):
    y, sr = librosa.load(file_path, sr=16000, duration=2.0)
    if len(y) < 32000: y = np.pad(y, (0, 32000 - len(y)))
    ps = librosa.feature.melspectrogram(y=y, sr=16000, n_mels=64, hop_length=512)
    ps_db = librosa.power_to_db(ps, ref=np.max)
    ps_db = (ps_db - ps_db.min()) / (ps_db.max() - ps_db.min() + 1e-6)
    return ps_db[np.newaxis, ..., np.newaxis]

def generate_flight_path():
    print("Démarrage de la simulation de vol spatio-temporelle...")
    
    # Charger l'IA
    model = tf.keras.models.load_model(MODEL_PATH)
    df_samples = load_data_samples()
    
    # Création de la trajectoire
    lats = np.linspace(START_LAT, END_LAT, N_POINTS)
    lons = np.linspace(START_LON, END_LON, N_POINTS)
    alts = ALTITUDE_BASE + np.sin(np.linspace(0, 10, N_POINTS)) * 200 # Légères variations d'altitude
    
    results = []
    
    # Zones de turbulence fictives sur la trajectoire (pour le scénario)
    # Zone 1: milieu du vol, Zone 2: fin du vol
    turb_zones = [(30, 50), (80, 95)]
    
    for i in range(N_POINTS):
        # Choisir un échantillon audio selon la zone
        is_turb_zone = any(start <= i <= end for start, end in turb_zones)
        
        if is_turb_zone:
            sample_row = df_samples[df_samples['label'] == 1].sample(1).iloc[0]
        else:
            sample_row = df_samples[df_samples['label'] == 0].sample(1).iloc[0]
            
        file_path = os.path.join(HYBRID_DATA_DIR, sample_row['filename'])
        spectro = extract_spectrogram(file_path)
        
        # Inférence IA
        prediction = model.predict(spectro, verbose=0)[0][0]
        
        results.append({
            "timestamp": i * 10, # 10 secondes entre chaque point
            "lat": lats[i],
            "lon": lons[i],
            "alt": alts[i],
            "intensity": prediction,
            "detected": 1 if prediction > 0.5 else 0,
            "sample_used": sample_row['filename']
        })
        
        if i % 10 == 0: print(f"Traitement point {i}/{N_POINTS}...")

    # Sauvegarde
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_res = pd.DataFrame(results)
    df_res.to_csv(OUTPUT_CSV, index=False)
    print(f"Trajectoire 3D générée avec succès : {OUTPUT_CSV}")

if __name__ == "__main__":
    generate_flight_path()
