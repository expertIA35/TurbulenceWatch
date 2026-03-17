import os
import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# CONFIGURATION
DATA_DIR = os.path.join("..", "data", "raw")
METADATA_PATH = os.path.join("..", "data", "metadata", "metadata.csv")
OUTPUT_CSV = os.path.join("..", "data", "processed", "features.csv")
SCALER_PATH = os.path.join("..", "models", "scaler.joblib")

def extract_features(file_path):
    """
    Extrait les caractéristiques acoustiques (Audio Features Extraction)
    Focus : MFCC, Energie, et Bandes Infrasonores
    """
    y, sr = librosa.load(file_path, sr=None)
    
    # 1. MFCC (13 coeffs)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    
    # 2. Spectral Centroid & Rolloff
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    
    # 3. RMS Energy
    rms = librosa.feature.rms(y=y)
    
    # 4. Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    
    # 5. Band Energy (PhD Focus)
    # On calcule le FFT pour extraire l'énergie spécifique du vent/infrasons
    stft = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    
    # Band 1: 1-30Hz (Infrasons)
    infrasound_energy = np.mean(stft[(freqs >= 1) & (freqs <= 30), :])
    # Band 2: 30-200Hz (Basses fréquences)
    low_energy = np.mean(stft[(freqs > 30) & (freqs <= 200), :])
    
    # Concaténation
    features = np.concatenate([
        mfcc_mean, mfcc_std,
        [np.mean(centroid), np.mean(rolloff), np.mean(rms), np.mean(zcr)],
        [infrasound_energy, low_energy]
    ])
    
    return features

def main():
    if not os.path.exists(METADATA_PATH):
        print("Erreur : metadata.csv introuvable.")
        return
        
    df = pd.read_csv(METADATA_PATH)
    all_features = []
    
    print(f"Extraction des features pour {len(df)} fichiers...")
    
    for idx, row in df.iterrows():
        f_path = os.path.join(DATA_DIR, row['filename'])
        try:
            feat = extract_features(f_path)
            all_features.append(np.append(feat, row['label']))
        except Exception as e:
            print(f"Erreur sur {f_path}: {e}")
            
    # Création du DataFrame final
    columns = [f"mfcc_mean_{i}" for i in range(13)] + \
              [f"mfcc_std_{i}" for i in range(13)] + \
              ["centroid", "rolloff", "rms", "zcr", "infrasound_energy", "low_energy", "label"]
              
    features_df = pd.DataFrame(all_features, columns=columns)
    
    # Normalisation (StandardScaler)
    X = features_df.drop('label', axis=1)
    y = features_df['label']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Sauvegarde
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    
    scaled_df = pd.DataFrame(X_scaled, columns=columns[:-1])
    scaled_df['label'] = y
    
    scaled_df.to_csv(OUTPUT_CSV, index=False)
    joblib.dump(scaler, SCALER_PATH)
    
    print(f"Features sauvegardées dans : {OUTPUT_CSV}")
    print(f"Scaler sauvegardé dans : {SCALER_PATH}")

if __name__ == "__main__":
    main()
