import numpy as np
import scipy.io.wavfile as wav
import os
import pandas as pd
import librosa

# CONFIGURATION
DATA_RAW_DIR = os.path.join("..", "data", "raw")
DATA_HYBRID_DIR = os.path.join("..", "data", "hybrid")
METADATA_PATH = os.path.join("..", "data", "metadata", "metadata.csv")
HYBRID_METADATA_PATH = os.path.join("..", "data", "metadata", "metadata_hybrid.csv")

os.makedirs(os.path.join(DATA_HYBRID_DIR, "turbulence"), exist_ok=True)
os.makedirs(os.path.join(DATA_HYBRID_DIR, "calme"), exist_ok=True)

def generate_cockpit_ambient_noise(duration, fs):
    """
    Simule un bruit de fond de cockpit réaliste (Bruit rose + Hum moteur).
    """
    t = np.linspace(0, duration, int(fs * duration))
    
    # 1. Bruit rose (Pink Noise) - Énergie décroissante 1/f
    white_noise = np.random.normal(0, 1, len(t))
    noise_fft = np.fft.fft(white_noise)
    freqs = np.fft.fftfreq(len(t), d=1/fs)
    with np.errstate(divide='ignore', invalid='ignore'):
        pink_filter = 1 / np.sqrt(np.abs(freqs))
        pink_filter[np.isinf(pink_filter)] = 0
    pink_noise_fft = noise_fft * pink_filter
    pink_noise = np.real(np.fft.ifft(pink_noise_fft))
    
    # 2. Hum moteur (Low frequency tones)
    engine_hum = 0.2 * np.sin(2 * np.pi * 100 * t) + 0.1 * np.sin(2 * np.pi * 200 * t)
    
    # 3. Mix
    ambient = 0.7 * pink_noise + 0.3 * engine_hum
    return ambient / np.max(np.abs(ambient)) * 0.15 # Niveau de fond

def hybridize():
    if not os.path.exists(METADATA_PATH):
        print("Erreur : metadata.csv introuvable.")
        return
        
    df = pd.read_csv(METADATA_PATH)
    hybrid_metadata = []
    
    print("Démarrage de l'Hybridation (Mixage avec bruit de cockpit)...")
    
    for idx, row in df.iterrows():
        raw_path = os.path.join(DATA_RAW_DIR, row['filename'])
        sr, audio = wav.read(raw_path)
        
        # S'assurer que c'est du float32 entre -1 et 1
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32) / 32768.0 if audio.dtype == np.int16 else audio

        # Générer le masque de bruit de fond cockpit
        ambient = generate_cockpit_ambient_noise(len(audio)/sr, sr)
        
        # Injection de bruit (Sim-to-Real Mitigation)
        # On ajoute du gain aléatoire (Domain Randomization)
        gain = np.random.uniform(0.8, 1.2)
        hybrid_audio = (audio * gain) + ambient
        
        # Normalisation finale
        hybrid_audio = hybrid_audio / np.max(np.abs(hybrid_audio)) * 0.95
        
        # Sauvegarde
        hybrid_filename = row['filename']
        hybrid_path = os.path.join(DATA_HYBRID_DIR, hybrid_filename)
        
        os.makedirs(os.path.dirname(hybrid_path), exist_ok=True)
        wav.write(hybrid_path, sr, hybrid_audio.astype(np.float32))
        
        hybrid_metadata.append({
            "filename": hybrid_filename,
            "label": row['label'],
            "intensity": row['intensity'],
            "type": "hybrid_sim_real"
        })

    # Sauvegarde des métadonnées hybrides
    pd.DataFrame(hybrid_metadata).to_csv(HYBRID_METADATA_PATH, index=False)
    print(f"Hybridation terminée. {len(hybrid_metadata)} fichiers créés dans {DATA_HYBRID_DIR}")

if __name__ == "__main__":
    hybridize()
