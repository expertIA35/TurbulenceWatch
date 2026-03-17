import numpy as np
import scipy.io.wavfile as wav
import os
import pandas as pd

# =============================================================================
# CONFIGURATION DU PROJET / PROJECT SETTINGS
# =============================================================================
DATA_DIR = os.path.join("..", "data", "raw")
METADATA_PATH = os.path.join("..", "data", "metadata", "metadata.csv")
SAMPLE_RATE = 16000  # 16kHz standard pour TinyML Audio
DURATION = 2.0       # 2 secondes par échantillon
N_SAMPLES = 250      # Par classe (Total 500)

# S'assurer que les dossiers existent / Ensure directories exist
os.makedirs(os.path.join(DATA_DIR, "turbulence"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "calme"), exist_ok=True)
os.makedirs(os.path.join("..", "data", "metadata"), exist_ok=True)

def generate_von_karman_noise(duration, fs, intensity='moderate'):
    """
    Simule la turbulence basée sur le spectre de Von Karman.
    Intensity: 'light', 'moderate', 'severe'
    """
    t = np.linspace(0, duration, int(fs * duration))
    
    # Paramètres physiques / Physical parameters
    params = {
        'light':    {'sigma': 0.5, 'L': 100, 'infrasound_boost': 1.2},
        'moderate': {'sigma': 1.2, 'L': 250, 'infrasound_boost': 2.5},
        'severe':   {'sigma': 2.5, 'L': 500, 'infrasound_boost': 5.0}
    }
    p = params[intensity]
    
    # Génération du bruit blanc / White noise generation
    noise = np.random.normal(0, 1, len(t))
    
    # Transformation de Fourier / Fourier Transform
    freqs = np.fft.fftfreq(len(t), d=1/fs)
    noise_fft = np.fft.fft(noise)
    
    # Filtre de Von Karman / Von Karman Filtering
    # S(f) = sigma^2 * L / (1 + (L*f/V)^2)^(5/6) -> simplifié pour simulation
    # On accentue les très basses fréquences (Infrasons 1-20Hz)
    with np.errstate(divide='ignore', invalid='ignore'):
        spectrum_filter = p['sigma']**2 * p['L'] / (1 + (p['L'] * np.abs(freqs))**2)**(5/6)
        spectrum_filter[np.isinf(spectrum_filter)] = 0
        
        # Boost Infrason : Originalité PhD / PhD Originality
        infrasound_mask = (np.abs(freqs) > 1) & (np.abs(freqs) < 30)
        spectrum_filter[infrasound_mask] *= p['infrasound_boost']

    # Application du filtre / Apply filter
    filtered_fft = noise_fft * np.sqrt(spectrum_filter)
    audio = np.real(np.fft.ifft(filtered_fft))
    
    # Ajouter des rafales (Modulation d'amplitude) / Add gusts (Amplitude Modulation)
    gust_env = 1 + 0.5 * np.sin(2 * np.pi * 0.2 * t) # Basse fréquence 0.2Hz
    audio *= gust_env
    
    # Normalisation / Normalization
    audio = audio / np.max(np.abs(audio)) * 0.9
    return audio

def generate_calm_noise(duration, fs):
    """
    Simule un environnement calme (bruit de fond cockpit stationnaire).
    """
    t = np.linspace(0, duration, int(fs * duration))
    # Bruit rose/blanc léger / Light pink/white noise
    noise = np.random.normal(0, 0.05, len(t))
    
    # Filtrer pour réduire les basses fréquences (pas de turbulence)
    freqs = np.fft.fftfreq(len(t), d=1/fs)
    noise_fft = np.fft.fft(noise)
    spectrum_filter = 1 / (1 + (0.1 * np.abs(freqs))) # Filtre passe-haut léger
    
    filtered_fft = noise_fft * spectrum_filter
    audio = np.real(np.fft.ifft(filtered_fft))
    
    return audio / np.max(np.abs(audio)) * 0.1 # Volume faible

# =============================================================================
# BOUCLE DE GÉNÉRATION / GENERATION LOOP
# =============================================================================
metadata = []

print(f"Démarrage de la génération du dataset dans: {DATA_DIR}")

# Génération Turbulence / Turbulence Generation
for i in range(N_SAMPLES):
    # Alterner entre intensités pour la diversité
    intensities = ['light', 'moderate', 'severe']
    lvl = intensities[i % 3]
    
    audio = generate_von_karman_noise(DURATION, SAMPLE_RATE, intensity=lvl)
    filename = f"turb_{lvl}_{i:03d}.wav"
    filepath = os.path.join(DATA_DIR, "turbulence", filename)
    
    wav.write(filepath, SAMPLE_RATE, audio.astype(np.float32))
    metadata.append({"filename": f"turbulence/{filename}", "label": 1, "intensity": lvl})

# Génération Calme / Calm Generation
for i in range(N_SAMPLES):
    audio = generate_calm_noise(DURATION, SAMPLE_RATE)
    filename = f"calme_{i:03d}.wav"
    filepath = os.path.join(DATA_DIR, "calme", filename)
    
    wav.write(filepath, SAMPLE_RATE, audio.astype(np.float32))
    metadata.append({"filename": f"calme/{filename}", "label": 0, "intensity": "none"})

# Sauvegarde des métadonnées / Save metadata
df = pd.DataFrame(metadata)
df.to_csv(METADATA_PATH, index=False)

print(f"Génération terminée. {len(df)} fichiers créés.")
print(f"Métadonnées sauvegardées dans: {METADATA_PATH}")
