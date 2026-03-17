import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pandas as pd

# CONFIGURATION
DATA_DIR = os.path.join("..", "data", "raw")
METADATA_PATH = os.path.join("..", "data", "metadata", "metadata.csv")
OUTPUT_DIR = os.path.join("..", "docs", "visuals")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_exploration():
    # Charger les métadonnées
    if not os.path.exists(METADATA_PATH):
        print(f"Erreur : {METADATA_PATH} introuvable. Lancez generate_turbulence_dataset.py d'abord.")
        return

    df = pd.read_csv(METADATA_PATH)
    
    # Prendre un échantillon de chaque classe
    sample_turb = df[df['label'] == 1].iloc[0]['filename']
    sample_calm = df[df['label'] == 0].iloc[0]['filename']
    
    plt.figure(figsize=(15, 10))
    
    # --- SUBPLOT 1: WAVEFORM TURBULENCE ---
    plt.subplot(2, 2, 1)
    y_turb, sr = librosa.load(os.path.join(DATA_DIR, sample_turb), sr=None)
    librosa.display.waveshow(y_turb, sr=sr, color='blue')
    plt.title(f"Waveform: Turbulence ({sample_turb})")
    plt.xlabel("Temps (s)")
    plt.ylabel("Amplitude")

    # --- SUBPLOT 2: WAVEFORM CALME ---
    plt.subplot(2, 2, 2)
    y_calm, sr = librosa.load(os.path.join(DATA_DIR, sample_calm), sr=None)
    librosa.display.waveshow(y_calm, sr=sr, color='green')
    plt.title(f"Waveform: Calme ({sample_calm})")
    plt.xlabel("Temps (s)")
    plt.ylabel("Amplitude")

    # --- SUBPLOT 3: SPECTROGRAMME TURBULENCE (Focus Infrasons) ---
    plt.subplot(2, 2, 3)
    D_turb = librosa.amplitude_to_db(np.abs(librosa.stft(y_turb)), ref=np.max)
    librosa.display.specshow(D_turb, sr=sr, x_axis='time', y_axis='hz')
    plt.ylim(0, 500) # On zoom sur les basses fréquences
    plt.colorbar(format='%+2.0f dB')
    plt.title("Spectrogramme: Turbulence (Focus 0-500Hz)")
    # Annoter la zone doctorale
    plt.axhline(y=30, color='r', linestyle='--', alpha=0.5)
    plt.text(0.1, 35, "Zone Infrasons (1-30Hz)", color='red', fontweight='bold')

    # --- SUBPLOT 4: SPECTROGRAMME CALME ---
    plt.subplot(2, 2, 4)
    D_calm = librosa.amplitude_to_db(np.abs(librosa.stft(y_calm)), ref=np.max)
    librosa.display.specshow(D_calm, sr=sr, x_axis='time', y_axis='hz')
    plt.ylim(0, 500)
    plt.colorbar(format='%+2.0f dB')
    plt.title("Spectrogramme: Calme (Focus 0-500Hz)")

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "data_exploration.png")
    plt.savefig(plot_path)
    print(f"Visualisation sauvegardée dans : {plot_path}")
    plt.show()

if __name__ == "__main__":
    plot_exploration()
