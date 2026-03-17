import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import librosa
import librosa.display

# CONFIGURATION
RAW_DIR = os.path.join("..", "data", "raw", "turbulence")
HYBRID_DIR = os.path.join("..", "data", "hybrid", "turbulence")
OUTPUT_DIR = os.path.join("..", "docs", "reports")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_report():
    # Prendre le premier fichier de turbulence
    try:
        files = [f for f in os.listdir(RAW_DIR) if f.endswith('.wav')]
    except FileNotFoundError:
        print(f"Erreur : Dossier {RAW_DIR} introuvable.")
        return
        
    if not files:
        print("Erreur : Aucun fichier trouvé.")
        return
        
    filename = files[0]
    raw_path = os.path.join(RAW_DIR, filename)
    hybrid_path = os.path.join(HYBRID_DIR, filename)

    # Chargement
    y_raw, sr = librosa.load(raw_path, sr=None)
    y_hybrid, sr = librosa.load(hybrid_path, sr=None)
    
    # Calcul des Spectres (FFT)
    fft_raw = np.abs(np.fft.rfft(y_raw))
    fft_hybrid = np.abs(np.fft.rfft(y_hybrid))
    # Correction de l'appel de fonction : np.fft.rfftfreq au lieu de rfftfrequencies
    freqs = np.fft.rfftfreq(len(y_raw), d=1/sr)

    # --- VISUALISATION ---
    plt.figure(figsize=(16, 12))

    # 1. Comparaison Temporelle
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(y_raw, sr=sr, alpha=0.5, color='blue', label='Synthétique Pure (Von Karman)')
    librosa.display.waveshow(y_hybrid, sr=sr, alpha=0.3, color='red', label='Hybride (Avec Bruit Cockpit)')
    plt.title(f"Analyse Temporelle : {filename}")
    plt.legend()

    # 2. Comparaison Fréquentielle (FFT) Log-Scale
    plt.subplot(3, 1, 2)
    plt.semilogy(freqs, fft_raw, alpha=0.8, color='blue', label='Spectre Turbulence Pure')
    plt.semilogy(freqs, fft_hybrid, alpha=0.6, color='red', label='Spectre Hybride (Sim-to-Real)')
    plt.title("Réponse Fréquentielle (FFT)")
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Magnitude (Log)")
    # On limite à 1000Hz pour bien voir les infrasons et le bruit moteur
    plt.xlim(0, 1000) 
    plt.legend()
    # Annoter le bruit moteur (vers 100-200Hz injecté dans hybridize)
    plt.annotate('Signature Infrasons Boostée', xy=(15, np.max(fft_raw)), xytext=(60, np.max(fft_raw)*2),
                 arrowprops=dict(facecolor='blue', shrink=0.05), fontsize=10, color='blue')

    # 3. Spectrogramme Hybride (Celle que l'IA va voir)
    plt.subplot(3, 1, 3)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y_hybrid)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
    plt.ylim(0, 500)
    plt.colorbar(format='%+2.0f dB')
    plt.title("Spectrogramme Hybride pour Entraînement TinyML (Focus 0-500Hz)")
    
    # Annoter la zone d'intérêt scientifique
    plt.axhline(y=30, color='yellow', linestyle='--', alpha=0.7)
    plt.text(0.2, 35, "Zone Infrasons (Signature Candidat Doctorat)", color='yellow', fontweight='bold')

    plt.tight_layout()
    report_path = os.path.join(OUTPUT_DIR, "spectral_comparison_report.png")
    plt.savefig(report_path)
    print(f"Rapport doctoral généré dans : {report_path}")

if __name__ == "__main__":
    generate_report()
