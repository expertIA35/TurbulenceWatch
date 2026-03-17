# 📊 Task 1.1 : Génération du Dataset Synthétique

## 🎯 Objectif
Créer une "vérité terrain" simulée pour entraîner nos modèles dans un environnement contrôlé.

## 📝 Spécifications Techniques
- **Format** : WAV (Mono, 16kHz)
- **Classes** : `Turbulence` (Classe 1) vs `Calme` (Classe 0)
- **Physique Simulée** :
    - Bruit aérodynamique large bande.
    - Rafales (modulations d'amplitude).
    - **Infrasons** (Composantes 5-50 Hz) pour la signature PhD.

## ✅ Liste de contrôle
- [ ] Créer le script `scripts/generate_turbulence_dataset.py`.
- [ ] Générer 250 fichiers par classe.
- [ ] Créer le fichier `metadata.csv` pour le suivi.
- [ ] Valider l'intégrité sonore (volume, clipping).
