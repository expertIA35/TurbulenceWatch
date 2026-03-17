# Task 1.4 : Intégration Données Réelles (NASA, Airbus, OpenSky)

## 🎯 Objectifs
Valider les modèles entraînés sur des données synthétiques/hybrides avec des données réelles issues de l'aéronautique commerciale et de recherche.

## 📊 Sources Identifiées
- **NASA DASHlink** : Sources audio cockpit (ASRS).
- **Airbus Open Data** : Données de vol d'essai (A350).
- **OpenSky Network** : Vecteurs d'état et rapports de turbulences (2020-2024).
- **DEMAND** : Bruit de fond cabine avion haute fidélité.

## 🛠️ Actions à mener
1. [ ] Demander les accès pour les datasets restreints (MASC, IAGOS).
2. [ ] Télécharger les échantillons open-source via `scripts/download_manager.py`.
3. [ ] Procéder à la normalisation (Resampling 16kHz, Normalisation Peak).
4. [ ] Effectuer un test d'inférence (Inference Test) du modèle CNN actuel sur ces données réelles pour mesurer le **Sim-to-Real Gap**.
