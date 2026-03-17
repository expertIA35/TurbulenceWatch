# ⚡ Task 2.3 : Optimisation TinyML & Quantification

## 🎯 Objectif
Réduire la taille du modèle pour qu'il puisse tourner sur un microcontrôleur sans perdre en précision.

## 📝 Spécifications Techniques
- **Cible** : TensorFlow Lite Micro
- **Quantification** : `Full Integer Quantization` (INT8)
- **Outil** : TFLite Converter

## ✅ Liste de contrôle
- [ ] Créer le script `scripts/quantize_model.py`.
- [ ] Convertir le modèle `.h5` en `.tflite`.
- [ ] Vérifier la taille (Cible : < 100 KB).
- [ ] Évaluer la perte de précision (Drop de max 2%).
