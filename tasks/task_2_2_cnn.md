# 🧠 Task 2.2 : Deep Learning (CNN sur Spectrogrammes)

## 🎯 Objectif
Passer d'une analyse de features tabulaires à une analyse visuelle du son pour capter les motifs complexes des turbulences.

## 📝 Spécifications Techniques
- **Modèle** : CNN 2D (Convolutional Neural Network)
- **Entrée** : Log-Mel Spectrogrammes
- **Architecture** :
    - Conv2D (16, 3x3) + ReLU
    - MaxPool2D
    - Conv2D (32, 3x3) + ReLU
    - Dense (64) + Dropout
    - Output (Sigmoid pour binaire)

## ✅ Liste de contrôle
- [ ] Créer le script `scripts/train_cnn_model.py`.
- [ ] Pré-calculer les spectrogrammes.
- [ ] Entraîner avec Early Stopping.
- [ ] Comparer l'accuracy avec les modèles classiques.
