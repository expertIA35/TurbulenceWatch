# PROJET : TurbulenceWatch
## Statut : Candidat Doctorat (PhD Proposal & Research)

### 📚 Bibliographie de Base (SOTA)
1.  **NASA/TM—2011-217086** : "Development of an Infrasonic Microphone for Ground-based Detection of Clear-air Turbulence".
2.  **Shams, Q. A., et al.** : Recherche sur les signatures acoustiques des vortex de sillage.
3.  **Warden, P., & Situnayake, D.** : "TinyML: Machine Learning with TensorFlow Lite on Arduino and Ultra-Low-Power Microcontrollers".

### 🧪 Hypothèse de Recherche
"Est-il possible de prédire l'entrée en zone de turbulence de sillage ou de sillage atmosphérique en analysant les variations d'énergie dans les bandes de fréquences infrasonores (1-20Hz) via un réseau de neurones convolutif (CNN) optimisé pour l'embarqué ?"

### ⚖️ Analyse Comparative (SOTA)

| Approche         | Méthode                 | Coût      | Limite actuelle                     | Apport TurbulenceWatch          |
| :--------------- | :---------------------- | :-------- | :---------------------------------- | :------------------------------ |
| **NASA Langley** | Micros Infrasons au sol | Élevé     | Détection terrestre uniquement      | **Embarqué (On-board)**         |
| **Airbus / NCAR**| Accéléromètres          | Propriétaire| Réactif (mesure après impact)      | **Prédictif (Acoustique)**      |
| **JAXA / Boeing**| LIDAR Laser             | Très Élevé| Taille et poids importants          | **Low-cost & Léger (MEMS)**     |
| **SOTA Audio AI**| CNN Standard (GPU)      | Moyen     | Non optimisé pour le TinyML         | **Quantification INT8 (Edge)**  |

### 🕵️ Empreinte d'Originalité (PhD Thesis Gap)

1.  **TinyML Infrasonore** : Premier système de traitement temps-réel des infrasons aéronautiques dans un environnement contraint (<256 Ko RAM).
2.  **Explicabilité Aéronautique (XAI)** : Introduction de métriques d'explicabilité pour la certification des systèmes critiques en TinyML.
3.  **Démocratisation** : Scalabilité aux drones (UAV) et aviation générale, segments actuellement ignorés par les solutions Radar/Lidar coûteuses.

### 🔒 Protection de la Propriété Intellectuelle
Le projet adopte une stratégie d'**Open-Research Sécurisée** :
- **Framework & Méthodologie** : Public (GPL-3.0) pour validation académique.
- **Poids & Inférence Optimisée** : Privé (Réservé aux partenaires de recherche).
- **Preuve d'Antériorité** : Garantie par l'horodatage GitHub (Copyright Proof).

### 🛠️ Travail en cours
- [x] Initialisation de l'espace de travail séparé sur le bureau.
- [x] Analyse comparative SOTA et définition du gap doctoral.
- [x] Génération du Dataset de Simulation (Modèle physique de Von Karman).
- [x] Création du moteur d'Hybridation (Mix Simulation + Bruits Réels).
- [x] Entraînement du premier CNN de détection.
- [x] Quantification TinyML (Preuve de portabilité Edge).
- [x] Export du pont C++ pour microcontrôleur.
- [x] Cartographie 3D interactive (Plotly).
- [x] Stratégie de communication (LinkedIn/GitHub) & Protection PI.
- [ ] Inférence sur données réelles NASA (Task 1.4).
