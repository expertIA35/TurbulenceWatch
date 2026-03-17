# 🦅 MASTER PLAN : TurbulenceWatch (PhD Research & Edge AI)

Ce document sert de fil conducteur pour la réalisation du projet, de la simulation à la validation doctorale.

## 📌 VISION DU PROJET
Développer un système de détection prédictive de turbulences basé sur l'analyse spectrale (infrasons) via Edge AI, déployable sur microcontrôleurs.

---

## 📅 ROADMAP DES SPRINT (TASKS)

### 🔬 SPRINT 1 : Fondations & Simulation (DATA)
- [x] **Task 1.1** : [Génération du Dataset Synthétique](file:///C:/Users/chaou/Desktop/TurbulenceWatch_PHD/tasks/task_1_1_data_gen.md)
- [x] **Task 1.2** : [Exploration & Visualisation Spectrale](file:///C:/Users/chaou/Desktop/TurbulenceWatch_PHD/tasks/task_1_2_eda.md)
- [x] **Task 1.3** : [Extraction de Features (MFCC & Infrasons)](file:///C:/Users/chaou/Desktop/TurbulenceWatch_PHD/tasks/task_1_3_features.md)
- [ ] **Task 1.4** : [Intégration Données Réelles (NASA, Airbus, OpenSky)](file:///C:/Users/chaou/Desktop/TurbulenceWatch_PHD/tasks/task_1_4_real_data.md)

### 🧠 SPRINT 2 : Intelligence & Modélisation (AI)
- [x] **Task 2.1** : [Entraînement des Baselines (ML Classique)](file:///C:/Users/chaou/Desktop/TurbulenceWatch_PHD/tasks/task_2_1_baselines.md)
- [x] **Task 2.2** : [Deep Learning (CNN sur Spectrogrammes)](file:///C:/Users/chaou/Desktop/TurbulenceWatch_PHD/tasks/task_2_2_cnn.md)
- [x] **Task 2.3** : [Optimisation TinyML & Quantification](file:///C:/Users/chaou/Desktop/TurbulenceWatch_PHD/tasks/task_2_3_tinyml.md)

### ⚡ SPRINT 3 : Déploiement & Interface (EDGE)
- [x] **Task 3.1** : [Génération du Pont C++ (Header)](file:///C:/Users/chaou/Desktop/TurbulenceWatch_PHD/tasks/task_3_1_cpp_deploy.md)
- [ ] **Task 3.2** : [Interface Streamlit (Monitoring Temps Réel)](file:///C:/Users/chaou/Desktop/TurbulenceWatch_PHD/tasks/task_3_2_dashboard.md)

### 🎓 SPRINT 4 : Validation Doctorale (RECHERCHE)
- [ ] **Task 4.1** : [Rédaction du Rapport Technique & Biblio](file:///C:/Users/chaou/Desktop/TurbulenceWatch_PHD/tasks/task_4_1_report.md)
- [ ] **Task 4.2** : [Préparation du Post LinkedIn (Showcase)](file:///C:/Users/chaou/Desktop/TurbulenceWatch_PHD/tasks/task_4_2_linkedin.md)

---

## 🛠️ STACK TECHNIQUE
- **Langages** : Python (3.10+), C++ (Arduino/ESP-IDF)
- **IA/ML** : TensorFlow/Keras, Scikit-Learn, TensorFlow Lite Micro
- **Audio** : Librosa, Scipy.signal
- **Interface** : Streamlit
