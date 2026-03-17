# 🛡️ Stratégie de Propriété Intellectuelle - TurbulenceWatch

Ce document définit la politique de partage du projet pour maximiser la visibilité académique tout en protégeant les actifs sensibles.

## 1. Classification des Actifs

| Actif | Publication | Raison |
| :--- | :--- | :--- |
| **Méthodologie & Architecture** | Publique | Prouve l'originalité et la rigueur scientifique. |
| **Scripts de Simulation (Von Karman)** | Publique | Permet la reproductibilité des résultats (indispensable en thèse). |
| **Logiciel de Visualisation (Plotly 3D)** | Publique | Outil de démonstration et d'impact visuel. |
| **Poids du Modèle (.h5, .tflite)** | **PRIVÉ** | Le "cerveau" final du système. Ne doit pas être accessible sans collaboration. |
| **Fichiers C++ Inférence (.h)** | **PRIVÉ / TEASER** | Contient le savoir-faire de l'implémentation Edge concrète. |
| **Datasets Hybrides (data/hybrid)** | **PRIVÉ** | Évite le vol de données labellisées prêtes à l'emploi. |

## 2. Licence de Diffusion
Le projet sera publié sur GitHub sous licence **GPL-3.0**. 
- **Obligation** : Toute dérivation doit être open-source et citer l'auteur original.
- **Protection** : Interdiction d'utilisation commerciale sans accord écrit.

## 3. Mécanisme de Protection (Gitignore)
Les fichiers suivants sont exclus de tout dépôt public :
- `models/*.h5`
- `models/*.tflite`
- `models/*.joblib`
- `models/*.h` (Sauf header d'exemple)
- `data/raw/*`
- `data/hybrid/*`
- `data/processed/*`

## 4. Message pour les Partenaires
*"Le code complet est partagé pour démonstration technique. Les poids entraînés et les modules d'inférence C++ haute performance sont réservés aux collaborations de recherche académique ou industrielle."*
