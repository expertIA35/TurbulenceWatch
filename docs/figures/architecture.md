# 🏗️ Architecture du Système TurbulenceWatch

Ce document présente les diagrammes d'architecture du projet, utilisant le format Mermaid.

## 📡 Flux de Données Global (Data Pipeline)

```mermaid
graph TD
    subgraph "PHASE 1: DATA SIMULATION"
        A[Von Karman Physical Model] -->|Synthetic Signal| B(Infrasonic Noise Gen)
        C[Cockpit Ambient Noise] -->|Real Samples| D(Hybridization Engine)
        B --> D
    end

    subgraph "PHASE 2: EDGE AI PIPELINE"
        D --> E{Feature Extraction}
        E -->|Mel-Spectrograms| F[Quantized CNN - 70KB]
        F -->|Inference| G[Turbulence Probability]
    end

    subgraph "PHASE 3: SPATIAL MAPPING"
        G --> H[GPS Data Fusion]
        H --> I[Plotly 3D Interactive Map]
    end

    subgraph "PHASE 4: DEPLOYMENT"
        F -->|Export| J[C++ Header]
        J --> K[Arduino / ESP32 Hardware]
    end
```

## 🧠 Structure du Modèle CNN (TinyML)

```mermaid
graph LR
    Input[Audio Input 2s] --> Pre[Preprocessing / STFT]
    Pre --> C1[Conv2D + Relu]
    C1 --> P1[MaxPooling]
    P1 --> C2[Conv2D + Relu]
    C2 --> P2[MaxPooling]
    P2 --> F[Flatten]
    F --> D1[Dense 16]
    D1 --> Out[Sigmoid Output]
```

## 🛠️ Stack Technique
- **Logiciel** : Python, TensorFlow, Plotly, Scipy, Librosa.
- **Matériel Cible** : ARM Cortex-M4 / ESP32.
- **Format d'Export** : TFLite INT8 / C++ Header.
