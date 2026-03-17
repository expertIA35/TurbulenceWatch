# 🦅 TurbulenceWatch: Predictive Edge AI for Aviation Safety

[![PhD Research](https://img.shields.io/badge/Status-PhD%20Candidate%20Research-blue.svg)](https://github.com/yourusername/TurbulenceWatch)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![TinyML](https://img.shields.io/badge/AI-TinyML%20%2F%20Edge-orange.svg)](https://www.tensorflow.org/lite/micro)

## 📌 Project Overview
**TurbulenceWatch** is an on-board predictive system designed to detect Clear Air Turbulence (CAT) and wake vortices using **Infrasonic Spectral Analysis (1-20Hz)** and **Edge AI**. 

Unlike conventional LIDAR/RADAR systems, TurbulenceWatch focuses on high-frequency acoustic signatures processed in real-time on resource-constrained microcontrollers (TinyML).

### 🚀 Key Innovations
- **Infrasonic Detection**: Real-time processing of signals below 20Hz.
- **Hybrid Trainning**: Sim-to-Real mitigation using Von Karman physical models mixed with real cockpit ambient noise.
- **Ultra-Lightweight CNN**: Optimized model footprint (<80 KB) for deployment on ARM Cortex-M4 (Arduino Nano 33 BLE Sense / ESP32).
- **Spatial Awareness**: 3D mapping of turbulence encounters integrated with GPS trajectories.

---

## 🔬 Scientific Foundation
This project bridges the gap between ground-based infrasound research (NASA Langley) and embedded aeronautical applications. 

### Core Hypothesis
*"The energy shift in infrasonic frequency bands can be used as a predictive indicator of turbulence entry when processed by a quantized Convolutional Neural Network (CNN)."*

---

## 🛠️ Project Structure
- `scripts/`: Python framework for data generation, training, and 3D visualization.
- `tasks/`: Detailed PhD roadmap and technical requirements.
- `models/`: (Architecture shared, weights private) 
- `docs/`: Technical reports, spectral analysis, and 3D flight maps.

---

## 📊 Visualizations
### 📍 3D Turbulence Mapping
Interactive mapping of detected turbulence zones along a flight path (Toulouse - Paris simulation).
*(Insert link or GIF to turbulence_3d_map.html)*

### 📈 Spectral Analysis
Comparison between theoretical Von Karman turbulence and cockpit-noise hybridized signals.

---

## 🛡️ Intellectual Property Notice
This repository contains the full scientific framework and simulation methodology.
- **Academic Use**: Free under GPL-3.0.
- **Commercial Use**: Requires explicit agreement.
- **Trained Weights & C++ Inference Engine**: Available upon request for verified research laboratories (NASA, ONERA, etc.) and industry partners.

---

## 📬 Contact
**[Votre Nom]**  
*Chercheur-Innovateur en IA Embarquée & Sécurité Aérienne*  
[Lien vers votre LinkedIn]
