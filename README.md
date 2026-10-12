# 🤖 AI-Assisted Threat Detection

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![ML](https://img.shields.io/badge/ML-scikit--learn-orange.svg)
![Domain](https://img.shields.io/badge/Domain-Cybersecurity-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A machine learning tool that classifies log events as **malicious** or **benign** using a Random Forest classifier trained on engineered features.

Built for **Cybersecurity** + **AI** applications — the kind of tool used in modern SOC environments.

## ✨ Features

- 🧠 **ML Classifier** — Random Forest (200 trees) with cross-validation
- 🔧 **Feature Engineering** — one-hot encoding, scaling, IP/user/port flags
- 📊 **Rich Evaluation** — accuracy, precision, recall, F1, confusion matrix
- 🎯 **Feature Importance** — shows which features matter most
- 📁 **Model Persistence** — save/load trained model with joblib
- 📄 **HTML Report** — clean, professional report with confidence bars
- 🌐 **Batch Prediction** — score entire CSV files of events

## 🛠️ Requirements

- Python 3.8+
- scikit-learn, pandas, numpy, jinja2, joblib

Install:

```bash
pip install -r requirements.txt