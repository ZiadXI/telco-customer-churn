---
title: Telco Customer Churn
emoji: 📉
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
app_port: 7860
---

# 📉 Telco Customer Churn Prediction System

![CI/CD Pipeline](https://github.com/ziadkassem/telco-churn-app/actions/workflows/ci_pipeline.yml/badge.svg)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/ziadkassem/telco-churn-app)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED)](https://www.docker.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)

An End-to-End MLOps project that predicts customer churn using XGBoost. This system features a robust production architecture including strict data validation (Pydantic & Pandera), a FastAPI backend, a user-friendly Gradio UI, and a fully automated CI/CD pipeline deploying to Hugging Face Spaces.

## 🚀 Key Features

* **🧠 Advanced Machine Learning:** XGBoost classifier with optimized feature engineering and threshold tuning.
* **🛡️ Dual-Layer Validation:**
    * **Training Time:** Strict DataFrame validation using `Pandera`/Custom logic to ensure data quality before training.
    * **Inference Time:** `Pydantic` schemas ensure API inputs match expected formats and constraints (No garbage in, no garbage out).
* **⚡ High-Performance API:** Built with **FastAPI** for asynchronous, high-throughput predictions.
* **🎨 Interactive UI:** Embedded **Gradio** interface (Dark Mode) mounted directly on the API for easy testing and demos.
* **🐳 Containerized:** Fully Dockerized application for consistent execution across environments.
* **🔄 CI/CD Automation:** GitHub Actions pipeline that automatically tests, builds, and pushes the Docker image to Hugging Face Spaces on every commit.
* **📊 Experiment Tracking:** Integrated with **MLflow** for tracking model metrics and parameters.

## 🛠️ Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Model** | XGBoost | Gradient Boosting for classification |
| **API Backend** | FastAPI | Async REST API with auto-generated docs |
| **Frontend** | Gradio | Interactive web interface for model inference |
| **Validation** | Pydantic | Runtime data validation for API requests |
| **Container** | Docker | Application isolation and deployment |
| **CI/CD** | GitHub Actions | Automated testing and deployment pipeline |
| **Hosting** | Hugging Face | Cloud hosting for the Docker container |

## 📂 Project Structure

```text
├── .github/workflows
│   └── ci_pipeline.yml    # CI/CD Configuration
├── artifacts/             # Trained models and feature maps (Git LFS)
├── src
│   ├── data/              # Data loading and preprocessing logic
│   ├── features/          # Feature engineering scripts
│   ├── models/            # Training and evaluation scripts
│   └── serving            # Production code
│       ├── app.py         # Main FastAPI entry point
│       ├── gradio_app.py  # Frontend UI logic
│       ├── schema.py      # Pydantic validation models
│       └── validator.py   # Custom business logic validation
├── tests/                 # Unit tests
├── Dockerfile             # Container definition
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation