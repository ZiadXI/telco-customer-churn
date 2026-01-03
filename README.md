---
title: Telco Customer Churn
emoji: 📉
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
app_port: 7860
---

# 📉 Telco Customer Churn Prediction — End-to-End MLOps System

![CI/CD Pipeline](https://github.com/ziadkassem/telco-churn-app/actions/workflows/ci_pipeline.yml/badge.svg)
[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/ziadkassem/telco-churn-app)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED)](https://www.docker.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue)](https://mlflow.org/)

A **production-ready MLOps project** that predicts customer churn for a telecom provider using **XGBoost**.  
The system demonstrates **real-world ML deployment practices**, including strict data validation, experiment tracking, containerization, CI/CD automation, and cloud deployment on **Hugging Face Spaces**.

> 🎯 **Goal:** Showcase how to move a machine learning model from experimentation to a reliable, monitored, and deployable production system.

---

## 🔍 Problem Overview

Customer churn is a critical business problem for telecom companies.  
This project predicts whether a customer is likely to churn based on demographics, service usage, and billing information — enabling proactive retention strategies.

---

## 🚀 Key Features

### 🧠 Machine Learning
- **XGBoost Classifier** with tuned hyperparameters
- Custom feature engineering & decision threshold optimization
- Reproducible training pipeline

### 🛡️ Enterprise-Grade Validation
- **Training-time validation:**  
  Pandera + custom checks ensure clean, consistent datasets before model fitting
- **Inference-time validation:**  
  Pydantic schemas enforce strict API contracts  
  → *No garbage in, no garbage out*

### ⚡ Scalable Serving
- **FastAPI** backend with async endpoints
- Auto-generated OpenAPI / Swagger docs
- Clean separation between serving, validation, and UI layers

### 🎨 Interactive UI
- **Gradio** frontend (Dark Mode)
- Mounted directly on FastAPI for demos & testing
- Zero frontend setup required

### 🔄 CI/CD & Deployment
- **GitHub Actions** pipeline:
  - Run tests
  - Validate Docker build
  - Deploy automatically to Hugging Face Spaces
- **Dockerized** for consistent execution across environments

### 📊 Experiment Tracking
- **MLflow** for tracking:
  - Metrics
  - Parameters
  - Model artifacts

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-----|-----------|---------|
| Model | XGBoost | Gradient boosting classifier |
| API | FastAPI | High-performance inference |
| UI | Gradio | Interactive web interface |
| Validation | Pandera / Pydantic | Data quality enforcement |
| Tracking | MLflow | Experiment management |
| Containerization | Docker | Environment consistency |
| CI/CD | GitHub Actions | Automated testing & deployment |
| Hosting | Hugging Face Spaces | Production deployment |

---

## 📂 Project Structure

```text
├── .github/workflows
│   └── ci_pipeline.yml        # CI/CD pipeline
├── artifacts/                 # Models & feature maps (Git LFS)
├── src
│   ├── data/                  # Data loading & preprocessing
│   ├── features/              # Feature engineering
│   ├── models/                # Training & evaluation
│   └── serving                # Production serving
│       ├── app.py             # FastAPI entry point
│       ├── gradio_app.py      # Gradio UI
│       ├── schema.py          # Pydantic models
│       └── validator.py       # Business rules validation
├── tests/                     # Unit & integration tests
├── Dockerfile                 # Container definition
├── requirements.txt           # Dependencies
└── README.md
