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
[![🤗 Hugging Face Spaces](https://img.shields.io/badge/Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/ziadkassem/telco-churn-app)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED)](https://www.docker.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue)](https://mlflow.org/)

A **production-ready MLOps project** for predicting customer churn in the telecom domain using **XGBoost**.  
This repository demonstrates how to move a machine learning model from **experimentation to deployment** using real-world MLOps practices.

> 🎯 **Goal:** Build a reliable, validated, reproducible, and deployable ML system — not just a notebook model.

---

## 🔍 Problem Overview

Customer churn is a major business challenge for telecom companies.  
This system predicts whether a customer is likely to churn based on demographics, service usage, and billing data, enabling proactive retention strategies.

---

## 🚀 Key Features

### 🧠 Machine Learning
- XGBoost classifier optimized for tabular data
- Feature engineering pipeline shared between training and inference
- Tuned decision threshold
- Fully reproducible training workflow

### 🛡️ Dual-Layer Validation
**Training Time**
- Pandera + custom validation logic
- Schema enforcement and constraint checks

**Inference Time**
- Pydantic schemas for strict API contracts
- Business-rule validation layer  
→ *No garbage in, no garbage out*

### ⚡ Production Serving
- FastAPI backend with async endpoints
- Auto-generated OpenAPI / Swagger docs
- Clean separation of API, validation, and UI layers

### 🎨 Interactive UI
- Gradio web interface (Dark Mode)
- Mounted directly on FastAPI
- Ideal for demos and manual testing

### 🐳 Containerization
- Fully Dockerized application
- Identical behavior across environments

### 🔄 CI/CD Automation
- GitHub Actions pipeline:
  - Run tests
  - Validate Docker build
  - Deploy automatically to Hugging Face Spaces
- GitOps-style workflow

### 📊 Experiment Tracking
- MLflow for tracking metrics, parameters, and artifacts

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-----|-----------|---------|
| Model | XGBoost | Gradient boosting classifier |
| API | FastAPI | High-performance inference |
| UI | Gradio | Interactive frontend |
| Validation | Pandera, Pydantic | Data quality enforcement |
| Tracking | MLflow | Experiment management |
| Containerization | Docker | Reproducibility |
| CI/CD | GitHub Actions | Automated deployment |
| Hosting | Hugging Face Spaces | Production hosting |

---

## 📂 Project Structure

```text
├── .github/workflows
│   └── ci_pipeline.yml        # CI/CD configuration
├── artifacts/                 # Trained models & feature maps (Git LFS)
├── src
│   ├── data/                  # Data loading & preprocessing
│   ├── features/              # Feature engineering
│   ├── models/                # Training & evaluation
│   └── serving                # Production serving
│       ├── app.py             # FastAPI entry point
│       ├── gradio_app.py      # Gradio UI
│       ├── schema.py          # Pydantic schemas
│       └── validator.py       # Business validation
├── tests/                     # Unit & integration tests
├── Dockerfile                 # Container definition
├── requirements.txt           # Dependencies
└── README.md
````

---

## 💻 Installation & Usage

### Option 1 — Run with Docker (Recommended)

No local Python setup required.

```bash
docker build -t churn-app .
docker run -p 7860:7860 churn-app
```

* **UI:** [http://localhost:7860](http://localhost:7860)
* **API Docs:** [http://localhost:7860/docs](http://localhost:7860/docs)

---

### Option 2 — Run Locally (Python)

```bash
pip install -r requirements.txt
uvicorn src.serving.app:app --host 0.0.0.0 --port 7860 --reload
```

---

## 🔌 API Documentation

**Endpoint:** `POST /predict`

### Example Request

```json
{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.0,
  "TotalCharges": 840.0
}
```

### Example Response

```json
{
  "churn_probability": 0.73,
  "churn_prediction": true
}
```

---

## 🔄 CI/CD Pipeline (GitOps Flow)

1. **Push** → Code pushed to `main`
2. **Test** → GitHub Actions runs automated tests
3. **Build** → Docker image is validated
4. **Deploy** → Synced to Hugging Face Hub → automatic rebuild

This guarantees every deployment is tested and reproducible.

---

## 🔮 Future Improvements

* 📉 Data drift detection (EvidentlyAI)
* 🗂️ Remote MLflow Model Registry
* 📈 Online monitoring & alerting
* 🧪 Expanded edge-case testing
* 🔐 Authentication & rate limiting

---

## 👤 Author

**Ziad Kassem**
Computer Science & Data Science
Focus: **MLOps, Production ML Systems, and Scalable AI**

---

⭐ If you find this project useful or inspiring, consider starring the repository.

```
