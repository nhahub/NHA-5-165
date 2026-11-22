# THERMOX AI: Building Energy Efficiency Prediction & 3D Digital Twin

[![Live Web Application](https://img.shields.io/badge/Vercel-Live%20App-000000?style=for-the-badge&logo=vercel&logoColor=white)](https://thermoxai.vercel.app/)
[![API Swagger Docs](https://img.shields.io/badge/Render-API%20Docs-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://thermox-ai.onrender.com/docs)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-Research%20Notebook-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1trevsvjMaN1i5Mov2enUExiX9mxFGKLk?usp=sharing#scrollTo=eGxbrK_s1uYY)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18+-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://reactjs.org)
[![Three.js](https://img.shields.io/badge/Three.js-WebGL%203D-black?style=for-the-badge&logo=three.js&logoColor=white)](https://threejs.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-EB5424?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)

**Digital Egypt Pioneers Initiative (DEPI) — Graduation Project**  
**Track:** AWS Machine Learning Track  
**Team Code:** NHA-5-165 | **Team Name:** CloudXAi  

### Team Members
- **Ahmed AbdElrahman Ahmed** (Team Leader)
- **Manar Mahmoud Ahmed**
- **Ali Ahmed Ali Elmeleegy**
- **Ahmed Mohamed Sayed Shehata**
- **Eman Abo Al Qassem Abo Al Hjajj Omar**

---

## 🚀 Live Deployments & Cloud Resources

| Resource | Platform | URL / Direct Link |
| :--- | :--- | :--- |
| **Frontend Web App** | Vercel | [**thermoxai.vercel.app**](https://thermoxai.vercel.app/) |
| **Backend API (Swagger Docs)** | Render | [**thermox-ai.onrender.com/docs**](https://thermox-ai.onrender.com/docs) |
| **Backend Health Endpoint** | Render | [**thermox-ai.onrender.com/api/health**](https://thermox-ai.onrender.com/api/health) |
| **Interactive Research Colab** | Google Colab | [**Open in Google Colab**](https://colab.research.google.com/drive/1trevsvjMaN1i5Mov2enUExiX9mxFGKLk?usp=sharing#scrollTo=eGxbrK_s1uYY) |

---

## 1. Project Overview

Heating and cooling account for over 40% of residential energy consumption. Architectural choices made at early design stages—such as building compactness, glazing ratio, envelope surface area, and height—permanently lock in operational thermal performance.

**THERMOX AI** delivers an end-to-end, production-calibrated machine learning system that models and predicts **Heating Load** ($Y_1$) and **Cooling Load** ($Y_2$) from 8 architectural envelope parameters. The system features:

1. **3D WebGL Digital Twin Simulation:** Real-time Three.js residential house model whose dimensions, stories, glazing panels, and cardinal orientation deform parametrically as you adjust geometric sliders.
2. **Thermal Infrared Thermography Mode:** False-color infrared heatmap visualizing surface heat loss, thermal bridging, and fenestration radiant heat gain directly on the 3D model.
3. **Modular Machine Learning Pipeline:** Leakage-free preprocessing, domain-informed interaction ratios (`Wall_Surface_Ratio`, `Glazing_x_Height`), and 5-fold cross-validation.
4. **10 Calibrated Production Models:** 5 algorithms trained independently across both Heating and Cooling targets (Linear Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost), with champion models reaching $R^2 > 0.994$.
5. **FastAPI Inference Microservice:** Production REST API deployed on **Render** (and Heroku), serving predictions in $\approx 5\text{ms}$ directly from a unified model bundle.
6. **Modern React 18 + Vite SaaS UI:** Deployed on **Vercel** with full Dark/Light mode, European Energy Performance Certificate (EPC) grade badges, sensitivity sweep optimizer, and live model comparison benchmarks.

---

## 2. Directory Structure

```
energy-efficiency-prediction/
│
├── data/
│   ├── raw/
│   │   └── energy_efficiency_data.csv       # Original dataset (768 records)
│   └── processed/
│       ├── train.csv                        # 80% train split (614 rows)
│       └── test.csv                         # 20% test split (154 rows)
│
├── notebooks/
│   ├── Energy_project.ipynb                 # Complete 37-cell exploratory research notebook
│   ├── 01_eda.ipynb                         # Exploratory Data Analysis & visual plots
│   ├── 02_preprocessing_feature_engineering.ipynb # Pipelines, scaling & interaction features
│   ├── 03_basic_modeling.ipynb              # Baseline & Tree Modeling
│   └── 04_advanced_modeling_tuning.ipynb    # Gradient Boosting, XGBoost Tuning & Bundle export
│
├── src/
│   └── energy_efficiency/                   # Core Python Package
│       ├── __init__.py                      # Package exports
│       ├── config.py                        # Constants, paths, hyperparameters
│       ├── data_loader.py                   # Data ingestion utilities
│       ├── preprocessing.py                 # Pipeline builder (Linear & Tree)
│       ├── feature_engineering.py           # Interaction features calculation
│       ├── evaluation.py                    # Multi-metric evaluation (MAE, RMSE, R²)
│       └── model_export.py                  # Single compressed bundle exporter
│
├── artifacts/
│   ├── candidates/                          # 10 individual candidate models
│   │   ├── linear_regression_heating.joblib
│   │   ├── linear_regression_cooling.joblib
│   │   ├── decision_tree_heating.joblib
│   │   ├── decision_tree_cooling.joblib
│   │   ├── random_forest_heating.joblib
│   │   ├── random_forest_cooling.joblib
│   │   ├── gradient_boosting_heating.joblib
│   │   ├── gradient_boosting_cooling.joblib
│   │   ├── xgboost_heating.joblib
│   │   └── xgboost_cooling.joblib
│   │
│   └── production/
│       └── model_bundle.joblib              # Unified production bundle (13.4 MB compressed)
│
├── reports/
│   ├── eda/                                 # Data quality report & plots
│   │   ├── data_quality_report.csv
│   │   ├── eda_summary.txt
│   │   └── plots/
│   ├── preprocessing/                       # Feature correlation & experiment logs
│   │   └── feature_experiments.csv
│   └── modeling/                            # Holdout results & cross-validation logs
│       ├── basic_models_results.csv
│       ├── advanced_models_results.csv
│       ├── cross_validation_results.csv
│       └── hyperparameter_results.csv
│
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                          # FastAPI endpoints, CORS & ping routes
│   │   ├── schemas.py                       # Pydantic request/response validation
│   │   ├── service.py                       # Inference & optimization engine
│   │   └── config.py                        # Server settings, sys.path & paths
│   ├── requirements.txt                     # Backend-specific dependencies
│   └── tests/
│       ├── test_api.py                      # REST API endpoint tests
│       └── test_prediction.py               # Model prediction & logic tests
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Header.jsx                   # Brand logo, live indicator & dark mode toggle
│   │   │   ├── Logo.jsx                     # 3D isometric architectural logo
│   │   │   ├── BuildingViewer3D.jsx         # Three.js 3D WebGL residential house digital twin
│   │   │   ├── ThermalDistribution.jsx      # Heat vs Cool ratio & EPC grade badge
│   │   │   ├── BuildingForm.jsx             # Architectural envelope sliders & presets
│   │   │   ├── PredictionCard.jsx           # Thermal load inference metric cards
│   │   │   ├── OptimizerTool.jsx            # Multi-variable parameter sweep tool
│   │   │   └── ModelComparisonTable.jsx     # Comprehensive model benchmark suite
│   │   ├── context/
│   │   │   └── ThemeContext.jsx             # Dark / Light theme provider & design tokens
│   │   ├── pages/
│   │   │   ├── SimulatorPage.jsx            # Main 3D simulation page
│   │   │   ├── OptimizerPage.jsx            # Sensitivity optimization page
│   │   │   ├── BenchmarksPage.jsx           # Model benchmark evaluation page
│   │   │   └── DocsPage.jsx                 # Technical documentation page
│   │   ├── services/
│   │   │   └── prediction.js                # API client with auto-normalized URL & fallback
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── package.json
│   ├── vite.config.js
│   └── vercel.json                          # Vercel SPA routing configuration
│
├── docs/
│   ├── data_dictionary.md                   # Feature definitions & ranges
│   ├── methodology.md                       # Pipeline & ML workflow
│   ├── model_card.md                        # Model cards & benchmark results
│   ├── api_documentation.md                 # REST API documentation
│   ├── limitations.md                       # Boundary conditions
│   └── deployment_guide.md                  # Step-by-step Render & Vercel deployment guide
│
├── Documentations/                          # Original complete project documentation
│   ├── 01_Project_Charter/
│   ├── 02_Project_Plan/
│   ├── 03_Weekly_Progress/
│   └── 04_Final_Report/
│
├── tests/
│   ├── test_preprocessing.py                # Preprocessor unit tests
│   └── test_feature_engineering.py          # Feature engineering unit tests
│
├── render.yaml                              # Render 1-click blueprint configuration
├── Procfile                                 # Heroku deployment directive
├── requirements.txt                         # Full dependencies
├── .gitignore
└── README.md
```

---

## 3. Model Benchmark Results (All 10 Models)

Evaluated on the independent 20% holdout test partition (154 samples):

| # | Algorithm | Target Load | MAE ($\text{kWh/m}^2$) | RMSE ($\text{kWh/m}^2$) | $R^2$ Score | Status |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| 1 | **Linear Regression** | Heating Load | 1.8728 | 2.8002 | 0.9248 | Baseline |
| 2 | **Linear Regression** | Cooling Load | 2.1165 | 3.0471 | 0.8998 | Baseline |
| 3 | **Decision Tree** | Heating Load | 0.4071 | 0.5873 | 0.9967 | Candidate |
| 4 | **Decision Tree** | Cooling Load | 1.3253 | 2.2609 | 0.9448 | Candidate |
| 5 | **Random Forest** | Heating Load | 0.3446 | 0.4919 | 0.9977 | Candidate |
| 6 | **Random Forest** | Cooling Load | 1.1659 | 1.9043 | 0.9609 | Candidate |
| 7 | **Gradient Boosting** | Heating Load | 0.3600 | 0.5100 | 0.9974 | Candidate |
| 8 | **Gradient Boosting** | Cooling Load | 1.0500 | 1.7300 | 0.9769 | Candidate |
| 9 | **XGBoost (Tuned)** | Heating Load | **0.2800** | **0.4100** | **0.9983** | **Production Champion** |
| 10 | **XGBoost (Tuned)** | Cooling Load | **0.6500** | **1.0200** | **0.9942** | **Production Champion** |

---

## 4. Local Installation & Quickstart

### Prerequisites
- Python 3.10+
- Node.js 18+

### Backend Setup
```bash
# Clone the repository
git clone https://github.com/blurz17/NHA-5-165.git
cd NHA-5-165

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run unit and API tests
pytest tests/ backend/tests/ -v

# Start FastAPI backend server
uvicorn backend.app.main:app --reload --port 8000
```
* API will be live at `http://localhost:8000`
* Interactive Swagger docs at `http://localhost:8000/docs`

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
* Frontend will be live at `http://localhost:5173`

---

## 5. Deployment Guide

Full detailed instructions are available in [docs/deployment_guide.md](docs/deployment_guide.md).

### A. Backend on Render ([Live API Docs](https://thermox-ai.onrender.com/docs))
* Uses the included [`render.yaml`](render.yaml) blueprint file for 1-click deployment.
* Start Command: `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT` (or `uvicorn app.main:app`)
* Health check: `/api/health`

### B. Frontend on Vercel ([Live Application](https://thermoxai.vercel.app/))
1. Import repository in Vercel.
2. In Project Settings, set **Root Directory** to `frontend`.
3. Add Environment Variable:
   * `VITE_API_URL` = `https://thermox-ai.onrender.com`
4. Deploy! Vercel handles automated builds and edge routing with [`frontend/vercel.json`](frontend/vercel.json).

---

## 6. License & Attribution
Developed for the **Digital Egypt Pioneers Initiative (DEPI)** - Ministry of Communications and Information Technology (MCIT), Egypt.  
Dataset: *Tsanas & Xifara (2012), Energy and Buildings, Oxford University / UCI Machine Learning Repository.*
