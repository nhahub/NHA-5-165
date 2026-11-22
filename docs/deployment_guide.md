# Cloud Deployment Guide: Render (Backend) & Vercel (Frontend)

This guide provides step-by-step instructions for deploying the **THERMOX AI** Building Energy Efficiency Prediction system.

---

## 🚀 Live Production Deployments

| Component | Platform | URL |
| :--- | :--- | :--- |
| **Frontend Web App** | Vercel | [**thermoxai.vercel.app**](https://thermoxai.vercel.app/) |
| **Backend API (Swagger Docs)** | Render | [**thermox-ai.onrender.com/docs**](https://thermox-ai.onrender.com/docs) |
| **Backend Health Endpoint** | Render | [**thermox-ai.onrender.com/api/health**](https://thermox-ai.onrender.com/api/health) |
| **Interactive Research Colab** | Google Colab | [**Open in Google Colab**](https://colab.research.google.com/drive/1trevsvjMaN1i5Mov2enUExiX9mxFGKLk?usp=sharing#scrollTo=eGxbrK_s1uYY) |

---

## 1. Backend Deployment on Render (FastAPI)

[Render](https://render.com) hosts the Python FastAPI application that runs real-time inference on the bundled machine learning models (`artifacts/production/model_bundle.joblib`).

### Option A: Automated Blueprint Deployment (Recommended)
1. Log in to your [Render Dashboard](https://dashboard.render.com/).
2. Click **New +** in the top navigation and select **Blueprint**.
3. Connect your GitHub account and select the repository: `blurz17/NHA-5-165`.
4. Render will automatically detect the [`render.yaml`](../render.yaml) configuration file at the repository root.
5. Click **Apply**. Render will automatically install dependencies from `requirements.txt`, start `uvicorn`, and run health checks at `/api/health`.

### Option B: Manual Web Service Setup
If creating the service manually:
1. In Render Dashboard, click **New +** $\rightarrow$ **Web Service**.
2. Select repository: `blurz17/NHA-5-165`.
3. Configure the following fields:
   * **Name**: `energy-efficiency-api` (or your choice)
   * **Language / Runtime**: `Python 3`
   * **Branch**: `main`
   * **Region**: Choose closest to your users (e.g., Frankfurt or Oregon)
   * **Build Command**: `pip install -r requirements.txt`
   * **Start Command**: `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT`
   * **Instance Type**: `Free`
4. Expand **Advanced Settings**:
   * **Health Check Path**: `/api/health`
   * **Add Environment Variable**:
     * `PYTHON_VERSION` = `3.12.0`
5. Click **Create Web Service**.

> [!NOTE]
> Once deployed, copy your assigned Render public URL (e.g. `https://energy-efficiency-api.onrender.com`). You will need this URL for the Vercel frontend.

---

## 2. Frontend Deployment on Vercel (React + Three.js)

[Vercel](https://vercel.com) hosts the high-performance React application with Three.js 3D WebGL digital twin rendering.

### Step-by-Step Vercel Setup
1. Log in to your [Vercel Dashboard](https://vercel.com/dashboard).
2. Click **Add New...** $\rightarrow$ **Project**.
3. Import your GitHub repository: `blurz17/NHA-5-165`.
4. In the **Configure Project** screen, configure the following:

   #### ⚠️ Critical Step: Root Directory
   * Click **Edit** next to **Root Directory**.
   * Select the `frontend` folder and click **Continue**.

   #### Build & Output Settings
   * **Framework Preset**: `Vite` (auto-detected)
   * **Build Command**: `vite build` (or `npm run build`)
   * **Output Directory**: `dist`

   #### Environment Variables
   * Expand **Environment Variables** and add:
     * **Key**: `VITE_API_URL`
     * **Value**: `https://energy-efficiency-api.onrender.com` (replace with your actual Render URL from Step 1)

5. Click **Deploy**.

---

## 3. Verification & Live Health Check

Once both services are deployed:
1. **Test Backend**: Visit `https://<your-render-app>.onrender.com/api/health` in your browser. It should return:
   ```json
   {
     "status": "ok",
     "version": "2.0.0",
     "models_loaded": true,
     "algorithms_available": ["Linear Regression", "Decision Tree", "Random Forest", "Gradient Boosting", "XGBoost"]
   }
   ```
2. **Interactive Swagger Docs**: Visit `https://<your-render-app>.onrender.com/docs` to test endpoints via the OpenAPI interactive UI.
3. **Open Frontend**: Visit your Vercel deployment URL (e.g., `https://energy-efficiency-frontend.vercel.app`).
   * Confirm the green badge in the top navigation shows **`LIVE ENGINE (v2.0.0)`**.
   * Adjust the architectural sliders (Surface Area, Height, Glazing) and verify that the 3D house model and thermal loads update dynamically.
