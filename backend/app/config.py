"""Backend configuration and environment settings."""

import os
import sys

# Root directory of the repository
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


def _load_dotenv():
    """Load root .env file into os.environ if present for local development."""
    env_path = os.path.join(BASE_DIR, ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

# Model bundle path (local fallback / offline bundle)
MODEL_BUNDLE_PATH = os.path.join(BASE_DIR, "artifacts", "production", "model_bundle.joblib")
CANDIDATES_DIR = os.path.join(BASE_DIR, "artifacts", "candidates")

# AWS SageMaker Serverless Endpoint settings
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
SAGEMAKER_ENDPOINT_NAME = os.getenv("SAGEMAKER_ENDPOINT_NAME", "thermox-ai-endpoint")
USE_AWS_SAGEMAKER = os.getenv("USE_AWS_SAGEMAKER", "false").lower() in ("true", "1", "yes", "on")

# Server settings
APP_NAME = "Building Energy Efficiency Prediction API"
APP_VERSION = "2.1.0"
DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")

# CORS allowed origins (configured for Vercel frontend and local development)
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://thermoxai.vercel.app",
    "https://*.vercel.app",
    "*",
]
