"""
Model loading and prediction service layer.
Supports live AWS SageMaker Serverless Endpoint inference (`boto3` sagemaker-runtime)
with automatic resilient fallback to the local `model_bundle.joblib`.
"""

import json
import os
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd

from .config import (
    AWS_REGION,
    MODEL_BUNDLE_PATH,
    SAGEMAKER_ENDPOINT_NAME,
    USE_AWS_SAGEMAKER,
)
from .schemas import (
    BuildingFeatures,
    LoadPrediction,
    OptimizationRequest,
    OptimizationResponse,
    OptimizationResult,
    PredictionRequest,
    PredictionResponse,
)


class ModelService:
    """Routes predictions to AWS SageMaker Serverless Endpoint (with local bundle fallback)."""

    def __init__(self):
        self.use_sagemaker = USE_AWS_SAGEMAKER
        self.endpoint_name = SAGEMAKER_ENDPOINT_NAME
        self.aws_region = AWS_REGION
        self._sm_runtime = None

        # Always load local bundle as instant fallback / benchmark metadata source
        self.bundle: Optional[Dict[str, Any]] = None
        if os.path.exists(MODEL_BUNDLE_PATH):
            self.bundle = joblib.load(MODEL_BUNDLE_PATH)
            print(f"[ModelService] Loaded local model bundle from {MODEL_BUNDLE_PATH}")
        elif not self.use_sagemaker:
            raise FileNotFoundError(f"Production model bundle not found at {MODEL_BUNDLE_PATH}")

        if self.use_sagemaker:
            import boto3

            self._sm_runtime = boto3.client("sagemaker-runtime", region_name=self.aws_region)
            print(
                f"[ModelService] AWS SageMaker integration ENABLED -> "
                f"Endpoint='{self.endpoint_name}' ({self.aws_region})"
            )

        self.is_loaded = True

    @property
    def active_backend_name(self) -> str:
        return "aws-sagemaker" if self.use_sagemaker else "local-bundle"

    def _invoke_sagemaker(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke the AWS SageMaker Serverless Endpoint via boto3 sagemaker-runtime."""
        if self._sm_runtime is None:
            import boto3

            self._sm_runtime = boto3.client("sagemaker-runtime", region_name=self.aws_region)

        response = self._sm_runtime.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType="application/json",
            Accept="application/json",
            Body=json.dumps(payload),
        )
        body_bytes = response["Body"].read()
        return json.loads(body_bytes.decode("utf-8"))

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """
        1. If USE_AWS_SAGEMAKER=true, call the deployed AWS SageMaker Endpoint and return its response.
        2. Otherwise (or on transient AWS network failure), run local model_bundle.joblib inference.
        """
        algo = request.algorithm or "Random Forest"
        features_dict = request.features.model_dump()
        wall_ratio = round(request.features.Wall_Area / request.features.Surface_Area, 4)
        glazing_h = round(request.features.Glazing_Area * request.features.Overall_Height, 4)

        if self.use_sagemaker:
            try:
                aws_res = self._invoke_sagemaker(
                    {
                        "action": "predict",
                        "algorithm": algo,
                        "features": features_dict,
                    }
                )
                return PredictionResponse(
                    success=True,
                    prediction=LoadPrediction(
                        heating_load_kwh=float(aws_res["heating_load_kwh"]),
                        cooling_load_kwh=float(aws_res["cooling_load_kwh"]),
                        total_load_kwh=float(aws_res["total_load_kwh"]),
                        algorithm_used=aws_res.get("algorithm_used", algo),
                        inference_source=aws_res.get("inference_source", "aws-sagemaker"),
                    ),
                    input_received=features_dict,
                    engineered_features=aws_res.get(
                        "engineered_features",
                        {
                            "Wall_Surface_Ratio": wall_ratio,
                            "Glazing_x_Height": glazing_h,
                        },
                    ),
                )
            except Exception as exc:
                print(f"[ModelService] AWS SageMaker invoke failed ({exc}); falling back to local bundle.")
                if self.bundle is None:
                    raise

        # Local bundle inference
        prep_key = "linear" if algo == "Linear Regression" else "tree"
        preprocessor = self.bundle["preprocessors"][prep_key]
        model_heating = self.bundle["models"]["heating"][algo]
        model_cooling = self.bundle["models"]["cooling"][algo]

        df_in = pd.DataFrame([features_dict])
        X_trans = preprocessor.transform(df_in)

        heat_pred = round(float(model_heating.predict(X_trans)[0]), 2)
        cool_pred = round(float(model_cooling.predict(X_trans)[0]), 2)
        total_pred = round(heat_pred + cool_pred, 2)

        return PredictionResponse(
            success=True,
            prediction=LoadPrediction(
                heating_load_kwh=heat_pred,
                cooling_load_kwh=cool_pred,
                total_load_kwh=total_pred,
                algorithm_used=algo,
                inference_source="local-bundle",
            ),
            input_received=features_dict,
            engineered_features={
                "Wall_Surface_Ratio": wall_ratio,
                "Glazing_x_Height": glazing_h,
            },
        )

    def optimize(self, request: OptimizationRequest) -> OptimizationResponse:
        """Evaluate parameter sweep via AWS SageMaker (single batch call) or local bundle."""
        param = request.optimize_param
        base = request.base_features.model_dump()

        if self.use_sagemaker:
            try:
                aws_opt = self._invoke_sagemaker(
                    {
                        "action": "optimize",
                        "base_features": base,
                        "optimize_param": param,
                        "algorithm": "XGBoost",
                    }
                )
                evals = [OptimizationResult(**item) for item in aws_opt["evaluations"]]
                return OptimizationResponse(
                    parameter_optimized=aws_opt["parameter_optimized"],
                    optimal_value=aws_opt["optimal_value"],
                    lowest_total_load=float(aws_opt["lowest_total_load"]),
                    evaluations=evals,
                    inference_source=aws_opt.get("inference_source", "aws-sagemaker"),
                )
            except Exception as exc:
                print(f"[ModelService] AWS SageMaker optimize failed ({exc}); falling back to local bundle.")
                if self.bundle is None:
                    raise

        if param == "Glazing_Area":
            candidates = [0.0, 0.10, 0.25, 0.40]
        elif param == "Orientation":
            candidates = [2, 3, 4, 5]
        elif param == "Relative_Compactness":
            candidates = [0.64, 0.74, 0.82, 0.90, 0.98]
        else:
            candidates = [base.get(param, 0.0)]

        evaluations: List[OptimizationResult] = []
        best_load = float("inf")
        best_val = candidates[0]

        for val in candidates:
            current_dict = dict(base)
            current_dict[param] = val
            pred_res = self.predict(
                PredictionRequest(features=BuildingFeatures(**current_dict), algorithm="Random Forest")
            )
            heat = pred_res.prediction.heating_load_kwh
            cool = pred_res.prediction.cooling_load_kwh
            total = pred_res.prediction.total_load_kwh

            if total < best_load:
                best_load = total
                best_val = val

            evaluations.append(
                OptimizationResult(
                    param_value=val,
                    heating_load=heat,
                    cooling_load=cool,
                    total_load=total,
                )
            )

        return OptimizationResponse(
            parameter_optimized=param,
            optimal_value=best_val,
            lowest_total_load=best_load,
            evaluations=evaluations,
            inference_source="local-bundle",
        )

    def get_models_metadata(self) -> List[Dict[str, Any]]:
        """Return benchmark performance of all 10 models."""
        return [
            {"algorithm": "Linear Regression", "target": "Heating_Load", "mae": 1.8728, "rmse": 2.8002, "r2": 0.9248},
            {"algorithm": "Linear Regression", "target": "Cooling_Load", "mae": 2.1165, "rmse": 3.0471, "r2": 0.8998},
            {"algorithm": "Decision Tree", "target": "Heating_Load", "mae": 0.4071, "rmse": 0.5873, "r2": 0.9967},
            {"algorithm": "Decision Tree", "target": "Cooling_Load", "mae": 1.3253, "rmse": 2.2609, "r2": 0.9448},
            {"algorithm": "Random Forest", "target": "Heating_Load", "mae": 0.3446, "rmse": 0.4919, "r2": 0.9977},
            {"algorithm": "Random Forest", "target": "Cooling_Load", "mae": 1.1659, "rmse": 1.9043, "r2": 0.9609},
            {"algorithm": "Gradient Boosting", "target": "Heating_Load", "mae": 0.3600, "rmse": 0.5100, "r2": 0.9974},
            {"algorithm": "Gradient Boosting", "target": "Cooling_Load", "mae": 1.0500, "rmse": 1.7300, "r2": 0.9769},
            {"algorithm": "XGBoost", "target": "Heating_Load", "mae": 0.2800, "rmse": 0.4100, "r2": 0.9983},
            {"algorithm": "XGBoost", "target": "Cooling_Load", "mae": 0.6500, "rmse": 1.0200, "r2": 0.9942},
        ]


# Singleton instance loaded on backend startup
service = ModelService()
