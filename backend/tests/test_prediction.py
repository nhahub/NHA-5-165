"""Tests for prediction service and model logic."""

import pytest
from backend.app.schemas import BuildingFeatures, OptimizationRequest, PredictionRequest
from backend.app.service import service


def test_prediction_service_linear():
    req = PredictionRequest(
        features=BuildingFeatures(
            Relative_Compactness=0.98,
            Surface_Area=514.5,
            Wall_Area=294.0,
            Roof_Area=110.25,
            Overall_Height=7.0,
            Orientation=2,
            Glazing_Area=0.1,
            Glazing_Area_Distribution=1
        ),
        algorithm="Linear Regression"
    )
    res = service.predict(req)
    assert res.success is True
    assert 5.0 <= res.prediction.heating_load_kwh <= 50.0
    assert 10.0 <= res.prediction.cooling_load_kwh <= 55.0


def test_prediction_service_random_forest():
    req = PredictionRequest(
        features=BuildingFeatures(
            Relative_Compactness=0.98,
            Surface_Area=514.5,
            Wall_Area=294.0,
            Roof_Area=110.25,
            Overall_Height=7.0,
            Orientation=2,
            Glazing_Area=0.0,
            Glazing_Area_Distribution=0
        ),
        algorithm="Random Forest"
    )
    res = service.predict(req)
    assert res.success is True
    # Ground truth for row 0 in energy efficiency dataset is heating=15.55, cooling=21.33
    assert abs(res.prediction.heating_load_kwh - 15.55) < 3.0
    assert abs(res.prediction.cooling_load_kwh - 21.33) < 4.0


def test_engineered_features_calculation():
    req = PredictionRequest(
        features=BuildingFeatures(
            Relative_Compactness=0.8,
            Surface_Area=600.0,
            Wall_Area=300.0,
            Roof_Area=150.0,
            Overall_Height=7.0,
            Orientation=3,
            Glazing_Area=0.25,
            Glazing_Area_Distribution=2
        )
    )
    res = service.predict(req)
    assert res.engineered_features["Wall_Surface_Ratio"] == 0.5
    assert res.engineered_features["Glazing_x_Height"] == 1.75


def test_sagemaker_endpoint_contract(monkeypatch):
    """Verify FastAPI service calls AWS SageMaker contract and returns aws-sagemaker source."""
    def fake_invoke(payload):
        if payload.get("action") == "optimize":
            return {
                "parameter_optimized": "Glazing_Area",
                "optimal_value": 0.0,
                "lowest_total_load": 36.88,
                "evaluations": [
                    {"param_value": 0.0, "heating_load": 15.55, "cooling_load": 21.33, "total_load": 36.88},
                    {"param_value": 0.1, "heating_load": 24.45, "cooling_load": 26.23, "total_load": 50.68},
                ],
                "inference_source": "aws-sagemaker",
            }
        return {
            "heating_load_kwh": 24.45,
            "cooling_load_kwh": 26.23,
            "total_load_kwh": 50.68,
            "algorithm_used": payload["algorithm"],
            "inference_source": "aws-sagemaker",
            "engineered_features": {"Wall_Surface_Ratio": 0.5714, "Glazing_x_Height": 0.7},
        }

    monkeypatch.setattr(service, "use_sagemaker", True)
    monkeypatch.setattr(service, "_invoke_sagemaker", fake_invoke)

    features = BuildingFeatures(
        Relative_Compactness=0.98,
        Surface_Area=514.5,
        Wall_Area=294.0,
        Roof_Area=110.25,
        Overall_Height=7.0,
        Orientation=2,
        Glazing_Area=0.10,
        Glazing_Area_Distribution=1,
    )

    pred_res = service.predict(PredictionRequest(features=features, algorithm="XGBoost"))
    assert pred_res.prediction.inference_source == "aws-sagemaker"
    assert pred_res.prediction.total_load_kwh == 50.68

    opt_res = service.optimize(OptimizationRequest(base_features=features, optimize_param="Glazing_Area"))
    assert opt_res.inference_source == "aws-sagemaker"
    assert opt_res.optimal_value == 0.0
