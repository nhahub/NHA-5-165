"""Pydantic schemas for request validation and response models."""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class BuildingFeatures(BaseModel):
    """Input features for building energy prediction."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Relative_Compactness": 0.98,
                "Surface_Area": 514.5,
                "Wall_Area": 294.0,
                "Roof_Area": 110.25,
                "Overall_Height": 7.0,
                "Orientation": 2,
                "Glazing_Area": 0.10,
                "Glazing_Area_Distribution": 1
            }
        }
    )

    Relative_Compactness: float = Field(..., ge=0.6, le=1.0, description="Relative compactness ratio (0.62 - 0.98)")
    Surface_Area: float = Field(..., ge=500.0, le=850.0, description="Total building surface area in m² (514.5 - 808.5)")
    Wall_Area: float = Field(..., ge=240.0, le=420.0, description="Total wall area in m² (245.0 - 416.5)")
    Roof_Area: float = Field(..., ge=100.0, le=230.0, description="Roof area in m² (110.25 - 220.5)")
    Overall_Height: float = Field(..., ge=3.0, le=8.0, description="Building height in meters (typically 3.5 or 7.0)")
    Orientation: int = Field(..., ge=2, le=5, description="Building orientation: 2=North, 3=East, 4=South, 5=West")
    Glazing_Area: float = Field(..., ge=0.0, le=0.5, description="Glazing area ratio: 0.0, 0.1, 0.25, or 0.40")
    Glazing_Area_Distribution: int = Field(..., ge=0, le=5, description="Glazing distribution mode: 0 (None) to 5")


class PredictionRequest(BaseModel):
    """Prediction request allowing feature input and algorithm selection."""
    features: BuildingFeatures
    algorithm: Optional[str] = Field("Random Forest", description="Algorithm name: 'Linear Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting', or 'XGBoost'")


class LoadPrediction(BaseModel):
    """Predicted thermal loads."""
    heating_load_kwh: float = Field(..., description="Heating load prediction in kWh/m²")
    cooling_load_kwh: float = Field(..., description="Cooling load prediction in kWh/m²")
    total_load_kwh: float = Field(..., description="Sum of heating and cooling loads")
    algorithm_used: str = Field(..., description="Model algorithm used for prediction")
    inference_source: str = Field("local-bundle", description="Inference execution provider: 'aws-sagemaker' or 'local-bundle'")


class PredictionResponse(BaseModel):
    """Full API prediction response."""
    success: bool = True
    prediction: LoadPrediction
    input_received: Dict[str, Any]
    engineered_features: Dict[str, float]


class OptimizationRequest(BaseModel):
    """Request to optimize glazing or orientation for minimum load."""
    base_features: BuildingFeatures
    optimize_param: str = Field("Glazing_Area", description="Parameter to optimize: 'Glazing_Area', 'Orientation', or 'Relative_Compactness'")


class OptimizationResult(BaseModel):
    """Individual candidate evaluated during optimization."""
    param_value: Any
    heating_load: float
    cooling_load: float
    total_load: float


class OptimizationResponse(BaseModel):
    """Response containing optimization results and recommended design."""
    parameter_optimized: str
    optimal_value: Any
    lowest_total_load: float
    evaluations: List[OptimizationResult]
    inference_source: str = Field("local-bundle", description="Inference execution provider: 'aws-sagemaker' or 'local-bundle'")


class ModelInfo(BaseModel):
    """Metadata about available models."""
    name: str
    target: str
    r2_score: float
    mae: float
    rmse: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    version: str
    models_loaded: bool
    inference_backend: str = "local-bundle"
    sagemaker_endpoint: Optional[str] = None
    algorithms_available: List[str]
