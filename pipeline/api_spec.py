"""
API Documentation and OpenAPI Specification.

Provides FastAPI-based REST API for inference service with automatic
OpenAPI/Swagger documentation.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum

# ==========================================
# REQUEST/RESPONSE MODELS
# ==========================================

class PredictionRequest(BaseModel):
    """Request model for batch prediction."""
    
    samples: List[Dict[str, float]] = Field(
        ...,
        description="List of samples with 590 features each",
        min_items=1,
        max_items=1000
    )
    return_probabilities: bool = Field(
        default=True,
        description="Whether to return class probabilities"
    )
    return_explanations: bool = Field(
        default=False,
        description="Whether to return SHAP explanations (slower)"
    )
    
    @validator('samples')
    def validate_features(cls, v):
        """Validate feature count."""
        for sample in v:
            if len(sample) != 590:
                raise ValueError(f"Each sample must have exactly 590 features, got {len(sample)}")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "samples": [
                    {f"feature_{i}": 0.5 for i in range(590)}
                ],
                "return_probabilities": True,
                "return_explanations": False
            }
        }


class PredictionResult(BaseModel):
    """Single prediction result."""
    
    prediction: int = Field(..., description="Predicted class (0=Pass, 1=Fail)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    probabilities: Optional[Dict[str, float]] = Field(
        None,
        description="Class probabilities"
    )
    explanation: Optional[Dict[str, Any]] = Field(
        None,
        description="SHAP explanation (if requested)"
    )


class PredictionResponse(BaseModel):
    """Response model for batch prediction."""
    
    predictions: List[PredictionResult] = Field(..., description="Prediction results")
    model_version: str = Field(..., description="Model version used")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {
                        "prediction": 0,
                        "confidence": 0.85,
                        "probabilities": {"pass": 0.85, "fail": 0.15}
                    }
                ],
                "model_version": "v1.2.0",
                "timestamp": "2025-12-16T19:00:00Z",
                "processing_time_ms": 45.2
            }
        }


class ModelInfo(BaseModel):
    """Model information response."""
    
    model_id: str
    model_name: str
    model_version: str
    model_type: str
    accuracy: float
    f1_score: float
    deployed_at: datetime
    is_active: bool


class HealthStatus(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Overall status (healthy/unhealthy)")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    services: Dict[str, bool] = Field(..., description="Individual service health")
    version: str = Field(..., description="API version")


class MetricsResponse(BaseModel):
    """Performance metrics response."""
    
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    total_predictions: int
    window_start: datetime
    window_end: datetime


# ==========================================
# OPENAPI SPECIFICATION
# ==========================================

OPENAPI_SPEC = {
    "openapi": "3.0.0",
    "info": {
        "title": "SECOM ML Pipeline API",
        "description": """
# SECOM Manufacturing Quality Prediction API

Real-time ML inference API for semiconductor manufacturing quality control.

## Features

- **Batch Prediction**: Predict quality outcomes for multiple samples
- **Model Information**: Get details about the active model
- **Performance Metrics**: Track model performance over time
- **Health Checks**: Monitor service health
- **SHAP Explanations**: Optional model interpretability

## Authentication

Currently no authentication required (development mode).
Production deployment should use API keys or JWT tokens.

## Rate Limits

- **Prediction**: 100 requests/minute
- **Other endpoints**: 1000 requests/minute

## Support

- **Documentation**: https://github.com/yourusername/distributed_ml_pipeline
- **Issues**: https://github.com/yourusername/distributed_ml_pipeline/issues
        """,
        "version": "1.0.0",
        "contact": {
            "name": "ML Engineering Team",
            "email": "ml-team@example.com"
        },
        "license": {
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT"
        }
    },
    "servers": [
        {
            "url": "http://localhost:8002",
            "description": "Development server"
        },
        {
            "url": "https://api.secom-ml.example.com",
            "description": "Production server"
        }
    ],
    "paths": {
        "/predict": {
            "post": {
                "summary": "Make predictions",
                "description": "Predict quality outcomes for a batch of samples",
                "operationId": "predict",
                "tags": ["Inference"],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/PredictionRequest"}
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Successful prediction",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/PredictionResponse"}
                            }
                        }
                    },
                    "400": {
                        "description": "Invalid request"
                    },
                    "500": {
                        "description": "Internal server error"
                    }
                }
            }
        },
        "/model/info": {
            "get": {
                "summary": "Get model information",
                "description": "Retrieve information about the active model",
                "operationId": "getModelInfo",
                "tags": ["Model"],
                "responses": {
                    "200": {
                        "description": "Model information",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ModelInfo"}
                            }
                        }
                    }
                }
            }
        },
        "/metrics": {
            "get": {
                "summary": "Get performance metrics",
                "description": "Retrieve model performance metrics for a time window",
                "operationId": "getMetrics",
                "tags": ["Monitoring"],
                "parameters": [
                    {
                        "name": "hours",
                        "in": "query",
                        "description": "Time window in hours",
                        "schema": {"type": "integer", "default": 24}
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Performance metrics",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/MetricsResponse"}
                            }
                        }
                    }
                }
            }
        },
        "/health": {
            "get": {
                "summary": "Health check",
                "description": "Check service health status",
                "operationId": "healthCheck",
                "tags": ["Monitoring"],
                "responses": {
                    "200": {
                        "description": "Service is healthy",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/HealthStatus"}
                            }
                        }
                    },
                    "503": {
                        "description": "Service is unhealthy"
                    }
                }
            }
        }
    },
    "components": {
        "schemas": {
            "PredictionRequest": PredictionRequest.schema(),
            "PredictionResponse": PredictionResponse.schema(),
            "PredictionResult": PredictionResult.schema(),
            "ModelInfo": ModelInfo.schema(),
            "HealthStatus": HealthStatus.schema(),
            "MetricsResponse": MetricsResponse.schema()
        }
    },
    "tags": [
        {
            "name": "Inference",
            "description": "Prediction endpoints"
        },
        {
            "name": "Model",
            "description": "Model management endpoints"
        },
        {
            "name": "Monitoring",
            "description": "Health and metrics endpoints"
        }
    ]
}


def save_openapi_spec(output_path: str = "docs/openapi.json"):
    """Save OpenAPI specification to JSON file."""
    import json
    from pathlib import Path
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(OPENAPI_SPEC, f, indent=2, default=str)
    
    print(f"OpenAPI specification saved to {output_path}")


if __name__ == "__main__":
    save_openapi_spec()
