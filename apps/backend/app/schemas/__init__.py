# Schemas module exports
from app.schemas.common import ErrorDetail, ErrorResponse, PaginationParams, ValidatedId
from app.schemas.auth import CurrentUser, TokenPayloadSchema
from app.schemas.inference import InferRequest, InferResponse, ModelInfo, ModelsListResponse
from app.schemas.analytics import (
    AnalyticsQueryParams,
    DataPoint,
    PredictionPoint,
    PredictionsResponse,
    ReadingsResponse,
    Resolution,
)

__all__ = [
    # Common
    "ErrorDetail",
    "ErrorResponse",
    "PaginationParams",
    "ValidatedId",
    # Auth
    "CurrentUser",
    "TokenPayloadSchema",
    # Inference
    "InferRequest",
    "InferResponse",
    "ModelInfo",
    "ModelsListResponse",
    # Analytics
    "AnalyticsQueryParams",
    "DataPoint",
    "PredictionPoint",
    "PredictionsResponse",
    "ReadingsResponse",
    "Resolution",
]
