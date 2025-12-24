"""
Pydantic Models for API Request/Response Validation

These define the "contract" between Frontend (Person 3) and Backend (Person 2).
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from enum import Enum
from typing import Literal


class CensorMethodEnum(str, Enum):
    """Available censorship methods."""
    BLUR = "blur"
    PIXELATE = "pixelate"
    BLACK_BAR = "black_bar"
    EMOJI = "emoji"


class DistanceMetricEnum(str, Enum):
    """Available distance metrics for face comparison."""
    EUCLIDEAN = "euclidean"  # L2 norm - standard for face recognition
    MANHATTAN = "manhattan"  # L1 norm - robust to outliers, faster
    COSINE = "cosine"        # Angular similarity - good for normalized vectors


# ============================================================================
# Enrollment Endpoint Models
# ============================================================================

class EnrollRequest(BaseModel):
    """Request body for user enrollment (opt-out)."""
    alias: Optional[str] = Field(
        None,
        description="Optional friendly name for the user",
        example="John Doe"
    )


class EnrollResponse(BaseModel):
    """Response from enrollment endpoint."""
    status: str = Field(..., example="success")
    message: str = Field(..., example="User enrolled successfully")
    user_id: str = Field(..., description="Unique identifier for the enrolled user")
    created_at: datetime
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "User enrolled successfully. Your face encoding has been stored. Original photo has been deleted.",
                "user_id": "usr_abc123def456",
                "created_at": "2024-01-15T10:30:00Z"
            }
        }


# ============================================================================
# Processing Endpoint Models
# ============================================================================

class FaceCoordinates(BaseModel):
    """Coordinates of a detected face."""
    x: int = Field(..., description="Top-left X coordinate")
    y: int = Field(..., description="Top-left Y coordinate")
    width: int = Field(..., description="Bounding box width")
    height: int = Field(..., description="Bounding box height")
    confidence: float = Field(1.0, description="Detection confidence (0-1)")


class ProcessedFace(BaseModel):
    """Information about a processed face."""
    coordinates: FaceCoordinates
    matched_user_id: Optional[str] = Field(
        None,
        description="ID of matched opt-out user, if any"
    )
    was_blurred: bool = Field(..., description="Whether this face was censored")
    match_distance: Optional[float] = Field(
        None,
        description="Distance score (lower = more similar)"
    )


class ProcessRequest(BaseModel):
    """Request parameters for image processing."""
    censor_method: CensorMethodEnum = Field(
        CensorMethodEnum.BLUR,
        description="Method to use for censorship"
    )
    threshold: float = Field(
        0.6,
        description="Match threshold (lower = stricter matching)",
        ge=0.0,
        le=1.0
    )
    return_coordinates: bool = Field(
        False,
        description="Include face coordinates in response"
    )


class ProcessResponse(BaseModel):
    """Response from image processing endpoint."""
    status: str = Field(..., example="success")
    faces_detected: int = Field(..., description="Total faces found in image")
    faces_redacted: int = Field(..., description="Number of faces that were censored")
    matched_users: List[str] = Field(
        default_factory=list,
        description="IDs of matched opt-out users"
    )
    processed_image: str = Field(
        ...,
        description="Base64-encoded processed image"
    )
    image_format: str = Field(..., example="jpeg")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    faces: Optional[List[ProcessedFace]] = Field(
        None,
        description="Details about each detected face (if return_coordinates=true)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "faces_detected": 5,
                "faces_redacted": 2,
                "matched_users": ["usr_abc123", "usr_def456"],
                "processed_image": "data:image/jpeg;base64,...",
                "image_format": "jpeg",
                "processing_time_ms": 234.5
            }
        }


# ============================================================================
# User Management Models
# ============================================================================

class UserInfo(BaseModel):
    """Information about an enrolled user."""
    user_id: str
    alias: Optional[str]
    created_at: datetime
    is_active: bool


class UserListResponse(BaseModel):
    """Response for listing enrolled users."""
    total_users: int
    users: List[UserInfo]


class DeleteUserResponse(BaseModel):
    """Response for user deletion."""
    status: str
    message: str
    user_id: str


# ============================================================================
# Health Check & Stats Models
# ============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., example="healthy")
    version: str = Field(..., example="1.0.0")
    database_connected: bool
    face_detector_loaded: bool
    total_enrolled_users: int


class StatsResponse(BaseModel):
    """System statistics response."""
    total_enrolled_users: int
    total_images_processed: int
    total_faces_detected: int
    total_faces_redacted: int
    uptime_seconds: float


# ============================================================================
# Error Models
# ============================================================================

class ErrorResponse(BaseModel):
    """Standard error response."""
    status: Literal["error"] = "error"
    error_code: str
    message: str
    details: Optional[dict] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "error",
                "error_code": "NO_FACE_DETECTED",
                "message": "No face was detected in the uploaded image",
                "details": {"image_size": "1920x1080"}
            }
        }


# Error codes
class ErrorCode:
    NO_FACE_DETECTED = "NO_FACE_DETECTED"
    MULTIPLE_FACES = "MULTIPLE_FACES"
    INVALID_IMAGE = "INVALID_IMAGE"
    USER_NOT_FOUND = "USER_NOT_FOUND"
    USER_ALREADY_EXISTS = "USER_ALREADY_EXISTS"
    PROCESSING_ERROR = "PROCESSING_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"