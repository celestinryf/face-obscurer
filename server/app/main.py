"""
Privacy Engine - FastAPI Backend (Local/dlib Version)
=====================================================

"Right to be Forgotten" face recognition and censorship system.

Uses dlib/face_recognition - same as Person 1's camera scanner.
This ensures face encodings are compatible across both systems.

Endpoints:
- POST /enroll      - Register a user for opt-out
- POST /process     - Process an image and blur matched faces
- GET  /users       - List enrolled users
- DELETE /users/{id} - Remove a user from opt-out list
- GET  /health      - Health check
- POST /sync-known-faces - Sync from known_faces directory
"""

import os
import io
import uuid
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Form, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from sqlalchemy.orm import Session
import cv2 
from fastapi.responses import StreamingResponse

# Local imports
from app.database import (
    get_db, init_db, add_optout_user, get_all_active_encodings,
    get_user_by_id, deactivate_user, hard_delete_user, get_user_count,
    OptOutUser
)
from app.face_utils import (
    FaceComparator, DetectedFace, MatchResult, DistanceMetric,
    DEFAULT_DETECTOR, load_known_faces_from_directory
)
from app.image_processor import (
    ImageProcessor, CensorMethod, image_to_base64, calculate_image_hash
)
from app.models import (
    EnrollResponse, ProcessResponse, ProcessedFace,
    FaceCoordinates, UserInfo, UserListResponse, DeleteUserResponse,
    HealthResponse, ErrorResponse, ErrorCode, CensorMethodEnum, DistanceMetricEnum
)


# ============================================================================
# Application Setup
# ============================================================================

START_TIME = time.time()

# Initialize face detector (dlib-based, same as Person 1's camera scanner)
face_detector = DEFAULT_DETECTOR()
image_processor = ImageProcessor()

# Path to known_faces directory (shared with Person 1's camera scanner)
KNOWN_FACES_DIR = Path(__file__).parent.parent / "known_faces"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    print("🚀 Privacy Engine starting up...")
    init_db()
    print("✅ Database initialized")
    print(f"✅ Face detector: {type(face_detector).__name__}")
    print(f"📁 Known faces directory: {KNOWN_FACES_DIR}")
    yield
    print("👋 Privacy Engine shutting down...")


app = FastAPI(
    title="Privacy Engine API",
    description="""
    ## Right to be Forgotten - Face Recognition Privacy System
    
    Uses **dlib** for face recognition - compatible with Person 1's camera scanner.
    
    ### How it works:
    1. **Enroll** users who want to opt-out (stores 128-dim face encoding)
    2. **Process** group photos to automatically blur enrolled users
    
    ### Encoding Compatibility:
    - Uses same 128-dimensional dlib embeddings as camera scanner
    - Faces enrolled here will be recognized by camera scanner and vice versa
    
    ### Distance Metrics:
    - `euclidean` (default): Standard dlib distance, threshold ~0.6
    - `manhattan`: More robust to outliers, threshold ~6.0
    - `cosine`: Angular similarity, threshold ~0.4
    """,
    version="2.0.0",
    lifespan=lifespan
)

# CORS - allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Health & Info Endpoints
# ============================================================================

@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Privacy Engine API",
        "version": "2.0.0",
        "description": "Right to be Forgotten - Face Recognition Privacy System",
        "face_detector": type(face_detector).__name__,
        "encoding_dimensions": 128,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health_check(db: Session = Depends(get_db)):
    """Check system health."""
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        database_connected=True,
        face_detector_loaded=face_detector is not None,
        total_enrolled_users=get_user_count(db)
    )


# ============================================================================
# Enrollment Endpoints
# ============================================================================

@app.post(
    "/enroll",
    response_model=EnrollResponse,
    responses={400: {"model": ErrorResponse}},
    tags=["Enrollment"]
)
async def enroll_user(
    photo: UploadFile = File(..., description="Photo containing the user's face"),
    alias: Optional[str] = Form(None, description="Optional friendly name"),
    db: Session = Depends(get_db)
):
    """
    Enroll a user for privacy protection (opt-out).
    
    **Process:**
    1. Extract 128-dim face encoding using dlib
    2. Store ONLY the encoding in encrypted database
    3. Original photo is never saved
    
    **Compatibility:**
    The encoding is compatible with Person 1's camera scanner.
    """
    try:
        image_bytes = await photo.read()
        
        # Generate 128-dim dlib encoding
        encoding = face_detector.generate_encoding(image_bytes)
        
        if encoding is None:
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": ErrorCode.NO_FACE_DETECTED,
                    "message": "No face detected. Please upload a clear photo of your face."
                }
            )
        
        # Generate unique ID
        user_id = f"usr_{uuid.uuid4().hex[:12]}"
        
        # Store in database
        user = add_optout_user(db, user_id, encoding, alias)
        
        return EnrollResponse(
            status="success",
            message=f"User enrolled successfully with 128-dim face encoding. Original photo deleted.",
            user_id=user_id,
            created_at=user.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error_code": ErrorCode.PROCESSING_ERROR, "message": str(e)}
        )

# Initialize camera globally (or inside a dependency)
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Here you would insert your blurring/detection logic
            # For now, we'll just stream the raw frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/sync-known-faces", tags=["Enrollment"])
async def sync_known_faces(db: Session = Depends(get_db)):
    """
    Sync faces from the known_faces/ directory into the database.
    
    This allows sharing faces between the camera scanner and the API.
    Faces in known_faces/ will be enrolled with their filename as the alias.
    """
    if not KNOWN_FACES_DIR.exists():
        KNOWN_FACES_DIR.mkdir(parents=True)
        return {
            "status": "warning",
            "message": f"Created {KNOWN_FACES_DIR}. Add photos and run again.",
            "synced": 0
        }
    
    encodings, names = load_known_faces_from_directory(str(KNOWN_FACES_DIR))
    
    synced = 0
    skipped = 0
    
    for encoding, name in zip(encodings, names):
        # Check if already exists by alias
        existing = db.query(OptOutUser).filter(
            OptOutUser.alias == name,
            OptOutUser.is_active == True
        ).first()
        
        if existing:
            skipped += 1
            continue
        
        user_id = f"usr_{uuid.uuid4().hex[:12]}"
        add_optout_user(db, user_id, encoding, alias=name)
        synced += 1
    
    return {
        "status": "success",
        "message": f"Synced {synced} new faces, skipped {skipped} existing",
        "synced": synced,
        "skipped": skipped,
        "total_enrolled": get_user_count(db)
    }


# ============================================================================
# Image Processing Endpoints
# ============================================================================

@app.post(
    "/process",
    response_model=ProcessResponse,
    responses={400: {"model": ErrorResponse}},
    tags=["Processing"]
)
async def process_image(
    image: UploadFile = File(..., description="Group photo to process"),
    censor_method: CensorMethodEnum = Form(
        CensorMethodEnum.BLUR,
        description="Censorship method"
    ),
    distance_metric: DistanceMetricEnum = Form(
        DistanceMetricEnum.EUCLIDEAN,  # dlib default
        description="Distance metric for face comparison"
    ),
    threshold: Optional[float] = Form(
        None,
        description="Match threshold (uses metric default if not specified)"
    ),
    return_coordinates: bool = Form(
        False,
        description="Include face coordinates in response"
    ),
    db: Session = Depends(get_db)
):
    """
    Process an image and blur faces of enrolled (opt-out) users.
    
    **Distance Metrics:**
    - `euclidean`: Default dlib distance (threshold ~0.6)
    - `manhattan`: Sum of absolute differences (threshold ~6.0)
    - `cosine`: Angular similarity (threshold ~0.4)
    
    **Compatibility:**
    Uses same 128-dim encodings as Person 1's camera scanner.
    """
    start_time = time.time()
    
    try:
        image_bytes = await image.read()
        
        # Detect faces using dlib
        detected_faces = face_detector.detect_faces(image_bytes)
        
        if not detected_faces:
            return ProcessResponse(
                status="success",
                faces_detected=0,
                faces_redacted=0,
                matched_users=[],
                processed_image=image_to_base64(image_bytes),
                image_format="jpeg",
                processing_time_ms=(time.time() - start_time) * 1000
            )
        
        # Get enrolled face encodings
        optout_encodings = get_all_active_encodings(db)
        
        # Compare faces
        comparator = FaceComparator(
            threshold=threshold,
            metric=DistanceMetric(distance_metric.value)
        )
        match_results = comparator.find_matches(detected_faces, optout_encodings)
        
        # Collect faces to blur
        faces_to_blur = []
        matched_users = []
        processed_faces = []
        
        for result in match_results:
            face_coords = FaceCoordinates(
                x=result.face.x,
                y=result.face.y,
                width=result.face.width,
                height=result.face.height,
                confidence=result.face.confidence
            )
            
            if result.is_match:
                faces_to_blur.append({
                    'x': result.face.x,
                    'y': result.face.y,
                    'width': result.face.width,
                    'height': result.face.height
                })
                matched_users.append(result.matched_user_id)
            
            if return_coordinates:
                processed_faces.append(ProcessedFace(
                    coordinates=face_coords,
                    matched_user_id=result.matched_user_id,
                    was_blurred=result.is_match,
                    match_distance=result.distance if result.is_match else None
                ))
        
        # Apply censorship
        processed_bytes, img_format = image_processor.process_image(
            image_bytes,
            faces_to_blur,
            method=CensorMethod(censor_method.value)
        )
        
        return ProcessResponse(
            status="success",
            faces_detected=len(detected_faces),
            faces_redacted=len(faces_to_blur),
            matched_users=matched_users,
            processed_image=image_to_base64(processed_bytes, img_format),
            image_format=img_format,
            processing_time_ms=(time.time() - start_time) * 1000,
            faces=processed_faces if return_coordinates else None
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error_code": ErrorCode.PROCESSING_ERROR, "message": str(e)}
        )


@app.post("/process/raw", tags=["Processing"])
async def process_image_raw(
    image: UploadFile = File(...),
    censor_method: CensorMethodEnum = Form(CensorMethodEnum.BLUR),
    distance_metric: DistanceMetricEnum = Form(DistanceMetricEnum.EUCLIDEAN),
    threshold: Optional[float] = Form(None),
    db: Session = Depends(get_db)
):
    """Process image and return raw bytes (no base64)."""
    try:
        image_bytes = await image.read()
        detected_faces = face_detector.detect_faces(image_bytes)
        
        if not detected_faces:
            return Response(content=image_bytes, media_type="image/jpeg")
        
        optout_encodings = get_all_active_encodings(db)
        comparator = FaceComparator(
            threshold=threshold,
            metric=DistanceMetric(distance_metric.value)
        )
        match_results = comparator.find_matches(detected_faces, optout_encodings)
        
        faces_to_blur = [
            {'x': r.face.x, 'y': r.face.y, 'width': r.face.width, 'height': r.face.height}
            for r in match_results if r.is_match
        ]
        
        processed_bytes, img_format = image_processor.process_image(
            image_bytes, faces_to_blur, method=CensorMethod(censor_method.value)
        )
        
        return Response(content=processed_bytes, media_type=f"image/{img_format}")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# User Management Endpoints
# ============================================================================

@app.get("/users", response_model=UserListResponse, tags=["User Management"])
async def list_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """List all enrolled users."""
    users = db.query(OptOutUser).filter(
        OptOutUser.is_active == True
    ).offset(skip).limit(limit).all()
    
    return UserListResponse(
        total_users=get_user_count(db),
        users=[
            UserInfo(
                user_id=u.id,
                alias=u.alias,
                created_at=u.created_at,
                is_active=u.is_active
            )
            for u in users
        ]
    )


@app.get("/users/{user_id}", response_model=UserInfo, tags=["User Management"])
async def get_user(user_id: str, db: Session = Depends(get_db)):
    """Get information about a specific enrolled user."""
    user = get_user_by_id(db, user_id)
    
    if not user:
        raise HTTPException(status_code=404, detail={"error_code": ErrorCode.USER_NOT_FOUND})
    
    return UserInfo(
        user_id=user.id,
        alias=user.alias,
        created_at=user.created_at,
        is_active=user.is_active
    )


@app.delete("/users/{user_id}", response_model=DeleteUserResponse, tags=["User Management"])
async def delete_user(
    user_id: str,
    permanent: bool = Query(False),
    db: Session = Depends(get_db)
):
    """Remove a user from the opt-out list."""
    success = hard_delete_user(db, user_id) if permanent else deactivate_user(db, user_id)
    
    if not success:
        raise HTTPException(status_code=404, detail={"error_code": ErrorCode.USER_NOT_FOUND})
    
    return DeleteUserResponse(
        status="success",
        message=f"User {'permanently deleted' if permanent else 'deactivated'}",
        user_id=user_id
    )


# ============================================================================
# Batch Processing
# ============================================================================

@app.post("/process/batch", tags=["Processing"])
async def process_batch(
    images: List[UploadFile] = File(...),
    censor_method: CensorMethodEnum = Form(CensorMethodEnum.BLUR),
    distance_metric: DistanceMetricEnum = Form(DistanceMetricEnum.EUCLIDEAN),
    threshold: Optional[float] = Form(None),
    db: Session = Depends(get_db)
):
    """Process multiple images in batch."""
    results = []
    optout_encodings = get_all_active_encodings(db)
    comparator = FaceComparator(
        threshold=threshold,
        metric=DistanceMetric(distance_metric.value)
    )
    
    for img_file in images:
        try:
            image_bytes = await img_file.read()
            detected_faces = face_detector.detect_faces(image_bytes)
            
            faces_to_blur = []
            if detected_faces and optout_encodings:
                match_results = comparator.find_matches(detected_faces, optout_encodings)
                faces_to_blur = [
                    {'x': r.face.x, 'y': r.face.y, 'width': r.face.width, 'height': r.face.height}
                    for r in match_results if r.is_match
                ]
            
            processed_bytes, img_format = image_processor.process_image(
                image_bytes, faces_to_blur, method=CensorMethod(censor_method.value)
            )
            
            results.append({
                "filename": img_file.filename,
                "status": "success",
                "faces_detected": len(detected_faces),
                "faces_redacted": len(faces_to_blur),
                "processed_image": image_to_base64(processed_bytes, img_format)
            })
            
        except Exception as e:
            results.append({
                "filename": img_file.filename,
                "status": "error",
                "error": str(e)
            })
    
    return {"status": "success", "total_processed": len(results), "results": results}


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)