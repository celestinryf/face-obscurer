"""
Face Detection & Recognition Utilities (dlib Version)

Uses dlib/face_recognition library - same as Person 1's camera scanner.
This ensures encodings are compatible between camera scanner and API.

Encoding: 128-dimensional face embeddings (industry standard)
"""

import io
import numpy as np
from typing import List, Tuple, Optional
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict
from PIL import Image

# Register HEIF/HEIC opener for PIL (Apple photo support)
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass


class DistanceMetric(str, Enum):
    """Supported distance metrics for face comparison."""
    EUCLIDEAN = "euclidean"  # L2 norm - dlib default, recommended
    MANHATTAN = "manhattan"  # L1 norm - robust to outliers
    COSINE = "cosine"        # Angular similarity


class DetectedFace(BaseModel):
    """
    Detected face with 128-dimensional dlib encoding.
    
    Using Pydantic for:
    - Automatic validation
    - JSON serialization
    - FastAPI integration
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    x: int = Field(..., description="Top-left X coordinate", ge=0)
    y: int = Field(..., description="Top-left Y coordinate", ge=0)
    width: int = Field(..., description="Bounding box width", gt=0)
    height: int = Field(..., description="Bounding box height", gt=0)
    encoding: np.ndarray = Field(..., description="128-dim face encoding vector")
    confidence: float = Field(default=1.0, description="Detection confidence", ge=0.0, le=1.0)
    
    def to_dict(self) -> dict:
        """Convert to dictionary (excluding encoding for JSON safety)."""
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "confidence": self.confidence
        }


class MatchResult(BaseModel):
    """Result of comparing a detected face against the opt-out database."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    face: DetectedFace
    matched_user_id: Optional[str] = None
    distance: float = Field(..., description="Distance score (lower = more similar)")
    is_match: bool = Field(..., description="Whether this face matched an opt-out user")
    
    def to_dict(self) -> dict:
        return {
            "coordinates": self.face.to_dict(),
            "matched_user_id": self.matched_user_id,
            "distance": round(self.distance, 4),
            "is_match": self.is_match
        }


class FaceComparator:
    """
    Core comparison engine for 128-dim dlib face encodings.
    
    Default thresholds:
    - Euclidean: 0.6 (dlib standard - what Person 1 uses)
    - Manhattan: 6.0 (approximate equivalent for 128-dim vectors)
    - Cosine: 0.4
    
    Person 1's camera scanner uses tolerance=0.4 (stricter than default).
    """
    
    # Default thresholds per metric
    DEFAULT_THRESHOLDS = {
        DistanceMetric.EUCLIDEAN: 0.6,   # dlib default
        DistanceMetric.MANHATTAN: 6.0,   # ~equivalent for 128-dim
        DistanceMetric.COSINE: 0.4,
    }
    
    def __init__(
        self,
        threshold: Optional[float] = None,
        metric: DistanceMetric = DistanceMetric.EUCLIDEAN  # Match dlib default
    ):
        self.metric = metric
        self.threshold = threshold if threshold is not None else self.DEFAULT_THRESHOLDS[metric]
    
    def calculate_distance(
        self,
        encoding1: np.ndarray,
        encoding2: np.ndarray
    ) -> float:
        """
        Calculate distance between two 128-dim face encodings.
        
        Formulas:
        - Euclidean (L2): d = √Σ(a_i - b_i)² — dlib default
        - Manhattan (L1): d = Σ|a_i - b_i|
        - Cosine: d = 1 - (a·b) / (‖a‖ × ‖b‖)
        """
        if self.metric == DistanceMetric.EUCLIDEAN:
            return float(np.linalg.norm(encoding1 - encoding2))
        
        elif self.metric == DistanceMetric.MANHATTAN:
            return float(np.sum(np.abs(encoding1 - encoding2)))
        
        elif self.metric == DistanceMetric.COSINE:
            dot_product = np.dot(encoding1, encoding2)
            norm_product = np.linalg.norm(encoding1) * np.linalg.norm(encoding2)
            
            if norm_product == 0:
                return 1.0
            
            return float(1 - (dot_product / norm_product))
        
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def is_match(self, distance: float) -> bool:
        """Determine if distance indicates a match."""
        return distance <= self.threshold
    
    def find_matches(
        self,
        detected_faces: List[DetectedFace],
        optout_encodings: List[Tuple[str, np.ndarray]]
    ) -> List[MatchResult]:
        """
        Compare all detected faces against the opt-out database.
        
        Args:
            detected_faces: Faces found in the image
            optout_encodings: List of (user_id, encoding) from database
        
        Returns:
            List of MatchResult for each detected face
        """
        results = []
        
        for face in detected_faces:
            best_match_id = None
            best_distance = float('inf')
            
            for user_id, db_encoding in optout_encodings:
                distance = self.calculate_distance(face.encoding, db_encoding)
                
                if distance < best_distance:
                    best_distance = distance
                    best_match_id = user_id
            
            matched = self.is_match(best_distance) if optout_encodings else False
            
            results.append(MatchResult(
                face=face,
                matched_user_id=best_match_id if matched else None,
                distance=best_distance if optout_encodings else 0.0,
                is_match=matched
            ))
        
        return results
    
    def get_faces_to_blur(
        self,
        match_results: List[MatchResult]
    ) -> List[DetectedFace]:
        """Extract only the faces that need to be blurred."""
        return [result.face for result in match_results if result.is_match]


# ============================================================================
# Face Detector Interface
# ============================================================================

class FaceDetectorInterface:
    """Interface that detection implementations must follow."""
    
    def detect_faces(self, image_bytes: bytes) -> List[DetectedFace]:
        raise NotImplementedError
    
    def generate_encoding(self, image_bytes: bytes) -> Optional[np.ndarray]:
        raise NotImplementedError


# ============================================================================
# dlib Implementation (matches Person 1's camera scanner)
# ============================================================================

try:
    import face_recognition
    
    class DlibFaceDetector(FaceDetectorInterface):
        """
        Face detector using dlib via face_recognition library.
        
        This produces 128-dimensional embeddings, same as Person 1's
        camera scanner. Encodings are fully compatible.
        
        Args:
            model: "hog" (faster, CPU) or "cnn" (more accurate, needs GPU)
        """
        
        def __init__(self, model: str = "hog"):
            self.model = model
        
        def detect_faces(self, image_bytes: bytes) -> List[DetectedFace]:
            """Detect all faces and generate 128-dim encodings."""
            # Load image from bytes
            image = face_recognition.load_image_file(io.BytesIO(image_bytes))
            
            # Detect face locations (bounding boxes)
            # Returns list of (top, right, bottom, left) tuples
            face_locations = face_recognition.face_locations(image, model=self.model)
            
            # Generate 128-dim encoding for each face
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            detected_faces = []
            for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
                detected_faces.append(DetectedFace(
                    x=left,
                    y=top,
                    width=right - left,
                    height=bottom - top,
                    encoding=encoding,
                    confidence=1.0  # dlib doesn't provide confidence scores
                ))
            
            return detected_faces
        
        def generate_encoding(self, image_bytes: bytes) -> Optional[np.ndarray]:
            """
            Generate 128-dim encoding for enrollment (single face expected).
            
            Returns the encoding of the first/largest face found.
            """
            image = face_recognition.load_image_file(io.BytesIO(image_bytes))
            
            # Find all faces
            face_locations = face_recognition.face_locations(image, model=self.model)
            
            if not face_locations:
                return None
            
            # Generate encodings
            encodings = face_recognition.face_encodings(image, face_locations)
            
            if not encodings:
                return None
            
            # If multiple faces, return the largest one
            if len(face_locations) > 1:
                # Find largest face by area
                areas = [(r - l) * (b - t) for (t, r, b, l) in face_locations]
                largest_idx = np.argmax(areas)
                return encodings[largest_idx]
            
            return encodings[0]
        
        def compare_faces(
            self,
            known_encodings: List[np.ndarray],
            face_encoding: np.ndarray,
            tolerance: float = 0.6
        ) -> Tuple[List[bool], np.ndarray]:
            """
            Compare a face encoding against a list of known encodings.
            
            This mirrors Person 1's camera scanner logic exactly.
            
            Returns:
                (matches, distances) where matches[i] is True if distance < tolerance
            """
            matches = face_recognition.compare_faces(
                known_encodings, 
                face_encoding, 
                tolerance=tolerance
            )
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            
            return matches, distances
    
    # Use dlib detector
    DEFAULT_DETECTOR = DlibFaceDetector
    print("✅ face_recognition (dlib) loaded successfully")

except ImportError as e:
    print(f"⚠️  face_recognition not available: {e}")
    print("   Install with: pip install face_recognition")
    print("   Using mock detector for testing...")
    
    class MockFaceDetector(FaceDetectorInterface):
        """Mock detector for testing without dlib."""
        
        def detect_faces(self, image_bytes: bytes) -> List[DetectedFace]:
            try:
                img = Image.open(io.BytesIO(image_bytes))
                w, h = img.size
            except:
                w, h = 640, 480
            
            return [
                DetectedFace(
                    x=int(w * 0.2), y=int(h * 0.2),
                    width=int(w * 0.2), height=int(h * 0.25),
                    encoding=np.random.randn(128),  # 128-dim like dlib
                    confidence=0.95
                ),
            ]
        
        def generate_encoding(self, image_bytes: bytes) -> Optional[np.ndarray]:
            return np.random.randn(128)
    
    DEFAULT_DETECTOR = MockFaceDetector


# ============================================================================
# Utility: Load known faces from directory (like Person 1's code)
# ============================================================================

def load_known_faces_from_directory(
    known_faces_dir: str
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Load known face encodings from a directory.
    
    Same logic as Person 1's camera scanner.
    
    Expected structure:
        known_faces/
            alice.jpg
            bob.heic
            charlie.png
    
    Label is extracted from filename (without extension).
    
    Returns:
        (encodings, names) tuples
    """
    from pathlib import Path
    
    known_encodings = []
    known_names = []
    
    known_dir = Path(known_faces_dir)
    if not known_dir.exists():
        print(f"⚠️  {known_faces_dir} does not exist. No known faces loaded.")
        return known_encodings, known_names
    
    # Support multiple image formats
    extensions = ["*.heic", "*.HEIC", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"]
    image_files = []
    for ext in extensions:
        image_files.extend(sorted(known_dir.glob(ext)))
    
    detector = DEFAULT_DETECTOR()
    
    for image_path in image_files:
        try:
            person_name = image_path.stem  # filename without extension
            
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            
            encoding = detector.generate_encoding(image_bytes)
            
            if encoding is not None:
                known_encodings.append(encoding)
                known_names.append(person_name)
                print(f"  ✅ Loaded {image_path.name} -> {person_name}")
            else:
                print(f"  ⚠️  No face found in {image_path.name}")
                
        except Exception as e:
            print(f"  ❌ Error loading {image_path.name}: {e}")
    
    print(f"Loaded {len(known_encodings)} face encodings for {len(set(known_names))} people")
    return known_encodings, known_names