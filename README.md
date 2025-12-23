# 🔒 Privacy Engine - Right to be Forgotten

A hackathon-ready face recognition privacy system that allows users to automatically opt-out of appearing in photos. Built with FastAPI, this addresses real-world privacy concerns like GDPR and CCPA compliance.

## 🎯 Project Overview

This is **Person 2's** contribution - the Backend & Database layer that:
- Stores encrypted face encodings (never actual photos)
- Orchestrates face comparison logic
- Applies censorship to matched faces
- Provides REST API for the frontend

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (Person 3)                      │
│                    React/Tailwind Dashboard                      │
└─────────────────────────────┬───────────────────────────────────┘
                              │ HTTP/REST
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Backend API (Person 2)                        │
│                         FastAPI Server                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   /enroll   │  │  /process   │  │     /users/*            │  │
│  └──────┬──────┘  └──────┬──────┘  └─────────────────────────┘  │
│         │                │                                       │
│         ▼                ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Face Comparison Engine                          ││
│  │         (Euclidean Distance < 0.6 = Match)                   ││
│  └─────────────────────────────────────────────────────────────┘│
│         │                │                                       │
│         ▼                ▼                                       │
│  ┌──────────────┐ ┌──────────────┐                              │
│  │   Encrypted  │ │    Image     │                              │
│  │   SQLite DB  │ │  Processor   │                              │
│  │  (Vectors)   │ │   (Blur)     │                              │
│  └──────────────┘ └──────────────┘                              │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CV Module (Person 1)                          │
│              face_recognition / dlib / MTCNN                     │
│         Provides: detect_faces() → [(x,y,w,h,encoding)]          │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- CMake (for dlib compilation)

### Installation

```bash
# Clone and navigate
cd privacy-engine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📡 API Endpoints

### Enrollment (Opt-Out)

```bash
# Enroll a user
curl -X POST "http://localhost:8000/enroll" \
  -F "photo=@face.jpg" \
  -F "alias=John Doe"
```

**Response:**
```json
{
  "status": "success",
  "message": "User enrolled successfully...",
  "user_id": "usr_abc123def456",
  "created_at": "2024-01-15T10:30:00Z"
}
```

### Process Image

```bash
# Process a group photo
curl -X POST "http://localhost:8000/process" \
  -F "image=@group_photo.jpg" \
  -F "censor_method=blur" \
  -F "threshold=0.6"
```

**Response:**
```json
{
  "status": "success",
  "faces_detected": 5,
  "faces_redacted": 2,
  "matched_users": ["usr_abc123", "usr_def456"],
  "processed_image": "data:image/jpeg;base64,...",
  "image_format": "jpeg",
  "processing_time_ms": 234.5
}
```

### Censorship Methods

| Method | Description |
|--------|-------------|
| `blur` | Gaussian blur (default, most natural) |
| `pixelate` | Mosaic/pixelation effect |
| `black_bar` | Solid black rectangle |
| `emoji` | Fun emoji overlay 😶 |

## 🔐 Privacy Features

### Zero-Knowledge Design

1. **No Photo Storage**: Original enrollment photos are processed in memory and immediately discarded
2. **Encrypted Vectors**: Face encodings are encrypted at rest using Fernet symmetric encryption
3. **Audit Logging**: Track processing events without storing images

### Database Schema

```
┌─────────────────────────────────────────┐
│            optout_users                  │
├─────────────────────────────────────────┤
│ id          │ VARCHAR (PK)              │
│ alias       │ VARCHAR (nullable)        │
│ face_encoding│ BLOB (encrypted)         │
│ created_at  │ DATETIME                  │
│ is_active   │ BOOLEAN                   │
└─────────────────────────────────────────┘
```

## 🔗 Integration with Team

### For Person 1 (CV Specialist)

Implement this interface in `face_utils.py`:

```python
class FaceDetectorInterface:
    def detect_faces(self, image_bytes: bytes) -> List[DetectedFace]:
        """Return list of DetectedFace with x, y, width, height, encoding"""
        pass
    
    def generate_encoding(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """Return 128-dim encoding for single face"""
        pass
```

### For Person 3 (Frontend)

**JavaScript fetch example:**

```javascript
// Enroll user
const enrollUser = async (photoFile, alias) => {
  const formData = new FormData();
  formData.append('photo', photoFile);
  formData.append('alias', alias);
  
  const response = await fetch('http://localhost:8000/enroll', {
    method: 'POST',
    body: formData
  });
  
  return response.json();
};

// Process image
const processImage = async (imageFile, method = 'blur') => {
  const formData = new FormData();
  formData.append('image', imageFile);
  formData.append('censor_method', method);
  
  const response = await fetch('http://localhost:8000/process', {
    method: 'POST',
    body: formData
  });
  
  const data = await response.json();
  // data.processed_image is a base64 data URL
  return data;
};
```

## 🧪 Testing

```bash
# Run tests
pytest tests/ -v

# Test with mock detector (no face_recognition required)
MOCK_DETECTOR=true pytest tests/ -v
```

## 📁 Project Structure

```
privacy-engine/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI application
│   ├── database.py       # SQLite + encryption
│   ├── face_utils.py     # Comparison logic
│   ├── image_processor.py # Blur/pixelate engine
│   └── models.py         # Pydantic schemas
├── tests/
│   └── test_api.py
├── requirements.txt
├── docker-compose.yml
└── README.md
```

## 🐳 Docker Deployment

```bash
docker-compose up -d
```

## 🏆 Hackathon Features

- [x] Core enrollment and processing
- [x] Multiple censorship methods
- [x] Encrypted vector storage
- [x] Batch processing endpoint
- [ ] Video frame processing
- [ ] Stable Diffusion "invisible" mode
- [ ] Real-time WebSocket streaming

## 📊 Performance

| Operation | Time (avg) |
|-----------|------------|
| Face encoding | ~100ms |
| Single comparison | ~0.1ms |
| Image blur | ~50ms |
| Full pipeline (5 faces) | ~300ms |

## 🤝 Contributing

This is a hackathon project. Each team member owns their domain:

- **Person 1**: `face_utils.py` detection implementation
- **Person 2**: Everything else in `/app`
- **Person 3**: Frontend (separate repo)

## 📜 License

MIT License - Built for [Hackathon Name] 2024