# 🔒 Privacy Engine - Local Version (dlib)

Face recognition privacy system using **dlib** - fully compatible with Person 1's camera scanner.

## ✅ Compatibility

| Component | Encoding | Compatible? |
|-----------|----------|-------------|
| This API | 128-dim dlib | ✅ |
| Camera Scanner | 128-dim dlib | ✅ |
| Known Faces folder | Shared | ✅ |

Faces enrolled via the API will be recognized by the camera scanner and vice versa!

---

## 🚀 Quick Start (Windows)

### Step 1: Install Visual Studio Build Tools

dlib requires C++ compilation. Download and install:

**Visual Studio Build Tools**: https://visualstudio.microsoft.com/visual-cpp-build-tools/

During installation, select:
- ✅ "Desktop development with C++"
- ✅ "MSVC v143 - VS 2022 C++ x64/x86 build tools"
- ✅ "Windows 10/11 SDK"

### Step 2: Install Python (if needed)

Download Python 3.11: https://www.python.org/downloads/

**Important**: Check ✅ "Add Python to PATH" during installation!

### Step 3: Install Dependencies

```powershell
# Navigate to server folder
cd E:\face-obscurer\server

# Create virtual environment (recommended)
python -m venv venv
.\venv\Scripts\Activate

# Install cmake first
pip install cmake

# Install all dependencies (dlib takes ~5-10 minutes to compile)
pip install -r requirements.txt
```

### Step 4: Run the Server

```powershell
# Make sure venv is activated
.\venv\Scripts\Activate

# Run the server
uvicorn app.main:app --reload --port 8000
```

The API is now running at **http://localhost:8000**

### Step 5: Test It

Open http://localhost:8000/docs in your browser to see the Swagger UI.

---

## 📁 Project Structure

```
server/
├── app/
│   ├── main.py           # FastAPI application
│   ├── face_utils.py     # dlib face detection (128-dim encodings)
│   ├── database.py       # SQLite + encryption
│   ├── image_processor.py # Blur/pixelate engine
│   └── models.py         # Pydantic schemas
├── known_faces/          # Shared with camera scanner!
│   ├── alice.jpg
│   ├── bob.heic
│   └── ...
├── requirements.txt
└── README.md
```

---

## 🔗 Sharing Faces with Camera Scanner

Both systems can share the `known_faces/` directory:

```
server/
├── known_faces/      ← Put photos here
│   ├── alice.jpg     ← Filename = label
│   ├── bob.heic
│   └── charlie.png
├── camera_face_scanner.py   ← Person 1's code (reads from known_faces/)
└── app/                     ← API code (can sync from known_faces/)
```

### Sync known_faces to database:

```bash
curl -X POST http://localhost:8000/sync-known-faces
```

This enrolls all faces from `known_faces/` into the database.

---

## 📡 API Endpoints

### Enroll a User

```bash
curl -X POST http://localhost:8000/enroll \
  -F "photo=@face.jpg" \
  -F "alias=John Doe"
```

### Process an Image

```bash
curl -X POST http://localhost:8000/process \
  -F "image=@group_photo.jpg" \
  -F "censor_method=blur" \
  -F "distance_metric=euclidean" \
  -F "threshold=0.6"
```

### List Enrolled Users

```bash
curl http://localhost:8000/users
```

---

## 🎛️ Distance Metrics

| Metric | Default Threshold | Description |
|--------|------------------|-------------|
| `euclidean` | 0.6 | dlib default, standard L2 distance |
| `manhattan` | 6.0 | L1 distance, robust to outliers |
| `cosine` | 0.4 | Angular similarity |

Person 1's camera scanner uses `euclidean` with tolerance `0.4` (stricter).

---

## 🎨 Censorship Methods

| Method | Description |
|--------|-------------|
| `blur` | Gaussian blur (default) |
| `pixelate` | Mosaic effect |
| `black_bar` | Solid black rectangle |
| `emoji` | 😶 overlay |

---

## 🔧 Frontend Integration (React)

```javascript
// Enroll user
const enrollUser = async (photoFile, alias) => {
  const formData = new FormData();
  formData.append('photo', photoFile);
  formData.append('alias', alias);
  
  const response = await fetch('http://localhost:8000/enroll', {
    method: 'POST',
    body: formData,
  });
  
  return response.json();
};

// Process image
const processImage = async (imageFile) => {
  const formData = new FormData();
  formData.append('image', imageFile);
  formData.append('censor_method', 'blur');
  formData.append('distance_metric', 'euclidean');
  formData.append('threshold', '0.5');
  
  const response = await fetch('http://localhost:8000/process', {
    method: 'POST',
    body: formData,
  });
  
  const data = await response.json();
  // data.processed_image is base64 data URL
  return data;
};
```

---

## ⚠️ Troubleshooting

### "pip is not recognized"
```powershell
python -m pip install -r requirements.txt
```

### dlib fails to compile
1. Install Visual Studio Build Tools (see Step 1)
2. Restart PowerShell
3. Try again

### "No module named face_recognition"
```powershell
pip install cmake
pip install dlib
pip install face_recognition
```

### Camera scanner doesn't recognize API-enrolled faces
Make sure both use the same database OR sync via `known_faces/`:
```bash
curl -X POST http://localhost:8000/sync-known-faces
```

---

## 🏃 Running Both Systems

Terminal 1 - API:
```powershell
cd server
.\venv\Scripts\Activate
uvicorn app.main:app --reload --port 8000
```

Terminal 2 - Camera Scanner:
```powershell
cd server
.\venv\Scripts\Activate
python camera_face_scanner.py --device 0
```

Both will recognize the same faces!

---

## 📊 Performance

| Operation | Time |
|-----------|------|
| Face encoding (dlib) | ~200ms |
| Face comparison | ~0.1ms per face |
| Image blur | ~50ms |
| Full pipeline (5 faces) | ~500ms |

---

## 🤝 Team Integration

- **Person 1 (CV)**: Camera scanner + known_faces management
- **Person 2 (Backend)**: This API
- **Person 3 (Frontend)**: React dashboard calling the API

All using compatible 128-dim dlib encodings!