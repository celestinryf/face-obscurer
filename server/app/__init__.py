"""
Privacy Engine - Right to be Forgotten (Local/dlib Version)
===========================================================

A face recognition privacy system that allows users to opt-out
of appearing in photos.

Uses dlib for 128-dim face encodings - compatible with Person 1's camera scanner.

Components:
- database.py: SQLite database for encrypted face encodings
- face_utils.py: dlib face detection and 128-dim encodings
- image_processor.py: Censorship application (blur, pixelate, etc.)
- models.py: Pydantic request/response models
- main.py: FastAPI application
"""

__version__ = "2.0.0"
__author__ = "Person 2 - Backend Team"