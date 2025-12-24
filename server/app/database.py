"""
Database module for storing face encodings (vectors only, never photos).
Uses SQLite for simplicity - can be swapped for PostgreSQL in production.
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import Optional, List, Tuple
from sqlalchemy import create_engine, Column, String, DateTime, LargeBinary, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from cryptography.fernet import Fernet

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./privacy_vault.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class EncryptionManager:
    """
    Handles encryption/decryption of face vectors.
    Even if the database is breached, vectors cannot be used to reconstruct faces.
    """
    
    def __init__(self):
        # In production, load this from environment variable or secure vault
        self.key = os.getenv("ENCRYPTION_KEY")
        if not self.key:
            # Generate a new key if none exists (for development)
            self.key = Fernet.generate_key()
            print(f"⚠️  Generated new encryption key. Set ENCRYPTION_KEY env var in production!")
        else:
            self.key = self.key.encode() if isinstance(self.key, str) else self.key
        
        self.cipher = Fernet(self.key)
    
    def encrypt_vector(self, vector: np.ndarray) -> bytes:
        """Encrypt a face encoding vector."""
        vector_bytes = vector.tobytes()
        return self.cipher.encrypt(vector_bytes)
    
    def decrypt_vector(self, encrypted_data: bytes) -> np.ndarray:
        """Decrypt a face encoding vector."""
        decrypted_bytes = self.cipher.decrypt(encrypted_data)
        return np.frombuffer(decrypted_bytes, dtype=np.float64)


# Global encryption manager
encryption_manager = EncryptionManager()


class OptOutUser(Base):
    """
    Stores users who have opted out of appearing in photos.
    
    PRIVACY DESIGN:
    - Only stores the mathematical face encoding (128 floats), never the actual photo
    - Vector is encrypted at rest
    - Original enrollment photo is deleted immediately after encoding
    """
    __tablename__ = "optout_users"
    
    id = Column(String, primary_key=True)  # UUID
    alias = Column(String, nullable=True)  # Optional friendly name
    face_encoding = Column(LargeBinary, nullable=False)  # Encrypted 128-dim vector
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)  # Soft delete capability
    
    def set_encoding(self, vector: np.ndarray):
        """Encrypt and store the face encoding."""
        self.face_encoding = encryption_manager.encrypt_vector(vector)
    
    def get_encoding(self) -> np.ndarray:
        """Decrypt and return the face encoding."""
        return encryption_manager.decrypt_vector(self.face_encoding)


class ProcessingLog(Base):
    """
    Audit log of processed images (without storing the images themselves).
    Useful for compliance reporting.
    """
    __tablename__ = "processing_logs"
    
    id = Column(String, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    faces_detected = Column(String)  # JSON list of count
    faces_redacted = Column(String)  # JSON list of matched user IDs
    image_hash = Column(String)  # SHA256 hash of input image (for deduplication)


def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Session:
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ============================================================================
# Database Operations (CRUD)
# ============================================================================

def add_optout_user(
    db: Session,
    user_id: str,
    face_encoding: np.ndarray,
    alias: Optional[str] = None
) -> OptOutUser:
    """
    Add a new user to the opt-out list.
    
    Args:
        db: Database session
        user_id: Unique identifier for the user
        face_encoding: 128-dimensional face encoding vector
        alias: Optional friendly name
    
    Returns:
        The created OptOutUser record
    """
    user = OptOutUser(id=user_id, alias=alias)
    user.set_encoding(face_encoding)
    
    db.add(user)
    db.commit()
    db.refresh(user)
    
    return user


def get_all_active_encodings(db: Session) -> List[Tuple[str, np.ndarray]]:
    """
    Retrieve all active face encodings for comparison.
    
    Returns:
        List of (user_id, face_encoding) tuples
    """
    users = db.query(OptOutUser).filter(OptOutUser.is_active == True).all()
    return [(user.id, user.get_encoding()) for user in users]


def get_user_by_id(db: Session, user_id: str) -> Optional[OptOutUser]:
    """Get a specific user by ID."""
    return db.query(OptOutUser).filter(OptOutUser.id == user_id).first()


def deactivate_user(db: Session, user_id: str) -> bool:
    """
    Soft-delete a user from the opt-out list.
    They can re-enroll later if needed.
    """
    user = get_user_by_id(db, user_id)
    if user:
        user.is_active = False
        db.commit()
        return True
    return False


def hard_delete_user(db: Session, user_id: str) -> bool:
    """
    Permanently delete a user's data.
    Used when user wants complete removal.
    """
    user = get_user_by_id(db, user_id)
    if user:
        db.delete(user)
        db.commit()
        return True
    return False


def get_user_count(db: Session) -> int:
    """Get total number of active opt-out users."""
    return db.query(OptOutUser).filter(OptOutUser.is_active == True).count()


# Initialize database on import
init_db()