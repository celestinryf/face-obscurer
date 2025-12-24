"""
Image Processing Module - The "Censor" Engine

Handles the visual surgery:
- Gaussian blur
- Pixelation (mosaic effect)
- Black bar overlay
- "Invisible" mode placeholder (for AI inpainting)

All operations work directly on image buffers to avoid
exposing uncensored data.
"""

import io
import base64
import hashlib
from enum import Enum
from typing import List, Optional, Tuple
from PIL import Image, ImageFilter, ImageDraw
import numpy as np

# Try to import OpenCV for advanced processing
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("⚠️  OpenCV not available. Using Pillow for all operations.")


class CensorMethod(Enum):
    """Available censorship methods."""
    GAUSSIAN_BLUR = "blur"
    PIXELATE = "pixelate"
    BLACK_BAR = "black_bar"
    EMOJI = "emoji"  # Fun hackathon feature


class ImageProcessor:
    """
    Core image processing engine.
    
    Takes an image and face coordinates, returns a censored version.
    The original image is never stored or returned.
    """
    
    def __init__(self, method: CensorMethod = CensorMethod.GAUSSIAN_BLUR):
        self.method = method
        self.blur_radius = 30  # Intensity of Gaussian blur
        self.pixel_size = 10   # Size of pixels in mosaic effect
    
    def process_image(
        self,
        image_bytes: bytes,
        faces_to_blur: List[dict],
        method: Optional[CensorMethod] = None
    ) -> Tuple[bytes, str]:
        """
        Apply censorship to specified face regions.
        
        Args:
            image_bytes: Raw image data
            faces_to_blur: List of dicts with {x, y, width, height}
            method: Override default censorship method
        
        Returns:
            Tuple of (processed_image_bytes, image_format)
        """
        method = method or self.method
        
        # Load image
        img = Image.open(io.BytesIO(image_bytes))
        original_format = img.format or "PNG"
        
        # Convert to RGB if necessary (for JPEG compatibility)
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')
        
        # Apply censorship to each face
        for face in faces_to_blur:
            x = face.get('x', face.get('left', 0))
            y = face.get('y', face.get('top', 0))
            width = face.get('width', face.get('w', 100))
            height = face.get('height', face.get('h', 100))
            
            # Ensure coordinates are within image bounds
            x = max(0, x)
            y = max(0, y)
            width = min(width, img.width - x)
            height = min(height, img.height - y)
            
            if width <= 0 or height <= 0:
                continue
            
            # Apply the chosen censorship method
            if method == CensorMethod.GAUSSIAN_BLUR:
                img = self._apply_blur(img, x, y, width, height)
            elif method == CensorMethod.PIXELATE:
                img = self._apply_pixelate(img, x, y, width, height)
            elif method == CensorMethod.BLACK_BAR:
                img = self._apply_black_bar(img, x, y, width, height)
            elif method == CensorMethod.EMOJI:
                img = self._apply_emoji(img, x, y, width, height)
        
        # Convert back to bytes
        output_buffer = io.BytesIO()
        save_format = "JPEG" if original_format.upper() in ["JPEG", "JPG"] else "PNG"
        img.save(output_buffer, format=save_format, quality=95)
        
        return output_buffer.getvalue(), save_format.lower()
    
    def _apply_blur(
        self,
        img: Image.Image,
        x: int, y: int, w: int, h: int
    ) -> Image.Image:
        """Apply Gaussian blur to a region."""
        box = (x, y, x + w, y + h)
        
        # Extract region
        face_region = img.crop(box)
        
        # Apply heavy blur
        blurred = face_region.filter(
            ImageFilter.GaussianBlur(radius=self.blur_radius)
        )
        
        # Paste back
        img.paste(blurred, box)
        return img
    
    def _apply_pixelate(
        self,
        img: Image.Image,
        x: int, y: int, w: int, h: int
    ) -> Image.Image:
        """Apply mosaic/pixelation effect to a region."""
        box = (x, y, x + w, y + h)
        
        # Extract region
        face_region = img.crop(box)
        
        # Shrink to create pixelation
        small_size = (
            max(1, w // self.pixel_size),
            max(1, h // self.pixel_size)
        )
        small = face_region.resize(small_size, Image.Resampling.NEAREST)
        
        # Scale back up
        pixelated = small.resize((w, h), Image.Resampling.NEAREST)
        
        # Paste back
        img.paste(pixelated, box)
        return img
    
    def _apply_black_bar(
        self,
        img: Image.Image,
        x: int, y: int, w: int, h: int
    ) -> Image.Image:
        """Apply black rectangle over the face (classic censorship)."""
        draw = ImageDraw.Draw(img)
        draw.rectangle([x, y, x + w, y + h], fill="black")
        return img
    
    def _apply_emoji(
        self,
        img: Image.Image,
        x: int, y: int, w: int, h: int
    ) -> Image.Image:
        """
        Apply emoji overlay (fun hackathon feature).
        Falls back to blur if emoji rendering isn't available.
        """
        try:
            from PIL import ImageFont
            
            draw = ImageDraw.Draw(img)
            
            # Center the emoji
            emoji = "😶"
            
            # Try to get a font that supports emoji
            try:
                font_size = min(w, h)
                font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf", font_size)
            except:
                # Fall back to blur if emoji font not available
                return self._apply_blur(img, x, y, w, h)
            
            # Calculate position to center emoji
            bbox = font.getbbox(emoji)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            
            text_x = x + (w - text_w) // 2
            text_y = y + (h - text_h) // 2
            
            # First black out the area, then draw emoji
            draw.rectangle([x, y, x + w, y + h], fill="white")
            draw.text((text_x, text_y), emoji, font=font, embedded_color=True)
            
            return img
            
        except Exception:
            # Fall back to blur
            return self._apply_blur(img, x, y, w, h)


class ImageProcessorCV:
    """
    OpenCV-based processor for higher performance.
    Particularly useful for video processing.
    """
    
    def __init__(self, method: CensorMethod = CensorMethod.GAUSSIAN_BLUR):
        if not OPENCV_AVAILABLE:
            raise ImportError("OpenCV is required for ImageProcessorCV")
        
        self.method = method
        self.blur_kernel = (99, 99)  # Must be odd numbers
        self.blur_sigma = 30
    
    def process_image(
        self,
        image_bytes: bytes,
        faces_to_blur: List[dict],
        method: Optional[CensorMethod] = None
    ) -> Tuple[bytes, str]:
        """Process image using OpenCV."""
        method = method or self.method
        
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        for face in faces_to_blur:
            x = face.get('x', face.get('left', 0))
            y = face.get('y', face.get('top', 0))
            w = face.get('width', face.get('w', 100))
            h = face.get('height', face.get('h', 100))
            
            # Bounds checking
            x = max(0, x)
            y = max(0, y)
            w = min(w, img.shape[1] - x)
            h = min(h, img.shape[0] - y)
            
            if w <= 0 or h <= 0:
                continue
            
            if method == CensorMethod.GAUSSIAN_BLUR:
                # Extract ROI
                roi = img[y:y+h, x:x+w]
                # Apply blur
                blurred = cv2.GaussianBlur(roi, self.blur_kernel, self.blur_sigma)
                # Replace
                img[y:y+h, x:x+w] = blurred
            
            elif method == CensorMethod.PIXELATE:
                roi = img[y:y+h, x:x+w]
                # Shrink and expand
                small = cv2.resize(roi, (max(1, w // 10), max(1, h // 10)))
                pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
                img[y:y+h, x:x+w] = pixelated
            
            elif method == CensorMethod.BLACK_BAR:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), -1)
        
        # Encode back
        _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return buffer.tobytes(), 'jpeg'


# ============================================================================
# Utility Functions
# ============================================================================

def image_to_base64(image_bytes: bytes, format: str = "jpeg") -> str:
    """Convert image bytes to base64 data URL."""
    b64 = base64.b64encode(image_bytes).decode('utf-8')
    return f"data:image/{format};base64,{b64}"


def base64_to_image(data_url: str) -> bytes:
    """Convert base64 data URL to image bytes."""
    # Remove header if present
    if ',' in data_url:
        data_url = data_url.split(',')[1]
    return base64.b64decode(data_url)


def calculate_image_hash(image_bytes: bytes) -> str:
    """Calculate SHA256 hash of image for deduplication."""
    return hashlib.sha256(image_bytes).hexdigest()


def get_image_dimensions(image_bytes: bytes) -> Tuple[int, int]:
    """Get width and height of an image."""
    img = Image.open(io.BytesIO(image_bytes))
    return img.size


# Default processor
def get_processor(use_opencv: bool = False) -> ImageProcessor:
    """Factory function to get the appropriate processor."""
    if use_opencv and OPENCV_AVAILABLE:
        return ImageProcessorCV()
    return ImageProcessor()