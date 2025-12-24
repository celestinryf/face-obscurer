#!/usr/bin/env python3
"""Camera scanner with face recognition and YOLO object detection.

Runs a live camera feed, detects faces and labels them, and detects animals/objects using YOLOv8.

Known faces (stored in known_faces/) are labeled by name; unknown faces are labeled as "Person 1", etc.

Expected folder structure:
    server/known_faces/
        alice.jpg
        bob.heic
        ...

Usage:
    python server/camera_face_scanner.py --device 0 --conf 0.5

Press 'q' to quit.
"""
from __future__ import annotations

import FreeSimpleGUI as sg

import argparse
import sys
from pathlib import Path

import cv2
import face_recognition
import numpy as np
from ultralytics import YOLO

# Setup SG theme
sg.theme('DarkBlack')

# Register HEIF/HEIC opener for PIL
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass  # pillow-heif not installed, HEIC support will fail gracefully

def create_loading_window() -> sg.Window:
    layout = [
        [sg.Text("Loading camera scanner...", key='LoadingText', font=("Lucida Console", 14))],
    ]
    return sg.Window("Loading...", layout, finalize=True)

def parse_args() -> argparse.Namespace:
    # Default to known_faces folder relative to this script
    script_dir = Path(__file__).parent
    default_faces_dir = script_dir / "known_faces"
    
    p = argparse.ArgumentParser(description="Camera scanner with face recognition and YOLO detection")
    p.add_argument("--device", type=int, default=0, help="Camera device index (default: 0)")
    p.add_argument("--width", type=int, default=640, help="Camera width")
    p.add_argument("--height", type=int, default=480, help="Camera height")
    p.add_argument("--known-faces-dir", type=str, default=str(default_faces_dir), help="Directory with known faces")
    p.add_argument("--face-tolerance", type=float, default=0.4, help="Face match tolerance (lower=stricter)")
    p.add_argument("--conf", type=float, default=0.5, help="YOLO confidence threshold")
    return p.parse_args()


def load_known_faces(known_faces_dir: str, loading_window: sg.Window) -> tuple[list[np.ndarray], list[str]]:
    """Load known face encodings from directory.
    
    Expected structure:
        known_faces/
            alice.jpg
            alice.png
            bob.heic
            charlie.jpg
            ...
    
    Label is extracted from the filename (without extension).
    Supports HEIC, JPG, JPEG, and PNG formats.
    
    Returns:
        (encodings, names) tuples where encodings[i] is the encoding for names[i]
    """
    known_encodings = []
    known_names = []

    known_dir = Path(known_faces_dir)
    if not known_dir.exists():
        print(f"INFO: {known_faces_dir} does not exist yet. No known faces loaded.", file=sys.stderr)
        return known_encodings, known_names

    # Support HEIC (Apple), JPG, and PNG formats
    image_files = sorted(known_dir.glob("*.png"))
    '''  
        With current implementation, all images are .png. To add more formats, uncomment below:
        (
        sorted(known_dir.glob("*.heic"))
        + sorted(known_dir.glob("*.HEIC"))
        + sorted(known_dir.glob("*.jpg"))
        + sorted(known_dir.glob("*.JPG"))
        + sorted(known_dir.glob("*.jpeg"))
        + sorted(known_dir.glob("*.JPEG"))
        + sorted(known_dir.glob("*.png"))
        + sorted(known_dir.glob("*.PNG"))
        )
    '''
    for image_path in image_files:
        try:
            # Extract label from filename (without extension)
            person_name = image_path.stem
            
            image = face_recognition.load_image_file(str(image_path))
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(person_name)
                loading_window['LoadingText'].update(f"Loaded {image_path.name} -> {person_name}")
                loading_window.refresh()
                print(f"  Loaded {image_path.name} -> {person_name}", file=sys.stderr)
            else:
                loading_window['LoadingText'].update(f"WARNING: No face found in {image_path.name}")
                loading_window.refresh()
                print(f"  WARNING: No face found in {image_path.name}", file=sys.stderr)
        except Exception as e:
            loading_window['LoadingText'].update(f"ERROR loading {image_path.name}: {e}")
            loading_window.refresh()
            print(f"  ERROR loading {image_path.name}: {e}", file=sys.stderr)

    loading_window['LoadingText'].update(f"Loaded {len(known_encodings)} face encodings for {len(set(known_names))} people")
    loading_window.refresh()
    print(f"Loaded {len(known_encodings)} face encodings for {len(set(known_names))} people", file=sys.stderr)
    return known_encodings, known_names


def main() -> int:

    loading_window = create_loading_window()

    event, values = loading_window.read(timeout=100)

    args = parse_args()

    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera device {args.device}", file=sys.stderr)
        return 2

    # Set capture resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # Load known faces
    known_encodings, known_names = load_known_faces(args.known_faces_dir, loading_window)

    # Load YOLO model
    loading_window['LoadingText'].update("Loading YOLOv8 model (this may take a moment)...")
    loading_window.refresh()
    print("Loading YOLOv8 model (this may take a moment)...", file=sys.stderr)
    try:
        model = YOLO("yolov8n.pt")  # nano model for speed; use yolov8s.pt, yolov8m.pt for better accuracy
    except Exception as e:
        print(f"ERROR: Failed to load YOLO model: {e}", file=sys.stderr)
        cap.release()
        return 3

    window_name = "Face & Object Scanner - press 'q' to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    loading_window['LoadingText'].update("Enjoy!")
    loading_window.refresh()

    unknown_person_counter = 0  # Counter for unknown faces in current frame

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("WARNING: Empty frame received", file=sys.stderr)
                break

            # Resize for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Detect faces in current frame
            face_locations = face_recognition.face_locations(rgb_small_frame, model="cnn")
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            # Run YOLO detection on full frame
            results = model(frame, conf=args.conf, verbose=False)

            # Reset unknown person counter for each frame
            unknown_person_counter = 0
            frame_labels = {}

            # Match faces
            for idx, face_encoding in enumerate(face_encodings):
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=args.face_tolerance)
                distances = face_recognition.face_distance(known_encodings, face_encoding)

                label = "Unknown"
                if len(distances) > 0:
                    best_match_idx = np.argmin(distances)
                    if matches[best_match_idx]:
                        label = known_names[best_match_idx]

                if label == "Unknown":
                    unknown_person_counter += 1
                    label = f"Person {unknown_person_counter}"

                frame_labels[idx] = label

                # Draw rectangles for faces with labels
                for idx, (top, right, bottom, left) in enumerate(face_locations):
                    # Scale back up (we detected on 0.25 scale)
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4

                    label = frame_labels.get(idx, "Unknown")
                    
                    # PRIVACY LOGIC: If the face is "Known" (one of your opt-out users), black it out
                    if label != "Unknown" and not label.startswith("Person "):
                        # Draw a solid black box over the face
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 0), -1)
                    else:
                        # Normal diagnostic box for unknown people
                        color = (0, 0, 255) # Red
                        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                        cv2.putText(
                            frame,
                            label,
                            (left + 6, bottom - 6),
                            cv2.FONT_HERSHEY_DUPLEX,
                            0.6,
                            (255, 255, 255),
                            1,
                        )

            # Draw YOLO detections
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = result.names[cls_id]

                    # Skip person class (handled by face recognition above)
                    if cls_name.lower() == "person":
                        continue

                    # Color based on object type
                    if "cat" in cls_name.lower() or "dog" in cls_name.lower():
                        color = (0, 128, 255)  # Orange for animals
                    else:
                        color = (255, 127, 0)  # Cyan for other objects

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label_text = f"{cls_name} {conf:.2f}"
                    cv2.rectangle(frame, (x1, y2 - 25), (x2, y2), color, cv2.FILLED)
                    cv2.putText(
                        frame,
                        label_text,
                        (x1 + 5, y2 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )

            
            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
