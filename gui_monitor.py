import FreeSimpleGUI as sg
import cv2
import requests
import io
from PIL import Image

# 1. Define the Layout
layout = [
    [sg.Text("🛡️ Privacy Engine Desktop Monitor", font=("Any", 18))],
    [sg.Image(filename="", key="-IMAGE-")], # This is our video screen
    [sg.Button("Sync Vault", size=(15, 1)), sg.Button("Exit", size=(10, 1))],
    [sg.StatusBar("System Status: Ready", key="-STATUS-")]
]

window = sg.Window("Privacy Monitor", layout, finalize=True)
cap = cv2.VideoCapture(0) # Open the user's camera

while True:
    event, values = window.read(timeout=20) # High frequency loop
    
    if event in (sg.WIN_CLOSED, "Exit"):
        break

    # 1. Capture frame from local camera
    ret, frame = cap.read()
    if not ret: continue

    if event == "Sync Vault":
        requests.post("http://127.0.0.1:8000/sync-known-faces")
        window["-STATUS-"].update("Vault Synced!")

    # 2. SEND TO BACKEND FOR OBSCURING
    # Convert OpenCV frame to bytes to send over the network
    _, img_encoded = cv2.imencode('.jpg', frame)
    files = {'image': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')}
    
    try:
        # Use your /process/raw endpoint to get the blurred bytes back
        response = requests.post("http://127.0.0.1:8000/process/raw", files=files)
        
        if response.status_code == 200:
            # 3. Display the returned processed bytes in the GUI
            window["-IMAGE-"].update(data=response.content)
    except Exception as e:
        window["-STATUS-"].update(f"Connection Error: {e}")

window.close()
cap.release()