import FreeSimpleGUI as sg
from PIL import Image
from pathlib import Path
import importlib.util

sg.theme('Dark Black')

'''
This module creates and manages the main application window for the Face Obscurer program.
It includes functions to create the initial window, transition to the main window,
and handle changes to the image file.
'''

def make_init_window():
    layout = [  
                [sg.Text("Face Obscurer - Privacy Unlocked", font=("Lucida Console", 40), text_color="yellow", visible=False, key='MainTitle')],
                [sg.VPush()],
                

                [sg.Text("Face Obscurer", font=("Lucida Console", 50), text_color="yellow", key='Title')],  
                [sg.Text("This application obscures your face using AI technology.", font=("Lucida Console", 20), text_color="yellow", key='Description')],
                [sg.Button('Enter at your own risk', font=("Lucida Console", 20), button_color="yellow")],

                [sg.Image(r'.\public\noImageSelected.png', key='ImageDisplay', subsample=2, visible=False)],
                [sg.Button('Select Images', font=("Lucida Console", 20), button_color="yellow", visible=False), sg.Button('Blur My Face', font=("Lucida Console", 20), button_color="yellow", visible=False)],

                [sg.VPush()]
        ]

    window = sg.Window('Face Obscurer', layout, resizable=True, element_justification='c', icon=r'.\public\minimalistIcon2.png').finalize()
    window.Maximize()
    return window

# Hide initial elements
def close_init_window(window):
    window['Title'].update(font=("Lucida Console", 1), visible=False)
    window['Description'].update(visible=False)
    window['Enter at your own risk'].update(visible=False)

# Show main elements
def open_main_window(window):
    window['MainTitle'].update(visible=True)
    window['Description'].update("Select images of your face to train the model, then click Blur My Face to continue.", visible=True)
    window['ImageDisplay'].update(visible=True)
    window['Select Images'].update(visible=True)
    window['Blur My Face'].update(visible=True)

# Handle image selection and processing
def process_image(window):
    image_files = sg.popup_get_file('Select an image file to open', multiple_files=True, no_window=True, file_types=(("Image Files", "*.png;*.jpg;*.jpeg;*.bmp"), ("All Files", "*.*")))

    i = 0
    for file in image_files:
        if file:
            image = Image.open(file)
            image.save(f"..\server\known_faces\known_face_{i}.png", format="PNG")
            i += 1

    

'''
Creates the window and handles the event loop for user interaction.
'''
def main():
    window = make_init_window()

    #cap = cv2.VideoCapture(0)

    # Display and interact with the Window using an Event Loop
    while True:
        event, values = window.read()
        # See if user wants to quit or window was closed
        if event == sg.WINDOW_CLOSED:
            break
        if event == 'Enter at your own risk':
            close_init_window(window)
            open_main_window(window)
        if event == 'Select Images':
            process_image(window)
        if event == 'Blur My Face':
            #sg.popup("Launching face obscuring... please wait.")

            import camera_face_scanner as cfs
            cfs.main()
            
                

    # Finish up by removing from the screen
    window.close()  


if __name__ == '__main__':
    main()