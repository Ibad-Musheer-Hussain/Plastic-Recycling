from fastai.vision.all import *
import numpy as np
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import keyboard  # Import keyboard library for global hotkeys
import queue  # Import queue to handle inter-thread communication

def load_existing_learner():
    model_path = Path(r'C:\Users\DELL\Desktop\seven_plastics\models\plastics_save_2.pkl')
    
    # Load the saved learner
    learn = load_learner(model_path)
    
    return learn

def predict_image(img_path, learn):
    img = PILImage.create(img_path)
    pred_class, pred_idx, outputs = learn.predict(img)
    print(f"Predicted class: {pred_class}")
    print(f"Prediction index: {pred_idx}")
    print(f"Output probabilities: {outputs}")

def open_image_dialog(learn):
    img_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")]
    )
    if img_path:
        predict_image(img_path, learn)
    else:
        print("No file selected. Press F8 to try again or F12 to exit.")

def main_loop(learn):
    root = tk.Tk()
    root.geometry('300x150')
    
    # Create a queue for communication between the main thread and hotkey handler
    q = queue.Queue()

    def process_queue():
        while not q.empty():
            command = q.get()
            if command == 'open_image':
                open_image_dialog(learn)
            elif command == 'exit':
                exit_program(root)
        root.after(100, process_queue)  # Schedule the next queue check

    # Register global hotkeys using the keyboard library
    keyboard.add_hotkey('F8', lambda: q.put('open_image'))
    keyboard.add_hotkey('F12', lambda: q.put('exit'))

    print("Press F8 to select an image or F12 to exit.")
    
    # Add a button to select an image
    select_button = tk.Button(root, text="Select Image", command=lambda: open_image_dialog(learn))
    select_button.pack(pady=20)
    
    # Add an exit button
    exit_button = tk.Button(root, text="Exit", command=lambda: exit_program(root))
    exit_button.pack(pady=10)

    root.after(100, process_queue)  # Start processing the queue
    root.mainloop()  # Tkinter event loop to keep the window running

def exit_program(root):
    print("Exiting...")
    root.quit()

if __name__ == "__main__":
    learn = load_existing_learner()
    main_loop(learn)