from fastai.vision.all import *
import numpy as np
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import keyboard  # Import keyboard library for global hotkeys
import queue  # Import queue to handle inter-thread communication

def setup_learner():
    directory = Path(r'C:\Users\DELL\Desktop\seven_plastics')

    tfms = aug_transforms(
        do_flip=False, 
        max_rotate=0, 
        max_zoom=1, 
        max_lighting=0, 
        max_warp=0
    )

    bs = 32
    np.random.seed(42)

    dls = ImageDataLoaders.from_folder(
        directory,
        valid_pct=0.05,
        item_tfms=Resize(128),
        batch_tfms=tfms,
        bs=bs
    )

    model_dir = Path(r'C:\Users\DELL\Desktop\seven_plastics\models')
    model_dir.mkdir(parents=True, exist_ok=True)

    learn = cnn_learner(dls, resnet50, model_dir=model_dir, metrics=error_rate)

    learn.lr_find()
    learn.fit_one_cycle(4, slice(1e-03, 4e-3))
    learn.save('plastics_save_1')

    learn.unfreeze()
    learn.lr_find()
    learn.fit_one_cycle(6, slice(1e-03, 3e-4))
    learn.save('plastics_save_2')

    learn.export('plastics_save_2.pkl')

    return learn, dls

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

def main_loop(learn, dls):
    root = tk.Tk()
    root.geometry('200x100')
    
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
    root.after(100, process_queue)  # Start processing the queue
    root.mainloop()  # Tkinter event loop to keep the window running

def exit_program(root):
    print("Exiting...")
    root.quit()

if __name__ == "__main__":
    learn, dls = setup_learner()
    main_loop(learn, dls)
