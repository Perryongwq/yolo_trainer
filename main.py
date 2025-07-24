import multiprocessing
import tkinter as tk
from yolo_trainer_gui import YOLOTrainerGUI
import sys
import os

def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)

def main():
    multiprocessing.freeze_support()
    root = tk.Tk()
    root.title("YOLO Training GUI")
    
    # Set application icon
    try:
        icon_path = resource_path("assets/yolo_icon.ico")
        if os.path.exists(icon_path):
            root.iconbitmap(icon_path)
    except Exception:
        pass
    
    app = YOLOTrainerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()