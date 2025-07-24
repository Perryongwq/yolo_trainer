import tkinter as tk
from tkinter import ttk, messagebox
import threading
import os
from ui.dataset_tab import DatasetTab
from ui.training_tab import TrainingTab
from ui.status_tab import StatusTab
from utils.training_manager import TrainingManager
from ui.evaluation_tab import EvaluationTab
from ui.auto_annotation_tab import AutoAnnotationTab

class YOLOTrainerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Training GUI")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # Variables
        self.yaml_path = tk.StringVar()
        self.model_path = tk.StringVar(value="yolo11l.pt")  # Default to YOLOv8
        self.epochs = tk.IntVar(value=100)
        self.imgsz = tk.IntVar(value=640)
        self.lr0 = tk.DoubleVar(value=0.001)
        self.lrf = tk.DoubleVar(value=0.2)
        self.patience = tk.IntVar(value=0)
        self.optimizer = tk.StringVar(value="Adam")
        
        self.dataset_content = None
        self.classes = []
        self.training_in_progress = False
        
        # Create tabs
        self.tab_control = ttk.Notebook(root)
        self.tab_control.pack(expand=1, fill="both")
        
        # Initialize tabs
        self.dataset_tab = DatasetTab(self.tab_control, self)
        self.training_tab = TrainingTab(self.tab_control, self)
        self.status_tab = StatusTab(self.tab_control, self)
        self.evaluation_tab = EvaluationTab(self.tab_control, self)
        self.auto_annotation_tab = AutoAnnotationTab(self.tab_control, self)
        # Initialize training manager
        self.training_manager = TrainingManager(self)
        
        # Log application startup
        self.update_status(f"Application started. Current directory: {os.getcwd()}")
    
    def start_training(self):
        if self.training_in_progress:
            messagebox.showwarning("Warning", "Training is already in progress")
            return
        
        if not self.yaml_path.get():
            messagebox.showerror("Error", "Please select a YAML file")
            return
        
        # Get advanced options from training tab
        advanced_options = self.training_tab.get_advanced_options()
        
        # Switch to status tab
        self.tab_control.select(2)  # Index of status tab
        
        # Start training in a separate thread
        self.training_in_progress = True
        threading.Thread(
            target=self.training_manager.run_training, 
            args=(advanced_options,),
            daemon=True
        ).start()
    
    def update_status(self, message):
        self.status_tab.update_status(message)