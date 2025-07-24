import tkinter as tk
from tkinter import ttk, filedialog
import torch

class TrainingTab:
    def __init__(self, parent, main_app):
        self.main_app = main_app
        
        # Create tab
        self.tab = ttk.Frame(parent)
        parent.add(self.tab, text="Training Parameters")
        
        # Build UI components
        self.create_training_tab()
    
    def create_training_tab(self):
        # Training parameters frame
        params_frame = ttk.LabelFrame(self.tab, text="Training Parameters")
        params_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Model selection
        ttk.Label(params_frame, text="Model:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        # Include YOLOv11 and YOLOv8 models
        yolo_models = [
            # YOLOv11 models - 'v' style naming
            "yolov11n.pt", "yolov11s.pt", "yolov11m.pt", "yolov11l.pt", "yolov11x.pt",
            # YOLOv11 models - alternative naming
            "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
            # YOLOv8 models
            "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"
        ]
        
        model_combo = ttk.Combobox(params_frame, textvariable=self.main_app.model_path, values=yolo_models)
        model_combo.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Add a browse button for model file selection
        ttk.Button(params_frame, text="Browse", command=self.browse_model_file).grid(
            row=0, column=2, padx=5, pady=5)
        
        # Epochs
        ttk.Label(params_frame, text="Epochs:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        ttk.Spinbox(params_frame, from_=1, to=1000, textvariable=self.main_app.epochs, width=10).grid(
            row=1, column=1, padx=5, pady=5, sticky="w")
        
        # Image size
        ttk.Label(params_frame, text="Image Size:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        ttk.Spinbox(params_frame, from_=32, to=1280, increment=32, textvariable=self.main_app.imgsz, width=10).grid(
            row=2, column=1, padx=5, pady=5, sticky="w")
        
        # Learning rate
        ttk.Label(params_frame, text="Initial Learning Rate:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        lr_entry = ttk.Entry(params_frame, textvariable=self.main_app.lr0, width=10)
        lr_entry.grid(row=3, column=1, padx=5, pady=5, sticky="w")
        
        # Final learning rate factor
        ttk.Label(params_frame, text="Final LR Factor:").grid(row=4, column=0, padx=5, pady=5, sticky="w")
        lrf_entry = ttk.Entry(params_frame, textvariable=self.main_app.lrf, width=10)
        lrf_entry.grid(row=4, column=1, padx=5, pady=5, sticky="w")
        
        # Patience for early stopping
        ttk.Label(params_frame, text="Patience (0 to disable):").grid(row=5, column=0, padx=5, pady=5, sticky="w")
        ttk.Spinbox(params_frame, from_=0, to=200, textvariable=self.main_app.patience, width=10).grid(
            row=5, column=1, padx=5, pady=5, sticky="w")
        
        # Optimizer
        ttk.Label(params_frame, text="Optimizer:").grid(row=6, column=0, padx=5, pady=5, sticky="w")
        optimizer_combo = ttk.Combobox(params_frame, textvariable=self.main_app.optimizer, 
                                      values=["SGD", "Adam", "AdamW", "RMSProp"])
        optimizer_combo.grid(row=6, column=1, padx=5, pady=5, sticky="w")
        
        # YOLOv11 specific parameters section (UI only, these don't affect training)
        yolov11_frame = ttk.LabelFrame(params_frame, text="YOLOv11 Features (Visual Only)")
        yolov11_frame.grid(row=7, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
        
        # Note that these parameters don't actually affect training
        ttk.Label(yolov11_frame, text="Note: These settings are for UI purposes only and don't affect training").grid(
            row=0, column=0, columnspan=2, padx=5, pady=2, sticky="w")
        
        # Enhanced backbone option for YOLOv11
        self.enhanced_backbone = tk.BooleanVar(value=True)
        ttk.Checkbutton(yolov11_frame, text="Enhanced Backbone", 
                       variable=self.enhanced_backbone).grid(row=1, column=0, padx=5, pady=5, sticky="w")
        
        # Feature fusion level for YOLOv11
        ttk.Label(yolov11_frame, text="Feature Fusion Level:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.fusion_level = tk.IntVar(value=3)
        ttk.Spinbox(yolov11_frame, from_=1, to=5, textvariable=self.fusion_level, width=5).grid(
            row=2, column=1, padx=5, pady=5, sticky="w")
        
        # Add a frame for additional options
        advanced_frame = ttk.LabelFrame(self.tab, text="Advanced Training Options")
        advanced_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Add a checkbox for pretrained weights
        self.pretrained = tk.BooleanVar(value=True)
        ttk.Checkbutton(advanced_frame, text="Use Pretrained Weights", 
                       variable=self.pretrained).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        # Add a checkbox for saving all models
        self.save_period = tk.IntVar(value=0)
        ttk.Label(advanced_frame, text="Save checkpoint every:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        ttk.Spinbox(advanced_frame, from_=0, to=100, textvariable=self.save_period, width=5).grid(
            row=1, column=1, padx=5, pady=5, sticky="w")
        ttk.Label(advanced_frame, text="epochs (0 to disable)").grid(row=1, column=2, padx=5, pady=5, sticky="w")
        
        # Add device selection - UPDATED for GPU
        ttk.Label(advanced_frame, text="Device:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        
        # Set default device based on GPU availability
        default_device = "0" if torch.cuda.is_available() else "cpu"
        self.device = tk.StringVar(value=default_device)
        
        # Create device options
        device_options = ["cpu"]
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            # Add single GPU options
            for i in range(gpu_count):
                device_options.insert(0, str(i))  # Add at the beginning to prioritize GPUs
            # Add multi-GPU options if more than one GPU
            if gpu_count > 1:
                device_options.append(",".join(str(i) for i in range(gpu_count)))
                device_options.append("0,1")  # Common dual GPU setup
        device_options.append("auto")  # Add auto as the last option
        
        device_combo = ttk.Combobox(advanced_frame, textvariable=self.device, values=device_options)
        device_combo.grid(row=2, column=1, columnspan=2, padx=5, pady=5, sticky="w")
        
        # Add GPU info label
        gpu_info = "GPU available: Yes" if torch.cuda.is_available() else "GPU available: No (using CPU)"
        ttk.Label(advanced_frame, text=gpu_info, foreground="green" if torch.cuda.is_available() else "red").grid(
            row=2, column=3, padx=5, pady=5, sticky="w")
        
        # Add batch size
        ttk.Label(advanced_frame, text="Batch Size:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.batch_size = tk.IntVar(value=16)
        ttk.Spinbox(advanced_frame, from_=1, to=128, textvariable=self.batch_size, width=5).grid(
            row=3, column=1, padx=5, pady=5, sticky="w")
        
        # Start training button with some styling
        style = ttk.Style()
        style.configure("Train.TButton", font=("Arial", 10, "bold"))
        train_button = ttk.Button(
            self.tab, 
            text="Start Training", 
            command=self.main_app.start_training,
            style="Train.TButton"
        )
        train_button.pack(pady=20)
    
    def browse_model_file(self):
        """Allow the user to browse for a model file"""
        file_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("PyTorch Models", "*.pt"), ("All files", "*.*")]
        )
        if file_path:
            self.main_app.model_path.set(file_path)
            self.main_app.update_status(f"Selected model: {file_path}")
    
    def get_advanced_options(self):
        """Return a dictionary of advanced training options"""
        # Only include supported parameters
        options = {
            'pretrained': self.pretrained.get(),
            'save_period': self.save_period.get(),
            'device': self.device.get(),
            'batch': self.batch_size.get()
        }
        
        # Note: We no longer pass the YOLOv11-specific parameters to the training function
        # as they are not supported by Ultralytics
        
        return options