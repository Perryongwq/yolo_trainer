import os
import sys
import traceback
import torch
import io
import threading
import time
from contextlib import redirect_stdout, redirect_stderr

# CRITICAL: Set matplotlib backend BEFORE importing YOLO
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode

from ultralytics import YOLO

class CustomStream(io.StringIO):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def write(self, text):
        super().write(text)
        if text.strip():  # Only process non-empty text
            self.callback(text)
        
    def flush(self):
        super().flush()

class TrainingManager:
    def __init__(self, main_app):
        self.main_app = main_app
        self.stop_event = threading.Event()
        
    def run_training(self, advanced_options=None):
        try:
            # Ensure matplotlib is using non-interactive backend
            matplotlib.use('Agg')
            plt.ioff()
            
            self.main_app.update_status("Initializing training...")
            
            # Get model path
            model_path = self.main_app.model_path.get()
            
            self.main_app.update_status(f"Loading {model_path}...")
            
            # Resolve model path - check multiple possible locations
            resolved_path = self.resolve_model_path(model_path)
            
            if resolved_path:
                self.main_app.update_status(f"Found model at: {resolved_path}")
                model_path = resolved_path
            else:
                self.main_app.update_status(f"Warning: Model file not found in searched paths.")
                self.main_app.update_status(f"Searched in: {os.getcwd()}")
                self.main_app.update_status(f"Will try to let YOLO handle downloading...")
            
            # GPU/CUDA checks
            has_cuda = torch.cuda.is_available()
            device = advanced_options.get('device', 'auto') if advanced_options else 'auto'
            
            # Check device compatibility and adjust if needed
            if device != 'cpu' and not has_cuda:
                self.main_app.update_status(f"WARNING: Device '{device}' requested but CUDA is not available.")
                self.main_app.update_status("Switching to CPU. This will make training slower.")
                device = 'cpu'
                if advanced_options:
                    advanced_options['device'] = 'cpu'
            elif device == 'auto' and not has_cuda:
                self.main_app.update_status("Device 'auto' selected but CUDA is not available. Using CPU.")
                device = 'cpu'
                if advanced_options:
                    advanced_options['device'] = 'cpu'
            elif has_cuda and device.isdigit():
                gpu_id = int(device)
                if gpu_id >= torch.cuda.device_count():
                    self.main_app.update_status(f"WARNING: GPU {gpu_id} requested but only {torch.cuda.device_count()} GPUs available.")
                    self.main_app.update_status(f"Using GPU 0 instead.")
                    device = '0'
                    if advanced_options:
                        advanced_options['device'] = '0'
            
            # Initialize YOLO model
            model = YOLO(model_path)
            
            # Prepare training parameters - ONLY INCLUDE SUPPORTED PARAMETERS
            train_args = {
                'data': self.main_app.yaml_path.get(),
                'epochs': self.main_app.epochs.get(),
                'imgsz': self.main_app.imgsz.get(),
                'lr0': self.main_app.lr0.get(),
                'lrf': self.main_app.lrf.get(),
                'patience': self.main_app.patience.get(),
                'optimizer': self.main_app.optimizer.get(),
                'verbose': True,  # Ensure verbose output
                'plots': True,    # Keep plots enabled, but they'll be saved to disk
            }
            
            # Add advanced options if provided
            if advanced_options:
                # Common advanced options - ONLY ADD SUPPORTED PARAMETERS
                train_args.update({
                    'pretrained': advanced_options.get('pretrained', True),
                    'device': advanced_options.get('device', 'cpu'),
                    'batch': advanced_options.get('batch', 16),
                    'save_period': advanced_options.get('save_period', 0)
                })
            
            # Log training parameters
            self.main_app.update_status("Starting training with parameters:")
            for key, value in train_args.items():
                self.main_app.update_status(f"  {key}: {value}")
            
            # Log device information
            if device == 'cpu':
                self.main_app.update_status("Training on CPU")
            elif has_cuda:
                if device == '0' or device == 0:
                    gpu_name = torch.cuda.get_device_name(0)
                    self.main_app.update_status(f"Training on GPU: {gpu_name}")
                elif ',' in str(device):
                    self.main_app.update_status(f"Training on multiple GPUs: {device}")
                else:
                    self.main_app.update_status(f"Training on device: {device}")
            
            # Create a custom stream to capture stdout and stderr
            output_stream = CustomStream(lambda text: self.main_app.update_status(text.rstrip()))
            
            # Redirect stdout and stderr to our custom stream to capture all output
            with redirect_stdout(output_stream), redirect_stderr(output_stream):
                # Start training
                self.main_app.update_status("--- TRAINING OUTPUT ---")
                results = model.train(**train_args)
            
            self.main_app.update_status("--- END OF TRAINING OUTPUT ---")
            self.main_app.update_status("Training complete!")
            self.main_app.update_status("Check the 'runs/detect/train' folder for results.")
            
        except Exception as e:
            self.main_app.update_status(f"Error during training: {str(e)}")
            self.main_app.update_status(f"Current working directory: {os.getcwd()}")
            
            # Print detailed debug information for troubleshooting
            self.main_app.update_status(f"Exception details: {traceback.format_exc()}")
            
            # Provide specific guidance for common errors
            error_str = str(e).lower()
            if "cuda" in error_str:
                self.main_app.update_status("CUDA/GPU ERROR: There seems to be an issue with your GPU setup.")
                self.main_app.update_status("Try the following:")
                self.main_app.update_status("1. Set device to 'cpu' in Advanced Options")
                self.main_app.update_status("2. Check if your GPU drivers are up to date")
                self.main_app.update_status("3. Ensure PyTorch is installed with CUDA support")
                self.main_app.update_status("4. Run 'nvidia-smi' in command prompt to check GPU status")
            elif "out of memory" in error_str:
                self.main_app.update_status("GPU MEMORY ERROR: Your GPU ran out of memory.")
                self.main_app.update_status("Try the following:")
                self.main_app.update_status("1. Reduce batch size in Advanced Options")
                self.main_app.update_status("2. Use a smaller model (e.g., yolov8n instead of yolov8x)")
                self.main_app.update_status("3. Reduce image size")
            elif "main thread is not in main loop" in error_str:
                self.main_app.update_status("THREADING ERROR: Matplotlib backend conflict.")
                self.main_app.update_status("This should be fixed by setting the Agg backend.")
                self.main_app.update_status("If the error persists, try disabling plots with 'plots': False")
            
        finally:
            self.main_app.training_in_progress = False
            self.stop_event.set()
            # Ensure any remaining plots are closed safely
            try:
                plt.close('all')
            except:
                pass

    def resolve_model_path(self, model_name):
        """
        Try to find the model file in various possible locations.
        Also handles yolo11/yolov11 naming variants.
        """
        # Check for both naming formats
        model_names = [model_name]
        
        # Add alternate model name if applicable
        if "yolov11" in model_name.lower():
            model_names.append(model_name.lower().replace("yolov11", "yolo11"))
        elif "yolo11" in model_name.lower():
            model_names.append(model_name.lower().replace("yolo11", "yolov11"))
        
        # List of possible locations to check for each model name
        for name in model_names:
            self.main_app.update_status(f"Checking for model name: {name}")
            
            possible_paths = [
                # Exact path provided
                name,
                # Current working directory
                os.path.join(os.getcwd(), name),
                # Script directory
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", name),
                # Script directory parent
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", name),
                # Models subdirectory
                os.path.join(os.getcwd(), "models", name),
                # Weights subdirectory
                os.path.join(os.getcwd(), "weights", name),
            ]
            
            # Check if any of these paths exist
            for path in possible_paths:
                self.main_app.update_status(f"Checking for model at: {path}")
                if os.path.exists(path):
                    return os.path.abspath(path)
        
        return None