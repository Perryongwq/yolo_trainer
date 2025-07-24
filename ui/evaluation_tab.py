import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import cv2
import numpy as np
from ultralytics import YOLO
import random
from datetime import datetime

class EvaluationTab:
    def __init__(self, parent, main_app):
        self.main_app = main_app
        
        # Create tab
        self.tab = ttk.Frame(parent)
        parent.add(self.tab, text="Model Evaluation")
        
        # Model related variables
        self.model = None
        self.model_path = None
        self.class_names = []
        self.current_image_path = None
        self.batch_results = []
        self.confidence_threshold = tk.DoubleVar(value=0.4)  # Default 0.4 like Flask
        self.conf_display = tk.StringVar(value="0.40")
        self.show_labels = tk.BooleanVar(value=True)
        self.show_conf = tk.BooleanVar(value=True)
        self.show_measurements = tk.BooleanVar(value=True)
        
        # Class colors (for consistent coloring)
        self.class_colors = {}
        
        # Measurement constants (from Flask code)
        self.MICRONS_PER_PIXEL = 3.3
        self.BLOCK1_OFFSET = 0.0
        self.BLOCK2_OFFSET = 0.0
        self.judgment_criteria = {"good": 10, "acceptable": 20}
        
        # Special class names for edge detection (exactly from Flask)
        self.class_names_15type = ["block1_edge15", "block2_edge15", "block1_15", "block2_15", "cal_mark"]
        
        # Build UI components
        self.create_evaluation_tab()
        
        # Add trace to update confidence display when slider changes
        self.confidence_threshold.trace_add("write", self.update_conf_display)
    
    def update_conf_display(self, *args):
        """Update the confidence threshold display value"""
        self.conf_display.set(f"{self.confidence_threshold.get():.2f}")
    
    def create_evaluation_tab(self):
        # Main frame
        main_frame = ttk.Frame(self.tab)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Top control panel
        control_frame = ttk.LabelFrame(main_frame, text="Evaluation Controls")
        control_frame.pack(fill="x", padx=5, pady=5)
        
        # Create a 2x4 grid for controls
        for i in range(2):
            control_frame.columnconfigure(i, weight=1)
        
        # Model selection
        ttk.Label(control_frame, text="Model:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.model_entry = ttk.Entry(control_frame, width=40)
        self.model_entry.grid(row=0, column=1, padx=5, pady=5, sticky="we")
        
        # If model is already loaded in the training tab, use that
        if hasattr(self.main_app, 'model_path') and self.main_app.model_path.get():
            self.model_entry.insert(0, self.main_app.model_path.get())
        
        ttk.Button(control_frame, text="Browse", command=self.browse_model).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(control_frame, text="Load Model", command=self.load_model).grid(row=0, column=3, padx=5, pady=5)
        
        # Confidence threshold
        ttk.Label(control_frame, text="Confidence:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        ttk.Scale(control_frame, from_=0.0, to=1.0, variable=self.confidence_threshold, 
                 orient="horizontal").grid(row=1, column=1, padx=5, pady=5, sticky="we")
        
        ttk.Label(control_frame, textvariable=self.conf_display, width=4).grid(
            row=1, column=2, padx=5, pady=5, sticky="w")
        
        # Display options
        display_frame = ttk.Frame(control_frame)
        display_frame.grid(row=1, column=3, padx=5, pady=5, sticky="w")
        ttk.Checkbutton(display_frame, text="Show Labels", variable=self.show_labels).pack(side="left", padx=2)
        ttk.Checkbutton(display_frame, text="Show Confidence", variable=self.show_conf).pack(side="left", padx=2)
        ttk.Checkbutton(display_frame, text="Show Measurements", variable=self.show_measurements).pack(side="left", padx=2)
        
        # Notebook for single image vs batch evaluation
        self.eval_notebook = ttk.Notebook(main_frame)
        self.eval_notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Single image evaluation tab
        self.single_frame = ttk.Frame(self.eval_notebook)
        self.eval_notebook.add(self.single_frame, text="Single Image")
        self.create_single_image_tab(self.single_frame)
        
        # Batch evaluation tab
        self.batch_frame = ttk.Frame(self.eval_notebook)
        self.eval_notebook.add(self.batch_frame, text="Batch Evaluation")
        self.create_batch_evaluation_tab(self.batch_frame)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready. Load a model to start evaluation.")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill="x", padx=5, pady=5)
    
    def create_single_image_tab(self, parent):
        # Left side - Image selection and controls
        left_frame = ttk.Frame(parent)
        left_frame.pack(side="left", fill="y", padx=5, pady=5)
        
        # Image selection
        select_frame = ttk.LabelFrame(left_frame, text="Image Selection")
        select_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(select_frame, text="Browse Image", command=self.browse_image).pack(fill="x", padx=5, pady=5)
        ttk.Button(select_frame, text="Evaluate Image", command=self.evaluate_single_image).pack(fill="x", padx=5, pady=5)
        
        # Measurement settings
        measurement_frame = ttk.LabelFrame(left_frame, text="Measurement Settings")
        measurement_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(measurement_frame, text="Microns/Pixel:").pack(anchor="w", padx=5, pady=2)
        self.microns_var = tk.DoubleVar(value=self.MICRONS_PER_PIXEL)
        ttk.Entry(measurement_frame, textvariable=self.microns_var, width=10).pack(fill="x", padx=5, pady=2)
        
        ttk.Label(measurement_frame, text="Block 1 Offset:").pack(anchor="w", padx=5, pady=2)
        self.block1_offset_var = tk.DoubleVar(value=self.BLOCK1_OFFSET)
        ttk.Entry(measurement_frame, textvariable=self.block1_offset_var, width=10).pack(fill="x", padx=5, pady=2)
        
        ttk.Label(measurement_frame, text="Block 2 Offset:").pack(anchor="w", padx=5, pady=2)
        self.block2_offset_var = tk.DoubleVar(value=self.BLOCK2_OFFSET)
        ttk.Entry(measurement_frame, textvariable=self.block2_offset_var, width=10).pack(fill="x", padx=5, pady=2)
        
        # Right side - Image display
        right_frame = ttk.Frame(parent)
        right_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)
        
        # Image display
        image_frame = ttk.LabelFrame(right_frame, text="Image Preview")
        image_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Canvas for image display
        self.image_canvas = tk.Canvas(image_frame, bg="lightgray")
        self.image_canvas.pack(fill="both", expand=True)
        
        # Results display below the image
        results_frame = ttk.LabelFrame(right_frame, text="Detection Results")
        results_frame.pack(fill="x", padx=5, pady=5)
        
        # Results as a scrollable text widget
        self.results_text = tk.Text(results_frame, height=8, width=40, wrap=tk.WORD)
        self.results_text.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        results_scrollbar = ttk.Scrollbar(results_frame, command=self.results_text.yview)
        results_scrollbar.pack(side="right", fill="y")
        self.results_text.config(yscrollcommand=results_scrollbar.set)
    
    def create_batch_evaluation_tab(self, parent):
        # Top controls
        controls_frame = ttk.Frame(parent)
        controls_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(controls_frame, text="Images Folder:").pack(side="left", padx=5, pady=5)
        self.batch_folder_var = tk.StringVar()
        ttk.Entry(controls_frame, textvariable=self.batch_folder_var, width=40).pack(side="left", fill="x", expand=True, padx=5, pady=5)
        ttk.Button(controls_frame, text="Browse", command=self.browse_batch_folder).pack(side="left", padx=5, pady=5)
        ttk.Button(controls_frame, text="Evaluate Batch", command=self.evaluate_batch).pack(side="left", padx=5, pady=5)
        
        # Results display
        results_frame = ttk.LabelFrame(parent, text="Batch Results")
        results_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create a treeview for batch results
        columns = ("Image", "Objects", "Classes", "Y-Diff (microns)", "Judgment", "Time")
        self.batch_tree = ttk.Treeview(results_frame, columns=columns, show="headings")
        
        # Configure column headings
        for col in columns:
            self.batch_tree.heading(col, text=col)
            if col == "Image":
                width = 150
            elif col in ["Y-Diff (microns)", "Judgment"]:
                width = 120
            else:
                width = 100
            self.batch_tree.column(col, width=width)
        
        self.batch_tree.pack(side="left", fill="both", expand=True)
        
        # Add scrollbars
        y_scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.batch_tree.yview)
        y_scrollbar.pack(side="right", fill="y")
        self.batch_tree.configure(yscrollcommand=y_scrollbar.set)
        
        # Bind double-click to view the image
        self.batch_tree.bind("<Double-1>", self.view_batch_image)
        
        # Image viewer for batch results
        self.batch_viewer_frame = ttk.LabelFrame(parent, text="Selected Image Preview")
        self.batch_viewer_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.batch_canvas = tk.Canvas(self.batch_viewer_frame, bg="lightgray")
        self.batch_canvas.pack(fill="both", expand=True)
    
    def browse_model(self):
        """Browse for a model file"""
        file_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("PyTorch Models", "*.pt"), ("All files", "*.*")]
        )
        if file_path:
            self.model_entry.delete(0, tk.END)
            self.model_entry.insert(0, file_path)
    
    def load_model(self):
        """Load the selected model"""
        model_path = self.model_entry.get()
        if not model_path:
            messagebox.showerror("Error", "Please select a model file")
            return
        
        try:
            self.status_var.set(f"Loading model: {os.path.basename(model_path)}...")
            self.model = YOLO(model_path)
            self.model_path = model_path
            
            # Try to get class names from model
            try:
                self.class_names = self.model.names
            except:
                # Use the 15type class names as default
                self.class_names = {i: self.class_names_15type[i] if i < len(self.class_names_15type) else f"Class {i}" 
                                  for i in range(1000)}
            
            # Generate consistent colors for each class
            self.generate_class_colors()
            
            self.status_var.set(f"Model loaded: {os.path.basename(model_path)}")
            messagebox.showinfo("Success", "Model loaded successfully")
        except Exception as e:
            self.status_var.set(f"Error loading model: {str(e)}")
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    def generate_class_colors(self):
        """Generate consistent colors for each class"""
        random.seed(42)  # For reproducible colors
        self.class_colors = {}
        
        # Set specific colors for edge classes (matching Flask)
        specific_colors = {
            "block1_edge15": (255, 0, 0),      # Blue
            "block2_edge15": (0, 255, 255),    # Cyan
            "block1_15": (0, 255, 0),          # Green
            "block2_15": (255, 255, 0),        # Yellow
            "cal_mark": (255, 0, 255)          # Magenta
        }
        
        for cls_id, class_name in self.class_names.items():
            if class_name in specific_colors:
                self.class_colors[cls_id] = specific_colors[class_name]
            else:
                # Generate random color for other classes
                self.class_colors[cls_id] = (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255)
                )
    
    def browse_image(self):
        """Browse for a single image to evaluate"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"), 
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path, self.image_canvas)
            self.status_var.set(f"Image loaded: {os.path.basename(file_path)}")
    
    def browse_batch_folder(self):
        """Browse for a folder containing images for batch evaluation"""
        folder_path = filedialog.askdirectory(title="Select Images Folder")
        if folder_path:
            self.batch_folder_var.set(folder_path)
            self.status_var.set(f"Folder selected: {folder_path}")
    
    def process_detections_flask_style(self, image, results, ratio=1.0):
        """Process detections following Flask code logic exactly"""
        # Initialize variables (Flask style)
        block1_edge_y = block2_edge_y = None
        block1_box_y = block2_box_y = None
        calibration_marker_width_px = None
        microns_per_pixel = self.microns_var.get()
        
        # Process results like Flask code
        if len(results) > 0 and hasattr(results[0].boxes, 'xywh') and hasattr(results[0].boxes, 'cls'):
            for box, cls in zip(results[0].boxes.xywh, results[0].boxes.cls):
                # Get box properties
                x_center, y_center, width, height = box
                label = self.class_names.get(int(cls.item()), f"Class {int(cls.item())}")
                
                # Scale coordinates to resized image
                x_center = int(x_center * ratio)
                y_center = int(y_center * ratio)
                width = width * ratio
                height = height * ratio
                
                print(f"[DEBUG] cls: {cls}, index: {int(cls.item())}, label: {label}")
                
                # Process according to label (exactly like Flask)
                if label == "block1_edge15":
                    edge_y = int(y_center + height / 2)
                    block1_edge_y = edge_y + (self.block1_offset_var.get() / microns_per_pixel)
                    cv2.line(image, (int(x_center - 150), edge_y), (int(x_center + 150), edge_y), (255, 0, 0), 2)
                
                elif label == "block2_edge15":
                    edge_y = int(y_center + height / 2)
                    block2_edge_y = edge_y + (self.block2_offset_var.get() / microns_per_pixel)
                    cv2.line(image, (int(x_center - 150), edge_y), (int(x_center + 150), edge_y), (0, 255, 255), 2)
                
                elif label == "block1_15":
                    block1_box_y = int(y_center + height / 2)
                
                elif label == "block2_15":
                    block2_box_y = int(y_center + height / 2)
                
                elif label == "cal_mark":
                    calibration_marker_width_px = width.item()
        
        # Update microns per pixel if calibration mark found
        if calibration_marker_width_px and calibration_marker_width_px > 0:
            microns_per_pixel = 1000.0 / calibration_marker_width_px
            print(f"[Calibration] cal_mark width = {calibration_marker_width_px:.2f}px, microns/px = {microns_per_pixel:.2f}")
        
        return block1_edge_y, block2_edge_y, block1_box_y, block2_box_y, microns_per_pixel
    
    def display_image(self, image_path, canvas, detections=None):
        """Display an image on the specified canvas with optional detection boxes and measurements"""
        try:
            # Clear canvas
            canvas.delete("all")
            
            # Open and resize image to fit canvas
            img = Image.open(image_path)
            canvas_width = canvas.winfo_width() or 600
            canvas_height = canvas.winfo_height() or 400
            
            # Calculate resize ratio
            img_width, img_height = img.size
            width_ratio = canvas_width / img_width
            height_ratio = canvas_height / img_height
            ratio = min(width_ratio, height_ratio)
            
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)
            
            img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # If there are detections, draw boxes and measurements
            if detections is not None:
                # Convert PIL image to OpenCV format for drawing
                img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                
                # Process detections Flask-style
                block1_edge_y, block2_edge_y, block1_box_y, block2_box_y, microns_per_pixel = \
                    self.process_detections_flask_style(img_cv, detections, ratio)
                
                # Draw bounding boxes for all detections
                if len(detections) > 0:
                    for result in detections:
                        if hasattr(result.boxes, 'xyxy'):
                            for i in range(len(result.boxes)):
                                conf = result.boxes.conf[i].item()
                                
                                # Skip if confidence is below threshold
                                if conf < self.confidence_threshold.get():
                                    continue
                                
                                box = result.boxes.xyxy[i].tolist()
                                cls = int(result.boxes.cls[i].item())
                                
                                # Scale box to resized image
                                x1, y1, x2, y2 = [int(coord * ratio) for coord in box]
                                
                                # Get class name and color
                                class_name = self.class_names.get(cls, f"Class {cls}")
                                color = self.class_colors.get(cls, (0, 255, 0))
                                
                                # Draw bounding box
                                cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
                                
                                # Draw label if enabled
                                if self.show_labels.get():
                                    label = class_name
                                    if self.show_conf.get():
                                        label += f" {conf:.2f}"
                                    
                                    # Draw background for text
                                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                                    cv2.rectangle(img_cv, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
                                    
                                    # Draw text
                                    cv2.putText(img_cv, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
                # Draw measurements if both edges were detected (Flask logic)
                if self.show_measurements.get() and block1_edge_y is not None and block2_edge_y is not None:
                    # Calculate Y difference
                    y_diff_pixels = block1_edge_y - block2_edge_y
                    y_diff_microns = y_diff_pixels * microns_per_pixel
                    
                    # Determine judgment
                    if y_diff_microns < self.judgment_criteria["good"]:
                        judgment = "Good"
                        judgment_color = (0, 255, 0)
                    elif y_diff_microns < self.judgment_criteria["acceptable"]:
                        judgment = "Acceptable"
                        judgment_color = (0, 165, 255)
                    else:
                        judgment = "No Good"
                        judgment_color = (0, 0, 255)
                    
                    # Draw measurement text (Flask style)
                    text_x = img_cv.shape[1] // 2
                    text_y = int((block1_edge_y + block2_edge_y) / 2)
                    
                    cv2.putText(img_cv, f"{y_diff_microns:.2f} microns", 
                               (text_x - 100, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(img_cv, f"Judgment: {judgment}", 
                               (text_x - 100, text_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, judgment_color, 2)
                    
                    # Add timestamp
                    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    cv2.putText(img_cv, f"Checked on: {current_datetime}", 
                               (10, img_cv.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
                
                # Convert back to PIL for display
                img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            
            # Convert to PhotoImage for canvas
            self.tk_img = ImageTk.PhotoImage(img)
            
            # Add to canvas
            canvas.create_image(canvas_width // 2, canvas_height // 2, image=self.tk_img, anchor=tk.CENTER)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display image: {str(e)}")
    
    def evaluate_single_image(self):
        """Run inference on a single image"""
        if not self.model:
            messagebox.showerror("Error", "Please load a model first")
            return
        
        if not self.current_image_path:
            messagebox.showerror("Error", "Please select an image first")
            return
        
        try:
            self.status_var.set("Running inference...")
            
            # Run inference (Flask uses conf=0.4 by default)
            results = self.model.predict(source=self.current_image_path, conf=self.confidence_threshold.get(), save=False)
            
            # Display image with detections
            self.display_image(self.current_image_path, self.image_canvas, results)
            
            # Display results in text box
            self.results_text.delete(1.0, tk.END)
            
            # Check detection status
            block1_edge_found = False
            block2_edge_found = False
            cal_mark_found = False
            y_diff_microns = None
            judgment = None
            
            # Analyze results
            if len(results) > 0 and hasattr(results[0].boxes, 'cls'):
                class_counts = {}
                
                for cls in results[0].boxes.cls:
                    class_name = self.class_names.get(int(cls.item()), f"Class {int(cls.item())}")
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    
                    if class_name == "block1_edge15":
                        block1_edge_found = True
                    elif class_name == "block2_edge15":
                        block2_edge_found = True
                    elif class_name == "cal_mark":
                        cal_mark_found = True
                
                # Display detection counts
                self.results_text.insert(tk.END, f"Found {len(results[0].boxes)} objects:\n\n")
                
                # Display by class
                for class_name, count in class_counts.items():
                    self.results_text.insert(tk.END, f"- {class_name}: {count}\n")
                
                # Calculate measurement if both edges found
                if block1_edge_found and block2_edge_found:
                    # Re-process to get measurements
                    img = cv2.imread(self.current_image_path)
                    block1_edge_y, block2_edge_y, _, _, microns_per_pixel = \
                        self.process_detections_flask_style(img, results, 1.0)
                    
                    if block1_edge_y is not None and block2_edge_y is not None:
                        y_diff_pixels = block1_edge_y - block2_edge_y
                        y_diff_microns = y_diff_pixels * microns_per_pixel
                        
                        if y_diff_microns < self.judgment_criteria["good"]:
                            judgment = "Good"
                        elif y_diff_microns < self.judgment_criteria["acceptable"]:
                            judgment = "Acceptable"
                        else:
                            judgment = "No Good"
                
                # Display measurement status
                self.results_text.insert(tk.END, "\nMeasurement Status:\n")
                self.results_text.insert(tk.END, f"- Block 1 Edge: {'Found' if block1_edge_found else 'Not Found'}\n")
                self.results_text.insert(tk.END, f"- Block 2 Edge: {'Found' if block2_edge_found else 'Not Found'}\n")
                self.results_text.insert(tk.END, f"- Calibration Mark: {'Found' if cal_mark_found else 'Not Found'}\n")
                
                if block1_edge_found and block2_edge_found:
                    if y_diff_microns is not None:
                        self.results_text.insert(tk.END, f"\nMeasurement Result:\n")
                        self.results_text.insert(tk.END, f"- Y-Difference: {y_diff_microns:.2f} microns\n")
                        self.results_text.insert(tk.END, f"- Judgment: {judgment}\n")
                else:
                    self.results_text.insert(tk.END, "\n⚠ Not able to detect one or more edges.\n")
                    self.results_text.insert(tk.END, "Manual line drawing would be required.\n")
            else:
                self.results_text.insert(tk.END, "No objects detected.")
            
            self.status_var.set("Evaluation complete")
            
        except Exception as e:
            self.status_var.set(f"Error during evaluation: {str(e)}")
            messagebox.showerror("Error", f"Failed to evaluate image: {str(e)}")
    
    def evaluate_batch(self):
        """Run inference on a batch of images"""
        if not self.model:
            messagebox.showerror("Error", "Please load a model first")
            return
        
        folder_path = self.batch_folder_var.get()
        if not folder_path or not os.path.isdir(folder_path):
            messagebox.showerror("Error", "Please select a valid folder")
            return
        
        # Get image files
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
        image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                     if f.lower().endswith(image_extensions)]
        
        if not image_files:
            messagebox.showerror("Error", "No image files found in the selected folder")
            return
        
        # Clear previous results
        for item in self.batch_tree.get_children():
            self.batch_tree.delete(item)
        
        self.batch_results = []
        
        # Start batch processing in a separate thread
        threading.Thread(
            target=self._batch_processing_thread,
            args=(image_files,),
            daemon=True
        ).start()
    
    def _batch_processing_thread(self, image_files):
        """Process batch of images in a separate thread"""
        total_images = len(image_files)
        self.status_var.set(f"Processing {total_images} images...")
        
        for i, img_path in enumerate(image_files):
            try:
                # Update status
                self.status_var.set(f"Processing image {i+1}/{total_images}: {os.path.basename(img_path)}")
                
                # Run inference
                import time
                start_time = time.time()
                results = self.model.predict(source=img_path, conf=self.confidence_threshold.get(), save=False)
                elapsed = time.time() - start_time
                
                # Analyze results
                num_objects = len(results[0].boxes) if len(results) > 0 and hasattr(results[0].boxes, 'cls') else 0
                classes = {}
                block1_found = False
                block2_found = False
                y_diff_microns = None
                judgment = "N/A"
                
                if num_objects > 0:
                    for cls in results[0].boxes.cls:
                        class_name = self.class_names.get(int(cls.item()), f"Class {int(cls.item())}")
                        classes[class_name] = classes.get(class_name, 0) + 1
                        
                        if class_name == "block1_edge15":
                            block1_found = True
                        elif class_name == "block2_edge15":
                            block2_found = True
                    
                    # Calculate measurement if both edges found
                    if block1_found and block2_found:
                        img = cv2.imread(img_path)
                        block1_edge_y, block2_edge_y, _, _, microns_per_pixel = \
                            self.process_detections_flask_style(img, results, 1.0)
                        
                        if block1_edge_y is not None and block2_edge_y is not None:
                            y_diff_pixels = block1_edge_y - block2_edge_y
                            y_diff_microns = y_diff_pixels * microns_per_pixel
                            
                            if y_diff_microns < self.judgment_criteria["good"]:
                                judgment = "Good"
                            elif y_diff_microns < self.judgment_criteria["acceptable"]:
                                judgment = "Acceptable"
                            else:
                                judgment = "No Good"
                
                classes_str = ", ".join([f"{name} ({count})" for name, count in classes.items()])
                y_diff_str = f"{y_diff_microns:.2f}" if y_diff_microns is not None else "N/A"
                
                # Store result
                result_item = {
                    "path": img_path,
                    "objects": num_objects,
                    "classes": classes_str,
                    "y_diff": y_diff_str,
                    "judgment": judgment,
                    "time": f"{elapsed:.2f}s",
                    "results": results
                }
                self.batch_results.append(result_item)
                
                # Add to treeview
                self.batch_tree.insert("", "end", values=(
                    os.path.basename(img_path),
                    num_objects,
                    classes_str if classes_str else "None",
                    y_diff_str,
                    judgment,
                    f"{elapsed:.2f}s"
                ))
                
                # Update UI
                self.main_app.root.update_idletasks()
                
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
        
        self.status_var.set(f"Batch processing complete. Processed {len(self.batch_results)}/{total_images} images.")
    
    def view_batch_image(self, event):
        """Handle double-click on batch result to view the image"""
        # Get selected item
        selected_item = self.batch_tree.selection()
        if not selected_item:
            return
        
        # Get item index
        item_index = self.batch_tree.index(selected_item[0])
        if item_index < 0 or item_index >= len(self.batch_results):
            return
        
        # Get result
        result = self.batch_results[item_index]
        img_path = result["path"]
        results = result["results"]
        
        # Display the image with detections
        self.display_image(img_path, self.batch_canvas, results)