import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random
from ultralytics import YOLO

# Import SAM if installed
try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False

class AutoAnnotationTab:
    def __init__(self, parent, main_app):
        self.main_app = main_app
        
        # Create tab
        self.tab = ttk.Frame(parent)
        parent.add(self.tab, text="Auto Annotation")
        
        # Variables
        self.detector_model = None  # YOLO model for detection
        self.sam_model = None  # SAM model for segmentation
        self.sam_predictor = None
        self.current_image_path = None
        self.current_image = None
        self.image_folder = None
        self.image_files = []
        self.current_image_index = 0
        self.annotations = {}  # Dictionary to store annotations for each image
        self.confidence_threshold = tk.DoubleVar(value=0.25)
        self.annotation_mode = tk.StringVar(value="yolo")  # "yolo", "sam", or "hybrid"
        
        # Drawing variables
        self.drawing = False
        self.rect = None
        self.start_x = None
        self.start_y = None
        self.canvas_cid_press = None
        self.canvas_cid_release = None
        self.canvas_cid_motion = None
        
        # UI components
        self.canvas = None
        self.class_checkboxes = {} # Dictionary to hold tk.BooleanVar for each class
        self.class_frame = None # Frame to hold class checkboxes
        self.class_list = None
        self.fig = None
        self.ax = None
        self.drawing_label = None
        
        # Build UI components
        self.create_annotation_tab()
    
    def create_annotation_tab(self):
        # Main layout with left panel and right panel
        self.main_pane = ttk.PanedWindow(self.tab, orient=tk.HORIZONTAL)
        self.main_pane.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Left panel for controls
        left_frame = ttk.Frame(self.main_pane, width=300)
        self.main_pane.add(left_frame, weight=1)
        
        # Right panel for image and annotations
        right_frame = ttk.Frame(self.main_pane)
        self.main_pane.add(right_frame, weight=3)
        
        # Create components for left panel (controls)
        self.create_control_panel(left_frame)
        
        # Create components for right panel (image and annotations)
        self.create_image_panel(right_frame)
    
    def create_control_panel(self, parent):
        # Models section
        models_frame = ttk.LabelFrame(parent, text="Models")
        models_frame.pack(fill="x", padx=5, pady=5)
        
        # YOLO model selection
        ttk.Label(models_frame, text="YOLO Model:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.yolo_model_entry = ttk.Entry(models_frame, width=20)
        self.yolo_model_entry.grid(row=0, column=1, padx=5, pady=5, sticky="we")
        ttk.Button(models_frame, text="Browse", command=self.browse_yolo_model).grid(row=0, column=2, padx=5, pady=5)
        
        # If model is already loaded in the training tab, use that
        if hasattr(self.main_app, 'model_path') and self.main_app.model_path.get():
            self.yolo_model_entry.insert(0, self.main_app.model_path.get())
        
        # SAM model selection (if available)
        ttk.Label(models_frame, text="SAM Model:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.sam_model_entry = ttk.Entry(models_frame, width=20)
        self.sam_model_entry.grid(row=1, column=1, padx=5, pady=5, sticky="we")
        ttk.Button(models_frame, text="Browse", command=self.browse_sam_model, 
                         state="normal" if SAM_AVAILABLE else "disabled").grid(row=1, column=2, padx=5, pady=5)
        
        # SAM availability message
        if not SAM_AVAILABLE:
            ttk.Label(models_frame, text="SAM not installed. Run: pip install segment-anything", 
                             foreground="red").grid(row=2, column=0, columnspan=3, padx=5, pady=5)
        
        # Load models button
        ttk.Button(models_frame, text="Load Models", command=self.load_models).grid(
            row=3, column=0, columnspan=3, padx=5, pady=5, sticky="we")
        
        # Dataset section
        dataset_frame = ttk.LabelFrame(parent, text="Dataset")
        dataset_frame.pack(fill="x", padx=5, pady=5)
        
        # Image folder selection
        ttk.Label(dataset_frame, text="Image Folder:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.folder_entry = ttk.Entry(dataset_frame, width=20)
        self.folder_entry.grid(row=0, column=1, padx=5, pady=5, sticky="we")
        ttk.Button(dataset_frame, text="Browse", command=self.browse_image_folder).grid(row=0, column=2, padx=5, pady=5)
        
        # Output folder selection
        ttk.Label(dataset_frame, text="Output Folder:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.output_entry = ttk.Entry(dataset_frame, width=20)
        self.output_entry.grid(row=1, column=1, padx=5, pady=5, sticky="we")
        ttk.Button(dataset_frame, text="Browse", command=self.browse_output_folder).grid(row=1, column=2, padx=5, pady=5)
        
        # Annotation mode
        ttk.Label(dataset_frame, text="Mode:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        modes = ["YOLO Only", "SAM Only", "YOLO+SAM Hybrid"]
        mode_combo = ttk.Combobox(dataset_frame, values=modes, state="readonly", width=15)
        mode_combo.current(0)  # Default to YOLO only
        mode_combo.grid(row=2, column=1, padx=5, pady=5, sticky="we")
        mode_combo.bind("<<ComboboxSelected>>", self.update_annotation_mode)
        
        # Confidence threshold
        ttk.Label(dataset_frame, text="Confidence:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        conf_scale = ttk.Scale(dataset_frame, from_=0.0, to=1.0, variable=self.confidence_threshold, orient="horizontal")
        conf_scale.grid(row=3, column=1, padx=5, pady=5, sticky="we")
        self.conf_label = ttk.Label(dataset_frame, text="0.25")
        self.conf_label.grid(row=3, column=2, padx=5, pady=5)
        conf_scale.configure(command=lambda v: self.conf_label.configure(text=f"{float(v):.2f}"))
        
        # Action buttons
        actions_frame = ttk.LabelFrame(parent, text="Actions")
        actions_frame.pack(fill="x", padx=5, pady=5)
        
        # Navigation buttons
        nav_frame = ttk.Frame(actions_frame)
        nav_frame.pack(fill="x", padx=5, pady=5)
        ttk.Button(nav_frame, text="Previous", command=self.previous_image).pack(side="left", padx=5)
        ttk.Button(nav_frame, text="Next", command=self.next_image).pack(side="left", padx=5)
        ttk.Button(nav_frame, text="Auto-annotate Current", command=self.annotate_current).pack(side="left", padx=5)
        
        # Batch annotation button
        ttk.Button(actions_frame, text="Auto-annotate All Images", command=self.annotate_batch).pack(fill="x", padx=5, pady=5)
        
        # Save annotations button
        ttk.Button(actions_frame, text="Save Annotations", command=self.save_annotations).pack(fill="x", padx=5, pady=5)
        
        # Classes section (Modified for checkboxes)
        self.classes_frame = ttk.LabelFrame(parent, text="Classes to Annotate")
        self.classes_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Frame to contain the checkboxes and scrollbar
        self.class_checkbox_container = ttk.Frame(self.classes_frame)
        self.class_checkbox_container.pack(fill="both", expand=True)

        self.class_canvas = tk.Canvas(self.class_checkbox_container)
        self.class_canvas.pack(side="left", fill="both", expand=True)

        self.class_scrollbar = ttk.Scrollbar(self.class_checkbox_container, orient="vertical", command=self.class_canvas.yview)
        self.class_scrollbar.pack(side="right", fill="y")

        self.class_canvas.configure(yscrollcommand=self.class_scrollbar.set)
        self.class_canvas.bind('<Configure>', lambda e: self.class_canvas.configure(scrollregion = self.class_canvas.bbox("all")))

        self.class_frame = ttk.Frame(self.class_canvas)
        self.class_canvas.create_window((0, 0), window=self.class_frame, anchor="nw")

        # Select/Deselect All buttons
        select_all_frame = ttk.Frame(self.classes_frame)
        select_all_frame.pack(fill="x", padx=5, pady=2)
        ttk.Button(select_all_frame, text="Select All", command=self.select_all_classes).pack(side="left", expand=True)
        ttk.Button(select_all_frame, text="Deselect All", command=self.deselect_all_classes).pack(side="right", expand=True)

        # Status section
        self.status_var = tk.StringVar(value="Ready. Load models and select image folder to start.")
        status_bar = ttk.Label(parent, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill="x", padx=5, pady=5, side="bottom")
    
    def create_image_panel(self, parent):
        # Create a frame for the image display
        image_frame = ttk.LabelFrame(parent, text="Image Preview")
        image_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Drawing instructions label
        self.drawing_label = ttk.Label(image_frame, text="", foreground="blue")
        self.drawing_label.pack(anchor="w", padx=5, pady=2)
        
        # Create a matplotlib figure for image display with annotations
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.canvas = FigureCanvasTkAgg(self.fig, master=image_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Initialize drawing variables
        self.drawing = False
        self.rect = None
        
        # Create a frame for annotation details
        anno_frame = ttk.LabelFrame(parent, text="Annotation Details")
        anno_frame.pack(fill="x", padx=5, pady=5)
        
        # Create a treeview for displaying annotations
        columns = ("Class", "Confidence", "Box")
        self.anno_tree = ttk.Treeview(anno_frame, columns=columns, show="headings", height=6)
        
        # Configure column headings
        for col in columns:
            self.anno_tree.heading(col, text=col)
            self.anno_tree.column(col, width=100)
        
        self.anno_tree.column("Box", width=200)
        
        self.anno_tree.pack(side="left", fill="both", expand=True)
        
        # Add scrollbar to treeview
        anno_scroll = ttk.Scrollbar(anno_frame, orient="vertical", command=self.anno_tree.yview)
        anno_scroll.pack(side="right", fill="y")
        self.anno_tree.configure(yscrollcommand=anno_scroll.set)
        
        # Bind selection to highlight the selected box in the image
        self.anno_tree.bind("<<TreeviewSelect>>", self.highlight_selected_box)
        
        # Create edit/delete buttons
        edit_frame = ttk.Frame(anno_frame)
        edit_frame.pack(fill="x", padx=5, pady=5)
        ttk.Button(edit_frame, text="Delete Selected", command=self.delete_selected_annotation).pack(side="left", padx=5)
        ttk.Button(edit_frame, text="Edit Selected", command=self.edit_selected_annotation).pack(side="left", padx=5)
        ttk.Button(edit_frame, text="Add Manual Box", command=self.add_manual_box).pack(side="left", padx=5)
    
    # Model handling methods
    def browse_yolo_model(self):
        file_path = filedialog.askopenfilename(
            title="Select YOLO Model",
            filetypes=[("PyTorch Models", "*.pt"), ("All files", "*.*")]
        )
        if file_path:
            self.yolo_model_entry.delete(0, tk.END)
            self.yolo_model_entry.insert(0, file_path)
    
    def browse_sam_model(self):
        file_path = filedialog.askopenfilename(
            title="Select SAM Model",
            filetypes=[("PyTorch Models", "*.pth"), ("All files", "*.*")]
        )
        if file_path:
            self.sam_model_entry.delete(0, tk.END)
            self.sam_model_entry.insert(0, file_path)
    
    def load_models(self):
        # Load YOLO model
        yolo_path = self.yolo_model_entry.get()
        if yolo_path:
            try:
                self.status_var.set(f"Loading YOLO model: {os.path.basename(yolo_path)}...")
                self.detector_model = YOLO(yolo_path)
                
                # Populate class list with checkboxes
                self.update_class_checkboxes()
                
                self.status_var.set(f"YOLO model loaded: {os.path.basename(yolo_path)}")
            except Exception as e:
                self.status_var.set(f"Error loading YOLO model: {str(e)}")
                messagebox.showerror("Error", f"Failed to load YOLO model: {str(e)}")
                return False
        
        # Load SAM model if available and selected
        if SAM_AVAILABLE:
            sam_path = self.sam_model_entry.get()
            if sam_path:
                try:
                    self.status_var.set(f"Loading SAM model: {os.path.basename(sam_path)}...")
                    # Determine SAM model type from filename
                    model_type = "vit_h"  # Default to highest quality
                    if "vit_b" in sam_path.lower():
                        model_type = "vit_b"
                    elif "vit_l" in sam_path.lower():
                        model_type = "vit_l"
                    
                    # Load SAM model
                    self.sam_model = sam_model_registry[model_type](checkpoint=sam_path)
                    self.sam_predictor = SamPredictor(self.sam_model)
                    
                    self.status_var.set(f"SAM model loaded: {os.path.basename(sam_path)}")
                except Exception as e:
                    self.status_var.set(f"Error loading SAM model: {str(e)}")
                    messagebox.showerror("Error", f"Failed to load SAM model: {str(e)}")
                    return False
        
        return True
    
    def update_class_checkboxes(self):
        """Populate the class list with checkboxes based on the loaded YOLO model's class names."""
        # Clear existing checkboxes
        for widget in self.class_frame.winfo_children():
            widget.destroy()
        self.class_checkboxes.clear()

        if self.detector_model and hasattr(self.detector_model, 'names'):
            for idx, class_name in self.detector_model.names.items():
                var = tk.BooleanVar(value=True)  # Check all by default
                cb = ttk.Checkbutton(self.class_frame, text=f"{idx}: {class_name}", variable=var)
                cb.pack(anchor="w", padx=2, pady=1)
                self.class_checkboxes[idx] = var
            self.class_canvas.update_idletasks()
            self.class_canvas.config(scrollregion=self.class_canvas.bbox("all"))
        else:
            ttk.Label(self.class_frame, text="Load YOLO model to see classes.").pack(padx=5, pady=5)

    def get_selected_classes(self):
        """Returns a list of class IDs for currently selected checkboxes."""
        selected_classes = []
        for class_id, var in self.class_checkboxes.items():
            if var.get():
                selected_classes.append(class_id)
        return selected_classes

    def select_all_classes(self):
        """Checks all class checkboxes."""
        for var in self.class_checkboxes.values():
            var.set(True)

    def deselect_all_classes(self):
        """Unchecks all class checkboxes."""
        for var in self.class_checkboxes.values():
            var.set(False)

    # Dataset handling methods
    def browse_image_folder(self):
        folder_path = filedialog.askdirectory(title="Select Image Folder")
        if folder_path:
            self.folder_entry.delete(0, tk.END)
            self.folder_entry.insert(0, folder_path)
            
            # Set default output folder
            if not self.output_entry.get():
                output_path = os.path.join(folder_path, "labels")
                self.output_entry.delete(0, tk.END)
                self.output_entry.insert(0, output_path)
            
            # Load image list
            self.load_image_list(folder_path)
    
    def browse_output_folder(self):
        folder_path = filedialog.askdirectory(title="Select Output Folder")
        if folder_path:
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, folder_path)
    
    def load_image_list(self, folder_path):
        # Get all image files in the folder
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
        self.image_files = [os.path.join(folder_path, f) for f in sorted(os.listdir(folder_path)) 
                             if f.lower().endswith(image_extensions)]
        
        if not self.image_files:
            messagebox.showerror("Error", "No image files found in the selected folder")
            return
        
        self.image_folder = folder_path
        self.current_image_index = 0
        
        # Load first image
        self.load_image(self.image_files[0])
        
        # Initialize annotations dictionary
        for img_path in self.image_files:
            if img_path not in self.annotations:
                self.annotations[img_path] = []
            
            # Check if annotation file exists
            label_path = self.get_label_path(img_path)
            if os.path.exists(label_path):
                self.load_existing_annotations(img_path, label_path)
        
        self.status_var.set(f"Loaded {len(self.image_files)} images from {folder_path}")
    
    def get_label_path(self, img_path):
        """Convert image path to label path"""
        output_folder = self.output_entry.get()
        if not output_folder:
            # If no output folder specified, use same folder as image
            output_folder = os.path.dirname(img_path)
        
        # Get filename without extension and add .txt extension
        filename = os.path.basename(img_path)
        base_filename = os.path.splitext(filename)[0]
        return os.path.join(output_folder, f"{base_filename}.txt")
    
    def load_existing_annotations(self, img_path, label_path):
        """Load existing YOLO format annotations"""
        try:
            # Get image dimensions
            img = cv2.imread(img_path)
            height, width = img.shape[:2]
            
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            annotations = []
            for line in lines:
                # Parse YOLO format: class_id x_center y_center width height
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    # Convert normalized coordinates to pixel coordinates
                    x_center = float(parts[1]) * width
                    y_center = float(parts[2]) * height
                    w = float(parts[3]) * width
                    h = float(parts[4]) * height
                    
                    # Convert to xmin, ymin, xmax, ymax
                    xmin = int(x_center - w/2)
                    ymin = int(y_center - h/2)
                    xmax = int(x_center + w/2)
                    ymax = int(y_center + h/2)
                    
                    confidence = 1.0  # Assume 100% confidence for manual annotations
                    
                    annotations.append({
                        'class_id': class_id,
                        'confidence': confidence,
                        'bbox': [xmin, ymin, xmax, ymax]
                    })
            
            self.annotations[img_path] = annotations
            
        except Exception as e:
            print(f"Error loading annotations for {img_path}: {str(e)}")
    
    def load_image(self, img_path):
        """Load and display an image"""
        try:
            self.current_image_path = img_path
            self.current_image = cv2.imread(img_path)
            self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            
            # Reset drawing mode
            self.drawing = False
            self.rect = None
            self.canvas.get_tk_widget().config(cursor="")
            self.drawing_label.config(text="")
            
            # Disconnect any existing event handlers
            if self.canvas_cid_press:
                self.canvas.mpl_disconnect(self.canvas_cid_press)
                self.canvas_cid_press = None
            if self.canvas_cid_release:
                self.canvas.mpl_disconnect(self.canvas_cid_release)
                self.canvas_cid_release = None
            if self.canvas_cid_motion:
                self.canvas.mpl_disconnect(self.canvas_cid_motion)
                self.canvas_cid_motion = None
            
            # Clear the plot
            self.ax.clear()
            self.ax.imshow(self.current_image)
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            
            # Draw existing annotations
            self.draw_annotations(img_path)
            
            # Update the canvas
            self.canvas.draw()
            
            # Update status
            filename = os.path.basename(img_path)
            index = self.image_files.index(img_path) + 1
            self.status_var.set(f"Image {index}/{len(self.image_files)}: {filename}")
            
            # Update annotation treeview
            self.update_annotation_treeview(img_path)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def draw_annotations(self, img_path):
        """Draw bounding boxes for annotations"""
        if img_path in self.annotations:
            for i, anno in enumerate(self.annotations[img_path]):
                bbox = anno['bbox']
                class_id = anno['class_id']
                confidence = anno['confidence']
                
                # Get class name
                class_name = f"Class {class_id}"
                if hasattr(self.detector_model, 'names'):
                    class_name = self.detector_model.names.get(class_id, f"Class {class_id}")
                
                # Create random but consistent color for this class
                np.random.seed(class_id)
                color = np.random.rand(3)
                
                # Draw rectangle
                rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                                     fill=False, edgecolor=color, linewidth=2)
                self.ax.add_patch(rect)
                
                # Draw label
                label = f"{class_name} {confidence:.2f}"
                self.ax.text(bbox[0], bbox[1]-5, label, color=color, fontsize=8,
                             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0))
    
    def update_annotation_treeview(self, img_path):
        """Update the annotation treeview with annotations for the current image"""
        # Clear the treeview
        for item in self.anno_tree.get_children():
            self.anno_tree.delete(item)
        
        # Add annotations to the treeview
        if img_path in self.annotations:
            for i, anno in enumerate(self.annotations[img_path]):
                bbox = anno['bbox']
                class_id = anno['class_id']
                confidence = anno['confidence']
                
                # Get class name
                class_name = f"Class {class_id}"
                if hasattr(self.detector_model, 'names'):
                    class_name = self.detector_model.names.get(class_id, f"Class {class_id}")
                
                # Add to treeview with the index of the annotation as a tag
                self.anno_tree.insert("", "end", values=(
                    class_name,
                    f"{confidence:.2f}",
                    f"({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]})"
                ), tags=(str(i),)) # Store the index 'i' as a string tag
        
        # Add "No annotations found" message if no annotations
        if not self.anno_tree.get_children():
            self.anno_tree.insert("", "end", values=("No annotations found", "", ""))

    def highlight_selected_box(self, event):
        """Highlight the selected bounding box in the image."""
        selected_item = self.anno_tree.focus()
        if not selected_item:
            return

        # Clear previous highlights
        for patch in self.ax.patches:
            if hasattr(patch, '_original_edgecolor'):
                patch.set_edgecolor(patch._original_edgecolor)
                del patch._original_edgecolor
            if hasattr(patch, '_original_linewidth'):
                patch.set_linewidth(patch._original_linewidth)
                del patch._original_linewidth

        # Get the index of the selected annotation
        annotation_index = int(self.anno_tree.item(selected_item, "tags")[0])

        if self.current_image_path and annotation_index < len(self.annotations[self.current_image_path]):
            # Get the corresponding bounding box from the annotations list
            bbox = self.annotations[self.current_image_path][annotation_index]['bbox']
            
            # Find the patch corresponding to this bbox (assuming order is maintained)
            # This is a bit fragile if patches are reordered, but for now it works.
            # A more robust solution might involve storing the patch object with the annotation.
            for i, patch in enumerate(self.ax.patches):
                if isinstance(patch, plt.Rectangle):
                    # Check if the patch's coordinates match the annotation's bbox
                    # Allow for slight floating point differences
                    patch_bbox = [patch.get_x(), patch.get_y(), patch.get_x() + patch.get_width(), patch.get_y() + patch.get_height()]
                    if np.allclose(patch_bbox, bbox, atol=1.0): # Use a tolerance for comparison
                        patch._original_edgecolor = patch.get_edgecolor() # Store original color
                        patch._original_linewidth = patch.get_linewidth() # Store original linewidth
                        patch.set_edgecolor('red') # Highlight in red
                        patch.set_linewidth(3)
                        break
            self.canvas.draw()

    def delete_selected_annotation(self):
        """Delete the selected annotation from the current image."""
        selected_item = self.anno_tree.focus()
        if not selected_item:
            messagebox.showwarning("No Selection", "Please select an annotation to delete.")
            return

        # Get the annotation index from the item's tags
        try:
            annotation_index = int(self.anno_tree.item(selected_item, "tags")[0])
        except (IndexError, ValueError):
            messagebox.showerror("Error", "Could not retrieve annotation index.")
            return

        if self.current_image_path and 0 <= annotation_index < len(self.annotations[self.current_image_path]):
            confirm = messagebox.askyesno("Confirm Deletion", "Are you sure you want to delete this annotation?")
            if confirm:
                # Remove the annotation from the list
                del self.annotations[self.current_image_path][annotation_index]
                self.status_var.set(f"Annotation deleted from {os.path.basename(self.current_image_path)}")
                
                # Reload the current image to refresh the display and the treeview
                self.load_image(self.current_image_path)
        else:
            messagebox.showwarning("Invalid Selection", "The selected annotation is no longer valid or image path is missing.")

    def edit_selected_annotation(self):
        """Open a dialog to edit the selected annotation."""
        selected_item = self.anno_tree.focus()
        if not selected_item:
            messagebox.showwarning("No Selection", "Please select an annotation to edit.")
            return

        try:
            annotation_index = int(self.anno_tree.item(selected_item, "tags")[0])
        except (IndexError, ValueError):
            messagebox.showerror("Error", "Could not retrieve annotation index for editing.")
            return
        
        if self.current_image_path and 0 <= annotation_index < len(self.annotations[self.current_image_path]):
            current_anno = self.annotations[self.current_image_path][annotation_index]
            
            # Create a top-level window for editing
            edit_window = tk.Toplevel(self.tab)
            edit_window.title("Edit Annotation")
            
            # Class ID
            ttk.Label(edit_window, text="Class ID:").grid(row=0, column=0, padx=5, pady=5)
            class_id_var = tk.IntVar(value=current_anno['class_id'])
            ttk.Entry(edit_window, textvariable=class_id_var).grid(row=0, column=1, padx=5, pady=5)
            
            # Confidence (can be edited for manual annotations if desired, otherwise disabled for auto)
            ttk.Label(edit_window, text="Confidence:").grid(row=1, column=0, padx=5, pady=5)
            confidence_var = tk.DoubleVar(value=current_anno['confidence'])
            ttk.Entry(edit_window, textvariable=confidence_var, state="readonly").grid(row=1, column=1, padx=5, pady=5)
            
            # Bounding Box (xmin, ymin, xmax, ymax)
            ttk.Label(edit_window, text="XMin:").grid(row=2, column=0, padx=5, pady=2)
            xmin_var = tk.IntVar(value=current_anno['bbox'][0])
            ttk.Entry(edit_window, textvariable=xmin_var).grid(row=2, column=1, padx=5, pady=2)

            ttk.Label(edit_window, text="YMin:").grid(row=3, column=0, padx=5, pady=2)
            ymin_var = tk.IntVar(value=current_anno['bbox'][1])
            ttk.Entry(edit_window, textvariable=ymin_var).grid(row=3, column=1, padx=5, pady=2)

            ttk.Label(edit_window, text="XMax:").grid(row=4, column=0, padx=5, pady=2)
            xmax_var = tk.IntVar(value=current_anno['bbox'][2])
            ttk.Entry(edit_window, textvariable=xmax_var).grid(row=4, column=1, padx=5, pady=2)

            ttk.Label(edit_window, text="YMax:").grid(row=5, column=0, padx=5, pady=2)
            ymax_var = tk.IntVar(value=current_anno['bbox'][3])
            ttk.Entry(edit_window, textvariable=ymax_var).grid(row=5, column=1, padx=5, pady=2)

            def save_changes():
                try:
                    new_class_id = class_id_var.get()
                    new_confidence = confidence_var.get()
                    new_xmin = xmin_var.get()
                    new_ymin = ymin_var.get()
                    new_xmax = xmax_var.get()
                    new_ymax = ymax_var.get()

                    # Basic validation
                    if not (0 <= new_confidence <= 1.0) or \
                       not (0 <= new_xmin < new_xmax) or \
                       not (0 <= new_ymin < new_ymax):
                        messagebox.showerror("Input Error", "Please enter valid annotation values. XMin < XMax, YMin < YMax, Confidence between 0 and 1.")
                        return

                    # Update the annotation
                    self.annotations[self.current_image_path][annotation_index]['class_id'] = new_class_id
                    self.annotations[self.current_image_path][annotation_index]['confidence'] = new_confidence
                    self.annotations[self.current_image_path][annotation_index]['bbox'] = [new_xmin, new_ymin, new_xmax, new_ymax]
                    self.status_var.set(f"Annotation for {os.path.basename(self.current_image_path)} updated.")
                    edit_window.destroy()
                    self.load_image(self.current_image_path) # Reload to show changes
                except ValueError:
                    messagebox.showerror("Input Error", "Please enter valid numbers for all fields.")

            ttk.Button(edit_window, text="Save", command=save_changes).grid(row=6, column=0, columnspan=2, padx=5, pady=10)
        else:
            messagebox.showwarning("Invalid Selection", "The selected annotation is no longer valid or image path is missing.")

    def add_manual_box(self):
        """Enable drawing mode to manually add a bounding box."""
        if not self.detector_model:
            messagebox.showwarning("No Model", "Please load a YOLO model first to define classes for manual annotation.")
            return
        
        # Create a Toplevel window for class selection
        class_selection_window = tk.Toplevel(self.tab)
        class_selection_window.title("Select Class for Manual Annotation")
        
        selected_class_id = tk.IntVar()
        selected_class_id.set(-1) # Default to no selection

        # Populate with radio buttons for classes from the loaded YOLO model
        if hasattr(self.detector_model, 'names'):
            for idx, class_name in self.detector_model.names.items():
                rb = ttk.Radiobutton(class_selection_window, text=f"{idx}: {class_name}", 
                                     variable=selected_class_id, value=idx)
                rb.pack(anchor="w", padx=10, pady=2)
        else:
            ttk.Label(class_selection_window, text="No classes available. Load a YOLO model.").pack(padx=10, pady=10)
            ttk.Button(class_selection_window, text="OK", command=class_selection_window.destroy).pack(pady=10)
            return

        def start_drawing_with_class():
            chosen_class = selected_class_id.get()
            if chosen_class == -1:
                messagebox.showwarning("No Class Selected", "Please select a class for the new annotation.")
                return
            
            self.current_manual_class_id = chosen_class
            class_selection_window.destroy()
            self._enable_drawing_mode()

        ttk.Button(class_selection_window, text="Start Drawing", command=start_drawing_with_class).pack(pady=10)


    def _enable_drawing_mode(self):
        """Activate drawing mode on the canvas."""
        if self.current_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        self.status_var.set("Drawing mode active: Click and drag to draw a box.")
        self.drawing_label.config(text="Draw a bounding box for the selected class. Press 'Esc' to cancel.")
        self.canvas.get_tk_widget().config(cursor="cross")

        # Bind mouse events to the canvas
        self.canvas_cid_press = self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas_cid_release = self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas_cid_motion = self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.main_app.root.bind('<Escape>', self.cancel_drawing) # Bind Esc key to cancel drawing

    def cancel_drawing(self, event=None):
        """Cancel the current drawing operation and reset mode."""
        self.drawing = False
        self.rect = None
        self.status_var.set("Drawing cancelled.")
        self.drawing_label.config(text="")
        self.canvas.get_tk_widget().config(cursor="")

        # Disconnect event handlers
        if self.canvas_cid_press:
            self.canvas.mpl_disconnect(self.canvas_cid_press)
            self.canvas_cid_press = None
        if self.canvas_cid_release:
            self.canvas.mpl_disconnect(self.canvas_cid_release)
            self.canvas_cid_release = None
        if self.canvas_cid_motion:
            self.canvas.mpl_disconnect(self.canvas_cid_motion)
            self.canvas_cid_motion = None
        self.main_app.root.unbind('<Escape>', self.cancel_drawing) # Unbind Esc key

        # Reload image to remove any temporary drawing artifacts
        if self.current_image_path:
            self.load_image(self.current_image_path)

    def on_press(self, event):
        """Handle mouse button press event for drawing."""
        if event.inaxes == self.ax and event.button == 1:  # Left click inside axes
            self.drawing = True
            self.start_x = int(event.xdata)
            self.start_y = int(event.ydata)
            self.rect = plt.Rectangle((self.start_x, self.start_y), 0, 0,
                                      fill=False, edgecolor='cyan', linewidth=2, linestyle='--')
            self.ax.add_patch(self.rect)
            self.canvas.draw_idle()

    def on_motion(self, event):
        """Handle mouse motion event for drawing."""
        if self.drawing and event.inaxes == self.ax:
            current_x = int(event.xdata)
            current_y = int(event.ydata)
            
            # Ensure coordinates are within image bounds
            height, width = self.current_image.shape[:2]
            current_x = max(0, min(current_x, width))
            current_y = max(0, min(current_y, height))

            self.rect.set_width(current_x - self.start_x)
            self.rect.set_height(current_y - self.start_y)
            self.canvas.draw_idle()

    def on_release(self, event):
        """Handle mouse button release event for drawing."""
        if self.drawing and event.inaxes == self.ax and event.button == 1:
            self.drawing = False
            end_x = int(event.xdata)
            end_y = int(event.ydata)
            
            # Ensure coordinates are within image bounds
            height, width = self.current_image.shape[:2]
            end_x = max(0, min(end_x, width))
            end_y = max(0, min(end_y, height))

            xmin = min(self.start_x, end_x)
            ymin = min(self.start_y, end_y)
            xmax = max(self.start_x, end_x)
            ymax = max(self.start_y, end_y)

            # Validate box dimensions
            if (xmax - xmin) > 5 and (ymax - ymin) > 5: # Minimum size for a valid box
                new_annotation = {
                    'class_id': self.current_manual_class_id,
                    'confidence': 1.0, # Manual annotations have 100% confidence
                    'bbox': [xmin, ymin, xmax, ymax]
                }
                if self.current_image_path not in self.annotations:
                    self.annotations[self.current_image_path] = []
                self.annotations[self.current_image_path].append(new_annotation)
                self.status_var.set(f"Manual annotation added for class {self.current_manual_class_id}.")
            else:
                messagebox.showwarning("Small Box", "The drawn box is too small to be a valid annotation.")
                self.status_var.set("Drawing cancelled (box too small).")

            # Reset drawing state and reload image
            self.cancel_drawing() # This will reload the image and update display

    # Navigation methods
    def previous_image(self):
        if self.image_files and self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_image(self.image_files[self.current_image_index])
        else:
            self.status_var.set("No previous image.")
    
    def next_image(self):
        if self.image_files and self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.load_image(self.image_files[self.current_image_index])
        else:
            self.status_var.set("No next image.")

    # Annotation methods
    def update_annotation_mode(self, event=None):
        selected_mode = self.tab_control.nametowidget(event.widget).get() # Get selected mode from combobox
        if selected_mode == "YOLO Only":
            self.annotation_mode.set("yolo")
            self.drawing_label.config(text="")
            self.canvas.get_tk_widget().config(cursor="")
            # Disconnect drawing events if active
            self.cancel_drawing()
        elif selected_mode == "SAM Only":
            if not SAM_AVAILABLE or not self.sam_model:
                messagebox.showwarning("SAM Not Ready", "SAM is not installed or model not loaded. Please select YOLO Only mode.")
                # Revert combobox selection
                self.tab_control.nametowidget(event.widget).set("YOLO Only")
                self.annotation_mode.set("yolo")
                return
            self.annotation_mode.set("sam")
            self.drawing_label.config(text="Click on an object to segment it with SAM.")
            self.canvas.get_tk_widget().config(cursor="tcross")
            # Connect SAM click event
            self.canvas_cid_press = self.canvas.mpl_connect('button_press_event', self.on_sam_click)
        elif selected_mode == "YOLO+SAM Hybrid":
            if not SAM_AVAILABLE or not self.sam_model:
                messagebox.showwarning("SAM Not Ready", "SAM is not installed or model not loaded. Please select YOLO Only mode.")
                # Revert combobox selection
                self.tab_control.nametowidget(event.widget).set("YOLO Only")
                self.annotation_mode.set("yolo")
                return
            self.annotation_mode.set("hybrid")
            self.drawing_label.config(text="YOLO detections will be segmented by SAM. Click on a bounding box for refinement.")
            self.canvas.get_tk_widget().config(cursor="cross")
            # Connect SAM click event (for refinement)
            self.canvas_cid_press = self.canvas.mpl_connect('button_press_event', self.on_sam_click)

        self.status_var.set(f"Annotation mode set to: {selected_mode}")

    def on_sam_click(self, event):
        """Handle click event for SAM segmentation or refinement."""
        if event.inaxes == self.ax and self.sam_predictor and self.current_image is not None:
            x, y = int(event.xdata), int(event.ydata)
            
            self.status_var.set(f"SAM processing click at ({x}, {y})...")
            
            # Run SAM prediction in a separate thread to avoid freezing the GUI
            threading.Thread(target=self._run_sam_prediction, args=(x, y)).start()
        else:
            messagebox.showwarning("SAM Error", "SAM model not loaded or no image selected.")

    def _run_sam_prediction(self, x, y):
        try:
            self.sam_predictor.set_image(self.current_image)
            input_point = np.array([[x, y]])
            input_label = np.array([1]) # 1 for foreground
            
            masks, scores, logits = self.sam_predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True, # Get multiple masks for ambiguity
            )

            # Choose the best mask (e.g., highest score)
            best_mask_idx = np.argmax(scores)
            mask = masks[best_mask_idx]

            # Convert mask to bounding box (xmin, ymin, xmax, ymax)
            # Find coordinates where the mask is True
            coords = np.argwhere(mask)
            if coords.shape[0] > 0:
                ymin, xmin = coords.min(axis=0)
                ymax, xmax = coords.max(axis=0)
                bbox = [xmin, ymin, xmax + 1, ymax + 1] # +1 for inclusivity

                # Add to annotations (assign a dummy class or ask user)
                # For simplicity, let's assign a new "segmented_object" class or prompt user
                # For now, we'll ask the user for a class ID
                self.main_app.root.after(0, self._prompt_for_class_id, bbox, scores[best_mask_idx])
                
            self.main_app.root.after(0, self.status_var.set, "SAM prediction complete.")

        except Exception as e:
            self.main_app.root.after(0, messagebox.showerror, "SAM Error", f"Failed SAM prediction: {str(e)}")
            self.main_app.root.after(0, self.status_var.set, "SAM prediction failed.")

    def _prompt_for_class_id(self, bbox, confidence):
        """Opens a dialog to ask the user for a class ID for a newly created SAM annotation."""
        prompt_window = tk.Toplevel(self.tab)
        prompt_window.title("Assign Class ID")

        ttk.Label(prompt_window, text="Enter Class ID for new annotation:").pack(padx=10, pady=10)
        class_id_entry = ttk.Entry(prompt_window)
        class_id_entry.pack(padx=10, pady=5)
        
        # Populate with a dropdown of existing classes from YOLO model if available
        if self.detector_model and hasattr(self.detector_model, 'names'):
            class_names = [f"{idx}: {name}" for idx, name in self.detector_model.names.items()]
            class_combo = ttk.Combobox(prompt_window, values=class_names, state="readonly")
            class_combo.pack(padx=10, pady=5)
            def on_combo_select(event):
                selected_text = class_combo.get()
                class_id_str = selected_text.split(':')[0]
                try:
                    class_id_entry.delete(0, tk.END)
                    class_id_entry.insert(0, class_id_str)
                except Exception:
                    pass # Ignore if invalid format
            class_combo.bind("<<ComboboxSelected>>", on_combo_select)

        def add_annotation():
            try:
                class_id = int(class_id_entry.get())
                new_annotation = {
                    'class_id': class_id,
                    'confidence': confidence,
                    'bbox': bbox
                }
                if self.current_image_path not in self.annotations:
                    self.annotations[self.current_image_path] = []
                self.annotations[self.current_image_path].append(new_annotation)
                self.status_var.set(f"SAM annotation added with class ID {class_id}.")
                self.load_image(self.current_image_path) # Reload image to display new annotation
                prompt_window.destroy()
            except ValueError:
                messagebox.showerror("Input Error", "Please enter a valid integer for Class ID.")

        ttk.Button(prompt_window, text="Add Annotation", command=add_annotation).pack(pady=10)
        
        # Make the prompt window modal
        prompt_window.transient(self.tab.winfo_toplevel())
        prompt_window.grab_set()
        self.tab.winfo_toplevel().wait_window(prompt_window)


    def annotate_current(self):
        """Start auto-annotation for the current image."""
        if self.current_image is None:
            messagebox.showwarning("No Image", "Please load an image folder first.")
            return
        if self.detector_model is None and self.annotation_mode.get() in ["yolo", "hybrid"]:
            messagebox.showwarning("No YOLO Model", "Please load a YOLO model to perform detection.")
            return
        if self.sam_model is None and self.annotation_mode.get() in ["sam", "hybrid"]:
            messagebox.showwarning("No SAM Model", "Please load a SAM model to perform segmentation.")
            return

        self.status_var.set("Annotating current image...")
        # Run annotation in a separate thread to keep GUI responsive
        threading.Thread(target=self._run_annotation_for_current_image).start()
    
    def _run_annotation_for_current_image(self):
        try:
            image = self.current_image
            annotations_for_image = []
            selected_classes = self.get_selected_classes()

            if self.annotation_mode.get() == "yolo" or self.annotation_mode.get() == "hybrid":
                # Perform YOLO detection
                results = self.detector_model(image, conf=self.confidence_threshold.get())
                for r in results:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()
                    class_ids = r.boxes.cls.cpu().numpy()

                    for i in range(len(boxes)):
                        class_id = int(class_ids[i])
                        confidence = float(confs[i])
                        bbox = [int(x) for x in boxes[i]] # xmin, ymin, xmax, ymax

                        if class_id in selected_classes:
                            if self.annotation_mode.get() == "hybrid" and self.sam_predictor:
                                # Use SAM to refine the YOLO bounding box
                                self.sam_predictor.set_image(image)
                                input_box = np.array(bbox)
                                masks, _, _ = self.sam_predictor.predict(
                                    point_coords=None,
                                    point_labels=None,
                                    box=input_box[None, :],
                                    multimask_output=False,
                                )
                                if masks.shape[0] > 0:
                                    # Take the first mask and convert to bbox
                                    coords = np.argwhere(masks[0])
                                    if coords.shape[0] > 0:
                                        ymin, xmin = coords.min(axis=0)
                                        ymax, xmax = coords.max(axis=0)
                                        bbox = [xmin, ymin, xmax + 1, ymax + 1]
                                    else:
                                        # If SAM produces no mask, use original YOLO box
                                        pass 
                                
                            annotations_for_image.append({
                                'class_id': class_id,
                                'confidence': confidence,
                                'bbox': bbox
                            })

            elif self.annotation_mode.get() == "sam":
                # SAM only mode requires manual clicks, so no auto-annotation here
                # The _run_sam_prediction is called by on_sam_click
                # This branch would typically not have batch annotation
                pass

            self.annotations[self.current_image_path] = annotations_for_image
            
            # Update UI on main thread
            self.main_app.root.after(0, self.load_image, self.current_image_path)
            self.main_app.root.after(0, self.status_var.set, f"Annotation complete for {os.path.basename(self.current_image_path)}")

        except Exception as e:
            self.main_app.root.after(0, messagebox.showerror, "Annotation Error", f"Failed to annotate image: {str(e)}")
            self.main_app.root.after(0, self.status_var.set, "Annotation failed.")

    def annotate_batch(self):
        """Start auto-annotation for all images in the folder."""
        if not self.image_files:
            messagebox.showwarning("No Images", "Please load an image folder first.")
            return
        if self.detector_model is None and self.annotation_mode.get() in ["yolo", "hybrid"]:
            messagebox.showwarning("No YOLO Model", "Please load a YOLO model to perform detection.")
            return
        if self.sam_model is None and self.annotation_mode.get() in ["sam", "hybrid"]:
            messagebox.showwarning("No SAM Model", "Please load a SAM model to perform segmentation in hybrid/SAM-only mode.")
            return
        if self.annotation_mode.get() == "sam":
            messagebox.showwarning("SAM Only Mode", "Batch annotation is not supported in 'SAM Only' mode as it requires user interaction per object. Use 'YOLO Only' or 'YOLO+SAM Hybrid' for batch processing.")
            return

        confirm = messagebox.askyesno("Confirm Batch Annotation", 
                                      f"This will auto-annotate {len(self.image_files)} images. Proceed?")
        if not confirm:
            return

        self.status_var.set(f"Starting batch annotation for {len(self.image_files)} images...")
        # Run batch annotation in a separate thread
        threading.Thread(target=self._run_batch_annotation).start()

    def _run_batch_annotation(self):
        try:
            total_images = len(self.image_files)
            for i, img_path in enumerate(self.image_files):
                # Load image for processing (not for UI display yet)
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                annotations_for_image = []
                selected_classes = self.get_selected_classes()

                # Perform YOLO detection
                results = self.detector_model(image, conf=self.confidence_threshold.get())
                for r in results:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()
                    class_ids = r.boxes.cls.cpu().numpy()

                    for j in range(len(boxes)):
                        class_id = int(class_ids[j])
                        confidence = float(confs[j])
                        bbox = [int(x) for x in boxes[j]]

                        if class_id in selected_classes:
                            if self.annotation_mode.get() == "hybrid" and self.sam_predictor:
                                # Use SAM to refine the YOLO bounding box
                                self.sam_predictor.set_image(image)
                                input_box = np.array(bbox)
                                masks, _, _ = self.sam_predictor.predict(
                                    point_coords=None,
                                    point_labels=None,
                                    box=input_box[None, :],
                                    multimask_output=False,
                                )
                                if masks.shape[0] > 0:
                                    coords = np.argwhere(masks[0])
                                    if coords.shape[0] > 0:
                                        ymin, xmin = coords.min(axis=0)
                                        ymax, xmax = coords.max(axis=0)
                                        bbox = [xmin, ymin, xmax + 1, ymax + 1]
                                    else:
                                        pass # Use original YOLO box if SAM fails
                                
                            annotations_for_image.append({
                                'class_id': class_id,
                                'confidence': confidence,
                                'bbox': bbox
                            })

                self.annotations[img_path] = annotations_for_image
                self.main_app.root.after(0, self.status_var.set, f"Annotated image {i+1}/{total_images}: {os.path.basename(img_path)}")
            
            # After batch annotation, reload the current image in the UI
            self.main_app.root.after(0, self.load_image, self.image_files[self.current_image_index])
            self.main_app.root.after(0, self.status_var.set, f"Batch annotation complete. Processed {total_images} images.")

        except Exception as e:
            self.main_app.root.after(0, messagebox.showerror, "Batch Annotation Error", f"Failed to batch annotate: {str(e)}")
            self.main_app.root.after(0, self.status_var.set, "Batch annotation failed.")

    def save_annotations(self):
        """Save annotations for all images in YOLO format."""
        output_folder = self.output_entry.get()
        if not output_folder:
            messagebox.showerror("Error", "Please select an output folder to save annotations.")
            return
        
        os.makedirs(output_folder, exist_ok=True)
        
        saved_count = 0
        for img_path, annos in self.annotations.items():
            if not annos: # Skip if no annotations for this image
                continue

            label_path = self.get_label_path(img_path)
            
            # Get image dimensions to normalize bounding boxes
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_path}, skipping annotations.")
                continue
            height, width = img.shape[:2]
            
            with open(label_path, 'w') as f:
                for anno in annos:
                    bbox = anno['bbox']
                    class_id = anno['class_id']
                    
                    # Convert xmin, ymin, xmax, ymax to YOLO format (normalized x_center, y_center, width, height)
                    xmin, ymin, xmax, ymax = bbox
                    
                    # Clamp coordinates to image bounds to prevent issues with out-of-bounds annotations
                    xmin = max(0, min(xmin, width - 1))
                    ymin = max(0, min(ymin, height - 1))
                    xmax = max(0, min(xmax, width))
                    ymax = max(0, min(ymax, height))

                    # Ensure valid box if clamping changed it to invalid (e.g., xmax < xmin)
                    if xmax <= xmin: xmax = xmin + 1
                    if ymax <= ymin: ymax = ymin + 1
                    
                    bbox_width = xmax - xmin
                    bbox_height = ymax - ymin
                    
                    x_center = (xmin + xmax) / 2.0 / width
                    y_center = (ymin + ymax) / 2.0 / height
                    norm_width = bbox_width / width
                    norm_height = bbox_height / height
                    
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")
            saved_count += 1
        
        self.status_var.set(f"Saved annotations for {saved_count} images to {output_folder}")
        messagebox.showinfo("Save Complete", f"Saved annotations for {saved_count} images.")