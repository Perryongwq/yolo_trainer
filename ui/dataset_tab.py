import os
import yaml
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

class DatasetTab:
    def __init__(self, parent, main_app):
        self.main_app = main_app
        
        # Create tab
        self.tab = ttk.Frame(parent)
        parent.add(self.tab, text="Dataset Configuration")
        
        # Build UI components
        self.create_dataset_tab()
    
    def create_dataset_tab(self):
        # Dataset selection frame
        dataset_frame = ttk.LabelFrame(self.tab, text="Dataset Selection")
        dataset_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # YAML file selection
        ttk.Label(dataset_frame, text="Select YAML file:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(dataset_frame, textvariable=self.main_app.yaml_path, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(dataset_frame, text="Browse", command=self.browse_yaml).grid(row=0, column=2, padx=5, pady=5)
        
        # Quick dataset selection
        ttk.Label(dataset_frame, text="Quick select:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        dataset_combo = ttk.Combobox(dataset_frame, values=["03dataset.yaml", "15dataset.yaml", "32dataset.yaml"])
        dataset_combo.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        dataset_combo.bind("<<ComboboxSelected>>", lambda e: self.main_app.yaml_path.set(dataset_combo.get()))
        ttk.Button(dataset_frame, text="Load", command=self.load_yaml).grid(row=1, column=2, padx=5, pady=5)
        
        # Create new YAML button
        ttk.Button(dataset_frame, text="Create New", command=self.create_new_dataset).grid(
            row=2, column=0, columnspan=3, padx=5, pady=5)
        
        # Dataset details frame
        self.details_frame = ttk.LabelFrame(self.tab, text="Dataset Details")
        self.details_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Path configuration
        ttk.Label(self.details_frame, text="Dataset Path:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.path_entry = ttk.Entry(self.details_frame, width=50)
        self.path_entry.grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(self.details_frame, text="Browse", command=self.browse_dataset_path).grid(row=0, column=2, padx=5, pady=5)
        
        # Classes frame
        self.classes_frame = ttk.LabelFrame(self.details_frame, text="Classes")
        self.classes_frame.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="nsew")
        
        # Save changes button
        ttk.Button(self.details_frame, text="Save YAML Changes", command=self.save_yaml_changes).grid(
            row=2, column=0, columnspan=3, padx=5, pady=10)
    
    def browse_yaml(self):
        """Browse for a YAML file"""
        file_path = filedialog.askopenfilename(
            title="Select YAML file",
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")]
        )
        if file_path:
            self.main_app.yaml_path.set(file_path)
    
    def browse_dataset_path(self):
        """Browse for dataset directory"""
        folder_path = filedialog.askdirectory(title="Select Dataset Folder")
        if folder_path:
            self.path_entry.delete(0, tk.END)
            self.path_entry.insert(0, folder_path)
    
    def load_yaml(self):
        """Load a YAML file and update the UI"""
        yaml_file = self.main_app.yaml_path.get()
        if not yaml_file:
            messagebox.showerror("Error", "Please select a YAML file")
            return
        
        try:
            # Try to find the file in the current directory first
            if not os.path.isabs(yaml_file) and not os.path.isfile(yaml_file):
                possible_paths = [
                    yaml_file,
                    os.path.join(os.getcwd(), yaml_file),
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", yaml_file)
                ]
                
                for path in possible_paths:
                    if os.path.isfile(path):
                        yaml_file = path
                        break
            
            with open(yaml_file, 'r') as f:
                self.main_app.dataset_content = yaml.safe_load(f)
            
            # Update UI with dataset information
            self.path_entry.delete(0, tk.END)
            self.path_entry.insert(0, self.main_app.dataset_content.get('path', ''))
            
            # Clear previous class entries
            for widget in self.classes_frame.winfo_children():
                widget.destroy()
            
            # Add new class entries
            self.main_app.classes = self.main_app.dataset_content.get('names', [])
            for i, class_name in enumerate(self.main_app.classes):
                ttk.Label(self.classes_frame, text=f"Class {i}:").grid(row=i, column=0, padx=5, pady=2, sticky="w")
                entry = ttk.Entry(self.classes_frame, width=30)
                entry.insert(0, class_name)
                entry.grid(row=i, column=1, padx=5, pady=2)
            
            # Add button for adding a new class
            ttk.Button(self.classes_frame, text="Add Class", command=self.add_class).grid(
                row=len(self.main_app.classes), column=0, columnspan=2, padx=5, pady=5)
            
            self.main_app.update_status(f"Loaded YAML file: {os.path.basename(yaml_file)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load YAML file: {str(e)}")
            self.main_app.update_status(f"Error loading YAML: {str(e)}")
    
    def add_class(self):
        """Add a new class to the list"""
        # Get the number of classes currently displayed
        class_count = len(self.main_app.classes)
        
        # Add a new empty class entry
        ttk.Label(self.classes_frame, text=f"Class {class_count}:").grid(
            row=class_count, column=0, padx=5, pady=2, sticky="w")
        entry = ttk.Entry(self.classes_frame, width=30)
        entry.grid(row=class_count, column=1, padx=5, pady=2)
        
        # Move the "Add Class" button down
        for widget in self.classes_frame.winfo_children():
            if isinstance(widget, ttk.Button) and widget.cget("text") == "Add Class":
                widget.grid(row=class_count + 1, column=0, columnspan=2, padx=5, pady=5)
                break
        
        # Add an empty string to the classes list
        self.main_app.classes.append("")
    
    def save_yaml_changes(self):
        """Save changes to the YAML file"""
        if not self.main_app.dataset_content:
            messagebox.showerror("Error", "No YAML file loaded")
            return
        
        # Update path
        self.main_app.dataset_content['path'] = self.path_entry.get()
        
        # Update classes
        class_entries = [widget for widget in self.classes_frame.winfo_children() 
                         if isinstance(widget, ttk.Entry)]
        self.main_app.classes = [entry.get() for entry in class_entries]
        self.main_app.dataset_content['names'] = self.main_app.classes
        
        # Save to file
        yaml_file = self.main_app.yaml_path.get()
        try:
            with open(yaml_file, 'w') as f:
                yaml.dump(self.main_app.dataset_content, f, default_flow_style=False, sort_keys=False)
            messagebox.showinfo("Success", "YAML file updated successfully")
            self.main_app.update_status(f"Saved changes to {os.path.basename(yaml_file)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save YAML file: {str(e)}")
            self.main_app.update_status(f"Error saving YAML: {str(e)}")
    
    def create_new_dataset(self):
        """Create a new YAML dataset file"""
        file_path = filedialog.asksaveasfilename(
            title="Create New Dataset YAML",
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        # Create dataset structure dialog
        dialog = tk.Toplevel(self.tab)
        dialog.title("Create Dataset")
        dialog.geometry("400x300")
        dialog.transient(self.tab)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Dataset Path:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        path_var = tk.StringVar(value="./datasets/custom")
        ttk.Entry(dialog, textvariable=path_var, width=30).grid(row=0, column=1, padx=5, pady=5)
        
        # Dataset type selection (YOLOv8 or YOLOv11 format)
        ttk.Label(dialog, text="Dataset Format:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        format_var = tk.StringVar(value="YOLOv8")
        ttk.Combobox(dialog, textvariable=format_var, values=["YOLOv8", "YOLOv11"]).grid(
            row=1, column=1, padx=5, pady=5)
        
        # Class entry frame
        classes_frame = ttk.LabelFrame(dialog, text="Classes")
        classes_frame.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
        
        class_entries = []
        for i in range(3):  # Start with 3 empty class entries
            ttk.Label(classes_frame, text=f"Class {i}:").grid(row=i, column=0, padx=5, pady=2, sticky="w")
            entry = ttk.Entry(classes_frame, width=20)
            entry.grid(row=i, column=1, padx=5, pady=2)
            class_entries.append(entry)
        
        # Buttons frame
        btn_frame = ttk.Frame(dialog)
        btn_frame.grid(row=3, column=0, columnspan=2, padx=5, pady=10)
        
        def add_class():
            i = len(class_entries)
            ttk.Label(classes_frame, text=f"Class {i}:").grid(row=i, column=0, padx=5, pady=2, sticky="w")
            entry = ttk.Entry(classes_frame, width=20)
            entry.grid(row=i, column=1, padx=5, pady=2)
            class_entries.append(entry)
        
        def create_yaml():
            # Get class names
            classes = [entry.get() for entry in class_entries if entry.get()]
            if not classes:
                messagebox.showerror("Error", "At least one class is required")
                return
            
            # Create YAML content
            content = {
                'path': path_var.get(),
                'train': 'images/train',
                'val': 'images/val',
                'test': 'images/test',
                'names': classes
            }
            
            # Add YOLOv11 specific settings if selected
            if format_var.get() == "YOLOv11":
                content['format_version'] = 11
                content['advanced_augmentation'] = True
            
            try:
                with open(file_path, 'w') as f:
                    yaml.dump(content, f, default_flow_style=False, sort_keys=False)
                
                messagebox.showinfo("Success", f"Dataset YAML created: {file_path}")
                dialog.destroy()
                
                # Load the newly created YAML file
                self.main_app.yaml_path.set(file_path)
                self.load_yaml()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to create YAML file: {str(e)}")
        
        ttk.Button(btn_frame, text="Add Class", command=add_class).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(btn_frame, text="Create", command=create_yaml).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).grid(row=0, column=2, padx=5, pady=5)