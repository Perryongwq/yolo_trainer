import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import datetime
import re

class StatusTab:
    def __init__(self, parent, main_app):
        self.main_app = main_app
        
        # Create tab
        self.tab = ttk.Frame(parent)
        parent.add(self.tab, text="Training Status")
        
        # Initialize variables needed by update_status
        self.start_time = None
        self.current_epoch = 0
        self.total_epochs = 0
        
        # Regular expressions for colored output
        self.color_patterns = [
            (r'Epoch\s+\d+/\d+', 'blue'),  # Epoch headers
            (r'GPU_mem', 'dark green'),     # GPU memory
            (r'Class\s+Images\s+Instances', 'purple'),  # Validation headers
            (r'[0-9]+%\|[█▉▊▋▌▍▎▏ ]+\|', 'dark green'),  # Progress bars
            (r'mAP50-95', 'purple'),        # mAP metric
            (r'WARNING', 'orange'),         # Warnings
            (r'Error', 'red'),              # Errors
        ]
        
        # Build UI components
        self.create_status_tab()
    
    def create_status_tab(self):
        # Main frame with weights
        main_frame = ttk.Frame(self.tab)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Configure grid weights to make status_text expandable
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Status label
        status_label = ttk.Label(main_frame, text="Training Status:")
        status_label.grid(row=0, column=0, sticky="w", padx=5, pady=5)
        
        # Progress frame (showing current epoch, etc.)
        self.progress_frame = ttk.LabelFrame(main_frame, text="Training Progress")
        self.progress_frame.grid(row=0, column=1, padx=5, pady=5, sticky="e")
        
        # Epoch progress
        ttk.Label(self.progress_frame, text="Epoch:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.epoch_var = tk.StringVar(value="0/0")
        ttk.Label(self.progress_frame, textvariable=self.epoch_var).grid(row=0, column=1, padx=5, pady=2, sticky="w")
        
        # Training time
        ttk.Label(self.progress_frame, text="Elapsed:").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.time_var = tk.StringVar(value="00:00:00")
        ttk.Label(self.progress_frame, textvariable=self.time_var).grid(row=1, column=1, padx=5, pady=2, sticky="w")
        
        # Status text widget with syntax highlighting capabilities
        self.status_text = scrolledtext.ScrolledText(
            main_frame, 
            height=20, 
            width=100, 
            wrap=tk.WORD,
            font=("Consolas", 9)  # Monospaced font works best for console output
        )
        self.status_text.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
        
        # Create tags for syntax highlighting
        self.status_text.tag_configure("timestamp", foreground="gray")
        self.status_text.tag_configure("blue", foreground="blue")
        self.status_text.tag_configure("red", foreground="red")
        self.status_text.tag_configure("green", foreground="green")
        self.status_text.tag_configure("dark green", foreground="dark green")
        self.status_text.tag_configure("orange", foreground="orange")
        self.status_text.tag_configure("purple", foreground="purple")
        self.status_text.tag_configure("bold", font=("Consolas", 9, "bold"))
        
        # Buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="w")
        
        # Clear log button
        ttk.Button(buttons_frame, text="Clear Log", command=self.clear_status).pack(side="left", padx=5)
        
        # Save log button
        ttk.Button(buttons_frame, text="Save Log", command=self.save_log).pack(side="left", padx=5)
        
        # Autoscroll checkbox
        self.autoscroll = tk.BooleanVar(value=True)
        ttk.Checkbutton(buttons_frame, text="Auto-scroll", variable=self.autoscroll).pack(side="left", padx=20)
        
        # Initial status
        self.update_status("Ready to train. Configure dataset and parameters, then start training.")
    
    def update_status(self, message):
        """Add a message to the status log with timestamp and syntax highlighting"""
        # Start tracking time if this is the first training message
        if "Starting training for" in message and not self.start_time:
            self.start_time = datetime.datetime.now()
            # Extract total epochs from message
            match = re.search(r'Starting training for (\d+) epochs', message)
            if match:
                self.total_epochs = int(match.group(1))
                self.epoch_var.set(f"0/{self.total_epochs}")
        
        # Update epoch counter if this is an epoch line
        epoch_match = re.search(r'Epoch\s+(\d+)/(\d+)', message)
        if epoch_match:
            self.current_epoch = int(epoch_match.group(1))
            total_epochs = int(epoch_match.group(2))
            self.epoch_var.set(f"{self.current_epoch}/{total_epochs}")
        
        # Update elapsed time if training is in progress
        if self.start_time:
            elapsed = datetime.datetime.now() - self.start_time
            hours, remainder = divmod(elapsed.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            self.time_var.set(f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
        
        # Get current timestamp
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        
        # Apply syntax highlighting
        self.status_text.insert(tk.END, f"[{timestamp}] ", "timestamp")
        
        # Insert the message with appropriate formatting
        message_start_index = self.status_text.index(tk.END)
        self.status_text.insert(tk.END, f"{message}\n")
        message_end_index = self.status_text.index(tk.END)
        
        # Apply color patterns
        for pattern, color in self.color_patterns:
            start_pos = message_start_index
            while True:
                # Find the pattern in the message
                pattern_start = self.status_text.search(
                    pattern, start_pos, message_end_index, regexp=True
                )
                if not pattern_start:
                    break
                
                # Calculate end position of the match
                line, col = map(int, pattern_start.split('.'))
                pattern_end = f"{line}.{col + len(self.status_text.get(pattern_start, f'{line}.end'))}"
                
                # Apply the tag
                self.status_text.tag_add(color, pattern_start, pattern_end)
                
                # Move to search after this match
                start_pos = pattern_end
        
        # Auto-scroll if enabled
        if self.autoscroll.get():
            self.status_text.see(tk.END)
        
        # Update the GUI
        self.main_app.root.update_idletasks()
    
    def clear_status(self):
        """Clear the status log"""
        self.status_text.delete(1.0, tk.END)
        self.update_status("Log cleared")
        
        # Reset progress tracking
        self.current_epoch = 0
        self.total_epochs = 0
        self.epoch_var.set("0/0")
        self.time_var.set("00:00:00")
        self.start_time = None
    
    def save_log(self):
        """Save the status log to a file"""
        file_path = filedialog.asksaveasfilename(
            title="Save Log",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write(self.status_text.get(1.0, tk.END))
                self.update_status(f"Log saved to {file_path}")
            except Exception as e:
                self.update_status(f"Error saving log: {str(e)}")