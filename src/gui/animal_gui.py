"""
Animal Detection System GUI
Simple interface for detecting animals in images and videos.
"""
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.modules.animal_engine import AnimalEngine


class AnimalDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Animal Detection System")
        self.root.geometry("900x700")
        
        # Load model
        model_path = "models/animal_best.pt"
        if not os.path.exists(model_path):
            model_path = "yolov8n.pt"
        self.engine = AnimalEngine(model_path)
        
        self.create_widgets()
        
    def create_widgets(self):
        # Title
        title = tk.Label(self.root, text="Animal Detection System", font=("Arial", 16, "bold"))
        title.pack(pady=10)
        
        # Buttons frame
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)
        
        self.btn_image = tk.Button(btn_frame, text="Upload Image", command=self.load_image, width=15)
        self.btn_image.pack(side=tk.LEFT, padx=10)
        
        self.btn_video = tk.Button(btn_frame, text="Upload Video", command=self.load_video, width=15)
        self.btn_video.pack(side=tk.LEFT, padx=10)
        
        # Preview area
        self.preview_frame = tk.Frame(self.root, bd=2, relief=tk.SUNKEN)
        self.preview_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.preview_label = tk.Label(self.preview_frame, text="No image loaded", bg="lightgray")
        self.preview_label.pack(fill=tk.BOTH, expand=True)
        
        # Status label
        self.status_label = tk.Label(self.root, text="Ready", anchor=tk.W)
        self.status_label.pack(fill=tk.X, padx=20, pady=5)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not file_path:
            return
        
        self.status_label.config(text="Processing...")
        self.root.update()
        
        frame = cv2.imread(file_path)
        if frame is None:
            messagebox.showerror("Error", "Could not load image.")
            return
        
        processed_frame, count = self.engine.process_frame(frame)
        self.display_frame(processed_frame)
        
        self.status_label.config(text=f"Detected {count} carnivore(s)")
        messagebox.showinfo("Detection Complete", f"Detection Complete: {count} Carnivores identified.")

    def load_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if not file_path:
            return
        
        self.btn_image.config(state=tk.DISABLED)
        self.btn_video.config(state=tk.DISABLED)
        self.status_label.config(text="Processing video...")
        
        threading.Thread(target=self.process_video, args=(file_path,), daemon=True).start()

    def process_video(self, file_path):
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open video.")
            self.reset_buttons()
            return
        
        max_carnivores = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame, count = self.engine.process_frame(frame)
            max_carnivores = max(max_carnivores, count)
            self.display_frame(processed_frame)
            cv2.waitKey(1)
        
        cap.release()
        self.reset_buttons()
        self.status_label.config(text=f"Video complete - Max carnivores: {max_carnivores}")
        messagebox.showinfo("Video Complete", f"Video Scan Complete.\nMax Carnivores in a frame: {max_carnivores}")

    def reset_buttons(self):
        self.btn_image.config(state=tk.NORMAL)
        self.btn_video.config(state=tk.NORMAL)

    def display_frame(self, frame):
        h, w = frame.shape[:2]
        max_h = 500
        scale = max_h / h
        new_w, new_h = int(w * scale), int(h * scale)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        
        self.preview_label.config(image=img_tk, text="")
        self.preview_label.image = img_tk


if __name__ == "__main__":
    root = tk.Tk()
    app = AnimalDetectionApp(root)
    root.mainloop()
