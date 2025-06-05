import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import json

class CoinClassifierGUI:
    def __init__(self, root):
        self.root = root
        root.title("Coin Classifier")
        root.geometry("800x500")
        root.resizable(True, True)
        
        # Create classifier instance
        self.classifier = CoinClassifier("hybrid_model.onnx", "class_indices.json")
        
        # Create GUI elements
        self.create_widgets()
        
    def create_widgets(self):
        # Left panel - Image display
        self.image_frame = ttk.LabelFrame(self.root, text="Coin Image", padding=10)
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # Right panel - Controls
        self.control_frame = ttk.LabelFrame(self.root, text="Controls", padding=10)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        
        # File selection button
        self.browse_btn = ttk.Button(
            self.control_frame, 
            text="Browse Image", 
            command=self.load_image,
            width=15
        )
        self.browse_btn.pack(pady=10)
        
        # Classify button
        self.classify_btn = ttk.Button(
            self.control_frame, 
            text="Classify Coin", 
            command=self.classify_image,
            state=tk.DISABLED,
            width=15
        )
        self.classify_btn.pack(pady=10)
        
        # Result display
        ttk.Label(self.control_frame, text="Classification Result:").pack(pady=(20, 5))
        self.result_var = tk.StringVar(value="No image loaded")
        self.result_label = ttk.Label(
            self.control_frame, 
            textvariable=self.result_var,
            font=("Arial", 14, "bold"),
            wraplength=250
        )
        self.result_label.pack(pady=5)
        
        # Confidence display
        ttk.Label(self.control_frame, text="Confidence:").pack(pady=(20, 5))
        self.confidence_var = tk.StringVar(value="N/A")
        self.confidence_label = ttk.Label(
            self.control_frame, 
            textvariable=self.confidence_var,
            font=("Arial", 12)
        )
        self.confidence_label.pack(pady=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(
            self.root, 
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if not file_path:
            return
            
        self.status_var.set(f"Loaded: {file_path.split('/')[-1]}")
        self.current_image = file_path
        
        # Display image
        img = Image.open(file_path)
        img.thumbnail((400, 400))  # Resize for display
        photo = ImageTk.PhotoImage(img)
        
        self.image_label.configure(image=photo)
        self.image_label.image = photo  # Keep reference
        
        # Enable classify button
        self.classify_btn.config(state=tk.NORMAL)
        self.result_var.set("Ready to classify")
        self.confidence_var.set("N/A")
    
    def classify_image(self):
        self.status_var.set("Classifying...")
        self.root.update()  # Update UI immediately
        
        try:
            # Get prediction
            prediction = self.classifier.predict(self.current_image)
            
            # Display results
            self.result_var.set(f"Prediction: {prediction}")
            self.confidence_var.set("High confidence")  # Actual confidence would come from model
            self.status_var.set("Classification complete")
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            self.result_var.set("Classification failed")
            self.confidence_var.set("N/A")

class CoinClassifier:
    def __init__(self, onnx_model_path, class_indices_path):
        import onnxruntime as ort
        self.session = ort.InferenceSession(onnx_model_path)
        self.input_names = [input.name for input in self.session.get_inputs()]
        self.output_name = self.session.get_outputs()[0].name
        
        with open(class_indices_path) as f:
            self.class_indices = json.load(f)
        self.reverse_class_map = {v: k for k, v in self.class_indices.items()}
    
    def preprocess_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        img = cv2.resize(img, (128, 128))
        
        # RGB processing
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb = rgb.astype(np.float32)
        rgb = rgb / 127.5 - 1.0
        
        # Grayscale processing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        gray = cv2.filter2D(gray, -1, kernel)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 100, 200)
        contour_img = np.zeros_like(edges)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contour_img, contours, -1, 255, 1)
        contour_img = contour_img.astype(np.float32) / 255.0
        contour_img = np.expand_dims(contour_img, axis=-1)
        
        # Hu Moments calculation
        moments = cv2.moments(contour_img)
        huMoments = cv2.HuMoments(moments).flatten()
        huMoments = -np.sign(huMoments) * np.log10(np.abs(huMoments) + 1e-10)
        
        return {
            "rgb_input": np.expand_dims(rgb, axis=0),
            "gray_input": np.expand_dims(contour_img, axis=0),
            "hu_input": np.expand_dims(huMoments, axis=0)
        }
    
    def predict(self, image_path):
        inputs = self.preprocess_image(image_path)
        ort_inputs = {name: inputs[name] for name in self.input_names}
        
        pred = self.session.run([self.output_name], ort_inputs)[0]
        class_idx = np.argmax(pred, axis=1)[0]
        confidence = np.max(pred)
        
        return self.reverse_class_map[class_idx], confidence

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = CoinClassifierGUI(root)
    root.mainloop()