import os
import json
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import numpy as np
import cv2
import tensorflow as tf

#Load model and class indices
model = tf.keras.models.load_model("hybrid_model.h5")
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)
index_to_class = {v: k for k, v in class_indices.items()}

def preprocess_image(filepath, target_size=(128, 128)):
    img = cv2.imread(filepath)
    img = cv2.resize(img, target_size)

    #RGB preprocessing
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb = tf.keras.applications.mobilenet_v2.preprocess_input(rgb)

    #Grayscale preprocessing for contours
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
    cv2.drawContours(contour_img, contours, -1, color=255, thickness=1)
    contour_img = contour_img.astype(np.float32) / 255.0

    #Hu Moments
    hu_moments = cv2.HuMoments(cv2.moments(contour_img)).flatten()
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)

    return {
        "rgb_input": np.expand_dims(rgb.astype(np.float32), axis=0),
        "gray_input": np.expand_dims(contour_img[..., np.newaxis].astype(np.float32), axis=0),
        "hu_input": np.expand_dims(hu_moments.astype(np.float32), axis=0)
    }

class CoinClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Coin Classifier")

        self.label = Label(root, text="Upload a coin image")
        self.label.pack()

        self.image_label = Label(root)
        self.image_label.pack()

        self.result_label = Label(root, text="", font=("Arial", 14))
        self.result_label.pack()

        self.upload_button = Button(root, text="Upload Image", command=self.load_image)
        self.upload_button.pack()

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if not file_path:
            return

        img = Image.open(file_path)
        img = img.resize((200, 200))
        img_tk = ImageTk.PhotoImage(img)
        self.image_label.configure(image=img_tk)
        self.image_label.image = img_tk

        inputs = preprocess_image(file_path)
        preds = model.predict(inputs)[0]
        pred_class = index_to_class[np.argmax(preds)]
        confidence = np.max(preds) * 100

        self.result_label.config(text=f"Prediction: {pred_class} ({confidence:.2f}%)")

#Launch the app
root = tk.Tk()
app = CoinClassifierApp(root)
root.mainloop()
