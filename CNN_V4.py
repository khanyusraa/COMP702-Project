import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, Model, Input
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
import json

from tensorflow.keras.layers import Input, Dense, Concatenate, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model


#Hybrid Model Definition
def create_hybrid_model(input_shape_rgb=(128, 128, 3), input_shape_gray=(128, 128, 1),
                        input_shape_hu=(7,), num_classes=7):
    #RGB stream
    rgb_input = Input(shape=input_shape_rgb, name="rgb_input")
    mobilenet_base = MobileNetV2(include_top=False, weights="imagenet", input_shape=input_shape_rgb)
    mobilenet_base.trainable = False
    x_rgb = mobilenet_base(rgb_input)
    x_rgb = layers.GlobalAveragePooling2D()(x_rgb)

    #Grayscale stream
    gray_input = Input(shape=input_shape_gray, name="gray_input")
    x_gray = layers.Conv2D(32, (3, 3), activation='relu')(gray_input)
    x_gray = layers.MaxPooling2D((2, 2))(x_gray)
    x_gray = layers.Conv2D(64, (3, 3), activation='relu')(x_gray)
    x_gray = layers.MaxPooling2D((2, 2))(x_gray)
    x_gray = layers.GlobalAveragePooling2D()(x_gray)


    #Hu Moments stream
    hu_input = Input(shape=input_shape_hu, name="hu_input")
    x_hu = layers.Dense(32, activation='relu')(hu_input)
    #Combined
    combined = layers.concatenate([x_rgb, x_gray, x_hu])
    x = layers.Dense(128, activation='relu')(combined)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=[rgb_input, gray_input, hu_input], outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

#Dual Input Generator
class DualInputGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size, target_size, class_indices, label_column='denomination', shuffle=True):
        self.df = df.copy()
        self.batch_size = batch_size
        self.target_size = target_size
        self.class_indices = class_indices
        self.label_column = label_column
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __getitem__(self, index):
        batch_df = self.df.iloc[index * self.batch_size: (index + 1) * self.batch_size]
        rgb_batch = np.zeros((len(batch_df), *self.target_size, 3), dtype=np.float32)
        gray_batch = np.zeros((len(batch_df), *self.target_size, 1), dtype=np.float32)
        y_batch = np.zeros((len(batch_df), len(self.class_indices)), dtype=np.float32)
        hu_batch = np.zeros((len(batch_df), 7), dtype=np.float32)
        y_batch = to_categorical(batch_df['label_idx'], num_classes=len(self.class_indices))
        for i, (_, row) in enumerate(batch_df.iterrows()):
            img = cv2.imread(row['filename'])
            img = cv2.resize(img, self.target_size)

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb = preprocess_input(rgb)
            rgb_batch[i] = rgb

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
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
            # Hu Moments
            moments = cv2.moments(contour_img)
            huMoments = cv2.HuMoments(cv2.moments(contour_img)).flatten()
            huMoments = -np.sign(huMoments) * np.log10(np.abs(huMoments) + 1e-10)  # Normalize
            hu_batch[i] = huMoments
            
            # Save to batch arrays
            rgb_batch[i] = rgb
            gray_batch[i] = np.expand_dims(contour_img, axis=-1)  
            hu_batch[i] = huMoments
            # One-hot encode labels (adjust if your labels are different)
            label = row['label_idx']

        
        return {'rgb_input': rgb_batch, 'gray_input': gray_batch, 'hu_input': hu_batch}, y_batch

#Evaluation Function
def evaluate_model(model, val_gen, val_df, class_indices):
    pred_probs = model.predict(val_gen)
    y_true = np.array([class_indices[c] for c in val_df['denomination'].tolist()])
    y_pred = np.argmax(pred_probs, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_indices.keys(), yticklabels=class_indices.keys())
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    print(classification_report(y_true, y_pred, target_names=list(class_indices.keys())))

#Plot Training History
def plot_training_history(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Training vs Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#Cross Validation and Testing
def run_cross_validation(df, class_indices, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label_idx'])):
        print(f"--- Fold {fold + 1} ---")
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()

        model = create_hybrid_model(num_classes=len(class_indices))
        train_gen = DualInputGenerator(train_df, 32, (128, 128), class_indices)
        val_gen = DualInputGenerator(val_df, 32, (128, 128), class_indices, shuffle=False)

        history = model.fit(train_gen, validation_data=val_gen, epochs=10, verbose=0)
        _, acc = model.evaluate(val_gen, verbose=0)
        print(f"Fold {fold + 1} accuracy: {acc:.4f}")
        accuracies.append(acc)

    print(f"\nCross-validation accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")

#Test Set Evaluation
def evaluate_on_test(model, test_df, class_indices):
    test_gen = DualInputGenerator(test_df, 32, (128, 128), class_indices, shuffle=False)
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"Test accuracy: {test_acc:.4f}")
    evaluate_model(model, test_gen, test_df, class_indices)

#Saves the Model
def export_model_h5(model, path="hybrid_model.h5"):
    model.save(path)
    print(f"Model exported to {path}")

#Saves Class Mapping
def export_class_indices(class_indices, path="class_indices.json"):
    with open(path, 'w') as f:
        json.dump(class_indices, f, indent=4)
    print(f"Class indices saved to {path}")

#Load and prepare data ---
df = pd.read_csv('coin_labels_augmented.csv')
image_dir = 'augmented_dataset'
df['filename'] = df['filename'].apply(lambda x: os.path.join(image_dir, x))
df = df[['filename', 'denomination']].sample(frac=1).reset_index(drop=True)

class_names = sorted(df['denomination'].unique())
class_to_index = {c: i for i, c in enumerate(class_names)}
df['label_idx'] = df['denomination'].map(class_to_index)

train_val_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label_idx'], random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.2, stratify=train_val_df['label_idx'], random_state=42)

model = create_hybrid_model(num_classes=len(class_to_index))
train_gen = DualInputGenerator(train_df, batch_size=32, target_size=(128, 128), class_indices=class_to_index)
val_gen = DualInputGenerator(val_df, batch_size=32, target_size=(128, 128), class_indices=class_to_index, shuffle=False)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint("best_hybrid_model.keras", monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
]

history = model.fit(train_gen, validation_data=val_gen, epochs=15, callbacks=callbacks)

plot_training_history(history)
evaluate_model(model, val_gen, val_df, class_to_index)
evaluate_on_test(model, test_df, class_to_index)

export_model_h5(model)
export_class_indices(class_to_index)
run_cross_validation(df, class_indices=class_to_index, n_splits=5)
