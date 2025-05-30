import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import pandas as pd
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

#Bank Coin Preprocessing and Bank Coin Segmentation
df = pd.read_csv('coin_labels_augmented.csv')
image_dir = 'augmented_dataset'
df['filepath'] = df['filename'].apply(lambda x: os.path.join(image_dir, x))
df = df[['filepath', 'denomination']].sample(frac=1).reset_index(drop=True)

class_names = sorted(df['denomination'].unique())
class_to_index = {c: i for i, c in enumerate(class_names)}
df['label_idx'] = df['denomination'].map(class_to_index)

train_val_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label_idx'], random_state=42)
def grayscale_canny(image):
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_applied = clahe.apply(gray)
    normalized = cv2.normalize(clahe_applied, None, 0, 255, cv2.NORM_MINMAX)
    bilateral_filtered = cv2.bilateralFilter(normalized, d=9, sigmaColor=75, sigmaSpace=75)
    edges = cv2.Canny(bilateral_filtered, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = np.zeros_like(edges)
    cv2.drawContours(contour_img, contours, -1, color=255, thickness=1)
    contour_img = contour_img.astype(np.float32) / 255.0
    return np.expand_dims(contour_img, axis=-1)


datagen = ImageDataGenerator(preprocessing_function=grayscale_canny)

#Bank Coin Features Extraction
def create_model():
    model = tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), activation=None, input_shape=(128, 128, 3)),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(64, (3, 3), activation=None),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(128, (3, 3), activation=None),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Flatten(),
        layers.Dense(128),
        layers.Dropout(0.5),
        layers.Dense(len(class_names), activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

#Bank Coin Classification
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []
best_val_loss = np.inf
best_fold = -1
best_model = None
best_history = None

for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_df, train_val_df['label_idx'])):
    print(f"\n--- Fold {fold+1} ---")

    train_fold_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_fold_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    # Generators for this fold
    train_gen = datagen.flow_from_dataframe(
        train_fold_df,
        x_col='filepath',
        y_col='denomination',
        target_size=(128,128),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=True
    )
    val_gen = datagen.flow_from_dataframe(
        val_fold_df,
        x_col='filepath',
        y_col='denomination',
        target_size=(128,128),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=False
    )

    model = create_model()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ModelCheckpoint(f'best_model_fold{fold+1}.keras', monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=15,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate on val set
    val_loss, val_acc = model.evaluate(val_gen, verbose=0)
    print(f"Fold {fold+1} validation accuracy: {val_acc:.4f}, val loss: {val_loss:.4f}")

    fold_accuracies.append(val_acc)

    # Keep best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_fold = fold
        best_model = model
        best_history = history

print(f"\nCross-validation accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
print(f"Best fold: {best_fold+1} with val loss {best_val_loss:.4f}")

# Save best model
best_model.save('best_model_overall.keras')

# --- Evaluate best model on TEST set ---
test_gen = datagen.flow_from_dataframe(
    test_df.reset_index(drop=True),
    x_col='filepath',
    y_col='denomination',
    target_size=(128,128),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=False
)

test_loss, test_acc = best_model.evaluate(test_gen, verbose=1)
print(f"Test set accuracy: {test_acc:.4f}")

# Classification report on test set
test_preds_prob = best_model.predict(test_gen)
test_preds = np.argmax(test_preds_prob, axis=1)
true_test_labels = test_gen.classes

print("\nClassification Report on test set:")
print(classification_report(true_test_labels, test_preds, target_names=class_names))

# --- Plot training vs validation accuracy of best fold ---
plt.figure(figsize=(8,6))
plt.plot(best_history.history['accuracy'], label='Training Accuracy')
plt.plot(best_history.history['val_accuracy'], label='Validation Accuracy')
plt.title(f'Training and Validation Accuracy - Best Fold {best_fold+1}')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()
