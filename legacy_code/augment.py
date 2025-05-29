import os
import cv2
import numpy as np
import pandas as pd

#Directories
INPUT_DIR = "coin_dataset"
OUTPUT_DIR = "augmented_dataset"
CSV_PATH = "coin_labels.csv"
NEW_CSV_PATH = "coin_labels_augmented.csv"

#Augments
ROTATION_ANGLES = [90, 180, 270]
CONTRAST_FACTORS = [1.2, 1.5] 
BRIGHTNESS_FACTORS = [30, 60]  
BLUR_KERNELS = [(3, 3), (5, 5)]

os.makedirs(OUTPUT_DIR, exist_ok=True)
df = pd.read_csv(CSV_PATH)
augmented_rows = []

#Helper Methods
def adjust_contrast(image, factor):
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)

def adjust_brightness(image, value):
    return cv2.convertScaleAbs(image, alpha=1.0, beta=value)

def add_gaussian_noise(image, mean=0, stddev=10):
    noise = np.random.normal(mean, stddev, image.shape).astype(np.float32)
    noisy_img = image.astype(np.float32) + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img

#Apply all Augments
for _, row in df.iterrows():
    filename = row['filename']
    image_path = os.path.join(INPUT_DIR, filename)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Skipping missing image: {filename}")
        continue

    base_name = os.path.splitext(filename)[0]
    orig_filename = f"{base_name}_orig.jpg"
    cv2.imwrite(os.path.join(OUTPUT_DIR, orig_filename), image)
    augmented_rows.append({**row, "filename": orig_filename})

    #Rotation Augment
    for angle in ROTATION_ANGLES:
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        rotated_filename = f"{base_name}_rot{angle}.jpg"
        cv2.imwrite(os.path.join(OUTPUT_DIR, rotated_filename), rotated)
        augmented_rows.append({**row, "filename": rotated_filename})

    #Contrast Augment
    for factor in CONTRAST_FACTORS:
        contrast_img = adjust_contrast(image, factor)
        contrast_filename = f"{base_name}_contrast_{factor:.1f}.jpg"
        cv2.imwrite(os.path.join(OUTPUT_DIR, contrast_filename), contrast_img)
        augmented_rows.append({**row, "filename": contrast_filename})

    #Brightness Augment
    for value in BRIGHTNESS_FACTORS:
        bright_img = adjust_brightness(image, value)
        bright_filename = f"{base_name}_bright_{value}.jpg"
        cv2.imwrite(os.path.join(OUTPUT_DIR, bright_filename), bright_img)
        augmented_rows.append({**row, "filename": bright_filename})

    #Blurring
    for k in BLUR_KERNELS:
        blurred_img = cv2.GaussianBlur(image, k, 0)
        blur_filename = f"{base_name}_blur{k[0]}x{k[1]}.jpg"
        cv2.imwrite(os.path.join(OUTPUT_DIR, blur_filename), blurred_img)
        augmented_rows.append({**row, "filename": blur_filename})

    #Gaussian noise
    noisy_img = add_gaussian_noise(image)
    noise_filename = f"{base_name}_noise.jpg"
    cv2.imwrite(os.path.join(OUTPUT_DIR, noise_filename), noisy_img)
    augmented_rows.append({**row, "filename": noise_filename})

#Generates new CSV file
augmented_df = pd.DataFrame(augmented_rows)
augmented_df.to_csv(NEW_CSV_PATH, index=False)

print(f"Total augmented images saved: {len(augmented_df)}")
print(f"New CSV saved to: {NEW_CSV_PATH}")
