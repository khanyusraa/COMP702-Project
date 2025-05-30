from skimage.feature import hog
from skimage import exposure
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from skimage.transform import resize
from skimage import measure

image_paths = {
    "5c": "coin_dataset/5c_new_front_002.jpg",
    "10c": "coin_dataset/10c_new_front_013.jpg",
    "20c": "coin_dataset/20c_new_front_016.jpg",
    "50c": "coin_dataset/50c_new_front_012.jpg",
    "R1": "coin_dataset/R1_new_front_002.jpg",
    "R2": "coin_dataset/R2_new_front_009.jpg",
    "R5": "coin_dataset/R5_new_front_010.jpg",
}

hog_data = []

for label, path in image_paths.items():

    #Load and Preprocess Image
    image = cv2.imread(path)
    resized = cv2.resize(image, (512, 512))
    gray_scale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(gray_scale, -1, sharpen_kernel)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_applied = clahe.apply(sharpened)
    normalized = cv2.normalize(clahe_applied, None, 0, 255, cv2.NORM_MINMAX)
    bilateral_filtered = cv2.bilateralFilter(normalized, d=9, sigmaColor=75, sigmaSpace=75)

    #Edge Detection and Contour Mask
    edges = cv2.Canny(bilateral_filtered, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(bilateral_filtered)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
    roi = cv2.bitwise_and(gray_scale, gray_scale, mask=mask)
    roi_resized = cv2.resize(roi, (128, 128))
    roi_normalized = roi_resized.astype('float32') / 255.0
    hog_features, hog_image = hog(roi_normalized,
                                   orientations=9,
                                   pixels_per_cell=(8, 8),
                                   cells_per_block=(2, 2),
                                   block_norm='L2-Hys',
                                   visualize=True,
                                   feature_vector=True)
    hog_data.append((label, np.round(hog_features[:10], 5)))  #First 10 features for table
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(roi_resized, cmap='gray')
    axs[0].set_title(f"{label} ROI")
    axs[0].axis('off')
    axs[1].imshow(hog_image, cmap='gray')
    axs[1].set_title(f"{label} HOG Image")
    axs[1].axis('off')
    plt.suptitle(f"HOG Visualization for {label}")
    plt.tight_layout()
    plt.show()
fig, ax = plt.subplots(figsize=(12, 4))
table_data = []
row_labels = []
for label, hog_vals in hog_data:
    row_labels.append(label)
    table_data.append([f"{val:.5f}" for val in hog_vals])

column_labels = [f"HOG{i+1}" for i in range(1, 11)]

ax.axis('off')
ax.axis('tight')
table = ax.table(cellText=table_data,
                 rowLabels=row_labels,
                 colLabels=column_labels,
                 loc='center',
                 cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

plt.title("HOG Features per Coin Type (First 10)", fontsize=14, weight='bold', pad=20)
plt.tight_layout()
plt.show()
