import cv2
import numpy as np
import matplotlib.pyplot as plt

#Load and Preprocess Image
image = cv2.imread("coin_dataset/50c_new_front_012.jpg")
resized = cv2.resize(image, (512, 512))
gray_scale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
sharpen_kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
sharpened = cv2.filter2D(gray_scale, -1, sharpen_kernel)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_applied = clahe.apply(sharpened)
normalized = cv2.normalize(clahe_applied, None, 0, 255, cv2.NORM_MINMAX)
blurred = cv2.GaussianBlur(normalized, (5, 5), 0)
bilateral_filtered = cv2.bilateralFilter(blurred, d=9, sigmaColor=75, sigmaSpace=75)
#Display Bilateral Filtered
plt.figure(figsize=(6, 6))
plt.imshow(bilateral_filtered, cmap='gray')
plt.title("Bilateral Filtered")
plt.axis('off')
plt.show()

#Segmentation
edges = cv2.Canny(bilateral_filtered, threshold1=50, threshold2=150)

#Contour Detection
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_img = cv2.cvtColor(bilateral_filtered, cv2.COLOR_GRAY2BGR)
cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)

#Display Results
titles = ["Canny Edge Detection", "Contour Detection"]
images = [edges, contour_img]

plt.figure(figsize=(12, 6))
for i in range(2):
    plt.subplot(1, 2, i + 1)
    if i == 0:
        plt.imshow(images[i], cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()
