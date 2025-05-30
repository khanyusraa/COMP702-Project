import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("coin_dataset/50c_new_front_012.jpg")

#Image Enhancement and Preprocessing
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

#Display only preprocessing steps
titles = ["Original", "Grayscale", "Sharpened", "CLAHE Applied", "Normalized",
          "Blurred", "Bilateral Filtered"]
images = [resized, gray_scale, sharpened, clahe_applied, normalized,
          blurred, bilateral_filtered]

plt.figure(figsize=(18, 10))
for i in range(len(images)):
    plt.subplot(2, 4, i+1)
    if len(images[i].shape) == 2:
        plt.imshow(images[i], cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
