import cv2
import numpy as np
import matplotlib.pyplot as plt

#Preprocessing
image = cv2.imread("coin_dataset/50c_old_front_001.jpg")
resized = cv2.resize(image, (512, 512))
sharpen_kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
sharpened = cv2.filter2D(resized, -1, sharpen_kernel)
gray_scale = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
normalized = cv2.normalize(gray_scale, None, 0, 255, cv2.NORM_MINMAX)
equalized = cv2.equalizeHist(normalized)
gamma = 1.2
inv_gamma = 1.0 / gamma
gamma_table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")
gamma_corrected = cv2.LUT(equalized, gamma_table)

#Morphological Opening
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
opened_image = cv2.morphologyEx(gamma_corrected, cv2.MORPH_OPEN, kernel)

blurred = cv2.GaussianBlur(opened_image, (5, 5), 0)
titles = ["Original", "Sharpened", "Grayscale", "Normalized", "Histogram Equalized",
          "Gamma Corrected", "Morphological Opening", "Gaussian Blurred"]
images = [resized, sharpened, gray_scale, normalized, equalized,
          gamma_corrected, opened_image, blurred]

plt.figure(figsize=(20, 12))
for i in range(len(images)):
    plt.subplot(2, 4, i + 1)
    if len(images[i].shape) == 2:
        plt.imshow(images[i], cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
