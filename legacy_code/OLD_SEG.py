import cv2
import numpy as np
import matplotlib.pyplot as plt

#Load and Preprocess Image ---
image = cv2.imread("coin_dataset/50c_new_front_012.jpg")
resized = cv2.resize(image, (512, 512))
gray_scale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

#Sharpening
sharpen_kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
sharpened = cv2.filter2D(gray_scale, -1, sharpen_kernel)
#CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_applied = clahe.apply(sharpened)

#Normalization
normalized = cv2.normalize(clahe_applied, None, 0, 255, cv2.NORM_MINMAX)

#Bilateral Filtering
bilateral_filtered = cv2.bilateralFilter(normalized, d=9, sigmaColor=75, sigmaSpace=75)

#Display Bilateral Filtered
plt.figure(figsize=(6, 6))
plt.imshow(bilateral_filtered, cmap='gray')
plt.title("Bilateral Filtered")
plt.axis('off')
plt.show()

#Watershed Segmentation
color_img = cv2.cvtColor(bilateral_filtered, cv2.COLOR_GRAY2BGR)
ret, thresh = cv2.threshold(bilateral_filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
sure_bg = cv2.dilate(opening, kernel, iterations=3)

dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

ret, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

watershed_img = color_img.copy()
cv2.watershed(watershed_img, markers)
watershed_img[markers == -1] = [255, 0, 0]  # red boundaries

#Edge Direction (Angle) & Distance
sobelx = cv2.Sobel(bilateral_filtered, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(bilateral_filtered, cv2.CV_64F, 0, 1, ksize=5)

#Edge direction (angle)
angle = np.arctan2(sobely, sobelx) * (180 / np.pi)
angle = np.mod(angle + 360, 360)  # 0 to 360 degrees

#Edge distance (gradient Euclidean distance)
distance = np.hypot(sobelx, sobely)

#Plot angle histogram
plt.figure(figsize=(10, 4))
plt.hist(angle.ravel(), bins=36, range=(0, 360), color='teal', alpha=0.7)
plt.title("Edge Angle(Direction) Histogram")
plt.xlabel("Angle (degrees)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

#Plot edge distance histogram
plt.figure(figsize=(10, 4))
plt.hist(distance.ravel(), bins=50, color='purple', alpha=0.7)
plt.title("Edge Distance Histogram")
plt.xlabel("Distance")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

#Hough Circle Detection
circles = cv2.HoughCircles(bilateral_filtered,
                           cv2.HOUGH_GRADIENT,
                           dp=1.2,
                           minDist=30,
                           param1=50,
                           param2=30,
                           minRadius=10,
                           maxRadius=100)

circle_img = cv2.cvtColor(bilateral_filtered, cv2.COLOR_GRAY2BGR)
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv2.circle(circle_img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        cv2.circle(circle_img, (i[0], i[1]), 2, (0, 0, 255), 3)

#Display Final Results
titles = ["Watershed Segmentation", "Hough Circle Detection"]
images = [watershed_img, circle_img]

plt.figure(figsize=(12, 6))
for i in range(len(images)):
    plt.subplot(1, 2, i + 1)
    plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()
