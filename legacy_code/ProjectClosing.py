import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

image = cv.imread("R5_old_back_002.jpg")
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
kernel = np.ones((3, 3), np.uint8)

resized = cv.resize(image, (512, 512))
sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
sharpened = cv.filter2D(resized, -1, sharpen_kernel)

blurred = cv.GaussianBlur(image, (7, 7), 0)
closing = cv.morphologyEx(gray, cv.MORPH_CLOSE, kernel)
normalized = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX)

# Convert color images for Matplotlib
original = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
blurred = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
sharpened = cv.cvtColor(sharpened, cv.COLOR_BGR2GRAY)
normalized = cv.cvtColor(normalized, cv.COLOR_BGR2GRAY)

plt.figure(figsize=(18, 12))
'''
plt.subplot(2, 3, 1)
plt.imshow(original)
plt.title("Original")
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(gray, cmap='gray')
plt.title("Grayscale")
plt.axis('off')
'''
plt.subplot(2, 3, 3)
plt.imshow(closing, cmap='gray')
plt.title("Closed (Noise Removal)")
plt.axis('off')
'''
plt.subplot(2, 3, 4)
plt.imshow(blurred, cmap='gray')
plt.title("Blurred")
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(sharpened,  cmap='gray')
plt.title("Sharpened")
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(normalized,  cmap='gray')
plt.title("Normalized")
plt.axis('off')
'''
plt.tight_layout()
plt.show()


'''
#WaterShedding

ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
 
# sure background area
sure_bg = cv.dilate(opening,kernel,iterations=3)
 
# Finding sure foreground area
dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
 
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg,sure_fg)

# Marker labelling
ret, markers = cv.connectedComponents(sure_fg)
 
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
 
# Now, mark the region of unknown with zero
markers[unknown==255] = 0

markers = cv.watershed(image,markers)
image[markers == -1] = [255,0,0]

cv.imshow("Solo leveler", image)
cv.waitKey(0)
'''
