import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from skimage.transform import resize
from skimage import measure

#List of coin images
image_paths = {
    "5c": "coin_dataset/5c_new_front_002.jpg",
    "10c": "coin_dataset/10c_new_front_013.jpg",
    "20c": "coin_dataset/20c_new_front_016.jpg",
    "50c": "coin_dataset/50c_new_front_012.jpg",
    "R1": "coin_dataset/R1_new_front_002.jpg",
    "R2": "coin_dataset/R2_new_front_009.jpg",
    "R5": "coin_dataset/R5_new_front_010.jpg",
}

hu_data = []
contour_data = []
fourier_data = []
lbp_data = []
orb_data = []

def extract_fourier_descriptors(contour, num_coeffs=10):
    contour = contour.squeeze()  #shape (N, 2)
    if contour.ndim != 2 or contour.shape[1] != 2:
        raise ValueError("Contour shape invalid for Fourier Descriptors.")
    contour_complex = contour[:, 0] + 1j * contour[:, 1]
    fourier_result = np.fft.fft(contour_complex)
    descriptors = np.abs(fourier_result[:num_coeffs])
    return descriptors

#ORB detector
orb = cv2.ORB_create(nfeatures=50)

#Storage for 50c visuals
visual_50c = {}

for label, path in image_paths.items():
    image = cv2.imread(path)
    if image is None:
        print(f"Failed to load image: {path}")
        continue

    resized = cv2.resize(image, (512, 512))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    #Preprocessing and Enhancements
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, sharpen_kernel)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_applied = clahe.apply(sharpened)
    normalized = cv2.normalize(clahe_applied, None, 0, 255, cv2.NORM_MINMAX)
    bilateral_filtered = cv2.bilateralFilter(normalized, d=9, sigmaColor=75, sigmaSpace=75)

    #Edge Detection and Contour Segmentation
    edges = cv2.Canny(bilateral_filtered, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(bilateral_filtered)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    #Hu Moments
    moments = cv2.moments(mask)
    hu = cv2.HuMoments(moments).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    hu_data.append((label, hu_log))

    #Contour Features
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        compactness = (perimeter ** 2) / (area + 1e-10)
        contour_data.append((label, [area, perimeter, compactness]))

        #Fourier Descriptors
        fourier = extract_fourier_descriptors(c, num_coeffs=7)
        fourier_data.append((label, fourier))
    else:
        contour_data.append((label, [0, 0, 0]))
        fourier_data.append((label, [0]*7))

    #Local Binary Pattern (LBP)
    lbp = local_binary_pattern(mask, P=8, R=1, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    lbp_data.append((label, hist[:7]))

    #ORB Features
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    num_kp = len(keypoints) if keypoints is not None else 0
    if descriptors is not None:
        mean_desc = descriptors.mean(axis=0)
    else:
        mean_desc = np.zeros(32) 
    orb_data.append((label, np.hstack(([num_kp], mean_desc[:6]))))  #First 7 values: #kp + first 6 descriptor means

    if label == "50c":
        visual_50c['original'] = resized
        visual_50c['mask'] = mask
        visual_50c['lbp'] = lbp

def display_table(title, data, col_labels):
    fig, ax = plt.subplots(figsize=(10, 4))
    row_labels = []
    table_data = []
    for label, values in data:
        row_labels.append(label)
        table_data.append([f"{v:.5f}" for v in values])
    ax.axis('off')
    ax.axis('tight')
    table = ax.table(cellText=table_data,
                     rowLabels=row_labels,
                     colLabels=col_labels,
                     loc='center',
                     cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    plt.title(title, fontsize=14, weight='bold', pad=20)
    plt.tight_layout()
    plt.show()

display_table("Hu Moments per Coin Type", hu_data, [f"Hu{i+1}" for i in range(7)])
display_table("Contour Features per Coin Type", contour_data, ["Area", "Perimeter", "Compactness"])
display_table("Fourier Descriptors per Coin Type", fourier_data, [f"F{i+1}" for i in range(7)])
display_table("Local Binary Pattern Histogram per Coin Type", lbp_data, [f"LBP{i+1}" for i in range(7)])
display_table("ORB Features per Coin Type (#Keypoints + Desc Means)", orb_data, ["NumKP"] + [f"DescMean{i+1}" for i in range(6)])

