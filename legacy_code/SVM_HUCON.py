import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn import svm
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from skimage.feature import hog
import joblib

image = cv2.imread("coin_dataset/50c_old_front_001.jpg")
#Image Enhancement and Preprocessing
resized = cv2.resize(image, (512, 512))
sharpen_kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
sharpened = cv2.filter2D(resized, -1, sharpen_kernel)
gray_scale = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_applied = clahe.apply(gray_scale)
normalized = cv2.normalize(clahe_applied, None, 0, 255, cv2.NORM_MINMAX)
blurred = cv2.GaussianBlur(normalized, (5, 5), 0)
bilateral_filtered = cv2.bilateralFilter(blurred, d=9, sigmaColor=75, sigmaSpace=75)

#Image Segmentation
edges = cv2.Canny(bilateral_filtered, threshold1=30, threshold2=100)
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_img = cv2.cvtColor(bilateral_filtered, cv2.COLOR_GRAY2BGR)
cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)

#Display preprocessing steps
titles = ["Original", "Sharpened", "Grayscale", "CLAHE Applied", "Normalized",
          "Blurred", "Bilateral Filtered", "Edges", "Contours"]
images = [resized, sharpened, gray_scale, clahe_applied, normalized,
          blurred, bilateral_filtered, edges, contour_img]

plt.figure(figsize=(20, 12))
for i in range(len(images)):
    plt.subplot(3, 3, i + 1)
    if len(images[i].shape) == 2:
        plt.imshow(images[i], cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()

#Feature Extraction
def extract_hog_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features, _ = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), block_norm='L2-Hys',
                      visualize=True, feature_vector=True)
    return features
def extract_hu_moments(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    moments = cv2.moments(gray)
    hu_moments = cv2.HuMoments(moments).flatten()
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
    return hu_moments
def extract_features(image):
    hog_features = extract_hog_features(image)
    hu_features = extract_hu_moments(image)
    return np.concatenate((hog_features, hu_features))
csv_path = 'coin_labels_augmented.csv'
df = pd.read_csv(csv_path)

X, y = [], []
for idx, row in df.iterrows():
    img_path = os.path.join('augmented_dataset', row['filename'])
    img = cv2.imread(img_path)
    if img is None:
        continue
    img = cv2.resize(img, (128, 128))
    features = extract_features(img)
    X.append(features)
    y.append(row['denomination'])

X, y = np.array(X), np.array(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#Classification with SVM
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("SVM Classification Report:")
print(classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy:.4f}")

#Save model and scaler
joblib.dump(clf, 'svm_model_augmented2.pkl')
joblib.dump(scaler, 'scaler_augmented.pkl')

#Prediction on Processed Test Image
test_img = cv2.resize(image, (128, 128))
test_feat = extract_features(test_img).reshape(1, -1)
test_feat_scaled = scaler.transform(test_feat)

prediction = clf.predict(test_feat_scaled)
print(f"Predicted class for test image: {prediction[0]}")

#Cross-Validation for Robust Accuracy Estimate
cv_scores = cross_val_score(clf, X_scaled, y, cv=5)
print(f"Cross-validation Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
labels = sorted(np.unique(y))  # Ensure correct order

#Using seaborn heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()
