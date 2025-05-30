import cv2
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from skimage.feature import hog, local_binary_pattern
import joblib
import matplotlib.pyplot as plt



CSV_PATH = r"coin_labels_augmented.csv"
IMAGE_DIR = r"augmented_dataset"
TEST_IMG_PATH = r"augmented_coin_dataset\augmented_dataset\50c_old_front_005_orig.jpg"

#LBP Parameters
LBP_RADIUS = 1
LBP_POINTS = 8 * LBP_RADIUS
LBP_METHOD = 'uniform'

def compute_lbp_features(gray_img):
    lbp =local_binary_pattern(gray_img, P=LBP_POINTS, R=LBP_RADIUS, method=LBP_METHOD)
    hist, _ = np.histogram(lbp.ravel(), bins=32, range=(0, 256), density=True)
    entropy_lbp = -np.sum(hist * np.log2(hist + 1e-6))
    return np.mean(lbp), np.std(lbp),entropy_lbp

def preprocess(image):
    # Resize & Preprocess
    image = cv2.resize(image, (512, 512))
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, sharpen_kernel)
    gray =cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    norm = cv2.normalize(clahe_img, None, 0, 255, cv2.NORM_MINMAX)
    blur = cv2.GaussianBlur(norm, (5, 5), 0)
    filtered = cv2.bilateralFilter(blur, 9, 75, 75)

    # Canny & Contours
    edges = cv2.Canny(filtered, 30, 100)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Largest Contour
    c =max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

    # LBP Texture Features
    mean_lbp, std_lbp, entropy_lbp = compute_lbp_features(filtered)

    # Hu Moments
    hu_moments =cv2.HuMoments(cv2.moments(c)).flatten()
    hu_log = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)

    # HOG Features
    resized_gray = cv2.resize(gray, (128, 128))
    hog_features = hog(resized_gray, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False, feature_vector=True)

    # Combine All
    return np.hstack([ mean_lbp, std_lbp, entropy_lbp, hog_features])

#Load the dataset
df =pd.read_csv(CSV_PATH)
X, y = [], []

for idx, row in df.iterrows():
    img_path = os.path.join(IMAGE_DIR, row['filename'])
    img = cv2.imread(img_path)
    if img is None:
        print(f"Skipping missing image: {img_path}")
        continue
    features = preprocess(img)
    if features is None:
        print(f"Skipping image with no contours: {img_path}")
        continue
    X.append(features)
    y.append(row['denomination'])

X = np.array(X)
y = np.array(y)

#Train/Test Split 80/20
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

#Train Random Forest
rf =RandomForestClassifier(n_estimators=100, random_state=7)
rf.fit(X_train, y_train)

#Evaluation
y_pred =rf.predict(X_test)
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

#print first five and the cross validation accuracy scores
for actual, predicted in list(zip(y_test, y_pred))[:5]:
    print(f"Actual: {actual} \t Predicted: {predicted}")

cv_scores =cross_val_score(rf, X_train, y_train, cv=5)
print(f"Cross-validation Accuracy Scores: {cv_scores}")
print(f"Mean Cross-validation Accuracy: {np.mean(cv_scores):.4f}")
train_accuracy = accuracy_score(y_train, rf.predict(X_train))
print(f"Training Accuracy: {train_accuracy:.4f}")

#Save Model
joblib.dump(rf, "coin_classifier_combined_features.pkl")
print("Model saved as 'coin_classifier_combined_features.pkl'.")


# graph (learning curve)
from sklearn.model_selection import learning_curve

# Generate learning curve data
train_sizes, train_scores, val_scores = learning_curve(
    rf, X, y, cv=5, scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 10)
)

#Compute mean scores
train_scores_mean = np.mean(train_scores, axis=1)
val_scores_mean = np.mean(val_scores, axis=1)

# Plotting the learning curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', label='Training Accuracy')
plt.plot(train_sizes, val_scores_mean, 'o-', label='Validation Accuracy')
plt.title('Learning Curve (Random Forest)')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend(loc='best')
plt.tight_layout()
plt.show()


# === Test on Single Image ===
#test_img = cv2.imread(TEST_IMG_PATH)
#test_feat = preprocess(test_img)
#if test_feat is not None:
 #   prediction = rf.predict([test_feat])
 #   print(f"Prediction for test image: {prediction[0]}")
#else:
#    print("Test image had no contours.")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=rf.classes_)

#confusion matrix
disp =ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf.classes_)
plt.figure(figsize=(10, 8))
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title('Confusion Matrix')
plt.grid(False)
plt.tight_layout()
plt.show()