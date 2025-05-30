import os
import cv2
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog

def preprocess(image):
    resized = cv2.resize(image, (512, 512))
    sharpen_kernel = np.array([[0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]])
    sharpened = cv2.filter2D(resized, -1, sharpen_kernel)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_applied = clahe.apply(sharpened)
    normalized = cv2.normalize(clahe_applied, None, 0, 255, cv2.NORM_MINMAX)
    blurred = cv2.GaussianBlur(normalized, (5, 5), 0)
    bilateral_filtered = cv2.bilateralFilter(blurred, d=9, sigmaColor=75, sigmaSpace=75)
    return bilateral_filtered

def load_random_coin_image(df, coin_label=None, image_dir="augmented_dataset"):
    if coin_label is None:
        coin_label = random.choice(df['denomination'].unique())

    filtered_df = df[df['denomination'] == coin_label]

    if filtered_df.empty:
        raise ValueError(f"No images found for coin label: {coin_label}")

    selected_row = filtered_df.sample(n=1).iloc[0]
    img_path = os.path.join(image_dir, selected_row['filename'])

    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image: {img_path}")

    return image, img_path, coin_label

def compute_hu_moments(image_gray):
    _, binary_img = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    moments = cv2.moments(binary_img)
    hu = cv2.HuMoments(moments)
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

    return hu_log

def collect_average_hu_moments(coin_images_dict, num_samples=2):
    hu_averages = {}

    for coin_label in coin_images_dict:
        try:
            files = coin_images_dict[coin_label]["files"]
            path = coin_images_dict[coin_label]["path"]

            if len(files) < num_samples:
                raise ValueError(f"Not enough images in class {coin_label} to sample {num_samples}.")

            selected_files = random.sample(files, num_samples)
            hu_list = []

            for file in selected_files:
                img_path = os.path.join(path, file)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise ValueError(f"Could not load image: {img_path}")

                hu = compute_hu_moments(image)
                hu_list.append(hu)

            hu_avg = np.mean(hu_list, axis=0).flatten()
            hu_averages[coin_label] = hu_avg

        except Exception as e:
            print(f"Error processing {coin_label}: {e}")

    return hu_averages

def plot_average_hu_table(hu_averages):
    column_labels = ['Coin Type'] + [f'Hu[{i}]' for i in range(7)]
    table_data = []

    for coin_type, hu in hu_averages.items():
        row = [coin_type] + [round(float(val), 6) for val in hu]
        table_data.append(row)

    _, ax = plt.subplots(figsize=(8, 0.3 + 0.3 * len(table_data)))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(
        cellText=table_data,
        colLabels=column_labels,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.0)

    plt.title("Average Hu Moments per Coin Type", fontsize=14, weight='bold', pad=20)
    plt.subplots_adjust(top=0.85)
    plt.show()

def plot_hu_moments_table(hu_moment_dict):
    coin_labels = list(hu_moment_dict.keys())
    hu_matrix = np.array([hu_moment_dict[label].flatten() for label in coin_labels])

    hu_matrix_rounded = np.round(hu_matrix, 4)

    column_labels = [f'Hu[{i}]' for i in range(7)]

    _, ax = plt.subplots(figsize=(12, 0.5 + 0.5 * len(coin_labels)))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(
        cellText=hu_matrix_rounded,
        rowLabels=coin_labels,
        colLabels=column_labels,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.5, 1.5)

    plt.title("Hu Moments per Coin Type", fontsize=14, weight='bold', pad=20)
    plt.show()

def compute_orb_features(image, max_features=100):
    orb = cv2.ORB_create(nfeatures=max_features)
    keypoints, descriptors = orb.detectAndCompute(image, None)

    if descriptors is None:
        return np.zeros(max_features * 32, dtype=np.float32)

    descriptors = descriptors.flatten()

    desired_length = max_features * 32 
    if descriptors.shape[0] > desired_length:
        descriptors = descriptors[:desired_length]
    else:
        padding = np.zeros(desired_length - descriptors.shape[0], dtype=np.float32)
        descriptors = np.concatenate([descriptors, padding])

    return descriptors

def show_orb_keypoints(image, keypoints, title="ORB Keypoints"):
    img_with_kp = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=0)
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(img_with_kp, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

def extract_raw_pixels(image, size=(32, 32)):
    resized = cv2.resize(image, size)
    return resized.flatten()

def test_parameters(X_train, y_train):
    param_grid = {
        'max_depth': [10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5],
        'max_features': ['sqrt', 'log2', None]
    }

    grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    return grid_search.best_params_

def compute_edges(image):
    return cv2.Canny(image, threshold1=100, threshold2=200)

def compute_hog(image):
    edges = compute_edges(image)

    if edges.dtype != np.uint8:
        edges = edges.astype(np.uint8)
    
    if edges.shape != image.shape:
        raise ValueError("Mask and image size do not match!")

    masked_image = cv2.bitwise_and(image, image, mask=edges)

    features, _ = hog(
        masked_image, 
        orientations=9, 
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), 
        block_norm='L2-Hys',
        visualize=True
    )
    return features

def train_decision_tree_classifier(type, coin_images_dict, num_samples=1000):
    X = []
    y = []

    for coin_label in coin_images_dict:
        files = coin_images_dict[coin_label]["files"]
        path = coin_images_dict[coin_label]["path"]

        if len(files) < num_samples:
            continue  

        selected_files = random.sample(files, num_samples)

        for file in selected_files:
            img_path = os.path.join(path, file)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            preprocessed = preprocess(image)
            if type==0:
                features = compute_hu_moments(preprocessed).flatten()
                string_type = "Hu Moments"
            elif type==1:
                features = compute_orb_features(preprocessed).flatten()
                string_type = "Orb Features"
            elif type==2:
                features = extract_raw_pixels(preprocessed, size=(64, 64))
                string_type = "Raw Pixels"
            elif type==3:
                features = compute_edges(preprocessed).flatten()
                string_type = "Edges"
            elif type==4:
                features = compute_hog(preprocessed)
                string_type = "HOG Features"

            X.append(features)
            y.append(coin_label)

    if not X:
        raise ValueError("No training data collected.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = DecisionTreeClassifier(
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42
    )
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"\n=== Cross-Validation Accuracy Scores ===")
    for i, score in enumerate(cv_scores, 1):
        print(f"Fold {i}: {score:.4f}")
    print(f"Mean CV Accuracy: {np.mean(cv_scores):.4f}")
    print(f"Std Dev CV Accuracy: {np.std(cv_scores):.4f}")
    model.fit(X_train, y_train)

    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    plt.figure(figsize=(6, 4))
    plt.bar(['Training Accuracy', 'Validation Accuracy'], [train_accuracy, test_accuracy], color=['blue', 'green'])
    plt.ylim(0, 1)
    plt.title(f"{string_type} Decision Tree Accuracy")
    plt.ylabel('Accuracy')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    y_pred = model.predict(X_test)
    print(f"\n=== {string_type} Decision Tree Classification Report ===")
    print(classification_report(y_test, y_pred, zero_division=0))

    return model

def adjust_contrast(image, factor):
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)

def adjust_brightness(image, value):
    return cv2.convertScaleAbs(image, alpha=1.0, beta=value)

def add_gaussian_noise(image, mean=0, stddev=10):
    noise = np.random.normal(mean, stddev, image.shape).astype(np.float32)
    noisy_img = image.astype(np.float32) + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img

def main():
    INPUT_DIR = "augmented_coin_dataset/augmented_dataset"
    CSV_PATH = "augmented_coin_dataset/coin_labels_augmented.csv"

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    if 'filename' not in df.columns or 'denomination' not in df.columns:
        raise ValueError("CSV must contain 'filename' and 'denomination' columns.")

    coin_images = {}
    for _, row in df.iterrows():
        label = row['denomination']
        filename = row['filename']
        if label not in coin_images:
            coin_images[label] = {"files": [], "path": INPUT_DIR}
        coin_images[label]["files"].append(filename)
    
    #for k, v in coin_images.items():
        #print(f"{k}: {len(v['files'])} images")

    # Train the decision tree classifiers
    #hu_model = train_decision_tree_classifier(0, coin_images, num_samples=300)
    #orb_model = train_decision_tree_classifier(1, coin_images, num_samples=300)
    raw_model = train_decision_tree_classifier(2, coin_images, num_samples=300)
    #edge_model = train_decision_tree_classifier(3, coin_images, num_samples=300)
    #hog_model = train_decision_tree_classifier(4, coin_images, num_samples=300)
    joblib.dump(raw_model, "raw_pixels_decision_tree.pkl")


    '''
    image, _, label = load_random_coin_image(df, image_dir=INPUT_DIR)

    if type==0:
        features = compute_hu_moments(image).flatten()
    elif type==1:
        features = compute_orb_features(image).flatten()
    elif type==2:
        features = extract_raw_pixels(image, size=(64, 64))
    prediction = clf.predict([features])

    print(f"Predicted coin type: {prediction[0]}")
    print(f"Actual coin type: {label}")'''

if __name__ == "__main__":
    main()