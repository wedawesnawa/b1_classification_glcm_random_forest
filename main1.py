import os
import cv2
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from skimage.feature import graycomatrix, graycoprops
import zipfile
import random
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from sklearn.ensemble import RandomForestClassifier
import joblib

# Mengatur jalur dataset
dataset_path = "datasets(3s).zip"
dataset_dir = 'ckplus/datasets(3e)'

# Ekstrak file zip ke direktori 'ckplus'
with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
    zip_ref.extractall('ckplus')

# Daftar class labels yang ingin diklasifikasikan
class_labels = ['angry', 'happy', 'sad']

# Load image paths and labels
def load_image_paths(dataset_dir):
    image_paths = []
    labels = []
    for root, _, files in os.walk(dataset_dir):
        # Extract label dari nama folder
        label = os.path.basename(root)
        # Pastikan label termasuk dalam class_labels
        if label in class_labels:
            for file in files:
                # Pastikan file merupakan citra dengan ekstensi yang diinginkan
                if file.endswith(".png") or file.endswith(".jpg"):
                    image_paths.append(os.path.join(root, file))
                    labels.append(label)
    return image_paths, labels

# Fungsi untuk mengekstraksi fitur GLCM dari gambar
def extract_greycomatrix_features(image):
    # Menggunakan nilai angles dan distances yang tetap
    # angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    # distances = [1, 3, 4, 5, 7]
    # Hitung GLCM
    # glcm = graycomatrix(image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    glcm = graycomatrix(image, distances=[1, 3, 4, 5, 7], angles=[0], levels=256, symmetric=True, normed=True)

    # Ekstrak properti GLCM
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    asm = graycoprops(glcm, 'ASM')[0, 0]
    # contrast = graycoprops(glcm, 'contrast').ravel()
    # dissimilarity = graycoprops(glcm, 'dissimilarity').ravel()
    # homogeneity = graycoprops(glcm, 'homogeneity').ravel()
    # energy = graycoprops(glcm, 'energy').ravel()
    # correlation = graycoprops(glcm, 'correlation').ravel()
    # asm = graycoprops(glcm, 'ASM').ravel()

    # Gabungkan semua fitur menjadi satu vektor fitur
    features = np.hstack([contrast, dissimilarity, homogeneity, energy, correlation,asm])

    return features

# Fungsi untuk mengekstraksi fitur GLCM dari seluruh dataset
def extract_features_from_dataset(image_paths, labels):
    features_list = []
    for image_path in image_paths:
        # Baca gambar dalam skala abu-abu
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Resize gambar menjadi 48x48
        resized_img = cv2.resize(img, (48, 48))

        # Ekstraksi fitur GLCM
        glcm_features = extract_greycomatrix_features(resized_img)
        features_list.append(glcm_features)
    return np.array(features_list)

# Memuat jalur gambar dan label dari dataset
image_paths, labels = load_image_paths(dataset_dir)

# Mengekstraksi fitur GLCM dari dataset dengan pre-processing
glcm_features = extract_features_from_dataset(image_paths, labels)
print("Shape of GLCM features array:", glcm_features.shape)

# Split data menjadi training dan testing set
X_train, X_test, y_train, y_test = train_test_split(glcm_features, labels, test_size=0.2, random_state=42)

# Inisialisasi pipeline dengan Random Forest
pipeline_rf = Pipeline([
    ('scaler', StandardScaler()),  # Preprocessing dengan standard scaler
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42))  # Random Forest Classifier
])

# Inisialisasi LabelEncoder
label_encoder = LabelEncoder()

# Menggunakan LabelEncoder untuk mengubah label menjadi representasi numerik
y_train_encoded = label_encoder.fit_transform(y_train)

# Fit pipeline pada seluruh data training
pipeline_rf.fit(X_train, y_train_encoded)

# Simpan model yang telah dilatih
joblib.dump(pipeline_rf, 'glcm_rf_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

# Evaluasi model pada data testing
y_test_encoded = label_encoder.transform(y_test)
y_pred = pipeline_rf.predict(X_test)
print(classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_))

