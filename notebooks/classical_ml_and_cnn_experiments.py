"""
Classical ML and CNN Experiments for ASD Facial Image Classification
===================================================================

This script contains a cleaned experimental pipeline for ASD facial image
classification using both:

1. Classical machine learning (SVM)
2. CNN-based transfer learning (MobileNetV2 and EfficientNetB3)

Associated paper:
Taneera, S. N., Samadi, S., Özmen, S., & Alhajj, R. (2025).
Game-Based Diagnosing of Children with Autism Spectrum Disorder.

Implementation note:
- This script is a cleaned GitHub-ready version based on the original
  experimental notebook.
- The work builds upon the dataset and reference implementation originally
  provided by Gerry (2020).

Expected dataset structure
--------------------------
dataset_root/
    train/
        autistic/
        non_autistic/
    valid/
        autistic/
        non_autistic/
    test/
        autistic/
        non_autistic/

Recommended file name
---------------------
classical_ml_and_cnn_experiments.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

try:
    from tensorflow.keras.applications import EfficientNetB3
except ImportError:
    from efficientnet.tfkeras import EfficientNetB3

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
sns.set_style("darkgrid")


# =============================================================================
# Configuration
# =============================================================================

DATASET_DIR = "path_to_dataset_directory"
IMG_SIZE_CNN = (200, 200)
IMG_SIZE_MOBILENET = (192, 192)
BATCH_SIZE = 25
CNN_EPOCHS = 20
LEARNING_RATE = 0.002
SEED = 42


# =============================================================================
# Dataset preprocessing for classical ML
# =============================================================================

def validate_dataset_structure(dataset_dir: str) -> None:
    expected_subdirs = ["train", "valid", "test"]
    for subdir in expected_subdirs:
        path = Path(dataset_dir) / subdir
        if not path.exists():
            raise FileNotFoundError(
                f"Missing folder: {path}\n"
                "Please update DATASET_DIR to point to a folder containing "
                "'train', 'valid', and 'test'."
            )


def preprocess_dataframes(dataset_path: str, target_size: Tuple[int, int] = (200, 200)):
    """
    Build train/test/valid dataframes for classical ML experiments.

    Each image is:
    - resized
    - converted to grayscale
    - flattened into a 1D feature vector

    Labels are encoded as integers.
    """
    dataset_names = ["train", "test", "valid"]
    train_df, test_df, valid_df = None, None, None

    for dataset in dataset_names:
        set_path = os.path.join(dataset_path, dataset)

        data = []
        for klass in sorted(os.listdir(set_path)):
            class_path = os.path.join(set_path, klass)
            if not os.path.isdir(class_path):
                continue

            file_list = sorted(os.listdir(class_path))
            data += [(os.path.join(class_path, f), klass) for f in file_list]

        df = pd.DataFrame(data, columns=["filepaths", "labels"])

        image_data = []
        valid_indices = []

        for idx, fpath in enumerate(df["filepaths"]):
            try:
                img = cv2.imread(fpath)
                if img is None:
                    continue

                img = cv2.resize(img, target_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_flat = img.flatten()

                image_data.append(img_flat)
                valid_indices.append(idx)
            except Exception:
                continue

        df = df.iloc[valid_indices].reset_index(drop=True)
        df["image_data"] = image_data

        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(df["labels"])
        df["encoded_labels"] = encoded_labels

        if dataset == "train":
            train_df = df
        elif dataset == "test":
            test_df = df
        else:
            valid_df = df

    print("\nClassical ML dataset summary")
    print("-" * 60)
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(valid_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Classes: {sorted(train_df['labels'].unique())}")
    print("-" * 60)

    return train_df, test_df, valid_df


def plot_train_distribution(train_df: pd.DataFrame) -> None:
    plt.figure(figsize=(6, 4))
    train_df["encoded_labels"].plot(kind="hist", bins=20, title="Encoded Training Labels")
    plt.gca().spines[["top", "right"]].set_visible(False)
    plt.xlabel("Encoded Label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_conf_matrix(y_true, y_pred, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()


# =============================================================================
# Classical ML experiments
# =============================================================================

def run_svm_experiments(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run multiple SVM experiments and return a summary dataframe.
    """
    X_train = np.array(train_df["image_data"].tolist())
    y_train = np.array(train_df["encoded_labels"])

    X_test = np.array(test_df["image_data"].tolist())
    y_test = np.array(test_df["encoded_labels"])

    svm_configs = [
        {"kernel": "linear", "C": 1.0},
        {"kernel": "rbf", "C": 0.1},
        {"kernel": "poly", "C": 0.1},
    ]

    results = []

    for config in svm_configs:
        print("\n" + "=" * 70)
        print(f"Training SVM with kernel={config['kernel']} and C={config['C']}")
        print("=" * 70)

        clf = SVC(kernel=config["kernel"], C=config["C"])
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report")
        print(classification_report(y_test, y_pred, digits=4))

        plot_conf_matrix(
            y_test,
            y_pred,
            title=f"SVM Confusion Matrix ({config['kernel']}, C={config['C']})"
        )

        results.append({
            "model": "SVM",
            "kernel": config["kernel"],
            "C": config["C"],
            "test_accuracy": accuracy
        })

    results_df = pd.DataFrame(results)
    print("\nSVM comparison table")
    print(results_df.to_string(index=False))
    return results_df


# =============================================================================
# CNN preprocessing
# =============================================================================

def preprocess_images(filepaths, labels, target_size=(200, 200)):
    """
    Load RGB images and return NumPy arrays for CNN experiments.
    """
    images = []
    output_labels = []

    for filepath, label in zip(filepaths, labels):
        try:
            img = load_img(filepath, target_size=target_size)
            img_array = img_to_array(img)
            images.append(img_array)
            output_labels.append(label)
        except Exception:
            continue

    return np.array(images), np.array(output_labels)


def create_cnn_generators(X_train, y_train, X_valid, y_valid, X_test, y_test):
    """
    Create ImageDataGenerator objects for CNN training and evaluation.
    """
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        rescale=1.0 / 255.0
    )

    eval_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_gen = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True, seed=SEED)
    valid_gen = eval_datagen.flow(X_valid, y_valid, batch_size=BATCH_SIZE, shuffle=False)
    test_gen = eval_datagen.flow(X_test, y_test, batch_size=BATCH_SIZE, shuffle=False)

    return train_gen, valid_gen, test_gen


# =============================================================================
# CNN models
# =============================================================================

def build_mobilenet_model(img_size: Tuple[int, int]) -> Model:
    img_shape = (img_size[0], img_size[1], 3)

    base_model = MobileNetV2(
        input_shape=img_shape,
        include_top=False,
        weights="imagenet",
        pooling="max"
    )
    base_model.trainable = True

    x = base_model.output
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    x = Dense(
        256,
        activation="relu",
        kernel_regularizer=regularizers.l2(0.016),
        activity_regularizer=regularizers.l1(0.006),
        bias_regularizer=regularizers.l1(0.006)
    )(x)
    x = Dropout(rate=0.4, seed=SEED)(x)

    # Correct binary output layer
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(
        optimizer=Adamax(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_efficientnet_model(img_size: Tuple[int, int]) -> Model:
    img_shape = (img_size[0], img_size[1], 3)

    base_model = EfficientNetB3(
        input_shape=img_shape,
        include_top=False,
        weights="imagenet",
        pooling="max"
    )
    base_model.trainable = True

    x = base_model.output
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    x = Dense(
        256,
        activation="relu",
        kernel_regularizer=regularizers.l2(0.016),
        activity_regularizer=regularizers.l1(0.006),
        bias_regularizer=regularizers.l1(0.006)
    )(x)
    x = Dropout(rate=0.4, seed=SEED)(x)

    # Correct binary output layer
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(
        optimizer=Adamax(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def evaluate_cnn_model(model: Model, test_gen, y_true: np.ndarray, model_name: str) -> Dict[str, float]:
    """
    Predict and evaluate a CNN model on the test generator.
    """
    y_pred = model.predict(test_gen, verbose=1).ravel()
    y_pred_classes = (y_pred > 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred_classes)

    print(f"\n{model_name} Test Accuracy: {acc:.4f}")
    print("\nClassification Report")
    print(classification_report(y_true, y_pred_classes, digits=4))

    plot_conf_matrix(y_true, y_pred_classes, title=f"{model_name} Confusion Matrix")

    return {
        "model": model_name,
        "test_accuracy": acc
    }


# =============================================================================
# Main pipeline
# =============================================================================

def main():
    validate_dataset_structure(DATASET_DIR)

    # -------------------------------------------------------------------------
    # Part 1: Classical ML
    # -------------------------------------------------------------------------
    train_df, test_df, valid_df = preprocess_dataframes(DATASET_DIR)
    plot_train_distribution(train_df)
    svm_results_df = run_svm_experiments(train_df, test_df)

    # -------------------------------------------------------------------------
    # Part 2: CNN Experiments
    # -------------------------------------------------------------------------
    print("\nStarting CNN experiments...")
    print("=" * 70)

    # Reload labels and file paths for CNN-based RGB preprocessing
    train_filepaths = train_df["filepaths"]
    test_filepaths = test_df["filepaths"]
    valid_filepaths = valid_df["filepaths"]

    y_train = train_df["encoded_labels"].values
    y_test = test_df["encoded_labels"].values
    y_valid = valid_df["encoded_labels"].values

    # MobileNetV2 uses 192x192 in the original notebook
    X_train_mobile, y_train_mobile = preprocess_images(train_filepaths, y_train, target_size=IMG_SIZE_MOBILENET)
    X_test_mobile, y_test_mobile = preprocess_images(test_filepaths, y_test, target_size=IMG_SIZE_MOBILENET)
    X_valid_mobile, y_valid_mobile = preprocess_images(valid_filepaths, y_valid, target_size=IMG_SIZE_MOBILENET)

    train_gen_mobile, valid_gen_mobile, test_gen_mobile = create_cnn_generators(
        X_train_mobile, y_train_mobile,
        X_valid_mobile, y_valid_mobile,
        X_test_mobile, y_test_mobile
    )

    mobile_model = build_mobilenet_model(IMG_SIZE_MOBILENET)
    mobile_model.fit(
        train_gen_mobile,
        epochs=CNN_EPOCHS,
        verbose=1,
        validation_data=valid_gen_mobile,
        shuffle=False
    )

    mobile_results = evaluate_cnn_model(
        mobile_model,
        test_gen_mobile,
        y_test_mobile,
        model_name="MobileNetV2"
    )

    # EfficientNetB3 uses 200x200 in the original notebook
    X_train_eff, y_train_eff = preprocess_images(train_filepaths, y_train, target_size=IMG_SIZE_CNN)
    X_test_eff, y_test_eff = preprocess_images(test_filepaths, y_test, target_size=IMG_SIZE_CNN)
    X_valid_eff, y_valid_eff = preprocess_images(valid_filepaths, y_valid, target_size=IMG_SIZE_CNN)

    train_gen_eff, valid_gen_eff, test_gen_eff = create_cnn_generators(
        X_train_eff, y_train_eff,
        X_valid_eff, y_valid_eff,
        X_test_eff, y_test_eff
    )

    efficientnet_model = build_efficientnet_model(IMG_SIZE_CNN)
    efficientnet_model.fit(
        train_gen_eff,
        epochs=CNN_EPOCHS,
        verbose=1,
        validation_data=valid_gen_eff,
        shuffle=True
    )

    efficientnet_results = evaluate_cnn_model(
        efficientnet_model,
        test_gen_eff,
        y_test_eff,
        model_name="EfficientNetB3"
    )

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------
    cnn_results_df = pd.DataFrame([mobile_results, efficientnet_results])

    print("\nFinal CNN comparison")
    print(cnn_results_df.to_string(index=False))

    print("\nCombined experimental summary")
    print("-" * 70)
    print("SVM experiments:")
    print(svm_results_df.to_string(index=False))
    print("\nCNN experiments:")
    print(cnn_results_df.to_string(index=False))


if __name__ == "__main__":
    main()
