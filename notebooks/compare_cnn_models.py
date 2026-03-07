"""
CNN Model Comparison for ASD Facial Image Classification
========================================================

This script compares multiple transfer-learning CNN models for binary
ASD facial image classification.

Associated paper:
Taneera, S. N., Samadi, S., Özmen, S., & Alhajj, R. (2025).
Game-Based Diagnosing of Children with Autism Spectrum Disorder.

Implementation note:
- This workflow is a cleaned, GitHub-ready comparison script inspired by
  the experimental setup described in the paper.
- The work builds on the dataset and reference implementation originally
  provided by Gerry (2020).

Models compared
---------------
- Xception
- MobileNetV2
- ResNet50
- EfficientNetB3

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

Usage
-----
1. Update DATASET_DIR below.
2. Install the required packages.
3. Run the script.
4. The script will train and evaluate each model, then print a final
   comparison table.

Recommended file name
---------------------
compare_cnn_models.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

# Backbone models
from tensorflow.keras.applications import Xception, MobileNetV2, ResNet50

# EfficientNet import
try:
    from tensorflow.keras.applications import EfficientNetB3
except ImportError:
    # For older setups
    from efficientnet.tfkeras import EfficientNetB3

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
sns.set_style("darkgrid")


# =============================================================================
# Configuration
# =============================================================================

DATASET_DIR = "path_to_dataset_directory"
OUTPUT_DIR = "model_outputs"
IMG_SIZE = (200, 200)
BATCH_SIZE = 20
EPOCHS = 20
LEARNING_RATE = 0.002
SEED = 42

# Binary classification class names expected in folder names
# If your exact names differ, the script will still work as long as there are 2 classes.
EXPECTED_MODE = "binary"


# =============================================================================
# Dataset utilities
# =============================================================================

def validate_dataset_structure(dataset_dir: str) -> None:
    expected_subdirs = ["train", "valid", "test"]
    for subdir in expected_subdirs:
        subdir_path = Path(dataset_dir) / subdir
        if not subdir_path.exists():
            raise FileNotFoundError(
                f"Missing folder: {subdir_path}\n"
                "Please update DATASET_DIR to point to a dataset folder with "
                "'train', 'valid', and 'test' subfolders."
            )


def make_dataframes(dataset_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    dataset_names = ["train", "valid", "test"]
    dataframes = {}

    for dataset_name in dataset_names:
        set_path = Path(dataset_dir) / dataset_name
        filepaths = []
        labels = []

        classes = sorted([d.name for d in set_path.iterdir() if d.is_dir()])

        for klass in classes:
            class_path = set_path / klass
            file_list = sorted([f.name for f in class_path.iterdir() if f.is_file()])
            desc = f"{dataset_name:6s}-{klass:15s}"

            for file_name in tqdm(file_list, ncols=120, desc=desc):
                filepaths.append(str(class_path / file_name))
                labels.append(klass)

        dataframes[dataset_name] = pd.DataFrame({
            "filepaths": filepaths,
            "labels": labels
        })

    train_df = dataframes["train"]
    valid_df = dataframes["valid"]
    test_df = dataframes["test"]

    classes = sorted(train_df["labels"].unique())

    print("\nDataset summary")
    print("-" * 60)
    print(f"Classes: {classes}")
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(valid_df)}")
    print(f"Test samples: {len(test_df)}")
    print("-" * 60)

    return train_df, valid_df, test_df, classes


def create_generators(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    img_size: Tuple[int, int],
    batch_size: int
):
    """
    Use simple rescaling to keep preprocessing consistent across backbones.
    This also matches the paper's general description of image normalization/rescaling.
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False
    )

    eval_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_gen = train_datagen.flow_from_dataframe(
        train_df,
        x_col="filepaths",
        y_col="labels",
        target_size=img_size,
        class_mode=EXPECTED_MODE,
        color_mode="rgb",
        batch_size=batch_size,
        shuffle=True,
        seed=SEED
    )

    valid_gen = eval_datagen.flow_from_dataframe(
        valid_df,
        x_col="filepaths",
        y_col="labels",
        target_size=img_size,
        class_mode=EXPECTED_MODE,
        color_mode="rgb",
        batch_size=batch_size,
        shuffle=False
    )

    test_gen = eval_datagen.flow_from_dataframe(
        test_df,
        x_col="filepaths",
        y_col="labels",
        target_size=img_size,
        class_mode=EXPECTED_MODE,
        color_mode="rgb",
        batch_size=batch_size,
        shuffle=False
    )

    return train_gen, valid_gen, test_gen


# =============================================================================
# Model building
# =============================================================================

def build_transfer_model(backbone_name: str, img_size: Tuple[int, int]) -> Model:
    """
    Build a transfer-learning model with a shared custom classification head.
    """
    img_shape = (img_size[0], img_size[1], 3)

    if backbone_name == "Xception":
        base_model = Xception(
            include_top=False,
            weights="imagenet",
            input_shape=img_shape,
            pooling="max"
        )
    elif backbone_name == "MobileNetV2":
        base_model = MobileNetV2(
            include_top=False,
            weights="imagenet",
            input_shape=img_shape,
            pooling="max"
        )
    elif backbone_name == "ResNet50":
        base_model = ResNet50(
            include_top=False,
            weights="imagenet",
            input_shape=img_shape,
            pooling="max"
        )
    elif backbone_name == "EfficientNetB3":
        base_model = EfficientNetB3(
            include_top=False,
            weights="imagenet",
            input_shape=img_shape,
            pooling="max"
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    base_model.trainable = True

    x = base_model.output
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    x = Dense(
        256,
        activation="relu",
        kernel_regularizer=regularizers.l2(0.016),
        activity_regularizer=regularizers.l1(0.006),
        bias_regularizer=regularizers.l1(0.006),
    )(x)
    x = Dropout(rate=0.4, seed=SEED)(x)

    # Binary classification output
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=Adamax(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


def get_callbacks(model_name: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, f"{model_name}_best.keras")

    return [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            verbose=1
        ),
        ModelCheckpoint(
            checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]


# =============================================================================
# Evaluation helpers
# =============================================================================

def plot_training_history(history, model_name: str) -> None:
    train_acc = history.history["accuracy"]
    train_loss = history.history["loss"]
    val_acc = history.history["val_accuracy"]
    val_loss = history.history["val_loss"]

    epochs_range = range(1, len(train_acc) + 1)

    best_loss_epoch = np.argmin(val_loss) + 1
    best_acc_epoch = np.argmax(val_acc) + 1

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Loss
    axes[0].plot(epochs_range, train_loss, label="Train Loss")
    axes[0].plot(epochs_range, val_loss, label="Validation Loss")
    axes[0].scatter(best_loss_epoch, val_loss[best_loss_epoch - 1], s=120, label=f"Best Epoch: {best_loss_epoch}")
    axes[0].set_title(f"{model_name} - Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Accuracy
    axes[1].plot(epochs_range, train_acc, label="Train Accuracy")
    axes[1].plot(epochs_range, val_acc, label="Validation Accuracy")
    axes[1].scatter(best_acc_epoch, val_acc[best_acc_epoch - 1], s=120, label=f"Best Epoch: {best_acc_epoch}")
    axes[1].set_title(f"{model_name} - Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def evaluate_generator(model: Model, generator, class_names: List[str], split_name: str, model_name: str) -> Dict[str, float]:
    """
    Evaluate a trained model on validation or test generator.
    """
    predictions = model.predict(generator, verbose=1).ravel()
    y_pred = (predictions >= 0.5).astype(int)
    y_true = generator.labels

    loss, accuracy = model.evaluate(generator, verbose=0)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0
    )

    print(f"\n{model_name} - {split_name} metrics")
    print("-" * 60)
    print(f"Loss      : {loss:.4f}")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")

    if split_name.lower() == "test":
        print("\nClassification Report")
        print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f"{model_name} - Test Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.show()

    return {
        f"{split_name.lower()}_loss": float(loss),
        f"{split_name.lower()}_accuracy": float(accuracy),
        f"{split_name.lower()}_precision": float(precision),
        f"{split_name.lower()}_recall": float(recall),
        f"{split_name.lower()}_f1": float(f1),
    }


# =============================================================================
# Training loop for all models
# =============================================================================

def run_model_comparison():
    validate_dataset_structure(DATASET_DIR)

    train_df, valid_df, test_df, classes = make_dataframes(DATASET_DIR)
    train_gen, valid_gen, test_gen = create_generators(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    models_to_compare = [
        "Xception",
        "MobileNetV2",
        "ResNet50",
        "EfficientNetB3"
    ]

    all_results = []

    for model_name in models_to_compare:
        print("\n" + "=" * 80)
        print(f"Training model: {model_name}")
        print("=" * 80)

        model = build_transfer_model(model_name, IMG_SIZE)

        callbacks = get_callbacks(model_name, OUTPUT_DIR)

        history = model.fit(
            train_gen,
            validation_data=valid_gen,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1,
            shuffle=False
        )

        plot_training_history(history, model_name)

        val_metrics = evaluate_generator(model, valid_gen, classes, "Validation", model_name)
        test_metrics = evaluate_generator(model, test_gen, classes, "Test", model_name)

        result_row = {
            "model": model_name,
            **val_metrics,
            **test_metrics
        }
        all_results.append(result_row)

    results_df = pd.DataFrame(all_results)

    print("\nFinal comparison table")
    print("=" * 80)
    print(results_df.sort_values(by="test_accuracy", ascending=False).to_string(index=False))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_path = os.path.join(OUTPUT_DIR, "cnn_model_comparison_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved comparison results to: {results_path}")


if __name__ == "__main__":
    run_model_comparison()
