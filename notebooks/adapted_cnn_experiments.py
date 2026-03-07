"""
ASD Facial Image Classification Experiments
===========================================

This script contains a cleaned and beginner-friendly version of an experimental
pipeline for ASD facial image classification.

Associated paper:
Taneera, S. N., Samadi, S., Özmen, S., & Alhajj, R. (2025).
Game-Based Diagnosing of Children with Autism Spectrum Disorder.

Implementation note:
- This workflow builds upon the dataset and reference implementation
  originally provided by Gerry (2020).
- This version was cleaned for readability, reproducibility, and GitHub sharing.

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

How to use
----------
1. Update DATASET_DIR below.
2. Install required packages.
3. Run the script.
4. Training plots, confusion matrix, classification report, and the saved model
   will be produced automatically.

Recommended file name
---------------------
adapted_cnn_experiments.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

# Reduce TensorFlow log noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
sns.set_style("darkgrid")


# =============================================================================
# Configuration
# =============================================================================

# TODO: change this to your dataset folder before running the script
DATASET_DIR = "path_to_dataset_directory"

# Image size used for training
IMG_SIZE: Tuple[int, int] = (200, 200)

# Batch size for training/validation
BATCH_SIZE = 20

# Number of epochs
EPOCHS = 40

# Learning rate
LEARNING_RATE = 0.002

# Directory where trained models will be saved
OUTPUT_DIR = "saved_models"

# Random seed for reproducibility
SEED = 42


# =============================================================================
# Helper functions
# =============================================================================

def validate_dataset_structure(dataset_dir: str) -> None:
    """
    Check whether the dataset directory has the required structure.
    """
    expected_subdirs = ["train", "valid", "test"]
    for subdir in expected_subdirs:
        path = Path(dataset_dir) / subdir
        if not path.exists():
            raise FileNotFoundError(
                f"Expected folder not found: {path}\n"
                "Please make sure DATASET_DIR points to a folder containing "
                "'train', 'valid', and 'test' subfolders."
            )


def make_dataframes(dataset_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], int]:
    """
    Build train, validation, and test dataframes from the folder structure.

    Returns:
        train_df, test_df, valid_df, classes, class_count
    """
    dataset_names = ["train", "test", "valid"]
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
                file_path = class_path / file_name
                filepaths.append(str(file_path))
                labels.append(klass)

        df = pd.DataFrame({
            "filepaths": filepaths,
            "labels": labels
        })
        dataframes[dataset_name] = df

    train_df = dataframes["train"]
    test_df = dataframes["test"]
    valid_df = dataframes["valid"]

    classes = sorted(train_df["labels"].unique())
    class_count = len(classes)

    print("\nDataset summary")
    print("-" * 60)
    print(f"Classes: {classes}")
    print(f"Number of classes: {class_count}")
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(valid_df)}")
    print(f"Test samples: {len(test_df)}")
    print("-" * 60)

    return train_df, test_df, valid_df, classes, class_count


def estimate_average_image_size(train_df: pd.DataFrame, sample_size: int = 50) -> None:
    """
    Estimate average height/width from a random sample of images.
    This is just a quick dataset inspection helper.
    """
    sample_df = train_df.sample(n=min(sample_size, len(train_df)), random_state=SEED)

    total_h, total_w, count = 0, 0, 0

    for fpath in sample_df["filepaths"]:
        try:
            img = cv2.imread(fpath)
            if img is None:
                continue
            h, w = img.shape[:2]
            total_h += h
            total_w += w
            count += 1
        except Exception:
            continue

    if count > 0:
        avg_h = total_h / count
        avg_w = total_w / count
        aspect_ratio = avg_h / avg_w
        print(f"Estimated average image size: {avg_h:.1f} x {avg_w:.1f}")
        print(f"Estimated aspect ratio (h/w): {aspect_ratio:.3f}")
    else:
        print("Could not estimate average image size.")


def create_generators(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    img_size: Tuple[int, int],
    batch_size: int
):
    """
    Create Keras image generators for train, validation, and test data.

    We use Xception preprocessing because the model is based on Xception.
    """
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    valid_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    print("\nCreating data generators...")

    train_gen = train_datagen.flow_from_dataframe(
        train_df,
        x_col="filepaths",
        y_col="labels",
        target_size=img_size,
        class_mode="categorical",
        color_mode="rgb",
        shuffle=True,
        batch_size=batch_size,
        seed=SEED
    )

    valid_gen = valid_test_datagen.flow_from_dataframe(
        valid_df,
        x_col="filepaths",
        y_col="labels",
        target_size=img_size,
        class_mode="categorical",
        color_mode="rgb",
        shuffle=False,
        batch_size=batch_size
    )

    # Make sure the test generator covers the test set exactly once
    test_length = len(test_df)
    test_batch_size = sorted(
        [int(test_length / n) for n in range(1, test_length + 1)
         if test_length % n == 0 and test_length / n <= 80],
        reverse=True
    )[0]
    test_steps = test_length // test_batch_size

    test_gen = valid_test_datagen.flow_from_dataframe(
        test_df,
        x_col="filepaths",
        y_col="labels",
        target_size=img_size,
        class_mode="categorical",
        color_mode="rgb",
        shuffle=False,
        batch_size=test_batch_size
    )

    classes = list(train_gen.class_indices.keys())
    print(f"Class indices: {train_gen.class_indices}")
    print(f"Test batch size: {test_batch_size}")
    print(f"Test steps: {test_steps}")

    return train_gen, valid_gen, test_gen, test_batch_size, test_steps, classes


def save_class_names(classes: List[str], dataset_dir: str) -> None:
    """
    Save the class names to a small text file.
    """
    output_path = Path(dataset_dir) / "classes.txt"
    content = ",".join(classes)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Saved class names to: {output_path}")


def show_image_samples(generator, max_images: int = 25) -> None:
    """
    Display a sample of images from the training generator.
    """
    class_dict = generator.class_indices
    classes = list(class_dict.keys())

    images, labels = next(generator)

    plt.figure(figsize=(20, 20))
    num_images = min(len(labels), max_images)

    for i in range(num_images):
        plt.subplot(5, 5, i + 1)

        # Undo some preprocessing only for display
        image = images[i]
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)

        plt.imshow(image)
        class_index = np.argmax(labels[i])
        class_name = classes[class_index]
        plt.title(class_name, color="blue", fontsize=12)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def build_model(img_size: Tuple[int, int], class_count: int, learning_rate: float) -> Model:
    """
    Create a transfer learning model using Xception as the backbone.

    Notes:
    - include_top=False removes the original ImageNet classification head
    - pooling='max' adds a global max pooling layer
    - additional Dense/Dropout layers are added for this ASD classification task
    """
    img_shape = (img_size[0], img_size[1], 3)

    base_model = Xception(
        include_top=False,
        weights="imagenet",
        input_shape=img_shape,
        pooling="max"
    )

    # In this experimental version we keep the base model trainable
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

    output = Dense(class_count, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=Adamax(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def get_callbacks(output_dir: str) -> List[keras.callbacks.Callback]:
    """
    Standard callbacks for training:
    - EarlyStopping: stop when validation loss stops improving
    - ReduceLROnPlateau: reduce learning rate when validation loss plateaus
    - ModelCheckpoint: save best model weights
    """
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_path = os.path.join(output_dir, "best_xception_model.keras")

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=6,
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
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]
    return callbacks


def plot_training_history(history: keras.callbacks.History) -> None:
    """
    Plot training and validation loss/accuracy.
    """
    train_acc = history.history["accuracy"]
    train_loss = history.history["loss"]
    val_acc = history.history["val_accuracy"]
    val_loss = history.history["val_loss"]

    epochs_range = range(1, len(train_acc) + 1)

    best_loss_epoch = np.argmin(val_loss) + 1
    best_acc_epoch = np.argmax(val_acc) + 1

    plt.style.use("fivethirtyeight")
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 7))

    # Loss plot
    axes[0].plot(epochs_range, train_loss, label="Training Loss")
    axes[0].plot(epochs_range, val_loss, label="Validation Loss")
    axes[0].scatter(best_loss_epoch, val_loss[best_loss_epoch - 1], s=120, label=f"Best Epoch: {best_loss_epoch}")
    axes[0].set_title("Training and Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Accuracy plot
    axes[1].plot(epochs_range, train_acc, label="Training Accuracy")
    axes[1].plot(epochs_range, val_acc, label="Validation Accuracy")
    axes[1].scatter(best_acc_epoch, val_acc[best_acc_epoch - 1], s=120, label=f"Best Epoch: {best_acc_epoch}")
    axes[1].set_title("Training and Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def evaluate_model(model: Model, test_gen, class_names: List[str]) -> Tuple[int, int]:
    """
    Evaluate the model on the test set, show confusion matrix,
    and print the classification report.

    Returns:
        errors, total_tests
    """
    print("\nRunning predictions on test set...")
    predictions = model.predict(test_gen, verbose=1)

    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.labels

    errors = int(np.sum(y_pred != y_true))
    total_tests = len(y_true)
    accuracy = (1 - errors / total_tests) * 100

    print(f"\nThere were {errors} errors in {total_tests} test samples.")
    print(f"Test accuracy: {accuracy:.2f}%")

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="g", cmap="Blues", cbar=False)
    plt.xticks(np.arange(len(class_names)) + 0.5, class_names, rotation=45)
    plt.yticks(np.arange(len(class_names)) + 0.5, class_names, rotation=0)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
    print("\nClassification Report")
    print("-" * 60)
    print(report)

    return errors, total_tests


def save_final_model(model: Model, output_dir: str, accuracy: float) -> str:
    """
    Save the final trained model using the accuracy in the file name.
    """
    os.makedirs(output_dir, exist_ok=True)

    filename = f"xception_asd_classifier_{accuracy:.2f}.keras"
    save_path = os.path.join(output_dir, filename)

    model.save(save_path)
    print(f"Model saved to: {save_path}")

    return save_path


# =============================================================================
# Main script
# =============================================================================

def main():
    # 1. Validate dataset structure
    validate_dataset_structure(DATASET_DIR)

    # 2. Build dataframes
    train_df, test_df, valid_df, classes, class_count = make_dataframes(DATASET_DIR)

    # 3. Inspect average image size (optional but useful)
    estimate_average_image_size(train_df)

    # 4. Create generators
    train_gen, valid_gen, test_gen, test_batch_size, test_steps, classes = create_generators(
        train_df=train_df,
        test_df=test_df,
        valid_df=valid_df,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    # 5. Save class names for later use
    save_class_names(classes, DATASET_DIR)

    # 6. Show a sample of training images
    show_image_samples(train_gen)

    # 7. Build model
    model = build_model(
        img_size=IMG_SIZE,
        class_count=class_count,
        learning_rate=LEARNING_RATE
    )

    # Print model summary so users can inspect the architecture
    print("\nModel summary")
    print("-" * 60)
    model.summary()

    # 8. Define callbacks
    callbacks = get_callbacks(OUTPUT_DIR)

    # 9. Train model
    print("\nStarting training...")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=valid_gen,
        callbacks=callbacks,
        shuffle=False,
        verbose=1
    )

    # 10. Plot training history
    plot_training_history(history)

    # 11. Evaluate on test set
    errors, total_tests = evaluate_model(model, test_gen, classes)
    final_accuracy = (1 - errors / total_tests) * 100

    # 12. Save final model
    save_final_model(model, OUTPUT_DIR, final_accuracy)


if __name__ == "__main__":
    main()
