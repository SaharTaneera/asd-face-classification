"""
5-Fold Cross-Validation with EfficientNetB3 for ASD Facial Image Classification
===============================================================================

This script performs 5-fold cross-validation using EfficientNetB3 on the
training+validation portion of the ASD facial image dataset, then evaluates
a final model on the held-out test set.

Associated paper:
Taneera, S. N., Samadi, S., Özmen, S., & Alhajj, R. (2025).
Game-Based Diagnosing of Children with Autism Spectrum Disorder.

Implementation note:
- This script is a cleaned, GitHub-ready version inspired by the published work.
- The workflow builds upon the dataset and reference implementation originally
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
cross_validation_efficientnet.py
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
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
OUTPUT_DIR = "cross_validation_outputs"

IMG_SIZE: Tuple[int, int] = (200, 200)
BATCH_SIZE = 20
EPOCHS = 20
LEARNING_RATE = 0.002
N_SPLITS = 5
SEED = 42


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


def build_dataframe_from_folder(folder_path: Path) -> pd.DataFrame:
    filepaths = []
    labels = []

    classes = sorted([d.name for d in folder_path.iterdir() if d.is_dir()])

    for klass in classes:
        class_path = folder_path / klass
        for file in sorted(class_path.iterdir()):
            if file.is_file():
                filepaths.append(str(file))
                labels.append(klass)

    return pd.DataFrame({
        "filepaths": filepaths,
        "labels": labels
    })


def load_dataframes(dataset_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    train_df = build_dataframe_from_folder(Path(dataset_dir) / "train")
    valid_df = build_dataframe_from_folder(Path(dataset_dir) / "valid")
    test_df = build_dataframe_from_folder(Path(dataset_dir) / "test")

    classes = sorted(train_df["labels"].unique())

    print("\nDataset summary")
    print("-" * 60)
    print(f"Classes: {classes}")
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(valid_df)}")
    print(f"Test samples: {len(test_df)}")
    print("-" * 60)

    return train_df, valid_df, test_df, classes


def combine_train_and_valid(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> pd.DataFrame:
    combined_df = pd.concat([train_df, valid_df], axis=0).reset_index(drop=True)
    print(f"Combined train+valid samples for cross-validation: {len(combined_df)}")
    return combined_df


# =============================================================================
# Generator helpers
# =============================================================================

def make_train_datagen() -> ImageDataGenerator:
    return ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )


def make_eval_datagen() -> ImageDataGenerator:
    return ImageDataGenerator(rescale=1.0 / 255.0)


def dataframe_to_generator(
    df: pd.DataFrame,
    datagen: ImageDataGenerator,
    img_size: Tuple[int, int],
    batch_size: int,
    shuffle: bool
):
    return datagen.flow_from_dataframe(
        df,
        x_col="filepaths",
        y_col="labels",
        target_size=img_size,
        class_mode="binary",
        color_mode="rgb",
        batch_size=batch_size,
        shuffle=shuffle,
        seed=SEED
    )


# =============================================================================
# Model and callbacks
# =============================================================================

def build_efficientnet_model(img_size: Tuple[int, int]) -> Model:
    img_shape = (img_size[0], img_size[1], 3)

    base_model = EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=img_shape,
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
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=Adamax(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


def get_callbacks(output_dir: str, fold_name: str):
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, f"{fold_name}_best.keras")

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

def evaluate_binary_model(model: Model, generator, class_names: List[str], title: str) -> Dict[str, float]:
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

    print(f"\n{title}")
    print("-" * 60)
    print(f"Loss      : {loss:.4f}")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")

    return {
        "loss": float(loss),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def plot_confusion_matrix(model: Model, generator, class_names: List[str], title: str) -> None:
    predictions = model.predict(generator, verbose=1).ravel()
    y_pred = (predictions >= 0.5).astype(int)
    y_true = generator.labels

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    print("\nClassification Report")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))


def plot_fold_accuracies(results_df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(results_df["fold"], results_df["val_accuracy"], marker="o")
    plt.title("5-Fold Cross-Validation Accuracy")
    plt.xlabel("Fold")
    plt.ylabel("Validation Accuracy")
    plt.xticks(results_df["fold"])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


# =============================================================================
# Cross-validation pipeline
# =============================================================================

def run_cross_validation():
    validate_dataset_structure(DATASET_DIR)
    train_df, valid_df, test_df, class_names = load_dataframes(DATASET_DIR)

    combined_df = combine_train_and_valid(train_df, valid_df)

    X = combined_df["filepaths"].values
    y = combined_df["labels"].values

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    fold_results = []

    print("\nStarting 5-fold cross-validation with EfficientNetB3")
    print("=" * 80)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\nFold {fold_idx}/{N_SPLITS}")
        print("-" * 60)

        fold_train_df = combined_df.iloc[train_idx].reset_index(drop=True)
        fold_val_df = combined_df.iloc[val_idx].reset_index(drop=True)

        train_gen = dataframe_to_generator(
            df=fold_train_df,
            datagen=make_train_datagen(),
            img_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        val_gen = dataframe_to_generator(
            df=fold_val_df,
            datagen=make_eval_datagen(),
            img_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            shuffle=False
        )

        model = build_efficientnet_model(IMG_SIZE)

        callbacks = get_callbacks(OUTPUT_DIR, f"fold_{fold_idx}")

        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1,
            shuffle=False
        )

        metrics = evaluate_binary_model(
            model,
            val_gen,
            class_names,
            title=f"Fold {fold_idx} Validation Metrics"
        )

        fold_results.append({
            "fold": fold_idx,
            "val_loss": metrics["loss"],
            "val_accuracy": metrics["accuracy"],
            "val_precision": metrics["precision"],
            "val_recall": metrics["recall"],
            "val_f1": metrics["f1"],
            "epochs_ran": len(history.history["loss"])
        })

    results_df = pd.DataFrame(fold_results)

    print("\nCross-validation results by fold")
    print("=" * 80)
    print(results_df.to_string(index=False))

    mean_acc = results_df["val_accuracy"].mean()
    std_acc = results_df["val_accuracy"].std()

    print("\nCross-validation summary")
    print("=" * 80)
    print(f"Mean validation accuracy : {mean_acc:.4f}")
    print(f"Std validation accuracy  : {std_acc:.4f}")

    plot_fold_accuracies(results_df)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cv_results_path = os.path.join(OUTPUT_DIR, "efficientnetb3_cv_results.csv")
    results_df.to_csv(cv_results_path, index=False)
    print(f"\nSaved fold results to: {cv_results_path}")

    return train_df, valid_df, test_df, class_names, results_df


# =============================================================================
# Final model training and test evaluation
# =============================================================================

def train_final_model_and_evaluate(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    class_names: List[str]
):
    print("\nTraining final EfficientNetB3 model on original train/valid split")
    print("=" * 80)

    train_gen = dataframe_to_generator(
        df=train_df,
        datagen=make_train_datagen(),
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    valid_gen = dataframe_to_generator(
        df=valid_df,
        datagen=make_eval_datagen(),
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_gen = dataframe_to_generator(
        df=test_df,
        datagen=make_eval_datagen(),
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = build_efficientnet_model(IMG_SIZE)
    callbacks = get_callbacks(OUTPUT_DIR, "final_model")

    history = model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
        shuffle=False
    )

    # Validation metrics
    evaluate_binary_model(
        model,
        valid_gen,
        class_names,
        title="Final Model Validation Metrics"
    )

    # Test metrics
    evaluate_binary_model(
        model,
        test_gen,
        class_names,
        title="Final Model Test Metrics"
    )

    plot_confusion_matrix(
        model,
        test_gen,
        class_names,
        title="EfficientNetB3 Test Confusion Matrix"
    )

    final_model_path = os.path.join(OUTPUT_DIR, "efficientnetb3_final.keras")
    model.save(final_model_path)
    print(f"\nSaved final model to: {final_model_path}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    train_df, valid_df, test_df, class_names, cv_results = run_cross_validation()
    train_final_model_and_evaluate(train_df, valid_df, test_df, class_names)
