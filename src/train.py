"""
SVM training module for car vs. tram audio classification.

This module:
1. Loads the extracted audio features.
2. Splits the dataset into training, validation and test sets.
3. Creates an SVM classification pipeline.
4. Trains the SVM model.
5. Compares different SVM configurations using validation data.
6. Trains the final model using training and validation data.
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

def load_features(csv_path):
    """
    Load extracted audio features from a CSV file.

    Returns:
        X: Audio features.
        y: Class labels (car or tram).
        groups: Recording user identifiers.
    """

    df = pd.read_csv(csv_path)

    # Check required columns
    required_columns = ["filename", "recording_user", "label"]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Separate input features from metadata and labels
    X = df.drop(
        columns=["filename", "recording_user", "label"]
    )

    y = df["label"]
    groups = df["recording_user"]

    # Check that feature columns exist
    if X.shape[1] == 0:
        raise ValueError(
            "The dataset does not contain any audio features."
        )

    # Check that all feature values are numeric
    X = X.apply(pd.to_numeric, errors="raise")

    # Check for missing or infinite values
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError(
            "The dataset contains missing or infinite feature values."
        )

    # Check labels
    if not y.isin(["car", "tram"]).all():
        raise ValueError(
            "Labels must be either 'car' or 'tram'."
        )

    # Check recording users
    if (
        groups.isna().any()
        or groups.astype(str).str.strip().eq("").any()
    ):
        raise ValueError(
            "Some recordings are missing their recording_user."
        )

    # Check that filenames are present and unique
    if df["filename"].isna().any():
        raise ValueError(
            "Some recordings are missing their filename."
        )

    if df["filename"].duplicated().any():
        raise ValueError(
            "The dataset contains duplicate filenames."
        )

    print(f"Loaded {len(df)} audio recordings.")
    print(f"Number of features: {X.shape[1]}")

    print("\nClass distribution:")
    print(y.value_counts())

    return X, y, groups


# --------------------------------------------------
# 2. SPLIT DATASET
# --------------------------------------------------

def split_data(X, y, groups, random_state=42):
    """
    Split the dataset into training, validation and test sets.

    Uses stratified splitting to preserve class proportions.

    Approximate proportions:
        Training:   60%
        Validation: 20%
        Test:       20%

    The groups argument is kept for compatibility with main.py,
    but is not used for splitting.

    This split assumes that each row represents an independent
    recording. It does not guarantee independence between
    recording users.
    """

    # First split: 80% training + validation, 20% test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=random_state
    )

    # Second split: 75% training, 25% validation
    # This produces approximately 60/20/20 overall
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.25,
        stratify=y_train_val,
        random_state=random_state
    )

    # Check that both classes are present in every set
    for name, labels in [
        ("Training", y_train),
        ("Validation", y_val),
        ("Test", y_test)
    ]:
        if set(labels.unique()) != {"car", "tram"}:
            raise ValueError(
                f"{name} set does not contain both classes."
            )

    # Check that no recording appears in multiple sets
    train_indices = set(X_train.index)
    val_indices = set(X_val.index)
    test_indices = set(X_test.index)

    if (
        train_indices & val_indices
        or train_indices & test_indices
        or val_indices & test_indices
    ):
        raise ValueError(
            "Recordings overlap between dataset splits."
        )

    print("\nDataset split:")
    print(f"Training: {len(X_train)} recordings")
    print(f"Validation: {len(X_val)} recordings")
    print(f"Test: {len(X_test)} recordings")

    print("\nTraining class distribution:")
    print(y_train.value_counts())

    print("\nValidation class distribution:")
    print(y_val.value_counts())

    print("\nTest class distribution:")
    print(y_test.value_counts())

    return X_train, X_val, X_test, y_train, y_val, y_test


# --------------------------------------------------
# 3. CREATE SVM MODEL
# --------------------------------------------------

def create_svm_model(kernel="rbf", C=1.0):
    """
    Create an SVM classification pipeline.

    StandardScaler standardizes the audio features.
    SVC performs binary classification.

    The scaler is fitted only on the data passed to model.fit().
    """

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel=kernel,
            C=C
        ))
    ])

    return model


# --------------------------------------------------
# 4. TRAIN SVM MODEL
# --------------------------------------------------

def train_svm(X_train, y_train, kernel="rbf", C=1.0):
    """
    Train an SVM model using the training dataset.
    """

    model = create_svm_model(
        kernel=kernel,
        C=C
    )

    model.fit(X_train, y_train)

    print("\nSVM model trained successfully!")
    print(f"Kernel: {kernel}")
    print(f"C: {C}")

    return model


# --------------------------------------------------
# 5. COMPARE SVM MODELS
# --------------------------------------------------

def select_best_svm(X_train, y_train, X_val, y_val):
    """
    Compare different SVM configurations using validation data.

    Returns:
        The model with the highest validation accuracy.
    """

    # Parameters to compare
    kernels = ["linear", "rbf"]
    C_values = [0.1, 1, 10]

    # Variables to store the best model
    best_model = None
    best_accuracy = -1
    best_params = None

    print("\nSVM Model Comparison")
    print("-------------------------")

    # Try all combinations of kernel and C
    for kernel in kernels:
        for C in C_values:

            # Train the model
            model = train_svm(
                X_train,
                y_train,
                kernel=kernel,
                C=C
            )

            # Calculate validation accuracy
            accuracy = model.score(X_val, y_val)

            print(
                f"Kernel: {kernel}, "
                f"C: {C}, "
                f"Validation Accuracy: {accuracy:.4f}"
            )

            # Update the best model if accuracy improves
            if accuracy > best_accuracy:

                best_accuracy = accuracy
                best_model = model
                best_params = {
                    "kernel": kernel,
                    "C": C
                }

    # Display the best configuration
    print("\nBest SVM Configuration:")
    print(f"Kernel: {best_params['kernel']}")
    print(f"C: {best_params['C']}")
    print(f"Validation Accuracy: {best_accuracy:.4f}")

    return best_model


# --------------------------------------------------
# 6. TRAIN FINAL MODEL
# --------------------------------------------------

def train_final_model(
    best_model,
    X_train,
    y_train,
    X_val,
    y_val
):
    """
    Train the final SVM using training and validation data.

    The kernel and C value are taken from the selected model.

    The test dataset is not used during training.
    """

    # Get the selected SVM parameters
    best_kernel = best_model.named_steps["svm"].kernel
    best_C = best_model.named_steps["svm"].C

    # Combine training and validation data
    X_combined = pd.concat(
        [X_train, X_val],
        axis=0
    )

    y_combined = pd.concat(
        [y_train, y_val],
        axis=0
    )

    print("\nTraining Final SVM Model")
    print("-------------------------")
    print(f"Total training recordings: {len(X_combined)}")

    # Train a new model using the selected parameters
    final_model = train_svm(
        X_combined,
        y_combined,
        kernel=best_kernel,
        C=best_C
    )

    return final_model