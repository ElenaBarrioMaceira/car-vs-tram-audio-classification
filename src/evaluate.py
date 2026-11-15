"""
Evaluation module for car vs. tram audio classification.

This module:
1. Makes predictions using a trained model.
2. Calculates classification metrics.
3. Displays and saves a confusion matrix.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    ConfusionMatrixDisplay
)


# --------------------------------------------------
# 1. EVALUATE MODEL
# --------------------------------------------------

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained SVM model.

    Returns:
        Dictionary containing accuracy, precision and recall.
    """

    # Predict the classes
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    # Print results
    print("\nModel Evaluation")
    print("-------------------------")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=["car", "tram"],
            zero_division=0
        )
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    }


# --------------------------------------------------
# 2. CONFUSION MATRIX
# --------------------------------------------------

def plot_confusion_matrix(
    model,
    X_test,
    y_test,
    save_path="results/figures/confusion_matrix.png"
):
    """
    Plot and save the confusion matrix.
    """

    # Create output directory if necessary
    Path(save_path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Create confusion matrix
    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        labels=["car", "tram"],
        display_labels=["Car", "Tram"],
        cmap="Blues"
    )

    plt.title("SVM Confusion Matrix")
    plt.tight_layout()

    # Save figure
    plt.savefig(save_path, dpi=300)

    # Display figure
    plt.show()