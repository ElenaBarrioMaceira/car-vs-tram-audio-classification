"""
Main script for car vs. tram audio classification.

This script:
1. Loads the extracted audio features.
2. Splits the dataset by recording user.
3. Compares different SVM configurations.
4. Trains the final SVM model.
5. Evaluates the model on the test dataset.
6. Saves the confusion matrix.
"""

from pathlib import Path

from src.train import (
    load_features,
    split_data,
    select_best_svm,
    train_final_model
)

from src.evaluate import (
    evaluate_model,
    plot_confusion_matrix
)


def main():

    print("===================================")
    print("CAR VS. TRAM AUDIO CLASSIFICATION")
    print("===================================")

    # --------------------------------------------------
    # 1. LOAD FEATURES
    # --------------------------------------------------

    # Use test data until real features are available
    csv_path = Path("data/processed/test_features.csv")

    print("\n1. Loading audio features...")

    X, y, groups = load_features(csv_path)

    # --------------------------------------------------
    # 2. SPLIT DATASET
    # --------------------------------------------------
    print("\n2. Splitting dataset with stratification...")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X,
        y,
        groups
    )

    # --------------------------------------------------
    # 3. SELECT BEST SVM
    # --------------------------------------------------

    print("\n3. Comparing SVM configurations...")

    best_model = select_best_svm(
        X_train,
        y_train,
        X_val,
        y_val
    )

    # --------------------------------------------------
    # 4. TRAIN FINAL MODEL
    # --------------------------------------------------

    print("\n4. Training final SVM model...")

    final_model = train_final_model(
        best_model,
        X_train,
        y_train,
        X_val,
        y_val
    )

    # --------------------------------------------------
    # 5. EVALUATE FINAL MODEL
    # --------------------------------------------------

    print("\n5. Evaluating final model on test data...")

    results = evaluate_model(
        final_model,
        X_test,
        y_test
    )

    # --------------------------------------------------
    # 6. SAVE CONFUSION MATRIX
    # --------------------------------------------------

    print("\n6. Generating confusion matrix...")

    plot_confusion_matrix(
        final_model,
        X_test,
        y_test,
        save_path="results/figures/confusion_matrix.png"
    )

    # --------------------------------------------------
    # 7. FINAL SUMMARY
    # --------------------------------------------------

    print("\n===================================")
    print("FINAL RESULTS")
    print("===================================")

    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")

    print("\nConfusion matrix saved to:")
    print("results/figures/confusion_matrix.png")

    print("\nClassification completed!")


if __name__ == "__main__":
    main()