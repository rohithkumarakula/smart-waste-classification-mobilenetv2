from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


# Set the image loading options and the exact class order.
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# Find the test data, trained model, and output image relative to this script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = PROJECT_ROOT / "data" / "split" / "test"
MODEL_PATH = PROJECT_ROOT / "models" / "best_waste_classifier.keras"
CONFUSION_MATRIX_PATH = PROJECT_ROOT / "confusion_matrix.png"


def main() -> int:
    # Check that the model and held-out test folder are available.
    if not MODEL_PATH.is_file():
        print(f"Error: trained model not found: {MODEL_PATH}", file=sys.stderr)
        return 1
    if not TEST_DIR.is_dir():
        print(f"Error: test dataset folder not found: {TEST_DIR}", file=sys.stderr)
        return 1

    # Confirm that the test folder contains exactly the expected class folders.
    found_classes = {path.name for path in TEST_DIR.iterdir() if path.is_dir()}
    expected_classes = set(CLASS_NAMES)
    if found_classes != expected_classes:
        missing = sorted(expected_classes - found_classes)
        unexpected = sorted(found_classes - expected_classes)
        details = []
        if missing:
            details.append(f"missing class folders: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected folders: {', '.join(unexpected)}")
        print(f"Error in {TEST_DIR}: {'; '.join(details)}", file=sys.stderr)
        return 1

    # Load only the test images, using categorical labels and the fixed class order.
    try:
        test_dataset = tf.keras.utils.image_dataset_from_directory(
            TEST_DIR,
            labels="inferred",
            label_mode="categorical",
            class_names=CLASS_NAMES,
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )
    except (OSError, ValueError) as error:
        print(f"Error loading test images: {error}", file=sys.stderr)
        return 1

    # Apply the preprocessing expected by MobileNetV2.
    test_dataset = test_dataset.map(
        lambda images, labels: (
            tf.keras.applications.mobilenet_v2.preprocess_input(images),
            labels,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Load the saved model for inference without compiling or modifying it.
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    except (OSError, ValueError) as error:
        print(f"Error loading trained model from {MODEL_PATH}: {error}", file=sys.stderr)
        return 1

    # Collect the true labels and predict each test image's class.
    true_labels = np.concatenate(
        [np.argmax(labels.numpy(), axis=1) for _, labels in test_dataset]
    )
    probabilities = model.predict(test_dataset)
    predicted_labels = np.argmax(probabilities, axis=1)

    # Calculate overall accuracy and weighted precision, recall, and F1-score.
    test_accuracy = accuracy_score(true_labels, predicted_labels)
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=range(len(CLASS_NAMES)),
        average="weighted",
        zero_division=0,
    )
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Weighted precision: {precision:.4f}")
    print(f"Weighted recall: {recall:.4f}")
    print(f"Weighted F1-score: {f1_score:.4f}")

    # Show precision, recall, F1-score, and support for every class.
    print("\nClassification report:")
    print(
        classification_report(
            true_labels,
            predicted_labels,
            labels=range(len(CLASS_NAMES)),
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )

    # Create, save, and display a confusion matrix for the test predictions.
    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=range(len(CLASS_NAMES)),
    )
    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=range(len(CLASS_NAMES)),
        yticks=range(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion Matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Write each count inside its corresponding cell.
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")

    figure.tight_layout()
    try:
        figure.savefig(CONFUSION_MATRIX_PATH)
    except OSError as error:
        plt.close(figure)
        print(f"Error saving confusion matrix: {error}", file=sys.stderr)
        return 1

    print(f"\nConfusion matrix saved to: {CONFUSION_MATRIX_PATH}")
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())