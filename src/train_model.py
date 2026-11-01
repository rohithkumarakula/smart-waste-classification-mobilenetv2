from pathlib import Path
import random
import sys

import numpy as np
import tensorflow as tf


# Set fixed values so runs are easier to reproduce.
RANDOM_SEED = 42
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# Find the data and model folders relative to this script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_ROOT / "data" / "split" / "train"
VALIDATION_DIR = PROJECT_ROOT / "data" / "split" / "val"
MODEL_PATH = PROJECT_ROOT / "models" / "best_waste_classifier.keras"


def main() -> int:
    # Set seeds for Python, NumPy, and TensorFlow.
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    # Check that the training and validation folders have the expected classes.
    for folder in (TRAIN_DIR, VALIDATION_DIR):
        if not folder.is_dir():
            print(f"Error: dataset folder not found: {folder}", file=sys.stderr)
            return 1

        found_classes = {path.name for path in folder.iterdir() if path.is_dir()}
        expected_classes = set(CLASS_NAMES)
        if found_classes != expected_classes:
            missing = sorted(expected_classes - found_classes)
            unexpected = sorted(found_classes - expected_classes)
            details = []
            if missing:
                details.append(f"missing class folders: {', '.join(missing)}")
            if unexpected:
                details.append(f"unexpected folders: {', '.join(unexpected)}")
            print(f"Error in {folder}: {'; '.join(details)}", file=sys.stderr)
            return 1

    # Load training and validation images with the same class order.
    try:
        train_dataset = tf.keras.utils.image_dataset_from_directory(
            TRAIN_DIR,
            labels="inferred",
            label_mode="categorical",
            class_names=CLASS_NAMES,
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            shuffle=True,
            seed=RANDOM_SEED,
        )
        validation_dataset = tf.keras.utils.image_dataset_from_directory(
            VALIDATION_DIR,
            labels="inferred",
            label_mode="categorical",
            class_names=CLASS_NAMES,
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )
    except (OSError, ValueError) as error:
        print(f"Error loading training or validation images: {error}", file=sys.stderr)
        return 1

    # Scale pixels using the preprocessing expected by MobileNetV2.
    train_dataset = train_dataset.map(
        lambda images, labels: (
            tf.keras.applications.mobilenet_v2.preprocess_input(images),
            labels,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    validation_dataset = validation_dataset.map(
        lambda images, labels: (
            tf.keras.applications.mobilenet_v2.preprocess_input(images),
            labels,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    # Load MobileNetV2 with ImageNet weights and freeze its pretrained layers.
    base_model = tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
    )
    base_model.trainable = False

    # Add a small classification head for the six waste categories.
    inputs = tf.keras.Input(shape=(224, 224, 3))
    features = base_model(inputs, training=False)
    features = tf.keras.layers.GlobalAveragePooling2D()(features)
    features = tf.keras.layers.Dropout(0.2)(features)
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax")(features)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Compile the model with the requested optimizer, loss, and metric.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Save the best validation model and stop when validation loss stops improving.
    try:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        print(f"Error creating model folder: {error}", file=sys.stderr)
        return 1

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    # Train using only the training and validation datasets.
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # Print the accuracy values from the final training epoch.
    training_accuracy = history.history["accuracy"][-1]
    validation_accuracy = history.history["val_accuracy"][-1]
    print(f"\nTraining accuracy: {training_accuracy:.4f}")
    print(f"Validation accuracy: {validation_accuracy:.4f}")
    print(f"Best model saved to: {MODEL_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())