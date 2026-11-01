from pathlib import Path
import random
import shutil
import sys


# Find the dataset and output folders relative to this script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "dataset-resized"
OUTPUT_DIR = PROJECT_ROOT / "data" / "split"

# List the class folders and common image file types.
CLASSES = ("cardboard", "glass", "metal", "paper", "plastic", "trash")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}


def main() -> int:
    # Check that the dataset and all expected class folders exist.
    if not DATASET_DIR.is_dir():
        print(f"Error: dataset folder not found: {DATASET_DIR}", file=sys.stderr)
        return 1

    missing_folders = [
        class_name
        for class_name in CLASSES
        if not (DATASET_DIR / class_name).is_dir()
    ]
    if missing_folders:
        print(
            "Error: missing class folder(s): "
            + ", ".join(str(DATASET_DIR / name) for name in missing_folders),
            file=sys.stderr,
        )
        return 1

    # Create every split and class folder, including folders with no images.
    try:
        for split_name in ("train", "val", "test"):
            for class_name in CLASSES:
                (OUTPUT_DIR / split_name / class_name).mkdir(
                    parents=True, exist_ok=True
                )

        # Use a fixed seed so the same images are selected on every run.
        generator = random.Random(42)
        image_counts = {}

        # Shuffle and copy each class independently.
        for class_name in CLASSES:
            class_folder = DATASET_DIR / class_name
            images = sorted(
                (
                    image_path
                    for image_path in class_folder.iterdir()
                    if image_path.is_file()
                    and image_path.suffix.lower() in IMAGE_EXTENSIONS
                ),
                key=lambda image_path: image_path.name.lower(),
            )
            generator.shuffle(images)

            # Round training and validation down; put any remainder in testing.
            train_end = len(images) * 70 // 100
            val_end = train_end + len(images) * 15 // 100
            split_images = {
                "train": images[:train_end],
                "val": images[train_end:val_end],
                "test": images[val_end:],
            }

            image_counts[class_name] = {}
            for split_name, split_files in split_images.items():
                destination_folder = OUTPUT_DIR / split_name / class_name
                for image_path in split_files:
                    shutil.copy2(image_path, destination_folder / image_path.name)
                image_counts[class_name][split_name] = len(split_files)

    except OSError as error:
        print(f"Error while reading or copying dataset images: {error}", file=sys.stderr)
        return 1

    # Show the number of images copied for each class in each split.
    print("Image counts by class and split:")
    print(f"{'Class':<12}{'Train':>8}{'Val':>8}{'Test':>8}")
    for class_name in CLASSES:
        counts = image_counts[class_name]
        print(
            f"{class_name:<12}"
            f"{counts['train']:>8}"
            f"{counts['val']:>8}"
            f"{counts['test']:>8}"
        )
    print(f"\nSplit folders created in: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())