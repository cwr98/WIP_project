from pathlib import Path
import cv2
import matplotlib.pyplot as plt

from src.data_pipeline.labels import load_image_mask_pairs

{
    "python.defaultInterpreterPath": "C:\\Users\\corba\\WIP_project\\.venv\\Scripts\\python.exe",
    "python.terminal.activateEnvironment": True
}

def preview_one(name=None):
    pairs = load_image_mask_pairs()
    if not pairs:
        print("No pairs found!")
        return

    # pick a specific name or the first one
    if name is None:
        name, paths = next(iter(pairs.items()))
    else:
        paths = pairs.get(name)
        if paths is None:
            print(f"{name} not found in pairs")
            return

    # load with OpenCV
    img = cv2.imread(paths["image"], cv2.IMREAD_COLOR)
    mask = cv2.imread(paths["mask"], cv2.IMREAD_GRAYSCALE)

    print("Image shape:", img.shape)
    print("Mask shape:", mask.shape)
    print("Unique mask values:", set(mask.flatten().tolist()))

    # Show them side by side
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Mask")
    plt.axis("off")

    plt.show()

if __name__ == "__main__":
    preview_one()