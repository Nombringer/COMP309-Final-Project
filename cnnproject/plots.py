from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import os
from cnnproject.config import FIGURES_DIR, PROCESSED_DATA_DIR
import copy
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pickle
app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = FIGURES_DIR / "plot.png",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating plot from data...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Plot generation complete.")
    # -----------------------------------------
    #Load training dataset object from file
    with open(PROCESSED_DATA_DIR / 'train_dataset_downscaled.pkl', 'rb') as f:
        dataset = pickle.load(f)
    #Visualize augmentations
    visualize_augmentations(dataset)


#Useful utilities adapted from: https://albumentations.ai/docs/examples/pytorch_classification/
def display_image_grid(images_filepaths, predicted_labels=(), cols=5):
    rows = len(images_filepaths) // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    for i, image_filepath in enumerate(images_filepaths):
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        true_label = os.path.normpath(image_filepath).split(os.sep)[-2]
        predicted_label = predicted_labels[i] if predicted_labels else true_label
        color = "green" if true_label == predicted_label else "red"
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_title(predicted_label, color=color)
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()


def visualize_augmentations(dataset, idx=102, samples=10, cols=5):
    dataset = copy.deepcopy(dataset)
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    rows = samples // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    for i in range(samples):
        image, _ = dataset[idx]
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()    


if __name__ == "__main__":
    app()
