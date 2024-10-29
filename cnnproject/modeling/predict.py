from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
import pickle
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

from collections import defaultdict
from cnnproject.config import MODELS_DIR, PROCESSED_DATA_DIR
from cnnproject.config import REPORTS_DIR
from cnnproject import dataset
from cnnproject.dataset import ImageDataset
from cnnproject.modeling.train import basicNN
from torchvision.models.convnext import LayerNorm2d
from functools import partial
#import metrics from sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Performing inference for model...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Inference complete.")
    # -----------------------------------------
    #setup device
    device = ( "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using {device} device")

    # Load model from file
    logger.info("Loading model from file")
    with MODELS_DIR / "CNN_epoch_9.pth" as f:
        model = torch.load(f)
    
    #Load dataset from file
    logger.info("Loading test dataset from file")
    with open(PROCESSED_DATA_DIR / 'test_dataset.pkl', 'rb') as f:
        test_dataset = pickle.load(f)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    logger.info("Test dataset loaded")
    test(model, test_loader, device, "CNN_epoch_9")



def test(model, test_loader, device, model_name):
    model.eval()
    logger.info("Testing model")
    stream = tqdm(test_loader, leave=True)
    with torch.no_grad():
        for i, (images, labels) in enumerate(stream):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            if i == 0:
                all_predicted = predicted
                all_labels = labels
            else:
                all_predicted = torch.cat((all_predicted, predicted))
                all_labels = torch.cat((all_labels, labels))
    #print metrics to file
    file_name  = f"{model_name}_metrics.txt"
    file_out = REPORTS_DIR / file_name
    with open(file_out, 'w') as f:
        f.write("Accuracy: " + str(accuracy_score(all_labels, all_predicted)) + "\n")
        f.write("Precision: " + str(precision_score(all_labels, all_predicted, average='macro')) + "\n")
        f.write("Recall: " + str(recall_score(all_labels, all_predicted, average='macro')) + "\n")
        f.write("F1: " + str(f1_score(all_labels, all_predicted, average='macro')) + "\n")
        f.write("Classification Report: " + "\n" + classification_report(all_labels, all_predicted) + "\n")
    logger.success("Metrics saved to " + file_name)



if __name__ == "__main__":
    app()
