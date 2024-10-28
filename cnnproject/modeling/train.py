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
from cnnproject import dataset
from cnnproject.dataset import ImageDataset


app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR,
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    # -----------------------------------------
):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
        )
    logger.info(f"Using {device} device")

    model= basicNN(num_classes=3).to(device)
    print(model)

    #Load downscaled dataset
    with open(PROCESSED_DATA_DIR / 'train_dataset_downscaled.pkl', 'rb') as f:
        train_dataset = pickle.load(f)
    train_basic_NN(device)

    
def train_basic_NN(device):
    model= basicNN(num_classes=3).to(device)
    print(model)

    #Load downscaled dataset
    with open(PROCESSED_DATA_DIR / 'train_dataset_downscaled.pkl', 'rb') as f:
        train_dataset = pickle.load(f)
    train_dataloader = DataLoader(train_dataset, batch_size=64)

    with open(PROCESSED_DATA_DIR / 'val_dataset_downscaled.pkl', 'rb') as f:
        test_dataset = pickle.load(f)
    test_dataloader = DataLoader(test_dataset, batch_size=64)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch_size = 32

    epochs = 25
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, batch_size)
        validate(test_dataloader, model, loss_fn)
    logger.info("Done!")

    # Save the model
    torch.save(model , MODELS_DIR / "basicNN.pth")
    logger.success(f"Model saved to {MODELS_DIR / 'basicNN.pth'}")
    

#Train/Test loop based on the lovely code at pytorch
def train(dataloader, model, loss_fn, optimizer, batch_size):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def validate(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


class basicNN(nn.Module):
    def __init__(self, num_classes):
        super(basicNN, self).__init__()
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(128*128*3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.model(x)

class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"], float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )



if __name__ == "__main__":
    app()
