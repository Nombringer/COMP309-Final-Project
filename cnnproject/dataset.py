from pathlib import Path
import albumentations as A
import typer
from loguru import logger
from tqdm import tqdm
import os
import cv2
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from albumentations.pytorch import ToTensorV2
from cnnproject.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, SEED
from cnnproject import plots
import pickle


random.seed(SEED)
app = typer.Typer()




@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    # -----------------------------------------
    logger.info("Processing dataset...")
    
    image_paths = []
    for f in os.listdir(PROCESSED_DATA_DIR / 'resized'):
        image_paths.append(PROCESSED_DATA_DIR / 'resized' / f)
    # -----------------------------------------
    random.shuffle(image_paths)
    train_paths = image_paths[:4000]
    val_paths = image_paths[4000:-10]
    test_paths = image_paths[-10:]
    logger.info(f"Number of training images: {len(train_paths)}, validation images: {len(val_paths)}, test images: {len(test_paths)}")
    
    train_transform = A.Compose(
        [
            A.Blur(blur_limit=6, p=0.3),
            A.CLAHE(p=0.3),
            A.Downscale(p=0.1),
            A.GaussNoise(p=0.5),
            A.ChannelDropout(p=0.1),
            A.HueSaturationValue(p=0.1),
            A.Sharpen(p=0.2),
            A.GlassBlur(p=0.1),
            A.Spatter(p=0.1),
            A.RandomShadow(p=0.3),
            A.RandomBrightnessContrast(p=0.3),
            A.RandomToneCurve(p=0.3),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=25, p=0.5),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), #Taken from ImageNet
            ToTensorV2()
        ]
    )

    train_transform_downscaled = A.Compose(
        [
            A.SmallestMaxSize(max_size=160),
            A.RandomCrop(128, 128),
            A.Blur(blur_limit=6, p=0.3),
            A.CLAHE(p=0.3),
            A.Downscale(p=0.1),
            A.GaussNoise(p=0.5),
            A.ChannelDropout(p=0.1),
            A.HueSaturationValue(p=0.1),
            A.Sharpen(p=0.2),
            A.GlassBlur(p=0.1),
            A.Spatter(p=0.1),
            A.RandomShadow(p=0.3),
            A.RandomBrightnessContrast(p=0.3),
            A.RandomToneCurve(p=0.3),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=25, p=0.5),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), #Taken from ImageNet
            ToTensorV2() 
        ]
    )

    train_dataset = ImageDataset(train_paths, transform=train_transform)
    train_dataset_downscaled = ImageDataset(train_paths, transform=train_transform_downscaled)

    val_transform = A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), #Taken from ImageNet
            ToTensorV2()
        ]
    )
    val_transform_downscaled = A.Compose(
        [
            A.SmallestMaxSize(max_size=160),
            A.CenterCrop(128, 128),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), #Taken from ImageNet
            ToTensorV2()
        ]
    )
    val_dataset = ImageDataset(val_paths, transform=val_transform)
    val_dataset_downscaled = ImageDataset(val_paths, transform=val_transform_downscaled)
    #save datasets
    with open(PROCESSED_DATA_DIR / 'train_dataset.pkl', 'wb') as f:
        pickle.dump(train_dataset, f)
    with open(PROCESSED_DATA_DIR / 'val_dataset.pkl', 'wb') as f:
        pickle.dump(val_dataset, f)
    with open(PROCESSED_DATA_DIR / 'train_dataset_downscaled.pkl', 'wb') as f:
        pickle.dump(train_dataset_downscaled, f)
    with open(PROCESSED_DATA_DIR / 'val_dataset_downscaled.pkl', 'wb') as f:
        pickle.dump(val_dataset_downscaled, f)
    
    logger.info(f"Saved train and val datasets to {PROCESSED_DATA_DIR}")

    plots.visualize_augmentations(train_dataset, idx=10, samples=10, cols=5)
    plots.visualize_augmentations(train_dataset_downscaled, idx=10, samples=10, cols=5)

    
class ImageDataset:
    def __init__(self, image_paths, transform = None):
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def getlabel(self, image_path):
        #strawberry = 0, tomato = 1, cherry = 2
        img_name = image_path.split('/')[-1]
        if 'strawberry' in img_name:
            return 0
        elif 'tomato' in img_name:
            return 1
        elif 'cherry' in img_name:
            return 2
        else:
            logger.error(f'Error getting label for image {img_name}')
            return None
    
    def __getitem__(self, idx):
        image = cv2.imread(str(self.image_paths[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.getlabel(str(self.image_paths[idx]))
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, label



def resize_images():
    for folder in os.listdir(RAW_DATA_DIR / 'train_data'):
        #Check folder is a directory
        if not os.path.isdir(RAW_DATA_DIR / 'train_data' / folder):
            continue

        for image in os.listdir(RAW_DATA_DIR / 'train_data' / folder):
            #We have to flip the colorchannels because cv2 reads in BGR instead of RGB
            cvimage = cv2.imread(str(RAW_DATA_DIR / 'train_data' / folder / image) , cv2.COLOR_BGR2RGB)
            
            if cvimage is None:
                logger.error(f'Error reading image image in folder {folder}')
                continue
            if cvimage.shape != (300, 300, 3):
                #Resize image
                #We could use more complicated transforms here, but there are so few images that it's not worth it.
                #I'm mainly just including this to show where it would happen: You could also just delete these there are so few
                cvimage = cv2.resize(cvimage, (300, 300))
                logger.info(f'Resized image in folder {folder} to 300x300')
            #Save image
            cv2.imwrite(str(PROCESSED_DATA_DIR / 'resized' /f'{image}.jpg'), cvimage)
            logger.info(f'Saved image {image} in folder {folder} to PROCESSED_DATA_DIR / resized')    
            

        

    
    
    

        
    

def augment_image(image, image_name, directory):
    #Outputs randomly augemented images to the directory
    pass


    

if __name__ == "__main__":
    app()
