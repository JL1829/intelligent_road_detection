"""
Author: lu.zhiping@u.nus.edu

Training Script for U-Net
"""
import torch
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from unet import UNET
from dataset import DriveableDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from utils import (
    load_checkpoint,
    save_checkpoint,
    eval_fn_mIOU,
    eval_fn_loss,
    mIOU_epoch_end,
    get_loaders
)


# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 20
NUM_WORKERS = 2
IMAGE_HEIGHT = 160*2  # 1280 originally
IMAGE_WIDTH = 240*2  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_FILE = "/home/ubuntu/works/driveable/train.txt"
TRAIN_IMG_DIR = "/home/ubuntu/works/driveable/bdd100k/images/100k/train"
TRAIN_MASK_DIR = "/home/ubuntu/works/driveable/dataset/bdd100k_label/labels/drivable/masks/train"
VAL_IMG_DIR = "/home/ubuntu/works/driveable/bdd100k/images/100k/train"
VAL_MASK_DIR = "/home/ubuntu/works/driveable/dataset/bdd100k_label/labels/drivable/masks/train"
TEST_IMG_DIR = ""


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    loss_history = []

    for _, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE, dtype=torch.long)

        # forward pass with mix precision training
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        loss_history.append(loss.item())
    return loss_history


def main():
    with open(TRAIN_FILE) as file:
        FILE_LIST = file.read().splitlines()
    split = train_test_split(
        FILE_LIST, test_size=0.2, random_state=42
    )
    train_file_list, val_file_list = split[0], split[1]
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=3, out_channels=3).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # get training and validation generator
    train_loader, val_loader = get_loaders(
        train_file_list=train_file_list,
        val_file_list=val_file_list,
        train_dir=TRAIN_IMG_DIR,
        train_maskdir=TRAIN_MASK_DIR,
        val_dir=VAL_IMG_DIR,
        val_maskdir=VAL_MASK_DIR,
        batch_size=BATCH_SIZE,
        train_transform=train_transform,
        val_transform=val_transforms,
        pin_memory=True
    )
    
    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    # clean cache and enable mix presicion
    torch.cuda.empty_cache()
    scaler = torch.cuda.amp.GradScaler()

    # metrics record
    train_loss_each_epoch = []
    val_loss_each_epoch = []
    train_mIOU_each_epoch = []
    val_mIOU_each_epoch = []
    
    # train epoch
    epoch = 0
    for _ in range(NUM_EPOCHS):
        print(f"Epoch: {epoch} of {NUM_EPOCHS}")
        print("Training Stage")
        history_train = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        train_loss_each_epoch.append(sum(history_train) / len(history_train))
        train_mIOU = eval_fn_mIOU(train_loader, model, DEVICE)
        train_mIOU_each_epoch.append(mIOU_epoch_end(train_mIOU))
        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)
        print("Eval Stage")
        history_val = eval_fn_loss(val_loader, model, loss_fn, DEVICE)
        val_mIOU = eval_fn_mIOU(val_loader, model, DEVICE)
        val_loss_each_epoch.append(sum(history_val) / len(history_val))
        val_mIOU_each_epoch.append(mIOU_epoch_end(val_mIOU))
        
        epoch += 1

    logs = pd.DataFrame({
        "Training_Loss_Epoch": train_loss_each_epoch,
        "Val_Loss_Epoch": val_loss_each_epoch,
        "Training_mIOU_Epoch": train_mIOU_each_epoch,
        "Val_mIOU_Epoch": val_mIOU_each_epoch
    })
    logs.to_csv("training_logs.csv", index=False)

if __name__ == "__main__":
    main()
