"""
Author: lu.zhiping@u.nus.edu


"""
import torch
import torchvision
import numpy as np
from dataset import DriveableDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(
    train_file_list,
    val_file_list,
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    pin_memory=True):

    train_ds = DriveableDataset(
        file_list=train_file_list,
        image_path=train_dir,
        mask_path=train_maskdir,
        transform=train_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = DriveableDataset(
        file_list=val_file_list,
        image_path=val_dir,
        mask_path=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()


def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()


def mIOU(label, pred, num_classes):
    iou_list = []
    present_iou_list = []

    pred = pred.view(-1)
    label = label.view(-1)
    # Note: Following for loop goes from 0 to (num_classes-1)
    # and ignore_index is num_classes, thus ignore_index is
    # not considered in computation of IoU.
    for sem_class in range(num_classes):
        pred_inds = (pred == sem_class)
        target_inds = (label == sem_class)
        if target_inds.long().sum().item() == 0:
            iou_now = float('nan')
        else: 
            intersection_now = (pred_inds[target_inds]).long().sum().item()
            union_now = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection_now
            iou_now = float(intersection_now) / float(union_now)
            present_iou_list.append(iou_now)
        iou_list.append(iou_now)
    return np.mean(present_iou_list), iou_list, present_iou_list


def eval_fn_mIOU(loader, model, DEVICE):
    model.eval()
    
    loop = tqdm(loader)
    result = []
    with torch.no_grad():
        softmax = nn.Softmax(dim=1)
        for data, target in loop:
            data = data.to(device=DEVICE)
            target = target.to(device=DEVICE)
            pred = torch.argmax(softmax(model(data)), axis=1)
            result.append(mIOU(target, pred, num_classes=3))
    return result


def eval_fn_loss(loader, model, loss_fn, DEVICE):
    model.eval()
    loop = tqdm(loader)
    history = []
    
    with torch.no_grad():
        for _, (data, targets) in enumerate(loop):
            data = data.to(device=DEVICE)
            targets = targets.to(device=DEVICE, dtype=torch.long)
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            history.append(loss.item())
    return history
        


def mIOU_epoch_end(result):
    sumIOU = 0
    for item in result:
        sumIOU += item[0]
    return sumIOU / len(result)