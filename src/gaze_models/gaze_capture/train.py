import math, shutil, os, time, argparse
from pyexpat import model
import numpy as np
import scipy.io as sio
from scipy.misc import face

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from model import ITrackerModel
from constants import *

parser = argparse.ArgumentParser(description="iTracker-pytorch-Trainer.")
parser.add_argument(
    "--data_path",
    help="path to processed dataset",
)
parser.add_argument(
    "--checkpoint",
    default="",
    help="checkpoint file",
)
args = parser.parse_args()

BATCH_SIZE = torch.cuda.device_count() * 100

best_prec1 = 1e20
lr = BASE_LR


def train(train_loader, val_loader, model, criterion, optimizer, completed_epoch):
    global best_prec1

    losses = AverageMeter()

    for _epoch in range(NUM_EPOCHS):
        if _epoch < completed_epoch:
            lr_decay(optimizer, _epoch)
            continue

        lr_decay(optimizer, _epoch)

        model.train()

        for i, (_, face, eyeL, eyeR, grid, gaze) in enumerate(train_loader):
            face = face.cuda()
            eyeL = eyeL.cuda()
            eyeR = eyeR.cuda()
            grid = grid.cuda()
            gaze = gaze.cuda()

            face = torch.autograd.Variable(face, requires_grad=True)
            eyeL = torch.autograd.Variable(eyeL, requires_grad=True)
            eyeR = torch.autograd.Variable(eyeR, requires_grad=True)
            grid = torch.autograd.Variable(grid, requires_grad=True)
            gaze = torch.autograd.Variable(gaze, requires_grad=False)

            output = model(face, eyeL, eyeR, grid)
            loss = criterion(output, gaze)
            losses.update(loss.data.item(), face.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(
                f"Epoch [{_epoch}][{i}/{len(val_loader)}]\tLoss {losses.val:.4f} ({losses.avg:.4f})"
            )

        prec1 = validate(val_loader, model, criterion, _epoch)
        best_prec1 = min(prec1, best_prec1)
        save_checkpoint(
            {
                "epoch": _epoch + 1,
                "state_dict": model.state_dict(),
                "best_prec1": best_prec1,
            },
            prec1 < best_prec1,
        )


def validate(val_loader, model, criterion, epoch):
    losses = AverageMeter()

    model.eval()

    for i, (_, face, eyeL, eyeR, grid, gaze) in enumerate(val_loader):
        face = face.cuda()
        eyeL = eyeL.cuda()
        eyeR = eyeR.cuda()
        grid = grid.cuda()
        gaze = gaze.cuda()

        face = torch.autograd.Variable(face, requires_grad=False)
        eyeL = torch.autograd.Variable(eyeL, requires_grad=False)
        eyeR = torch.autograd.Variable(eyeR, requires_grad=False)
        grid = torch.autograd.Variable(grid, requires_grad=False)
        gaze = torch.autograd.Variable(gaze, requires_grad=False)

        with torch.no_grad():
            output = model(face, eyeL, eyeR, grid)

        loss = criterion(output, gaze)
        losses.update(loss.data.item(), face.size(0))

        print(
            f"Epoch [{epoch}][{i}/{len(val_loader)}]\tLoss {losses.val:.4f} ({losses.avg:.4f})"
        )

    return losses.avg


def load_checkpoint():
    checkpoint = torch.load(args.checkpoint)
    return checkpoint


def save_checkpoint(state, is_best):
    filename = args.checkpoint.split("/")[-1]
    dir = "/".join(args.checkpoint.split("/")[:-1])

    bestFilename = f"{dir}/best_{filename}"
    torch.save(state, args.checkpoint)
    if is_best:
        shutil.copyfile(args.checkpoint, bestFilename)


def lr_decay(optimizer, epoch):
    lr = BASE_LR * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def main():
    global best_prec1

    model = ITrackerModel()
    model = torch.nn.DataParallel(model)
    model.cuda()
    cudnn.benchmark = True

    completed_epoch = 0
    if args.checkpoint != "":
        checkpoint = load_checkpoint()
        try:
            model.module.load_state_dict(checkpoint["state_dict"])
        except:
            model.load_state_dict(checkpoint["state_dict"])
        completed_epoch = checkpoint["epoch"]
        best_prec1 = checkpoint["best_prec1"]

    dataTrain = ITrackerData(dataPath=args.data_path, split="train", imSize=IMAGE_SIZE)
    dataVal = ITrackerData(dataPath=args.data_path, split="val", imSize=IMAGE_SIZE)

    train_loader = torch.utils.data.DataLoader(
        dataTrain,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=WORKERS,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        dataVal,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=True,
    )

    criterion = nn.MSELoss().cuda()

    optimizer = torch.optim.SGD(
        model.parameters(), lr, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
    )

    train(train_loader, val_loader, model, criterion, optimizer, completed_epoch)


if __name__ == "__main__":
    main()
