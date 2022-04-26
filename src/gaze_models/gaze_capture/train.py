import shutil, argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

import wandb
from ITrackerDataModule import ITrackerData

from lib.model import ITrackerModel
from lib.constants import *

from tqdm import tqdm

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
        total_samples = len(train_loader)

        for i, (_, face, eyeL, eyeR, grid, gaze) in tqdm(enumerate(train_loader), total=len(train_loader)):
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
            if i % PRINT_FREQ == 0:
                wandb.log({"step": _epoch * total_samples + i, "train/loss_step": losses.avg})

        print(
            f"Epoch [{_epoch}]\tLoss {losses.val:.4f} ({losses.avg:.4f})"
        )
        wandb.log({'train/loss': losses.avg, 'epoch': _epoch})
        prec1 = validate(val_loader, model, criterion, _epoch)
        wandb.log({'val/loss': losses.avg, 'epoch': _epoch})
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
    L2losses = AverageMeter()

    model.eval()

    for i, (_, face, eyeL, eyeR, grid, gaze) in tqdm(enumerate(val_loader)):
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
        L2loss = torch.mean(torch.sqrt(torch.sum(torch.square(output - gaze), 1)))
        losses.update(loss.data.item(), face.size(0))
        L2losses.update(L2loss.item(), face.size(0))

        # if i % PRINT_FREQ == 0:
        # print(
        #     f"Epoch [{epoch}][{i}/{len(val_loader)}]\tLoss {losses.val:.4f} ({losses.avg:.4f})\tL2 Loss {L2losses.val:.4f} ({L2losses.avg:.4f})"
        # )
    print(
        f"Epoch [{epoch}][]\tLoss {losses.val:.4f} ({losses.avg:.4f})\tL2 Loss {L2losses.val:.4f} ({L2losses.avg:.4f})"
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

    wandb.login()
    wandb.init(project='CV')

    model = ITrackerModel()
    model.cuda()
    cudnn.benchmark = True

    completed_epoch = 0
    if args.checkpoint != "":
        checkpoint = load_checkpoint()
        model.load_state_dict(checkpoint["state_dict"])
        completed_epoch = checkpoint["epoch"]
        best_prec1 = checkpoint["best_prec1"]

    dataTrain = ITrackerData(data_path=args.data_path, split="train", im_size=IMAGE_SIZE)
    dataVal = ITrackerData(data_path=args.data_path, split="val", im_size=IMAGE_SIZE)

    train_loader = torch.utils.data.DataLoader(
        dataTrain,
        batch_size=BATCH_SIZE,
        shuffle=True,
        # num_workers=WORKERS,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        dataVal,
        batch_size=BATCH_SIZE,
        shuffle=False,
        # num_workers=WORKERS,
        pin_memory=True,
    )

    criterion = nn.MSELoss().cuda()

    optimizer = torch.optim.SGD(
        model.parameters(), lr, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
    )

    train(train_loader, val_loader, model, criterion, optimizer, completed_epoch)


if __name__ == "__main__":
    main()
