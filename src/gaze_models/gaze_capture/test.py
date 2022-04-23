import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

from constants import *
from model import ITrackerModel
from train import load_checkpoint, AverageMeter

parser = argparse.ArgumentParser(description="iTracker-pytorch-Tester.")
parser.add_argument(
    "--data_path",
    help="path to processed dataset",
)
parser.add_argument(
    "--best",
    default="",
    help="best checkpoint file",
)
args = parser.parse_args()

BATCH_SIZE = torch.cuda.device_count() * 100


def test(test_loader, model):
    L2losses = AverageMeter()

    model.eval()

    for _, (_, face, eyeL, eyeR, grid, gaze) in enumerate(test_loader):
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

        L2loss = torch.mean(torch.sqrt(torch.sum(torch.square(output - gaze), 1)))
        L2losses.update(L2loss.item(), face.size(0))

    return L2losses.avg


def main():
    global best_prec1

    model = ITrackerModel()
    model.cuda()
    cudnn.benchmark = True

    if args.checkpoint != "":
        checkpoint = load_checkpoint()
        model.load_state_dict(checkpoint["state_dict"])

    dataTest = ITrackerData(dataPath=args.data_path, split="test", imSize=IMAGE_SIZE)

    test_loader = torch.utils.data.DataLoader(
        dataTest,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=True,
    )

    best_prec1 = test(test_loader, model)
    print(f"Best L2 Loss {best_prec1:.4f}")


if __name__ == "__main__":
    main()
