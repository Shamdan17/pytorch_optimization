# Train resnet50 on Cifar10 using only 1 GPU
# Usage: python 003-ModelParallel.py

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader


from tqdm import tqdm, trange

from utils import get_parser
from model import get_model

# Dataset code
from data import get_data

# Log with tensorboard
from tensorboardX import SummaryWriter

# Set seed
torch.manual_seed(0)
np.random.seed(0)

# STEP 1: We implement a version of resnet18 which uses two GPUs
class myresnet18(nn.Module):
    def __init__(
        self, num_classes=10, devices=[torch.device("cuda:0"), torch.device("cuda:1")]
    ):
        super(myresnet18, self).__init__()
        model = get_model()

        # This is the forward of resnet18 provided by torchvision
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)

        # x = self.layer1(x)
        # x = self.layer2(x)

        # x = self.layer3(x)
        # x = self.layer4(x)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        # return x

        # We will create two modules, one for the layers until layer2, and one for the rest
        self.model_part1 = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
        )

        self.model_part2 = nn.Sequential(
            model.layer3,
            model.layer4,
            model.avgpool,
            nn.Flatten(),
            model.fc,
        )

        self.devices = devices
        self.model_part1.to(devices[0])
        self.model_part2.to(devices[1])

    def forward(self, x):
        # Compute the first part of the model
        x = self.model_part1(x)
        x = x.to(self.devices[1])
        x = self.model_part2(x)
        return x


def main():
    # Parse arguments
    parser = get_parser()
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # STEP 2: Make sure we have two GPUs
    assert torch.cuda.device_count() >= 2

    devices = [torch.device("cuda:0"), torch.device("cuda:1")]

    # STEP 3: Create model (resnet18 split in two)
    model = myresnet18(devices=devices)

    # Don't move model to GPU, we will do it manually in the model itself
    # model = model.cuda()

    # Log with tensorboard
    writer = SummaryWriter(comment="model_parallel_example")

    # Create data
    train_dataset, val_dataset = get_data()

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Create criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    best_val_accuracy = 0

    # Train model
    for epoch in trange(args.epochs, desc="Epoch"):
        # Train for one epoch
        model = model.train()
        training_progress_bar = tqdm(train_loader, desc="batch")
        training_count = 0
        training_loss = 0
        training_accuracy = 0

        for batch_idx, (data, target) in enumerate(training_progress_bar):
            # Zero out any existing gradients
            optimizer.zero_grad()

            # STEP 4: Move data to first device, and target to second device
            data = data.to(devices[0])
            target = target.to(devices[1])

            # Compute output
            output = model(data)
            loss = criterion(output, target)

            # Compute gradient and do SGD step
            loss.backward()
            optimizer.step()

            # Measure accuracy and record loss
            accuracy = (output.argmax(dim=1) == target).float().mean()

            training_count += data.size(0)
            training_loss += loss.item() * data.size(0)
            training_accuracy += accuracy.item() * data.size(0)

            # Update bar with statistics
            if batch_idx % args.print_freq == 0:
                training_progress_bar.set_postfix(loss=loss.item())

        # Evaluate model
        model = model.eval()
        validation_progress_bar = tqdm(val_loader, desc="batch")
        validation_count = 0
        validation_loss = 0
        validation_accuracy = 0

        for batch_idx, (data, target) in enumerate(validation_progress_bar):
            # STEP 5: Move data to first device, and target to second device for validation as well
            data = data.to(devices[0])
            target = target.to(devices[1])

            # Compute output. Disable gradient calculation to save memory
            with torch.no_grad():
                output = model(data)
                loss = criterion(output, target)

            # Measure accuracy and record loss
            accuracy = (output.argmax(dim=1) == target).float().mean()

            validation_count += data.size(0)
            validation_loss += loss.item() * data.size(0)
            validation_accuracy += accuracy.item() * data.size(0)

            # Update bar with statistics
            if batch_idx % args.print_freq == 0:
                validation_progress_bar.set_postfix(loss=loss.item())

        # Compute statistics
        training_loss /= training_count
        training_accuracy /= training_count
        validation_loss /= validation_count
        validation_accuracy /= validation_count

        # Log statistics
        writer.add_scalar("train/loss", training_loss, epoch)
        writer.add_scalar("train/accuracy", training_accuracy, epoch)
        writer.add_scalar("val/loss", validation_loss, epoch)
        writer.add_scalar("val/accuracy", validation_accuracy, epoch)

        # Save best model
        if epoch % args.save_freq == 0:
            if validation_accuracy > best_val_accuracy:
                best_val_accuracy = validation_accuracy
                torch.save(
                    model.state_dict(),
                    os.path.join(args.save_dir, f"model_{epoch}.pth"),
                )

    # How to load best model
    # model.load_state_dict(torch.load(os.path.join(args.save_dir, f"model_{epoch}.pth")))

    # Close tensorboard writer
    writer.close()


if __name__ == "__main__":
    main()
