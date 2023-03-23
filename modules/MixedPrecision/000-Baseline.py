# Train resnet50 on Cifar10 using only 1 GPU
# Usage: python train.py
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


def main():
    # Parse arguments
    parser = get_parser()
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Create model (resnet18)
    model = get_model()

    # Move model to GPU
    model = model.cuda()

    # Log with tensorboard
    writer = SummaryWriter(comment="full_precision_baseline")

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

            # Move data to GPU
            data, target = data.cuda(), target.cuda()

            # Compute output
            output = model(data)
            loss = criterion(output, target)

            # Compute gradient and do SGD step
            optimizer.zero_grad()
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
            # Move data to GPU
            data, target = data.cuda(), target.cuda()

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
