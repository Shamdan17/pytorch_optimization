# (Optional)
# Before running, you are also recommended to use the OMP_NUM_THREADS environment variable to control the number of threads used by PyTorch
# Set it to the number of threads/cores you have requested for your job divided by the number of GPUs you are using
# For example, for 4 GPUs with 20 cores each, you would set OMP_NUM_THREADS=5
# export OMP_NUM_THREADS=5

# USAGE, replace X with the number of GPUs you want to use
# torchrun --nproc_per_node=X 002-DistributedDataParallel.py

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

# DistributedDataParallel required imports
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from dist_utils import disable_printing
import os  # For os.environ

# For distributed training, we need to use the following function
# This function sets up the environment for distributed training, and lets every process know
# which process it is (rank) and how many processes there are (world_size)
def ddp_setup(args):
    # We get the rank and world size from the environment variables
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])

    print(
        "| Initializing process with rank {} out of {} processes |".format(
            args.rank, args.world_size
        ),
        flush=True,
    )

    # This is a useful hack I like to do sometimes. This DISABLES printing on nodes that are not rank 0, to make the output cleaner
    # disable_printing(args.rank == 0)

    torch.cuda.set_device(args.rank)

    init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        rank=args.rank,
        world_size=args.world_size,
    )


def main():

    # Parse arguments
    parser = get_parser()
    args = parser.parse_args()

    # STEP 1: Set up distributed training
    # This gets the rank and world size from the environment variables
    # And sets up the communication between the processes
    ddp_setup(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Create model (resnet18)
    model = get_model()

    # STEP 2: Wrap model in DistributedDataParallel
    # Similarly to DataParallel, we wrap the model with DistributedDataParallel
    # We use the local rank to make sure that each process uses the correct GPU
    model.to(args.rank)
    model = DDP(model, device_ids=[args.rank])

    # STEP 3: Convert batchnorm to SyncBatchNorm (If using batchnorm)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # print how many GPUs are being used
    # Since now we are using a different process for each GPU, we will see this message
    # the same number of times as the number of GPUs if we do not use the disable_printing function
    print("Using {} GPUs".format(torch.cuda.device_count()))

    # IMPORTANT: DistributedDataParallel uses the given batch size for every GPU
    #       The batch size given is PER GPU, and the total batch size is the number of GPUs times the batch size
    # If you want to use the given batch size, you must divide it by the number of GPUs
    assert (
        args.batch_size % torch.cuda.device_count() == 0
    ), "Batch size must be divisible by number of GPUs"
    args.batch_size = args.batch_size // torch.cuda.device_count()

    # Move model to GPU
    model = model.cuda()

    # Log with tensorboard
    # NOTE: In order to not log from every process, we use the following logic
    do_log = args.rank == 0  # only log from rank 0 to avoid cluttering the logs
    if do_log:
        writer = SummaryWriter(comment="distributed_data_parallel")

    # Create data
    train_dataset, val_dataset = get_data()

    # STEP 4: Use the distributed sampler class
    # This is a MUST: In order for every GPU to handle different parts of the data, we must use the distributed sampler class
    # This class will make sure that every GPU gets a different part of the data
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        shuffle=True,
    )
    # This is not necessary for validation, but if you do not do it, then you will run the same validation on every GPU
    # If you do use it, then we can use the DistributedDataParallel class to average the validation accuracy across all GPUs
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, shuffle=False
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
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
        # STEP 5: This is extremely important
        # You MUST call train_sampler.set_epoch(epoch) before every epoch
        # This is because the DistributedSampler class shuffles the data every epoch
        # Otherwise, the data will be the same for every epoch
        train_sampler.set_epoch(epoch)

        # Train for one epoch
        model = model.train()
        training_progress_bar = tqdm(train_loader, desc="training")
        training_count = 0
        training_loss = 0
        training_accuracy = 0

        for batch_idx, (data, target) in enumerate(training_progress_bar):
            # Zero out any existing gradients
            optimizer.zero_grad()

            # Move data to GPU
            data, target = data.cuda(), target.cuda()

            # Compute output separately for each GPU
            output = model(data)

            # Compute loss, again separately for each GPU
            loss = criterion(output, target)

            # Compute gradient and do SGD step
            # In DistributedDataParallel, loss.backward() will automatically average the gradients across all GPUs
            # This is why we don't need to divide the loss by the number of GPUs
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
        validation_progress_bar = tqdm(val_loader, desc="validation")
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

        # Get the average training/validation accuracy across all GPUs
        # This is done by using the all_reduce function from torch.distributed which sums the given tensor across all GPUs
        # NOTE: This is necessary as every GPU computes the validation accuracy separately for a different part of the data
        training_loss_tensor = torch.tensor(training_loss).to("cuda", non_blocking=True)
        training_accuracy_tensor = torch.tensor(training_accuracy).to(
            "cuda", non_blocking=True
        )
        validation_loss_tensor = torch.tensor(validation_loss).to(
            "cuda", non_blocking=True
        )
        validation_accuracy_tensor = torch.tensor(validation_accuracy).to(
            "cuda", non_blocking=True
        )

        # If you are curious:
        # print(
        #     "Before all_reduce: training_loss = {}, training_accuracy = {}, validation_loss = {}, validation_accuracy = {}, rank = {}".format(
        #         training_loss,
        #         training_accuracy,
        #         validation_loss,
        #         validation_accuracy,
        #         args.rank,
        #     ),
        #     flush=True,
        # )

        torch.distributed.all_reduce(validation_loss_tensor)
        torch.distributed.all_reduce(validation_accuracy_tensor)
        torch.distributed.all_reduce(training_loss_tensor)
        torch.distributed.all_reduce(training_accuracy_tensor)

        training_loss = training_loss_tensor.item() / args.world_size
        training_accuracy = training_accuracy_tensor.item() / args.world_size
        validation_loss = validation_loss_tensor.item() / args.world_size
        validation_accuracy = validation_accuracy_tensor.item() / args.world_size

        # If you are curious:
        # print(
        #     "After all_reduce: training_loss = {}, training_accuracy = {}, validation_loss = {}, validation_accuracy = {}, rank = {}".format(
        #         training_loss,
        #         training_accuracy,
        #         validation_loss,
        #         validation_accuracy,
        #         args.rank,
        #     ),
        #     flush=True,
        # )

        # Log statistics. We only do this on rank 0 typically with the do_log flag
        # since we don't want to log the same statistics multiple times
        if do_log:
            writer.add_scalar("train/loss", training_loss, epoch)
            writer.add_scalar("train/accuracy", training_accuracy, epoch)
            writer.add_scalar("val/loss", validation_loss, epoch)
            writer.add_scalar("val/accuracy", validation_accuracy, epoch)

        # Always save only on rank 0 to not overwrite the same file from multiple processes
        if args.rank == 0:
            if epoch % args.save_freq == 0:
                # Save best model only
                if validation_accuracy > best_val_accuracy:
                    best_val_accuracy = validation_accuracy
                    torch.save(
                        model.state_dict(),
                        os.path.join(args.save_dir, f"model_{epoch}.pth"),
                    )

    # How to load best model
    # model.load_state_dict(torch.load(os.path.join(args.save_dir, f"model_{epoch}.pth")))

    # IMPORTANT: If you are initializing the model again, you have two options:

    # Initialize model again, then wrap it with DDP before loading the state dict
    # model = get_model()
    # model = DDP(model)
    # model.load_state_dict(torch.load(os.path.join(args.save_dir, f"model_{epoch}.pth")))

    # OR, when saving the model state dict do the following:
    # torch.save(
    #     model.module.state_dict(), # <- This is the important part. You need to save the state dict of the model.module, not the model as it is now a DataParallel wrapper
    #     os.path.join(args.save_dir, f"model_{epoch}.pth"),
    # )

    # That way, when loading the model, you can do the following:
    # model = get_model()
    # model.load_state_dict(torch.load(os.path.join(args.save_dir, f"model_{epoch}.pth")))
    # model = DDP(model) # Now optionally wrap it with DDP

    # Close tensorboard writer
    writer.close()


if __name__ == "__main__":
    main()
