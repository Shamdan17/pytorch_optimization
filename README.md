# General code optimization with PyTorch

## Overview

This repository contains a brief introduction as well as examples of different possible optimizations that speed up training as well as decrease RAM and GPU VRAM usage.

The different modules are found within the [modules](./modules/) folder.

## Using multiple GPUs

An example of using multiple gpus using both DataParallel as well as DistributedDataParallel can be found within the [MultiGPU](./modules/MultiGPU/) module. 

## Speeding up training using half or mixed precision

An example of using CUDA amp as well as FP16 training can be found within the [MPTraining](./modules/MPTraining)

## Checking memory/cpu/vram usage
If you have a job on a node, you can always ssh to it to monitor your job. You can take a look at 

## Useful resources: 

- [PyTorch Distributed Overview](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [PyTorch DDP Examples](https://github.com/pytorch/examples/tree/main/distributed)

