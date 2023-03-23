# Utilizing multiple GPUs with PyTorch
In this folder, we train a small resnet-50 on the cifar-10 dataset. There are three scripts: 

- 000-Baseline.py
    - This is the original training with no changes
- 001-DataParallel.py
    - This is using DataParallel, which utilizes multiple GPUs using a single python process.
- 002-DistributedDataParallel.py
    - This is using DistributedDataParallel, which spawns a separate python process for every GPU.