# Project 2: Distributed Data Parallel Training on CIFAR-10
**Members:** hle14, jdk330, riyap06, tvignesh, xavierphan
**Names:** Hieu Le, Jack Kochan, Riya Pasupulati, Vignesh Thallam, Xavier Phan

---

## File Structure
```text
cmda3634-pr2/
├── .gitignore
├── README.md
├── cifar10_ddp.py
├── job.sh
├── output.log
└── cifar10.py
```

## 1. Process Topology and Rank Assignment
Our DDP implementation uses one Falcon node with two NVIDIA A30 GPUs. We launch one Python process per GPU.

Node 0
├── GPU 0 → local_rank = 0 → global_rank = 0
└── GPU 1 → local_rank = 1 → global_rank = 1

The global rank is computed as:
```
global_rank = args.node_id * args.num_gpus + local_rank
```
The world size is computed as:
```
world_size = args.num_nodes * args.num_gpus
```
For our successful run, we used `num_nodes = 1`, `node_id = 0`, and `num_gpus = 2`, so `world_size = 2`.

Each process is pinned to its assigned GPU using:
```
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")
```

The process group is initialized using the NCCL backend with `dist.init_process_group(...)`. This gives each process a unique rank and allows the two GPU workers to communicate during training.

## 2. Data Partitioning
The original single-GPU script used a normal shuffled DataLoader. In the DDP version, we use `DistributedSampler` so each GPU receives a different part of the CIFAR-10 dataset.

The training sampler uses the total number of workers, the current process rank, and shuffling. The validation sampler also uses the total number of workers and the current rank, but does not shuffle.

This prevents both GPUs from training on the same exact samples. Each process only sees its own shard of the dataset.

At the start of each epoch, we call:
```
train_sampler.set_epoch(epoch)
```
This changes the shuffle order each epoch while keeping all ranks synchronized. Without this, the sampler could use the same ordering every epoch.

## 3. Gradient Synchronization
The model is wrapped with PyTorch `DistributedDataParallel`:
```
model = DDP(model, device_ids=[local_rank])
```

Each GPU has its own copy of the WideResNet model. During the forward pass, each process works on its own mini-batch. During the backward pass, DDP averages the gradients across all GPUs using an all-reduce operation.

Then each process applies the same optimizer update. Because every process starts from the same weights and applies the same averaged gradient update, all model replicas stay synchronized.

## 4. Metric Aggregation
Training accuracy is computed by summing the correct predictions and total labels across all GPUs. This gives the true training accuracy across the full distributed batch.

Validation accuracy and validation loss are averaged across ranks using `dist.all_reduce` with `dist.ReduceOp.AVG`. This is appropriate because each rank computes validation metrics on its own shard, and the final result should represent the average validation performance across all shards.

Throughput, or images per second, is summed using `dist.reduce` with `dist.ReduceOp.SUM` and sent to rank 0. Throughput is summed because each GPU processes images independently. The total images per second represents the total work completed by the whole distributed job.

Only rank 0 prints the results so the log does not contain duplicate output from each process.

## 5. Performance and Scaling
| Metric | 1 GPU (Baseline) | 2 GPUs (Falcon A30) |
|--------|------------------:|---------------------:|
| Images/sec | 1627.227 | 3040.142 |
| Time to 85% Acc | 323.679 sec | 186.904 sec |
| Final Accuracy | 0.853 | 0.873 |

The target was validation accuracy greater than or equal to 0.85 for two consecutive epochs. The two-GPU run met the target at epochs 10 and 11.

Epoch 10 validation accuracy was 0.856.
Epoch 11 validation accuracy was 0.873.

The cumulative time at early stopping for the two-GPU run was 186.904 seconds.

The one-GPU baseline stopped after epoch 7 with a cumulative time of 502.202 seconds.

The observed speedup was:
```
323.679 / 186.904 = 1.73x
```

The DDP version was much faster than the single-GPU version. The speedup came from splitting the training work across two GPUs and increasing total throughput. The scaling is not perfectly ideal because DDP adds overhead from gradient synchronization, validation, data loading, CPU-to-GPU transfers, and process coordination.

## 6. Optimizations
We increased the batch size for the two-GPU run.

| Run | Batch Size | Samples/sec near end |
|-----|-----------:|--------------------:|
| 1 GPU | 32 | 709.185 |
| 2 GPUs | 64 | 3040.142 |

The larger batch size improved GPU utilization and increased throughput.

We also used these DataLoader settings:
```
num_workers=2
prefetch_factor=2
persistent_workers=True
pin_memory=True
```

These settings help data loading by keeping worker processes alive, prefetching batches, and using pinned memory for faster CPU-to-GPU transfer.

## Software Environment
On Falcon, we used:
```
module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
```
Inside the repository:
```
python3 -m venv pytorch
source pytorch/bin/activate
```
If PyTorch or torchvision were missing:
```
python3 -m pip install torch torchvision
```

The `.gitignore` excludes the Python virtual environment and CIFAR-10 dataset files.
