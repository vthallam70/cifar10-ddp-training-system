#import os  # may not be needed?
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# DDP specific imports
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# --------------------------------------
# Command line arguments
# --------------------------------------

parser = argparse.ArgumentParser(
    description="CIFAR-10 DDP example",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training per GPU")
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--target-accuracy", type=float, default=0.85, help="Target accuracy for early stopping")
parser.add_argument("--patience", type=int, default=2, help="Number of epochs that meet the target accuracy")
parser.add_argument("--model", type=str, default="WideResNet", choices=["SimpleConvNet", "WideResNet"])

# NEW DDP Arguments (Requirement R1.1)
parser.add_argument("--num-nodes", type=int, default=1, help="Total number of nodes")
parser.add_argument("--node-id", type=int, default=0, help="Unique ID for this node")
parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs per node")

# --------------------------------------
# Models (Unchanged - SimpleConvNet & WideResNet)
# --------------------------------------
# [NOTE: Keep SimpleConvNet and WideResNet exactly as they were in cifar10.py. 
# Omitted here to keep the skeleton clean, just copy-paste them back in.]
class SimpleConvNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        z = self.conv1(x)
        z = F.relu(z)
        z = self.conv2(z)
        z = F.relu(z)
        z = F.max_pool2d(z, 2)
        z = torch.flatten(z, 1)
        z = self.fc1(z)
        z = F.relu(z)
        y = self.fc2(z)
        return y


class cbrblock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(cbrblock, self).__init__()
        self.cbr = nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=3,
                stride=(1, 1),
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.cbr(x)
        return out


class conv_block(nn.Module):
    def __init__(self, input_channels, output_channels, scale_input):
        super(conv_block, self).__init__()
        self.scale_input = scale_input
        if self.scale_input:
            self.scale = nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=1,
                stride=(1, 1),
                padding="same",
            )
        self.layer1 = cbrblock(input_channels, output_channels)
        self.dropout = nn.Dropout(p=0.01)
        self.layer2 = cbrblock(output_channels, output_channels)

    def forward(self, x):
        residual = x
        out = self.layer1(x)
        out = self.dropout(out)
        out = self.layer2(out)
        if self.scale_input:
            residual = self.scale(residual)
        out = out + residual
        return out


class WideResNet(nn.Module):
    def __init__(self, num_classes):
        super(WideResNet, self).__init__()
        nChannels = [3, 16, 160, 320, 640]

        self.input_block = cbrblock(nChannels[0], nChannels[1])

        self.block1 = conv_block(nChannels[1], nChannels[2], True)
        self.block2 = conv_block(nChannels[2], nChannels[2], False)
        self.pool1 = nn.MaxPool2d(2)

        self.block3 = conv_block(nChannels[2], nChannels[3], True)
        self.block4 = conv_block(nChannels[3], nChannels[3], False)
        self.pool2 = nn.MaxPool2d(2)

        self.block5 = conv_block(nChannels[3], nChannels[4], True)
        self.block6 = conv_block(nChannels[4], nChannels[4], False)

        self.pool = nn.AvgPool2d(7)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(nChannels[4], num_classes)

    def forward(self, x):
        z = self.input_block(x)
        z = self.block1(z)
        z = self.block2(z)
        z = self.pool1(z)

        z = self.block3(z)
        z = self.block4(z)
        z = self.pool2(z)

        z = self.block5(z)
        z = self.block6(z)

        z = self.pool(z)
        z = self.flat(z)
        y = self.fc(z)
        return y
# --------------------------------------
# Training and Testing Loops
# --------------------------------------

def train(model, optimizer, train_loader, loss_fn, device, local_rank):
    total_labels = 0
    correct_labels = 0
    model.train()
    for images, labels in train_loader:
        labels = labels.to(device)
        images = images.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        predictions = torch.max(outputs, 1)[1]
        total_labels += len(labels)
        correct_labels += (predictions == labels).sum()

    # TODO Requirement R1.9: 
    # This currently only calculates accuracy for THIS specific GPU
    # You need to use dist.all_reduce() to average this across all GPUs
    correct_tensor = correct_labels.clone().detach().float()
    total_tensor = torch.tensor(total_labels, device=device, dtype=torch.float)

    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)

    t_accuracy = correct_tensor / total_tensor
    return t_accuracy


def test(model, test_loader, loss_fn, device, local_rank):
    total_labels = 0
    correct_labels = 0
    loss_total = 0
    model.eval()

    with torch.no_grad():
        for images, labels in test_loader:
            labels = labels.to(device)
            images = images.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            predictions = torch.max(outputs, 1)[1]
            total_labels += len(labels)
            correct_labels += (predictions == labels).sum()
            loss_total += loss

    # TODO Requirement R1.9: 
    # Use dist.all_reduce() with ReduceOp.AVG to average v_accuracy and v_loss across all ranks!
    v_accuracy = correct_labels / total_labels
    v_loss = loss_total / len(test_loader)

    dist.all_reduce(v_accuracy, op=dist.ReduceOp.AVG)
    dist.all_reduce(v_loss, op=dist.ReduceOp.AVG)

    return v_accuracy, v_loss

# --------------------------------------
# DDP Worker Process
# --------------------------------------

def main_worker(local_rank, args):
    # 1. Compute global rank and initialize process group (Requirement R1.2 & R1.3)
    global_rank = args.node_id * args.num_gpus + local_rank
    world_size = args.num_nodes * args.num_gpus

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=global_rank,
    )

    # 2. Pin process to specific GPU (Requirement R1.7)
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Fix random seed
    torch.manual_seed(123)

    # Transformations
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # TODO Requirement R1.4:
    # Only one process per node downloads the dataset first.
    # Then dist.barrier() makes the other processes wait until downloading is done.
    if local_rank == 0:
        torchvision.datasets.CIFAR10("./data", download=False, transform=transform_train)
        torchvision.datasets.CIFAR10(
            "./data", download=False, train=False, transform=transform_test
        )

    dist.barrier()

    train_set = torchvision.datasets.CIFAR10(
        "./data", download=False, transform=transform_train
    )
    test_set = torchvision.datasets.CIFAR10(
        "./data", download=False, train=False, transform=transform_test
    )

    # TODO Requirement R1.5:
    # Create DistributedSampler objects so each GPU process gets a different
    # shard of the training and validation datasets.
    train_sampler = DistributedSampler(
        train_set,
        num_replicas=world_size,
        rank=global_rank,
        shuffle=True,
    )

    test_sampler = DistributedSampler(
        test_set,
        num_replicas=world_size,
        rank=global_rank,
        shuffle=False,
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        drop_last=True,
        sampler=train_sampler,
        shuffle=False,
        num_workers=4,
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        drop_last=False,
        sampler=test_sampler,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Model Setup
    num_classes = 10
    if args.model == "SimpleConvNet":
        model = SimpleConvNet(num_classes).to(device)
    else:
        model = WideResNet(num_classes).to(device)

    # TODO Requirement R1.8:
    # Wrap the model with DistributedDataParallel so gradients are synchronized
    # during the backward pass.
    model = DDP(model, device_ids=[local_rank])

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if global_rank == 0:
        print("Arguments:", args, "\n")

    val_accuracy = []
    total_time = 0.0

    for epoch in range(args.epochs):
        # TODO Requirement R1.6:
        # This gives DistributedSampler a new shuffle seed each epoch.
        train_sampler.set_epoch(epoch)

        start_time = time.time()
        t_accuracy = train(model, optimizer, train_loader, loss_fn, device, local_rank)
        epoch_time = time.time() - start_time
        total_time += epoch_time

        # Calculate local throughput for this process.
        images_per_sec = len(train_loader) * args.batch_size / epoch_time
        images_per_sec_tensor = torch.tensor(
            images_per_sec, device=device, dtype=torch.float
        )

        # TODO Requirement R1.9:
        # Sum throughput across all GPUs and send the result to rank 0.
        dist.reduce(
            images_per_sec_tensor,
            dst=0,
            op=dist.ReduceOp.SUM,
        )

        v_accuracy, v_loss = test(model, test_loader, loss_fn, device, local_rank)
        val_accuracy.append(v_accuracy.item())

        # TODO Requirement R1.10:
        # Only rank 0 prints metrics so the log does not contain duplicate lines.
        if global_rank == 0:
            print(
                "Epoch = {:2d}: Epoch Time = {:5.3f}, Cumul. Time = {:5.3f}, Samples/Sec = {:5.3f}, Training Accuracy = {:5.3f}, Validation Loss = {:5.3f}, Validation Accuracy = {:5.3f}".format(
                    epoch + 1,
                    epoch_time,
                    total_time,
                    images_per_sec_tensor.item(),
                    t_accuracy.item(),
                    v_loss.item(),
                    val_accuracy[-1],
                )
            )

        if args.patience <= epoch and all(
            args.target_accuracy <= acc for acc in val_accuracy[-args.patience:]
        ):
            if global_rank == 0:
                print("Early stopping after epoch {}".format(epoch + 1))
            break

    # Cleanup DDP
    dist.destroy_process_group()

# --------------------------------------
# Main
# --------------------------------------

if __name__ == "__main__":
    args = parser.parse_args()
    
    # Requirement R1.11: Launch worker processes
    mp.spawn(
        main_worker,
        args=(args,),
        nprocs=args.num_gpus,
        join=True
    )
