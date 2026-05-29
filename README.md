# Distributed CIFAR-10 Training System

## Overview

This project implements distributed deep learning training for image classification on the CIFAR-10 dataset using PyTorch Distributed Data Parallel (DDP).

The goal of the project was to investigate the performance benefits of multi-GPU training compared to traditional single-GPU training while maintaining model accuracy. Training was performed on a high-performance computing (HPC) cluster using NVIDIA GPUs and Slurm job scheduling.

---

## Objectives

* Implement distributed training using PyTorch DDP
* Train image classification models on the CIFAR-10 dataset
* Compare single-GPU and multi-GPU performance
* Analyze training throughput and scalability
* Evaluate the impact of distributed computing on model training time

---

## Technologies Used

* Python
* PyTorch
* Distributed Data Parallel (DDP)
* CUDA
* Slurm
* Linux
* HPC Computing

---

## Dataset

The project uses the CIFAR-10 dataset, which contains 60,000 color images across 10 object categories:

* Airplanes
* Automobiles
* Birds
* Cats
* Deer
* Dogs
* Frogs
* Horses
* Ships
* Trucks

The dataset consists of:

* 50,000 training images
* 10,000 testing images

---

## Implementation

The training system includes:

### Distributed Training

* PyTorch Distributed Data Parallel (DDP)
* Multi-process GPU training
* Distributed data loading
* Gradient synchronization across GPUs

### Training Pipeline

* CIFAR-10 data preprocessing
* Distributed samplers
* Model training and validation
* Accuracy tracking
* Early stopping support

### HPC Deployment

* Slurm job scheduling
* Multi-GPU resource allocation
* Automated training execution
* Performance monitoring

---

## Performance Analysis

Experiments were conducted using both single-GPU and multi-GPU configurations.

Metrics evaluated include:

* Training throughput
* Images processed per second
* Time-to-target accuracy
* Final model accuracy
* Scalability across GPUs

The results demonstrated significant improvements in training throughput when utilizing multiple GPUs while maintaining comparable classification accuracy.

---

## Key Concepts Demonstrated

* Distributed Computing
* Parallel Processing
* Deep Learning
* GPU Programming
* High Performance Computing (HPC)
* Model Training Optimization
* Performance Benchmarking

---

## Future Improvements

Potential future enhancements include:

* Multi-node distributed training
* Mixed precision training
* Additional neural network architectures
* Hyperparameter optimization
* Larger image datasets
* Automated experiment tracking

---

## Author

Vignesh Thallam
Computational Modeling & Data Analytics
Virginia Tech
