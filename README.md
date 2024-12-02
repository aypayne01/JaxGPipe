# JaxGPipe

JaxGPipe is a **GPipe** implementation in [JAX](https://github.com/google/jax), enabling pipeline parallelism for training large neural network models across multiple devices (GPUs or TPUs). This allows for efficient utilization of hardware resources and the training of models that exceed the memory capacity of a single device.

## Features

- **Model Partitioning**: Split models into pipeline stages assigned to different devices.
- **Micro-batching**: Process smaller micro-batches for efficient pipeline execution.
- **Optimizers**: Integrated optimization routines for parameter updates.
- **Scalability**: Easily scale across multiple GPUs or TPUs.

## Installation

### Prerequisites

- Python 3.7 or higher
- [JAX](https://github.com/google/jax) with appropriate device support (GPU/TPU)

### Using pip

```bash
pip install gpipe-jax
