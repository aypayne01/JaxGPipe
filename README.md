# GPipe-JAX

GPipe-JAX is a pipeline parallelism framework implemented in JAX, enabling efficient training of large neural network models across multiple devices (GPUs or TPUs). Inspired by Google's GPipe, this project leverages JAX's parallelism primitives to facilitate model and data parallelism.

## Features

- **Model Partitioning**: Split models into sequential stages across devices.
- **Micro-batching**: Process smaller batches to maximize device utilization.
- **Optimized Pipeline Execution**: Efficient forward and backward passes.
- **Customizable Optimizers**: Integrate various optimization algorithms.

## Installation

### Prerequisites

- Python 3.8+
- JAX installed with support for your hardware (GPU/TPU).

### Steps

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/gpipe-jax.git
    cd gpipe-jax
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. (Optional) Install the package in editable mode:

    ```bash
    pip install -e .
    ```

## Usage

### Example: Training a Model

Refer to the [examples](./examples/example_training.py) directory for detailed examples.

```python
import jax.numpy as jnp
from gpipe_jax.pipeline import GPipe
from gpipe_jax.model import MyModel
from gpipe_jax.optimizer import SGDOptimizer

# Initialize model and optimizer
model = MyModel()
optimizer = SGDOptimizer(learning_rate=1e-3)

# Initialize GPipe with model stages
gpipe = GPipe(model, optimizer, devices=jax.devices())

# Training loop
for epoch in range(num_epochs):
    for batch_x, batch_y in data_loader:
        loss = gpipe.train_step(batch_x, batch_y)
        print(f"Epoch {epoch}, Loss: {loss}")
