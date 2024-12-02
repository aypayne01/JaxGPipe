import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap, lax
import numpy as np

def init_layer_params(rng, in_dim, out_dim):
    """Initialize parameters for a single layer."""
    w_key, b_key = random.split(rng)
    w = random.normal(w_key, (in_dim, out_dim)) / np.sqrt(in_dim)
    b = random.normal(b_key, (out_dim,))
    return w, b

def init_model_params(rng, layer_sizes):
    """Initialize model parameters."""
    params = []
    for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
        rng, layer_rng = random.split(rng)
        params.append(init_layer_params(layer_rng, in_dim, out_dim))
    return params

def layer_forward(params, x):
    """Forward pass for a single layer."""
    w, b = params
    return jnp.dot(x, w) + b

def model_forward(params, x):
    """Forward pass for the entire model."""
    activations = x
    for i, layer_params in enumerate(params[:-1]):
        activations = jax.nn.relu(layer_forward(layer_params, activations))
    # No activation on the last layer
    activations = layer_forward(params[-1], activations)
    return activations

def loss_fn(params, x, y):
    """Compute the mean squared error loss."""
    preds = model_forward(params, x)
    loss = jnp.mean((preds - y) ** 2)
    return loss

def update(params, x, y, lr):
    """Compute gradients and update parameters."""
    grads = grad(loss_fn)(params, x, y)
    new_params = [(w - lr * dw, b - lr * db) for (w, b), (dw, db) in zip(params, grads)]
    return new_params

@jit
def train_step(params, x, y, lr):
    """Perform a single training step."""
    grads = grad(loss_fn)(params, x, y)
    new_params = [(w - lr * dw, b - lr * db) for (w, b), (dw, db) in zip(params, grads)]
    return new_params

def main():
    rng = random.PRNGKey(0)
    num_devices = jax.device_count()
    print(f"Number of devices: {num_devices}")

    layer_sizes = [784, 512, 512, 10]  # Example layer sizes
    params = init_model_params(rng, layer_sizes)

    # Dummy data for demonstration
    x = random.normal(rng, (64, 784))  # Batch size of 64
    y = random.normal(rng, (64, 10))

    lr = 0.001
    microbatch_size = 16

    # Split data into microbatches
    num_microbatches = x.shape[0] // microbatch_size
    x_microbatches = jnp.split(x, num_microbatches)
    y_microbatches = jnp.split(y, num_microbatches)

    # Training loop
    for epoch in range(10):
        for x_mb, y_mb in zip(x_microbatches, y_microbatches):
            params = train_step(params, x_mb, y_mb, lr)
        epoch_loss = loss_fn(params, x, y)
        print(f"Epoch {epoch}, Loss: {epoch_loss}")

if __name__ == "__main__":
    main()
