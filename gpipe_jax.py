import jax
import jax.numpy as jnp
from jax import random, grad, lax
import numpy as np

def init_layer_params(rng, in_dim, out_dim):
    """Initialize parameters for a single layer."""
    w_key, b_key = random.split(rng)
    w = random.normal(w_key, (in_dim, out_dim)) / np.sqrt(in_dim)
    b = random.normal(b_key, (out_dim,))
    return w, b

def init_model_params(rng, layer_sizes, num_stages):
    """Initialize model parameters partitioned into stages."""
    layers_per_stage = (len(layer_sizes) - 1) // num_stages
    params = []
    start = 0
    for i in range(num_stages):
        if i == num_stages - 1:
            end = len(layer_sizes) - 1
        else:
            end = start + layers_per_stage
        stage_layer_sizes = layer_sizes[start:end+1]
        stage_params = []
        for j in range(len(stage_layer_sizes) - 1):
            rng, layer_rng = random.split(rng)
            stage_params.append(init_layer_params(
                layer_rng, stage_layer_sizes[j], stage_layer_sizes[j+1]))
        params.append(stage_params)
        start = end
    return params

def layer_forward(params, x):
    """Forward pass for a single layer."""
    w, b = params
    return jnp.dot(x, w) + b

def relu(x):
    """ReLU activation function."""
    return jnp.maximum(0, x)

def stage_forward(stage_params, x):
    """Forward pass for a stage."""
    activations = x
    for layer_params in stage_params:
        activations = relu(layer_forward(layer_params, activations))
    return activations

def model_forward(params, x):
    """Forward pass for the entire model."""
    activations = x
    for stage_params in params:
        activations = stage_forward(stage_params, activations)
    return activations

def loss_fn(params, x, y):
    """Compute the mean squared error loss."""
    preds = model_forward(params, x)
    loss = jnp.mean((preds - y) ** 2)
    return loss

def split_into_microbatches(x, y, microbatch_size):
    """Split the batch into microbatches."""
    num_microbatches = x.shape[0] // microbatch_size
    x_microbatches = jnp.split(x, num_microbatches)
    y_microbatches = jnp.split(y, num_microbatches)
    return x_microbatches, y_microbatches

def gpipe_forward_backward(params, x_microbatches, y_microbatches):
    """Perform forward and backward passes using pipeline parallelism."""
    num_microbatches = len(x_microbatches)
    num_stages = len(params)
    activations = [[None for _ in range(num_microbatches)] for _ in range(num_stages)]
    loss = 0.0

    # Forward pass
    for m in range(num_microbatches):
        activ = x_microbatches[m]
        for s in range(num_stages):
            activ = stage_forward(params[s], activ)
            activations[s][m] = activ
        loss += jnp.mean((activ - y_microbatches[m]) ** 2)
    loss /= num_microbatches

    # Backward pass
    grads = [None for _ in range(num_stages)]
    delta = None
    for s in reversed(range(num_stages)):
        grad_stage_params = []
        for m in reversed(range(num_microbatches)):
            if s == num_stages - 1:
                # Output layer
                delta = (activations[s][m] - y_microbatches[m]) / num_microbatches
            else:
                # Propagate delta to previous stage
                w_next = params[s + 1][0][0]
                delta = jnp.dot(delta, w_next.T)
            # Compute gradients for the stage
            activ_input = activations[s - 1][m] if s > 0 else x_microbatches[m]
            dw = jnp.dot(activ_input.T, delta)
            db = jnp.sum(delta, axis=0)
            grad_stage_params.append((dw, db))
        # Average gradients over microbatches
        avg_dw = sum([g[0] for g in grad_stage_params]) / num_microbatches
        avg_db = sum([g[1] for g in grad_stage_params]) / num_microbatches
        grads[s] = [(avg_dw, avg_db)]
    return loss, grads

def update_params(params, grads, lr):
    """Update model parameters."""
    new_params = []
    for stage_params, stage_grads in zip(params, grads):
        new_stage_params = []
        for (w, b), (dw, db) in zip(stage_params, stage_grads):
            new_w = w - lr * dw
            new_b = b - lr * db
            new_stage_params.append((new_w, new_b))
        new_params.append(new_stage_params)
    return new_params

def main():
    rng = random.PRNGKey(0)
    num_devices = jax.device_count()
    print(f"Number of devices: {num_devices}")

    num_stages = num_devices if num_devices > 0 else 1
    layer_sizes = [784, 512, 512, 10]  # Example layer sizes
    params = init_model_params(rng, layer_sizes, num_stages)

    # Dummy data for demonstration
    x = random.normal(rng, (64, 784))  # Batch size of 64
    y = random.normal(rng, (64, 10))

    lr = 0.001
    microbatch_size = 16
    x_microbatches, y_microbatches = split_into_microbatches(x, y, microbatch_size)

    # Training loop
    for epoch in range(10):
        loss, grads = gpipe_forward_backward(params, x_microbatches, y_microbatches)
        params = update_params(params, grads, lr)
        print(f"Epoch {epoch}, Loss: {loss}")

if __name__ == "__main__":
    main()
