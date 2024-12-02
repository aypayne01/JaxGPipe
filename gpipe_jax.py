import jax
import jax.numpy as jnp
from jax import random, grad, value_and_grad, jit, pmap
from functools import partial
import numpy as np
import optax

def init_layer_params(rng, in_dim, out_dim):
    """Initialize parameters for a single layer."""
    w_key, b_key = random.split(rng)
    w = random.normal(w_key, (in_dim, out_dim)) * np.sqrt(2.0 / in_dim)
    b = jnp.zeros((out_dim,))
    return {'w': w, 'b': b}

def init_model_params(rng, layer_sizes):
    """Initialize model parameters."""
    params = []
    for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
        rng, layer_rng = random.split(rng)
        params.append(init_layer_params(layer_rng, in_dim, out_dim))
    return params

def layer_forward(params, x):
    """Forward pass for a single layer."""
    return jnp.dot(x, params['w']) + params['b']

def forward_pass(params, x):
    """Forward pass through a partition of the model."""
    activations = x
    for layer_params in params:
        activations = layer_forward(layer_params, activations)
        activations = jax.nn.relu(activations)
    return activations

def model_forward(params, x):
    """Forward pass through the entire model."""
    activations = x
    for layer_params in params[:-1]:
        activations = layer_forward(layer_params, activations)
        activations = jax.nn.relu(activations)
    # Last layer without activation
    activations = layer_forward(params[-1], activations)
    return activations

def loss_fn(params, x, y):
    """Compute the mean squared error loss."""
    preds = model_forward(params, x)
    loss = jnp.mean((preds - y) ** 2)
    return loss

def split_params(params, num_partitions):
    """Split model parameters into partitions for pipeline parallelism."""
    param_partitions = []
    layers_per_partition = len(params) // num_partitions
    remainder = len(params) % num_partitions
    start = 0
    for i in range(num_partitions):
        extra = 1 if i < remainder else 0
        end = start + layers_per_partition + extra
        param_partitions.append(params[start:end])
        start = end
    return param_partitions

def split_batch(x, y, microbatch_size):
    """Split batch into microbatches."""
    num_microbatches = x.shape[0] // microbatch_size
    x_microbatches = jnp.split(x, num_microbatches)
    y_microbatches = jnp.split(y, num_microbatches)
    return x_microbatches, y_microbatches

def create_initial_states(params_partition, opt_state_partition):
    """Create initial states for each partition."""
    return {'params': params_partition, 'opt_state': opt_state_partition}

@partial(jax.pmap, axis_name='devices')
def train_step(state, x, y):
    """Perform a training step using pipeline parallelism."""
    params = state['params']
    opt_state = state['opt_state']

    def loss_fn_partition(params, x, y):
        preds = forward_pass(params, x)
        return jnp.mean((preds - y) ** 2)

    loss, grads = value_and_grad(loss_fn_partition)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    state = {'params': params, 'opt_state': opt_state}
    return state, loss

def main():
    rng = random.PRNGKey(0)
    num_devices = jax.device_count()
    print(f"Number of devices: {num_devices}")

    # Model configuration
    layer_sizes = [784, 1024, 1024, 1024, 10]  # Example layer sizes
    num_partitions = num_devices  # One partition per device

    # Initialize model parameters
    params = init_model_params(rng, layer_sizes)

    # Partition the model parameters
    param_partitions = split_params(params, num_partitions)

    # Replicate parameters and optimizer states across devices
    replicated_params = jax.device_put_replicated(param_partitions, jax.devices())
    optimizer = optax.sgd(learning_rate=0.001)
    opt_state = optimizer.init(replicated_params)
    replicated_opt_state = jax.device_put_replicated(opt_state, jax.devices())

    # Create initial state
    state = create_initial_states(replicated_params, replicated_opt_state)

    # Dummy data for demonstration
    batch_size = 64
    x = random.normal(rng, (batch_size, 784))
    y = random.normal(rng, (batch_size, 10))

    # Split data into microbatches
    microbatch_size = 16
    x_microbatches, y_microbatches = split_batch(x, y, microbatch_size)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        total_loss = 0.0
        for x_mb, y_mb in zip(x_microbatches, y_microbatches):
            # Replicate data across devices
            x_mb_rep = jax.device_put_replicated(x_mb, jax.devices())
            y_mb_rep = jax.device_put_replicated(y_mb, jax.devices())

            state, loss = train_step(state, x_mb_rep, y_mb_rep)
            total_loss += jax.tree_leaves(loss)[0]  # Get scalar loss from devices

        avg_loss = total_loss / len(x_microbatches)
        print(f"Epoch {epoch+1}, Loss: {avg_loss}")

if __name__ == "__main__":
    main()
