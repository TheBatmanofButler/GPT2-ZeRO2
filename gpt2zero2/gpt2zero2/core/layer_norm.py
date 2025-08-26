import jax.numpy as jnp


def layer_norm(x, layer_norm_epsilon):
    mean = jnp.mean(x)
    std = jnp.std(x)
    x = (x - mean) / jnp.sqrt(std + layer_norm_epsilon)

    return x
