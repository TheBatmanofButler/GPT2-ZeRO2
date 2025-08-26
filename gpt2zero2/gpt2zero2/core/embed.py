import jax
import jax.numpy as jnp


def embed_init(key, d_vocab, d_embedding, initialized_std_range):
    key, W_key = jax.random.split(key, 2)

    return (
        jax.random.normal(
            key=W_key,
            shape=(d_vocab, d_embedding),
        )
        * initialized_std_range
    )


def embed_forward(params, x):
    return params[x]


def pos_embed_init(context_window, d_embedding):
    positions = jnp.arange(context_window)[:, None]
    dims = jnp.arange(d_embedding)[None, :]
    angles = positions / 10000 ** (2 * (dims // 2) / d_embedding)

    return jnp.where(
        dims % 2 == 0,
        jnp.sin(angles),
        jnp.cos(angles),
    )


def pos_embed_forward(params, x):
    return params[x]


def unembed_init(key, d_vocab, d_embedding, initialized_std_range):
    key, W_key = jax.random.split(key, 2)

    return (
        jax.random.normal(
            key=W_key,
            shape=(d_embedding, d_vocab),
        )
        * initialized_std_range
    )


def unembed_forward(params, x):
    return jnp.matmul(x, params)
