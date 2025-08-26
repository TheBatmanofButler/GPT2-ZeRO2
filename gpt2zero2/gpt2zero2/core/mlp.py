import jax
import jax.numpy as jnp


def init(key, d_embedding, d_mlp):
    key, l0_W_key, l0_b_key, l1_W_key, l1_b_key = jax.random.split(key, 5)
    scale = 0.02

    params = {
        "l0_W": jax.random.normal(
            key=l0_W_key,
            shape=(d_embedding, d_mlp),
        )
        * scale,
        "l0_b": jax.random.normal(
            key=l0_b_key,
            shape=(d_mlp,),
        )
        * scale,
        "l1_W": jax.random.normal(
            key=l1_W_key,
            shape=(d_mlp, d_embedding),
        )
        * scale,
        "l1_b": jax.random.normal(
            key=l1_b_key,
            shape=(d_embedding,),
        )
        * scale,
    }

    return params


def forward(params, x):
    x = jnp.matmul(x, params["l0_W"]) + params["l0_b"]
    x = jax.nn.gelu(x)
    x = jnp.matmul(x, params["l1_W"]) + params["l1_b"]

    return x
