import jax
import jax.numpy as jnp


def init(key, d_embedding):
    key, q_key, k_key, v_key, o_key = jax.random.split(key, 5)
    scale = 0.02

    params = {
        "W_Q": jax.random.normal(
            key=q_key,
            shape=(d_embedding, d_embedding),
        )
        * scale,
        "W_K": jax.random.normal(
            key=k_key,
            shape=(d_embedding, d_embedding),
        )
        * scale,
        "W_V": jax.random.normal(
            key=v_key,
            shape=(d_embedding, d_embedding),
        )
        * scale,
        "W_O": jax.random.normal(
            key=o_key,
            shape=(d_embedding, d_embedding),
        )
        * scale,
        "b_Q": jnp.zeros((d_embedding,)),
        "b_K": jnp.zeros((d_embedding,)),
        "b_V": jnp.zeros((d_embedding,)),
        "b_O": jnp.zeros((d_embedding,)),
    }

    return params


def forward(params, x, d_head):
    q = jnp.matmul(x, params["W_Q"]) + params["b_Q"]
    k = jnp.matmul(x, params["W_K"]) + params["b_K"]
    v = jnp.matmul(x, params["W_V"]) + params["b_V"]

    k_t = k.transpose()

    attention_scores = jnp.matmul(q, k_t)
    attention_scores = attention_scores / jnp.sqrt(d_head)

    mask = jnp.tril(jnp.ones_like(attention_scores))
    attention_scores = jnp.where(mask, attention_scores, -jnp.inf)

    attention_weights = jax.nn.softmax(attention_scores, axis=-1)
    attention_output = jnp.matmul(attention_weights, v)

    output = jnp.matmul(attention_output, params["W_O"]) + params["b_O"]

    return output
