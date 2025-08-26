import jax
import jax.numpy as jnp


def shard_and_put_pytree(pytree, device_sharding, replicated_sharding):
    return jax.tree_util.tree_map(
        lambda x: jax.device_put(x, device_sharding)
        if len(x.shape) > 0
        else jax.device_put(x, replicated_sharding),
        pytree,
    )


def split_pytree2(pytree, sharding):
    return jax.tree_util.tree_map(
        lambda x: jax.device_put(x, sharding) if len(x.shape) > 0 else x, pytree
    )


def split_pytree(pytree, num_devices):
    flat_split_pytrees = [[] for _ in range(num_devices)]

    flat_pytree, treedef = jax.tree_util.tree_flatten(pytree)

    for leaf in flat_pytree:
        for i, split in enumerate(jnp.array_split(leaf, num_devices)):
            flat_split_pytrees[i].append(split)

    split_pytrees = [
        jax.tree_util.tree_unflatten(treedef, leaves) for leaves in flat_split_pytrees
    ]

    return split_pytrees


# test cases

# x = {
#     "a": jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
#     "b": {
#         "c": jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
#         "d": jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
#     },
# }

# print(split_pytree(split_pytree, 5))
