import jax

import gpt2zero2.core.config as config
import gpt2zero2.core.embed as embed_lib
import gpt2zero2.core.block as block_lib


def init(key):
    key, embed_key = jax.random.split(key, 2)
    block_keys = jax.random.split(key, config.gpt2_config.num_blocks)

    return {
        "embed": embed_lib.embed_init(
            key=embed_key,
            d_vocab=config.gpt2_config.d_vocab,
            d_embedding=config.gpt2_config.d_embedding,
            initialized_std_range=config.gpt2_config.initialized_std_range,
        ),
        "pos_embed": embed_lib.pos_embed_init(
            config.gpt2_config.context_window,
            config.gpt2_config.d_embedding,
        ),
        "blocks": [
            block_lib.init(
                block_keys[i],
                config.gpt2_config.d_embedding,
                config.gpt2_config.d_mlp,
            )
            for i in range(config.gpt2_config.num_blocks)
        ],
        "unembed": embed_lib.unembed_init(
            key=embed_key,
            d_vocab=config.gpt2_config.d_vocab,
            d_embedding=config.gpt2_config.d_embedding,
            initialized_std_range=config.gpt2_config.initialized_std_range,
        ),
    }


def forward(params, sample):
    x = sample[0]

    x = jax.vmap(embed_lib.embed_forward, in_axes=(None, 0))(
        params["embed"], x
    ) + jax.vmap(embed_lib.pos_embed_forward, in_axes=(None, 0))(params["pos_embed"], x)

    for block in params["blocks"]:
        x = block_lib.forward(
            block,
            x,
            config.gpt2_config.layer_norm_epsilon,
            config.gpt2_config.d_head,
        )

    x = embed_lib.unembed_forward(params["unembed"], x)

    return x
