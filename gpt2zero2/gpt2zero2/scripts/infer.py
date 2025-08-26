import argparse
import jax
import jax.numpy as jnp

import gpt2zero2.data.load as load
import gpt2zero2.core_eqx.model as model_lib
import gpt2zero2.core_eqx.config as config
import gpt2zero2.data.tokenizer as tokenizer


key = jax.random.PRNGKey(1)
key, model_key = jax.random.split(key, 2)
gpt2 = model_lib.GPT2(model_key)

loaded_gpt2 = load.load_model(gpt2, config.gpt2_config)


def sample(key: jax.random.PRNGKey, prompt: str, max_tokens_generated=100):
    tokens = tokenizer.tokenizer(prompt)
    input_ids = jnp.array([tokens["input_ids"]])
    key, sampling_key = jax.random.split(key, 2)

    for _ in range(max_tokens_generated):
        logits = loaded_gpt2(input_ids)
        logits = logits * config.gpt2_config.temperature

        sampling_key, subkey = jax.random.split(sampling_key, 2)
        next_token = jax.random.categorical(subkey, logits[-1], shape=(1,)).reshape(
            1, 1
        )
        input_ids = jnp.concat([input_ids, next_token], axis=-1)

        print(tokenizer.tokenizer.decode(next_token[0, 0]))

    return logits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prompt", help="Prompt for GPT-2")
    args = parser.parse_args()

    key, sampling_key = jax.random.split(key, 2)
    sample(sampling_key, args.prompt)
