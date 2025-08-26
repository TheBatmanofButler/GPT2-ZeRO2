import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax
import jax.numpy as jnp
import optax
import wandb

import gpt2zero2.core.model as model_lib
import gpt2zero2.core.config as config
import gpt2zero2.data.load as load
import gpt2zero2.data.save as save
import gpt2zero2.data.sharding as sharding

key = jax.random.PRNGKey(1)
key, model_key = jax.random.split(key)
params = model_lib.init(model_key)

train_dataloader, test_dataloader = load.build_dataloaders(config.gpt2_config)

mesh = jax.make_mesh(
    (config.gpt2_config.num_devices,),
    axis_names=("device",),
)
replicated_sharding = jax.NamedSharding(mesh, jax.sharding.PartitionSpec())
device_sharding = jax.NamedSharding(mesh, jax.sharding.PartitionSpec("device"))

optimizer = optax.adamw(learning_rate=config.gpt2_config.learning_rate)
optimizer_state = optimizer.init(params)
optimizer_state = sharding.shard_and_put_pytree(
    optimizer_state,
    device_sharding,
    replicated_sharding,
)


optimizer_in_specs = jax.tree_util.tree_map(lambda x: x.sharding.spec, optimizer_state)

params = jax.device_put(params, replicated_sharding)


def loss_fn(params, sample):
    logits = model_lib.forward(params, sample)

    predictions = logits[:-1]
    targets = sample[0][1:]
    attention_mask = sample[1][1:]

    losses = jnp.array(
        [
            -jax.nn.log_softmax(predictions[i], axis=-1)[targets[i]]
            for i in range(len(predictions))
        ]
    )

    masked_losses = losses * attention_mask
    total_loss = jnp.sum(masked_losses)
    total_tokens = jnp.sum(attention_mask)

    return jnp.where(total_tokens > 0, total_loss / total_tokens, 0.0)


def param_loss_fn(params, batch):
    losses = jax.vmap(lambda sample: loss_fn(params, sample))(batch)

    return jnp.mean(losses)


def compute_loss_and_grads(params, batch):
    loss, grads = jax.value_and_grad(param_loss_fn)(params, batch)

    loss = jax.lax.pmean(loss, axis_name="device")
    grads = jax.lax.pmean(grads, axis_name="device")

    device_grads = jax.lax.psum_scatter(grads, axis_name="device", tiled=True)
    device_grads = jax.tree_util.tree_map(lambda x: x / 4, device_grads)

    return loss, device_grads


def compute_updated_params(device_grads, params, device_optimizer_state):
    device_id = jax.lax.axis_index("device")

    params = sharding.split_pytree(params, config.gpt2_config.num_devices)
    params_fns = [lambda split=split: split for split in params]
    device_params = jax.lax.switch(device_id, params_fns)

    updates, optimizer_state = optimizer.update(
        device_grads, device_optimizer_state, device_params
    )
    updated_params = optax.apply_updates(device_params, updates)

    return updated_params, optimizer_state


@jax.jit
def update_step(params, optimizer_state, batch):
    loss, device_grads = jax.shard_map(
        compute_loss_and_grads,
        mesh=mesh,
        in_specs=(replicated_sharding.spec, device_sharding.spec),
        out_specs=(replicated_sharding.spec, device_sharding.spec),
    )(params, batch)

    optimizer_out_specs = jax.tree_util.tree_map(
        lambda x: replicated_sharding.spec if x.ndim == 0 else device_sharding.spec,
        optimizer_state,
    )

    device_params, optimizer_state = jax.shard_map(
        compute_updated_params,
        mesh=mesh,
        in_specs=(
            device_sharding.spec,
            replicated_sharding.spec,
            optimizer_in_specs,
        ),
        out_specs=(device_sharding.spec, optimizer_out_specs),
    )(device_grads, params, optimizer_state)

    params = jax.shard_map(
        lambda x: jax.lax.all_gather(
            x,
            axis_name="device",
            tiled=True,
        ),
        mesh=mesh,
        in_specs=device_sharding.spec,
        out_specs=replicated_sharding.spec,
        check_vma=False,
    )(device_params)

    return params, optimizer_state, loss


# wandb.init(
#     entity=config.gpt2_config.wandb_entity,
#     project=config.gpt2_config.wandb_project,
#     config=config.gpt2_config.to_dict(),
# )


def evaluate(params, sample):
    logits = model_lib.forward(params, sample)

    predictions = logits[:-1]
    targets = sample[0][1:]
    correct = jnp.argmax(predictions, axis=-1) == targets

    total_correct = jnp.sum(correct)
    total_tokens = correct.size

    return total_correct / total_tokens


for batch_idx, batch in enumerate(train_dataloader):
    batch = jax.device_put(batch, device_sharding)

    step = batch_idx + 1

    params, optimizer_state, loss = update_step(params, optimizer_state, batch)

    print(f"Batch {step}: Loss = {loss:.4f}")
    # wandb.log(
    #     data={"train_loss": loss},
    #     step=step,
    # )

    if step % 100 == 0:
        test_batch = next(iter(test_dataloader))
        test_batch = jax.device_put(test_batch, device_sharding)

        eval_fn = jax.shard_map(
            lambda params, batch: jax.vmap(lambda sample: evaluate(params, sample))(
                batch
            ),
            mesh=mesh,
            in_specs=(replicated_sharding.spec, device_sharding.spec),
            out_specs=device_sharding.spec,
        )

        batch_accuracies = eval_fn(params, test_batch)
        accuracy = jnp.mean(batch_accuracies)

        print(f"Accuracy = {accuracy}")
        # wandb.log(
        #     data={"accuracy": accuracy},
        #     step=step,
        # )

        print("Saving model")
        cpu_params = jax.device_get(params)
        save.save_model(cpu_params, config.gpt2_config.saved_model_name)

# wandb.finish()
