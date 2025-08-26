import equinox as eqx
import jax.numpy as jnp
import datasets
import torch
import os

import gpt2zero2.data.tokenizer as tokenizer
from gpt2zero2.core.config import GPT2Config


def build_dataloaders(config: GPT2Config):
    dataset_dict = datasets.load_dataset(
        config.dataset_path,
        config.dataset_name,
    )

    train_dataset = dataset_dict[datasets.Split.TRAIN]
    val_dataset = dataset_dict[datasets.Split.VALIDATION]

    #  Each sample will be padded or truncated to the size of the context window.
    def tokenize(sample: str):
        return tokenizer.tokenizer(
            sample["text"],
            max_length=config.context_window,
            padding="max_length",
            truncation=True,
        )

    train_dataset_path = os.path.join(os.getcwd(), "dataset_train")
    val_dataset_path = os.path.join(os.getcwd(), "dataset_val")
    if not os.path.exists(train_dataset_path):
        # Tokenize datasets.
        train_dataset = (
            train_dataset.select(range(config.num_train_samples))
            .filter(lambda x: len(x["text"]) > 2)
            .map(
                tokenize,
                batched=True,
            )
        )
        val_dataset = val_dataset.filter(lambda x: len(x["text"]) > 2).map(
            tokenize,
            batched=True,
        )
        train_dataset.save_to_disk(train_dataset_path)
        val_dataset.save_to_disk(val_dataset_path)
    else:
        train_dataset = datasets.load_from_disk(train_dataset_path)
        val_dataset = datasets.load_from_disk(val_dataset_path)

    print("train_dataset size", len(train_dataset))

    def collate(batch):
        return jnp.array(
            [jnp.array([sample[key] for key in sample]) for sample in batch]
        )

    # Create dataloader iterables for training and validation.
    train_dataset = train_dataset.select_columns(
        [
            "input_ids",
            "attention_mask",
        ]
    ).with_format("torch")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.num_sequences_per_batch,
        collate_fn=collate,
    )

    val_dataset = val_dataset.select_columns(
        [
            "input_ids",
            "attention_mask",
        ]
    ).with_format("torch")
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=4,
        collate_fn=collate,
    )

    return train_dataloader, val_dataloader


def load_model(model, config: GPT2Config):
    return eqx.tree_deserialise_leaves(config.saved_model_name, model)
