from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class GPT2Config:
    # Core dimensions
    d_embedding: int = 384
    d_vocab: int = 50260

    # Sequence length
    context_window: int = 256

    # Transformer blocks
    num_blocks: int = 12
    num_heads: int = 12
    d_head: int = d_embedding // num_heads

    # Feed-forward network
    d_mlp: int = d_embedding * 4

    # Optimization
    learning_rate: float = 3e-4

    # Training
    num_devices = 4
    num_sequences_per_batch_per_device: int = 16
    num_sequences_per_batch: int = num_devices * num_sequences_per_batch_per_device

    num_train_samples: int = 2**10  # 2119489

    # Weight initialization
    initialized_std_range: float = 0.02

    # Normalization
    layer_norm_epsilon: float = 1e-5

    # Inference
    temperature: float = 2.0

    # Dataset
    dataset_path: str = "roneneldan/TinyStories"
    dataset_name: str | None = None

    # Tokenizer
    tokenizer_path: str = "openai-community/gpt2"

    # Instrumentation
    wandb_entity: str = "the-ganesh-ravichandran-none"
    wandb_project: str = "gpt2-jax"

    saved_model_name: str = (
        f"gpt2-multi-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.eqx"
    )

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return asdict(self)


gpt2_config = GPT2Config()
