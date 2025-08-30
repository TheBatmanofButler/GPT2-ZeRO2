# GPT2-ZeRO2

This is a JAX implementation of GPT-2 with ZeRO-2 (see [the paper](https://arxiv.org/pdf/1910.02054) for more details). In ZeRO-2, we shard the optimizer state and gradients, while leaving parameters and activations unsharded.

To run the training script:
```
cd GPT2-DDP/gpt2ddp/gpt2ddp
uv run scripts/train.py
```

To modify the model/training configuration, see [`gpt2ddp/core/config.py`](gpt2ddp/core/config.py).

Here's a memory profile of 16 training steps. Compared to my [experiments with GPT2-DDP](https://github.com/TheBatmanofButler/gpt2-ddp), the max memory usage is about the same, but we see an ~400 MB reduction in average memory use:
<img width="1143" height="750" alt="image" src="https://github.com/user-attachments/assets/a695d072-3562-4b98-9c3d-083aee693d3b" />
