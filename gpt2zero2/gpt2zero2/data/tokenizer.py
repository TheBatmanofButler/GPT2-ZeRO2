import transformers
import gpt2zero2.core.config as config


class Tokenizer:
    def __init__(self):
        self.tokenizer = transformers.GPT2Tokenizer.from_pretrained(
            config.gpt2_config.tokenizer_path,
        )
        self.tokenizer.pad_token = self.tokenizer.unk_token
        if len(self.tokenizer.get_vocab()) % config.gpt2_config.num_devices != 0:
            self.tokenizer.add_special_tokens(
                {
                    "additional_special_tokens": [
                        f"<UNUSED_{i}>"
                        for i in range(
                            config.gpt2_config.num_devices
                            - len(self.tokenizer.get_vocab())
                            % config.gpt2_config.num_devices
                        )
                    ]
                }
            )


tokenizer = Tokenizer().tokenizer
