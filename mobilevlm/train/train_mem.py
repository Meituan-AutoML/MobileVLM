# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# Need to call this before importing transformers.
from mobilevlm.train.llama_flash_attn import replace_llama_attn_with_flash_attn

replace_llama_attn_with_flash_attn()

from mobilevlm.train.train import train

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

if __name__ == "__main__":
    train()
