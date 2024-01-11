import os
import sys
import torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoTokenizer, AutoConfig, BitsAndBytesConfig
sys.path.append(str(Path(__file__).parent.parent.resolve()))
from mobilevlm.model.mobilellama import MobileLlamaForCausalLM
from mobilevlm.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def merge_lora(model_base, model_path, save_path):

    kwargs = {"device_map": "auto", "torch_dtype": torch.float16}

    lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
    # Loading weight from base model
    model = MobileLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
    print("🔜 Don't worry, we will load vision-tower weight soon later...")
    token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
    if model.lm_head.weight.shape[0] != token_num:
        model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
        model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
    # Loading additional non-lora weights
    non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
    non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
    if any(k.startswith('model.model.') for k in non_lora_trainables):
        non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
    model.load_state_dict(non_lora_trainables, strict=False)
    # Loading lora weights and merge
    model = PeftModel.from_pretrained(model, model_path)
    model = model.merge_and_unload()
    # Loading vision-tower weights
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=model.device, dtype=torch.float16)
    print("✅ The vision-tower is loaded successful!")
    # save
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

merge_lora(sys.argv[1], sys.argv[2], sys.argv[3])
