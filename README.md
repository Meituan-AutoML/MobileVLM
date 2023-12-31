#  MobileVLM: A Fast, Strong and Open Vision Language Assistant for Mobile Devices

<a href='https://github.com/Meituan-AutoML/MobileVLM'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='https://arxiv.org/abs/2312.16886'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)

We present MobileVLM, a competent multimodal vision language model (MMVLM) targeted to run on mobile devices. It is an amalgamation of a myriad of architectural designs and techniques that are mobile-oriented, which comprises a set of language models at the scale of 1.4B and 2.7B parameters, trained from scratch, a multimodal vision model that is pre-trained in the CLIP fashion, cross-modality interaction via an efficient projector. We evaluate MobileVLM on several typical VLM benchmarks. Our models demonstrate on par performance compared with a few much larger models. More importantly, we measure the inference speed on both a Qualcomm Snapdragon 888 CPU and an NVIDIA Jeston Orin GPU, and we obtain state-of-the-art performance of 21.5 tokens and 65.3 tokens per second, respectively.

<p align="center"><img src="assets/mobilevlm_arch.png"></p>

The MobileVLM architecture (right) utilizes MobileLLaMA as its language model, intakes $\mathbf{X}_v$ and $\mathbf{X}_q$ which are image and language instructions as respective inputs and gives $\mathbf{Y}_a$ as the output language response. LDP refers to a lightweight downsample projector (left).


## 📸 Release

* ⏳ MobileLLaMA Pre-training code.
* ⏳ MobileLLaMA SFT training code.
* ⏳ MobileVLM training code.
* **`Dec. 31st, 2023`**: Our MobileVLM weights are uploaded on the HuggingFace website. We also provide inference examples for the MobileLLaMA/MobileVLM model so that anyone can enjoy [them](https://huggingface.co/mtgv/) early.
* **`Dec. 29th, 2023`**: Our MobileLLaMA weights are uploaded on the HuggingFace website. Enjoy [them](https://huggingface.co/mtgv/) !
* **`Dec. 28th, 2023`:** 🔥🔥🔥 We release **MobileVLM: A Fast, Strong and Open Vision Language Assistant for Mobile Devices** on arxiv. Refer to **[our paper](https://arxiv.org/abs/2312.16886)** for more details !

## 🦙 Model Zoo

- [MobileLLaMA-1.4B-Base](https://huggingface.co/mtgv/MobileLLaMA-1.4B-Base)
- [MobileLLaMA-1.4B-Chat](https://huggingface.co/mtgv/MobileLLaMA-1.4B-Chat)
- [MobileLLaMA-2.7B-Base](https://huggingface.co/mtgv/MobileLLaMA-2.7B-Base)
- [MobileLLaMA-2.7B-Chat](https://huggingface.co/mtgv/MobileLLaMA-2.7B-Chat)
- [MobileVLM-1.7B](https://huggingface.co/mtgv/MobileVLM-1.7B)
- [MobileVLM-3B](https://huggingface.co/mtgv/MobileVLM-3B)

🔔 **Usage and License Notices**: This project utilizes certain datasets and checkpoints that are subject to their respective original licenses. Users must comply with all terms and conditions of these original licenses. This project is licensed permissively under the Apache 2.0 license and does not impose any additional constraints. <sup>[LLaVA](https://github.com/haotian-liu/LLaVA/tree/main?tab=readme-ov-file#release)</sup>

<!--
🧭 Please check out our [ModelZoo](https://github.com/Meituan-AutoML/MobileVLM/blob/main/MODELZOO.md) for detailed information and instructions of these public weights. \[TBA\]
-->

## 🛠️ Install

1. Clone this repository and navigate to MobileVLM folder
   ```bash
   git clone https://github.com/Meituan-AutoML/MobileVLM.git
   cd MobileVLM
   ```

2. Install Package
    ```Shell
    conda create -n mobilevlm python=3.10 -y
    conda activate mobilevlm
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

## 🗝️ Quick Start

#### Example for MobileLLaMA model inference

```python
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

model_path = 'mtgv/MobileLLaMA-1.4B-Base'

tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map='auto',
)

prompt = 'Q: What is the largest animal?\nA:'
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

generation_output = model.generate(
    input_ids=input_ids, max_new_tokens=32
)
print(tokenizer.decode(generation_output[0]))
```
* For more advanced usage, please follow the [transformers LLaMA documentation](https://huggingface.co/docs/transformers/main/model_doc/llama).

#### Example for MobileVLM model inference

```python
from scripts.inference import inference_once

model_path = "mtgv/MobileVLM-1.7B"
image_file = "assets/samples/demo.jpg"
prompt_str = "Who is the author of this book?\nAnswer the question using a single word or phrase."
# (or) What is the title of this book?
# (or) Is this book related to Education & Teaching?

args = type('Args', (), {
    "model_path": model_path,
    "image_file": image_file,
    "prompt": prompt_str,
    "conv_mode": "v1",
    "temperature": 0, 
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512,
})()

inference_once(args)
```

<!--

## Evaluation

### Data
### Evaluation on popular benchmarks
#### Evaluating MobileLLaMA with LM-Eval-Harness
The model can be evaluated with [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness). 

## Training
### MobileVLM
### MobileLLaMA SFT
### MobileLLaMA Pre-training
-->

## 🤝 Acknowledgments

- [LLaVA](https://github.com/haotian-liu/LLaVA): the codebase we built upon. Thanks for their wonderful work! 👏
- [Vicuna](https://github.com/lm-sys/FastChat): the amazing open-sourced large language model!


## ✏️ Reference

If you find MobileVLM or MobileLLaMA useful in your research or applications, please consider giving a star ⭐ and citing using the following BibTeX:
```
@misc{chu2023mobilevlm,
      title={MobileVLM: A Fast, Strong and Open Vision Language Assistant for Mobile Devices}, 
      author={Xiangxiang Chu and Limeng Qiao and Xinyang Lin and Shuang Xu and Yang Yang and Yiming Hu and Fei Wei and Xinyu Zhang and Bo Zhang and Xiaolin Wei and Chunhua Shen},
      year={2023},
      eprint={2312.16886},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
