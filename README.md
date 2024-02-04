# <center> MobileVLM: A Fast, Strong and Open <br> Vision Language Assistant for Mobile Devices

<a href='https://github.com/Meituan-AutoML/MobileVLM'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='https://arxiv.org/abs/2312.16886'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)

We present MobileVLM, a competent multimodal vision language model (MMVLM) targeted to run on mobile devices. It is an amalgamation of a myriad of architectural designs and techniques that are mobile-oriented, which comprises a set of language models at the scale of 1.4B and 2.7B parameters, trained from scratch, a multimodal vision model that is pre-trained in the CLIP fashion, cross-modality interaction via an efficient projector. We evaluate MobileVLM on several typical VLM benchmarks. Our models demonstrate on par performance compared with a few much larger models. More importantly, we measure the inference speed on both a Qualcomm Snapdragon 888 CPU and an NVIDIA Jeston Orin GPU, and we obtain state-of-the-art performance of 21.5 tokens and 65.3 tokens per second, respectively.

<p align="center"><img src="assets/mobilevlm_arch.png"></p>

The MobileVLM architecture (right) utilizes MobileLLaMA as its language model, intakes $\mathbf{X}_v$ and $\mathbf{X}_q$ which are image and language instructions as respective inputs and gives $\mathbf{Y}_a$ as the output language response. LDP refers to a lightweight downsample projector (left).

## 📸 Release

* ⏳ MobileLLaMA Pre-training code.
* ⏳ MobileLLaMA SFT training code.
* **`Jan. 23th, 2024`**: 🚀🚀🚀 **MobileVLM** is officially supported by [`llama.cpp`](https://github.com/ggerganov/llama.cpp/blob/master/examples/llava/MobileVLM-README.md) now ! Have a try !
* **`Jan. 15th, 2024`**: Customized `llama.cpp` for **MobileVLM** and its [deployment instruction](#deployment-on-mobile-devices) on mobile devices.
* **`Jan. 11st, 2024`**: The training and evaluation codes of MobileVLM are available now! Follow these  step-by-step instructions below to easily train your own mobileVLM in **5 hours** ⚡️ !
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

model_path = 'mtgv/MobileLLaMA-1.4B-Chat'

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
    "load_8bit": False,
    "load_4bit": False,
})()

inference_once(args)
```

## 🪜 Step-by-step Tutorial

### MobileVLM

The training process of MobileVLM is divided into two stages:

- stage I: feature alignment pretrain
  - ❄️ frozen vision encoder + 🔥 **learnable** LDP projector + ❄️ frozen LLM
  - this training process takes around **1~1.5 hours** for MobileVLM-1.7B/3B on 8x A100 (80G) with a batch size of 256 and an average of approximately 15G/19G of GPU memory required.
- stage II: visual instruction tuning
  - ❄️ frozen vision encoder + 🔥 **learnable** LDP projector + 🔥 **learnable** LLM
  - this training process takes around **2~3.5 hours** for MobileVLM-1.7B/3B on 8x A100 (80G) with a batch size of 128 and an average of approximately 46G/52G of GPU memory required.

Note: To train on fewer GPU memory or cards, you can reduce the `per_device_train_batch_size` and increase the `gradient_accumulation_steps` accordingly. Always keep the `global batch size` the same: `per_device_train_batch_size` x `gradient_accumulation_steps` x `num_gpus`.

#### 1️⃣ Prepare MobileLLaMA checkpoints

Download MobileLLaMA chatbot checkpoints from huggingface website (🤗 [1.7B](https://huggingface.co/mtgv/MobileLLaMA-1.4B-Chat), [2.7B](https://huggingface.co/mtgv/MobileLLaMA-3B-Chat)). Please note that this is **optional** (it depends on your working environment), run the training script we provide below and the model will be automatically downloaded by the `transformers` library.

#### 2️⃣ Prepare data
- For convenience, assume your working directory `/path/to/project/mobilevlm` as `work_dir`: 
  - `cd ${work_dir} && mkdir -p data/pretrain_data data/finetune_data data/benchmark_data`
- prepare alignment pre-training data
  - `cd ${work_dir}/data/pretrain_data`
  - download the LLaVA-558K from [here](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain), which is provided by LLaVA team.
- prepare  instruction tuning data
  - `cd ${work_dir}/data/finetune_data`
  - download the annotation of the LLaVA mixture instruction tuning data [here](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json), and download the images from constituting datasets: [COCO](http://images.cocodataset.org/zips/train2017.zip), [GQA](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip), [OCR-VQA](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing), [TextVQA](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip), [VisualGnome](https://cs.stanford.edu/people/rak248/VG_100K_2) ([Part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [Part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip))
- prepare benchmark data
  - We evaluate models on a diverse set of 6 benchmarks, *i.e.* GQA, MMBench, MME, POPE, SQA, TextVQA. We do not evaluate using beam search to make the inference process consistent with the chat demo of real-time outputs. You should follow these instructions to manage the datasets.
  - <details>
    <summary> Data Download Instructions </summary>

    - download some useful [data/scripts](https://github.com/Meituan-AutoML/MobileVLM/releases/download/v0.1/benchmark_data.zip) pre-collected by us.
      - `unzip benchmark_data.zip && cd benchmark_data`
      - `bmk_dir=${work_dir}/data/benchmark_data`
    - gqa
      - download its image data following the official instructions [here](https://cs.stanford.edu/people/dorarad/gqa/download.html)
      - `cd ${bmk_dir}/gqa && ln -s /path/to/gqa/images images`
    - mme
      - download the data following the official instructions [here](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation).
      - `cd ${bmk_dir}/mme && ln -s /path/to/MME/MME_Benchmark_release_version images`
    - pope
      - download coco from POPE following the official instructions [here](https://github.com/AoiDragon/POPE/tree/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco).
      - `cd ${bmk_dir}/pope && ln -s /path/to/pope/coco coco && ln -s /path/to/coco/val2014 val2014`
    - sqa
      - download images from the `data/scienceqa` folder of the ScienceQA [repo](https://github.com/lupantech/ScienceQA).
      - `cd ${bmk_dir}/sqa && ln -s /path/to/sqa/images images`
    - textvqa
      - download images following the instructions [here](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip).
      - `cd ${bmk_dir}/textvqa && ln -s /path/to/textvqa/train_images train_images`
    - mmbench
      - no action is needed.

    </details>


- organize the `data` directory as follows after downloading all of them: 
  - <details>
    <summary> Data Structure Tree </summary>

    ```
    .
    ├── benchmark_data
    │   ├── gqa
    │   │   ├── convert_gqa_for_eval.py
    │   │   ├── eval.py
    │   │   ├── images -> /path/to/your/gqa/images
    │   │   ├── llava_gqa_testdev_balanced.jsonl
    │   │   └── testdev_balanced_questions.json
    │   ├── mmbench
    │   │   ├── convert_mmbench_for_submission.py
    │   │   ├── eval.py
    │   │   └── mmbench_dev_en_20231003.tsv
    │   ├── mme
    │   │   ├── calculation.py
    │   │   ├── convert_answer_to_mme.py
    │   │   ├── images -> /path/to/your/MME/MME_Benchmark_release_version
    │   │   └── llava_mme.jsonl
    │   ├── pope
    │   │   ├── coco -> /path/to/your/pope/coco
    │   │   ├── eval.py
    │   │   ├── llava_pope_test.jsonl
    │   │   └── val2014 -> /path/to/your/coco/val2014
    │   ├── sqa
    │   │   ├── eval.py
    │   │   ├── images -> /path/to/your/scienceqa/images
    │   │   ├── llava_test_CQM-A.json
    │   │   ├── pid_splits.json
    │   │   └── problems.json
    │   └── textvqa
    │       ├── eval.py
    │       ├── llava_textvqa_val_v051_ocr.jsonl
    │       ├── m4c_evaluator.py
    │       ├── TextVQA_0.5.1_val.json
    │       └── train_images -> /path/to/your/textvqa/train_images
    ├── finetune_data
    │    ├── llava_v1_5_mix665k.json
    │    ├── coco
    │    │   └── train2017
    │    ├── gqa
    │    │   └── images
    │    ├── ocr_vqa
    │    │   └── images
    │    ├── textvqa
    │    │   └── train_images
    │    └── vg
    │        ├── VG_100K
    │        └── VG_100K_2
    ├── pretrain_data
    │    ├── images
    │    └── blip_laion_cc_sbu_558k.json
    ```
    </details>

#### 3️⃣ Run everything with one click!
```shell
LANGUAGE_MODEL=/path/to/your/MobileLLaMA-1.4B-Chat  # or 2.7B
VISION_MODEL=/path/to/your/clip-vit-large-patch14-336
bash run.sh mobilevlm1.7b pretrain-finetune-test ${LANGUAGE_MODEL} ${VISION_MODEL}

# (test-only) bash run.sh mobilevlm1.7b test /path/to/your/own/checkpoint
# (3B) bash run.sh mobilevlm3b pretrain-finetune-test ${LANGUAGE_MODEL} ${VISION_MODEL}
```

- Note 🧭: We place all running commands in `run.sh` so they can be run with one click for simplification. If you would like to modify some super-parameters to observe their impact, please dive into `run.sh` to explore.

## <h2 id="deployment-on-mobile-devices">📲 Deployment on Mobile Devices </h2>
**MobileVLM** now is officially supported by `llama.cpp`. We are looking for more cooperation with open-source communities on the deployment of mobile devices.
- [llama.cpp](https://github.com/ggerganov/llama.cpp): the repository of official `llama.cpp`. Step-by-step deployment instructions are provided [here](https://github.com/ggerganov/llama.cpp/blob/master/examples/llava/MobileVLM-README.md).
## 🤝 Acknowledgments

- [LLaVA](https://github.com/haotian-liu/LLaVA): the codebase we built upon. Thanks for their wonderful work! 👏
- [Vicuna](https://github.com/lm-sys/FastChat): the amazing open-sourced large language model!
- [llama.cpp](https://github.com/ggerganov/llama.cpp): the great open-sourced framework for the inference of LLaMA model in pure C/C++!


## ✏️ Reference

If you find MobileVLM or MobileLLaMA useful in your research or applications, please consider giving a star ⭐ and citing using the following BibTeX:
```
@article{chu2023mobilevlm,
  title={Mobilevlm: A fast, reproducible and strong vision language assistant for mobile devices},
  author={Chu, Xiangxiang and Qiao, Limeng and Lin, Xinyang and Xu, Shuang and Yang, Yang and Hu, Yiming and Wei, Fei and Zhang, Xinyu and Zhang, Bo and Wei, Xiaolin and others},
  journal={arXiv preprint arXiv:2312.16886},
  year={2023}
}
```


## 🌟 Star History
<picture>
  <source
    media="(prefers-color-scheme: dark)"
    srcset="
      https://api.star-history.com/svg?repos=Meituan-AutoML/MobileVLM&type=Date&theme=dark
    "
  />

  <source
    media="(prefers-color-scheme: light)"
    srcset="
      https://api.star-history.com/svg?repos=Meituan-AutoML/MobileVLM&type=Date
    "
  />

  <img
    alt="Star History Chart"
    src="https://api.star-history.com/svg?repos=Meituan-AutoML/MobileVLM&type=Date"
  />
</picture>

